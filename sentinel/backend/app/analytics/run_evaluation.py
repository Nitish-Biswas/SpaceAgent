"""run_evaluation.py

Runs all checkpoint candidates through the SENTINEL evaluator and writes
per-candidate result JSONs into an ``evaluation/`` folder.

Candidate discovery
-------------------
The script looks for candidate JSONL files in two ways:

1. **Named files** listed in CANDIDATE_FILES below (edit this list to add
   new checkpoints).  Each entry is a dict::

       {
           "name":  "sentinel_phi_v1_ckpt16",   # used as output filename stem
           "path":  "sentinel_phi_v1_ckpt16.jsonl",
           "latency_col": "latency_ms",          # optional: key in each record
       }

2. **Auto-discovery**: any ``*.jsonl`` file in the working directory whose
   name is not in the exclusion list is picked up automatically with a
   sanitised name derived from its filename.

JSONL format expected from each candidate file
-----------------------------------------------
Each line must be a JSON object with at minimum::

    {
        "true_fault_type": "EPS_SOLAR_UNDERVOLT",   # ground-truth label
        "response":        "<raw assistant JSON string>",
        "latency_ms":      312.5                 # optional float
    }

Alternatively the file may follow the *training* format (messages list), in
which case the evaluator extracts ``true_fault_type`` from the user message
and ``response`` from the assistant message automatically.

Output
------
``evaluation/<candidate_name>_results.json`` for each candidate, plus
``evaluation/comparison_summary.json`` with all candidates ranked by
fault_class_accuracy.

Usage
-----
    python run_evaluation.py [--candidates-dir DIR] [--output-dir DIR] [--verbose]
"""

import argparse
import json
import math
import os
import re
import sys
import time
from typing import Any, Optional

from evaluator import (
    SentinelEvaluator,
    GROUND_TRUTH_REGISTRY,
    DEMO_FAULT_TYPES,
)

# ---------------------------------------------------------------------------
# Candidate registry — edit this list to add / remove checkpoints
# ---------------------------------------------------------------------------

CANDIDATE_FILES: list[dict[str, Any]] = [
    # Example entries — add your own checkpoint files here:
    # {
    #     "name": "base_gemini",
    #     "path": "base_gemini_responses.jsonl",
    #     "latency_col": "latency_ms",
    # },
    # {
    #     "name": "sentinel_phi_v1_ckpt16",
    #     "path": "sentinel_phi_v1_ckpt16_responses.jsonl",
    #     "latency_col": "latency_ms",
    # },
]

# JSONL files to skip during auto-discovery (training data, split files, etc.)
AUTO_DISCOVERY_EXCLUDE: set[str] = {
    "sentinel_training.jsonl",
    "train.jsonl",
    "valid.jsonl",
    "test_output.jsonl",
}

# Fault-type keywords used to extract the label from a crash dump user message
_FAULT_TYPE_RE = re.compile(
    r"\b(EPS_SOLAR_UNDERVOLT|ADCS_GYRO_SEU|OBC_WATCHDOG_OVERFLOW"
    r"|TCS_THERMAL_RUNAWAY|COMMS_TRANSPONDER_LOSS|MULTI_CASCADE)\b"
)

# Fault-register → fault-type mapping (fallback when no label field present)
_FAULT_REGISTER_MAP: dict[str, str] = {
    "0x00000002": "EPS_SOLAR_UNDERVOLT",
    "0x00000042": "ADCS_GYRO_SEU",
    "0x00000040": "OBC_WATCHDOG_OVERFLOW",
    "0x00000010": "TCS_THERMAL_RUNAWAY",
    "0x00000008": "COMMS_TRANSPONDER_LOSS",
    "0x00000046": "MULTI_CASCADE",
}

# ---------------------------------------------------------------------------
# JSONL parsing helpers
# ---------------------------------------------------------------------------

def _extract_fault_type(record: dict) -> Optional[str]:
    """Try to determine the ground-truth fault type from a record."""

    # 1. Explicit field
    for key in ("true_fault_type", "fault_type", "label", "ground_truth"):
        if key in record and record[key] in GROUND_TRUTH_REGISTRY:
            return record[key]

    # 2. Scan the "messages" list (training / chat format)
    for msg in record.get("messages", []):
        content = msg.get("content", "")
        # Check assistant content for a JSON root_cause
        if msg.get("role") == "assistant":
            try:
                parsed = json.loads(content)
                hyps = parsed.get("hypotheses", [])
                rank1 = next((h for h in hyps if h.get("rank") == 1), None)
                if rank1 and rank1.get("root_cause") in GROUND_TRUTH_REGISTRY:
                    return rank1["root_cause"]
            except (json.JSONDecodeError, TypeError):
                pass
        # Check user content for fault register line
        if msg.get("role") == "user":
            fr_match = re.search(r"Fault Register:\s*(0x[0-9A-Fa-f]+)", content)
            if fr_match:
                ft = _FAULT_REGISTER_MAP.get(fr_match.group(1).lower())
                if ft:
                    return ft
            # Also scan for fault type string directly
            ft_match = _FAULT_TYPE_RE.search(content)
            if ft_match:
                return ft_match.group(1)

    # 3. Fault register field directly
    fr = record.get("fault_register", "")
    if fr in _FAULT_REGISTER_MAP:
        return _FAULT_REGISTER_MAP[fr]

    return None


def _extract_response(record: dict) -> Optional[str]:
    """Extract the raw model response string from a record."""

    # 1. Explicit field
    for key in ("response", "assistant_response", "output", "completion"):
        if key in record:
            val = record[key]
            return val if isinstance(val, str) else json.dumps(val)

    # 2. Messages list — take the assistant turn
    for msg in record.get("messages", []):
        if msg.get("role") == "assistant":
            return msg.get("content", "")

    return None


def _extract_latency(record: dict, latency_col: str = "latency_ms") -> Optional[float]:
    """Try to read latency from the record."""
    val = record.get(latency_col) or record.get("latency_ms") or record.get("latency")
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Candidate discovery
# ---------------------------------------------------------------------------

def _sanitise_name(path: str) -> str:
    """Turn a file path into a safe candidate name."""
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = re.sub(r"[^A-Za-z0-9_\-]", "_", stem)
    # Strip common suffixes like "_responses", "_outputs"
    stem = re.sub(r"_(responses?|outputs?|results?)$", "", stem, flags=re.IGNORECASE)
    return stem


def discover_candidates(candidates_dir: str = ".") -> list[dict[str, Any]]:
    """Return the merged list of registered + auto-discovered candidates."""
    known_paths = {c["path"] for c in CANDIDATE_FILES}
    auto: list[dict[str, Any]] = []

    for fname in sorted(os.listdir(candidates_dir)):
        if not fname.endswith(".jsonl"):
            continue
        if fname in AUTO_DISCOVERY_EXCLUDE:
            continue
        full = os.path.join(candidates_dir, fname)
        if full in known_paths or fname in known_paths:
            continue
        auto.append({
            "name": _sanitise_name(fname),
            "path": full,
            "latency_col": "latency_ms",
        })

    return CANDIDATE_FILES + auto


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def evaluate_candidate(
    candidate: dict[str, Any],
    output_dir: str,
    verbose: bool = False,
) -> Optional[dict[str, Any]]:
    """Run one candidate file through the evaluator and write results.

    Returns the report dict, or None if the file could not be read.
    """
    name        = candidate["name"]
    path        = candidate["path"]
    latency_col = candidate.get("latency_col", "latency_ms")

    if not os.path.exists(path):
        print(f"  [SKIP] {name}: file not found — {path}")
        return None

    evaluator = SentinelEvaluator(name)
    n_skipped  = 0
    n_total    = 0

    with open(path, "r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            n_total += 1

            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                if verbose:
                    print(f"    Line {line_no}: JSON parse error — {exc}")
                # Count as a malformed response; we need a fault type to score it.
                # Without one we skip it entirely.
                n_skipped += 1
                continue

            fault_type = _extract_fault_type(record)
            response   = _extract_response(record)
            latency    = _extract_latency(record, latency_col)

            if fault_type is None:
                if verbose:
                    print(f"    Line {line_no}: could not determine fault type — skipped")
                n_skipped += 1
                continue

            if response is None:
                if verbose:
                    print(f"    Line {line_no}: could not extract response — treating as malformed")
                response = ""   # scored as malformed / invalid JSON

            evaluator.add(response, fault_type, latency_ms=latency)

    if evaluator._scores == []:
        print(f"  [WARN] {name}: no scorable responses found in {path}")
        return None

    report = evaluator.report()
    report["source_file"]  = path
    report["lines_total"]  = n_total
    report["lines_skipped"] = n_skipped

    # Write individual result file
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{name}_results.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    _print_summary(report, out_path, verbose)
    return report


def _print_summary(report: dict, out_path: str, verbose: bool) -> None:
    """Print a one-line (or verbose) summary for a completed candidate."""
    c   = report["candidate"]
    n   = report["n_samples"]
    acc = report["fault_class_accuracy"]
    cal = report["confidence_calibration"]
    rec = report["recovery_plan_adequacy"]
    jvr = report["json_validity_rate"]
    mfr = report["retry_malformed_rate"]
    lat = report.get("mean_latency_ms")
    demo= report["demo_scenario_success_rate"]
    rhr = report["requires_human_review_correct"]

    lat_str = f"{lat:.1f}ms" if lat is not None else "N/A"

    print(f"  [{c}]  n={n}  acc={acc:.3f}  cal={cal:.3f}  "
          f"rec={rec:.3f}  jvr={jvr:.3f}  mfr={mfr:.3f}  "
          f"rhr={rhr:.3f}  demo={demo:.3f}  lat={lat_str}")
    print(f"    → {out_path}")

    if verbose:
        print("    Per-fault breakdown:")
        for ft, ft_stats in report.get("per_fault_type", {}).items():
            print(f"      {ft:<25s}  acc={ft_stats['fault_class_accuracy']:.2f}  "
                  f"rec={ft_stats['recovery_plan_adequacy']:.2f}  "
                  f"demo={ft_stats['demo_scenario_pass_rate']:.2f}")


def write_comparison_summary(
    reports: list[dict[str, Any]],
    output_dir: str,
) -> None:
    """Write a ranked comparison of all candidates to comparison_summary.json."""
    if not reports:
        return

    # Rank by fault_class_accuracy descending, then by demo_scenario_success_rate
    ranked = sorted(
        reports,
        key=lambda r: (r["fault_class_accuracy"], r["demo_scenario_success_rate"]),
        reverse=True,
    )

    summary = {
        "total_candidates": len(ranked),
        "ranking": [
            {
                "rank":                          i + 1,
                "candidate":                     r["candidate"],
                "n_samples":                     r["n_samples"],
                "fault_class_accuracy":          r["fault_class_accuracy"],
                "confidence_calibration":        r["confidence_calibration"],
                "requires_human_review_correct": r["requires_human_review_correct"],
                "recovery_plan_adequacy":        r["recovery_plan_adequacy"],
                "json_validity_rate":            r["json_validity_rate"],
                "retry_malformed_rate":          r["retry_malformed_rate"],
                "mean_latency_ms":               r.get("mean_latency_ms"),
                "demo_scenario_success_rate":    r["demo_scenario_success_rate"],
            }
            for i, r in enumerate(ranked)
        ],
    }

    out_path = os.path.join(output_dir, "comparison_summary.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nComparison summary written to {out_path}")


# ---------------------------------------------------------------------------
# Synthetic self-evaluation using training data (fallback / smoke test)
# ---------------------------------------------------------------------------

def _run_self_eval(output_dir: str, verbose: bool) -> Optional[dict]:
    """Score sentinel_training.jsonl against itself as a smoke test.

    This gives a ~perfect baseline (ground-truth responses evaluated against
    ground-truth labels) to confirm the evaluator is working correctly.
    """
    training_file = "sentinel_training.jsonl"
    if not os.path.exists(training_file):
        return None

    print(f"\nRunning self-evaluation on {training_file} ...")
    candidate = {
        "name":        "ground_truth_baseline",
        "path":        training_file,
        "latency_col": "latency_ms",
    }
    return evaluate_candidate(candidate, output_dir, verbose=verbose)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SENTINEL checkpoint candidates through the evaluator."
    )
    parser.add_argument(
        "--candidates-dir",
        default=".",
        help="Directory to scan for candidate JSONL files (default: current dir)",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation",
        help="Directory to write result JSONs (default: evaluation/)",
    )
    parser.add_argument(
        "--self-eval",
        action="store_true",
        default=True,
        help="Also score sentinel_training.jsonl as a ground-truth baseline (default: on)",
    )
    parser.add_argument(
        "--no-self-eval",
        dest="self_eval",
        action="store_false",
        help="Skip the ground-truth self-evaluation",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-fault-type breakdowns",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    candidates = discover_candidates(args.candidates_dir)
    reports: list[dict] = []

    # Ground-truth self-evaluation first
    if args.self_eval:
        r = _run_self_eval(args.output_dir, args.verbose)
        if r:
            reports.append(r)

    # Registered + auto-discovered candidates
    if candidates:
        print(f"\nEvaluating {len(candidates)} candidate(s) ...")
        for cand in candidates:
            r = evaluate_candidate(cand, args.output_dir, verbose=args.verbose)
            if r:
                reports.append(r)
    else:
        if not args.self_eval:
            print(
                "No candidate JSONL files found.  Add entries to CANDIDATE_FILES in "
                "run_evaluation.py, or drop response JSONL files into the working directory."
            )

    if reports:
        write_comparison_summary(reports, args.output_dir)
        print(f"\n{len(reports)} result file(s) written to {args.output_dir}/")
    else:
        print("No results to write.")
        sys.exit(1)


if __name__ == "__main__":
    main()
