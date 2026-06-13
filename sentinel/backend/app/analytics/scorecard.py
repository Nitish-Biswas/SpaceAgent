"""scorecard.py

Reads all *_results.json files from the evaluation/ folder and prints a
formatted comparison scorecard table — the basis for choosing the final model.

Columns
-------
  Candidate           model / checkpoint name
  Acc                 held-out fault-class accuracy        (higher is better)
  JSON%               JSON validity rate                   (higher is better)
  Recovery            recovery-plan adequacy               (higher is better)
  Demo                demo scenario success rate           (higher is better)
  Latency             mean inference latency (ms)          (lower is better, N/A if unknown)
  Decision            DEPLOY / REVIEW / REJECT  (see rules below)

Decision rules
--------------
  DEPLOY  — Acc >= 0.90 AND JSON% == 1.00 AND Demo >= 0.90
  REVIEW  — Acc >= 0.75 AND JSON% >= 0.95 (meets minimum bar but not deploy threshold)
  REJECT  — anything below REVIEW threshold

Usage
-----
    python scorecard.py                        # reads evaluation/ in current dir
    python scorecard.py --dir path/to/eval     # custom folder
    python scorecard.py --md                   # also emit a Markdown table
"""

import argparse
import json
import os
import sys

# ---------------------------------------------------------------------------
# Decision thresholds
# ---------------------------------------------------------------------------

DEPLOY_ACC      = 0.90
DEPLOY_JSON     = 1.00
DEPLOY_DEMO     = 0.90
REVIEW_ACC      = 0.75
REVIEW_JSON     = 0.95


def _decision(acc: float, json_valid: float, demo: float) -> str:
    if acc >= DEPLOY_ACC and json_valid >= DEPLOY_JSON and demo >= DEPLOY_DEMO:
        return "DEPLOY"
    if acc >= REVIEW_ACC and json_valid >= REVIEW_JSON:
        return "REVIEW"
    return "REJECT"


def _decision_symbol(decision: str) -> str:
    return {"DEPLOY": "✓ DEPLOY", "REVIEW": "~ REVIEW", "REJECT": "✗ REJECT"}[decision]


# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------

def load_results(eval_dir: str) -> list[dict]:
    """Read all *_results.json files from *eval_dir*. Returns list of dicts."""
    rows = []
    if not os.path.isdir(eval_dir):
        print(f"Error: directory '{eval_dir}' not found.", file=sys.stderr)
        sys.exit(1)

    for fname in sorted(os.listdir(eval_dir)):
        if not fname.endswith("_results.json"):
            continue
        path = os.path.join(eval_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                rows.append(json.load(fh))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Warning: could not read {path} — {exc}", file=sys.stderr)

    return rows


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------

# Column widths (minimum)
_COL_WIDTHS = {
    "candidate": 30,
    "acc":        8,
    "json":       8,
    "recovery":   10,
    "demo":       8,
    "latency":    10,
    "decision":   10,
}

_HEADERS = {
    "candidate": "Candidate",
    "acc":       "Acc",
    "json":      "JSON%",
    "recovery":  "Recovery",
    "demo":      "Demo",
    "latency":   "Latency",
    "decision":  "Decision",
}


def _fmt_pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _fmt_lat(v) -> str:
    if v is None:
        return "N/A"
    return f"{v:.0f} ms"


def _row_cells(r: dict) -> dict[str, str]:
    acc      = r.get("fault_class_accuracy",      0.0)
    json_v   = r.get("json_validity_rate",         0.0)
    recovery = r.get("recovery_plan_adequacy",     0.0)
    demo     = r.get("demo_scenario_success_rate", 0.0)
    latency  = r.get("mean_latency_ms")
    decision = _decision(acc, json_v, demo)
    return {
        "candidate": r.get("candidate", "unknown"),
        "acc":       _fmt_pct(acc),
        "json":      _fmt_pct(json_v),
        "recovery":  _fmt_pct(recovery),
        "demo":      _fmt_pct(demo),
        "latency":   _fmt_lat(latency),
        "decision":  _decision_symbol(decision),
    }


def _col_width(key: str, all_rows: list[dict[str, str]]) -> int:
    """Return the max of the header width, minimum width, and all data widths."""
    return max(
        _COL_WIDTHS[key],
        len(_HEADERS[key]),
        *(len(row[key]) for row in all_rows),
    )


def print_table(rows: list[dict], emit_md: bool = False) -> None:
    """Print the scorecard table to stdout, sorted by Acc descending."""

    # Sort: DEPLOY first, then REVIEW, then REJECT; within each group by Acc desc
    _order = {"DEPLOY": 0, "REVIEW": 1, "REJECT": 2}
    rows_sorted = sorted(
        rows,
        key=lambda r: (
            _order[_decision(
                r.get("fault_class_accuracy", 0.0),
                r.get("json_validity_rate", 0.0),
                r.get("demo_scenario_success_rate", 0.0),
            )],
            -r.get("fault_class_accuracy", 0.0),
        ),
    )

    cells = [_row_cells(r) for r in rows_sorted]
    keys  = list(_HEADERS.keys())
    widths = {k: _col_width(k, cells) for k in keys}

    # ---- Plain-text table ----
    sep  = "+-" + "-+-".join("-" * widths[k] for k in keys) + "-+"
    head = "| " + " | ".join(_HEADERS[k].ljust(widths[k]) for k in keys) + " |"

    print()
    print("SENTINEL Checkpoint Scorecard")
    print("=" * len(sep))
    print(sep)
    print(head)
    print(sep.replace("-", "="))

    for i, (row_data, row_cells) in enumerate(zip(rows_sorted, cells)):
        line = "| " + " | ".join(row_cells[k].ljust(widths[k]) for k in keys) + " |"
        print(line)
        if i < len(rows_sorted) - 1:
            print(sep)

    print(sep)

    # ---- Decision key ----
    print()
    print(f"  Decision rules:  "
          f"DEPLOY = Acc≥{DEPLOY_ACC*100:.0f}% & JSON={DEPLOY_JSON*100:.0f}% & Demo≥{DEPLOY_DEMO*100:.0f}%  |  "
          f"REVIEW = Acc≥{REVIEW_ACC*100:.0f}% & JSON≥{REVIEW_JSON*100:.0f}%  |  "
          f"REJECT = below REVIEW")
    print()

    # ---- Per-fault breakdown for DEPLOY/REVIEW candidates ----
    for row in rows_sorted:
        acc    = row.get("fault_class_accuracy", 0.0)
        json_v = row.get("json_validity_rate", 0.0)
        demo   = row.get("demo_scenario_success_rate", 0.0)
        dec    = _decision(acc, json_v, demo)
        if dec in ("DEPLOY", "REVIEW") and row.get("per_fault_type"):
            name = row.get("candidate", "unknown")
            print(f"  Per-fault breakdown — {name}:")
            pft_keys = ["fault_class_accuracy", "recovery_plan_adequacy", "demo_scenario_pass_rate"]
            pft_header = f"    {'Fault Type':<26} {'Acc':>6}  {'Recovery':>9}  {'Demo':>6}"
            print(pft_header)
            print("    " + "-" * (len(pft_header) - 4))
            for ft, stats in row["per_fault_type"].items():
                ft_acc  = stats.get("fault_class_accuracy", 0.0)
                ft_rec  = stats.get("recovery_plan_adequacy", 0.0)
                ft_demo = stats.get("demo_scenario_pass_rate", 0.0)
                print(f"    {ft:<26} {ft_acc*100:>5.1f}%  {ft_rec*100:>8.1f}%  {ft_demo*100:>5.1f}%")
            print()

    # ---- Markdown table (optional) ----
    if emit_md:
        _print_markdown(rows_sorted, cells, keys, widths)


def _print_markdown(rows_sorted, cells, keys, widths) -> None:
    """Emit a GitHub-flavoured Markdown table."""
    print("## SENTINEL Checkpoint Scorecard\n")

    # Header row
    md_header = "| " + " | ".join(_HEADERS[k] for k in keys) + " |"
    md_sep    = "| " + " | ".join(
        (":---:" if k not in ("candidate",) else ":---") for k in keys
    ) + " |"
    print(md_header)
    print(md_sep)

    for row_cells in cells:
        # Escape pipe characters in candidate names
        clean = {k: v.replace("|", "\\|") for k, v in row_cells.items()}
        print("| " + " | ".join(clean[k] for k in keys) + " |")

    print()
    print(f"> **Decision rules:** "
          f"✓ DEPLOY = Acc≥{DEPLOY_ACC*100:.0f}% & JSON={DEPLOY_JSON*100:.0f}% & Demo≥{DEPLOY_DEMO*100:.0f}%  "
          f"| ~ REVIEW = Acc≥{REVIEW_ACC*100:.0f}% & JSON≥{REVIEW_JSON*100:.0f}%  "
          f"| ✗ REJECT = below REVIEW\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print a comparison scorecard for SENTINEL checkpoint results."
    )
    parser.add_argument(
        "--dir", "-d",
        default="evaluation",
        help="Folder containing *_results.json files (default: evaluation/)",
    )
    parser.add_argument(
        "--md", "--markdown",
        action="store_true",
        help="Also emit a GitHub-flavoured Markdown table",
    )
    args = parser.parse_args()

    rows = load_results(args.dir)
    if not rows:
        print(f"No *_results.json files found in '{args.dir}'.")
        print("Run  python run_evaluation.py  first to generate results.")
        sys.exit(1)

    print_table(rows, emit_md=args.md)


if __name__ == "__main__":
    main()
