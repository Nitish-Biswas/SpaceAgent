"""evaluator.py

Core evaluation module for the SENTINEL LLM agent.

Provides:
  - GROUND_TRUTH_REGISTRY  : canonical labels for all 6 fault types
  - SentinelEvaluator      : computes 8 metrics against a set of model responses
  - evaluate_response()    : score a single response dict

Metrics
-------
1. fault_class_accuracy         — rank-1 root_cause matches ground truth
2. confidence_calibration       — |predicted_confidence - gt_confidence| (lower = better)
3. requires_human_review_correct— requires_human_review flag matches expected value
4. recovery_plan_adequacy       — fraction of GT recovery keywords covered
5. json_validity_rate           — response parsed as valid JSON
6. retry_malformed_rate         — fraction of responses that were malformed / needed retry
7. mean_latency_ms              — average inference latency in milliseconds
8. demo_scenario_success_rate   — all 6 canonical demo scenarios passed simultaneously

No external dependencies — stdlib only.
"""

import json
import math
import statistics
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Ground-truth registry
# ---------------------------------------------------------------------------

GROUND_TRUTH_REGISTRY: dict[str, dict[str, Any]] = {
    "EPS_SOLAR_UNDERVOLT": {
        "root_cause":              "EPS_SOLAR_UNDERVOLT",
        "subsystem":               "EPS",
        "confidence":              0.95,
        "risk_level":              "HIGH",
        "requires_human_review":   True,   # HIGH risk → review required
        "recovery_keywords": [
            "solar array", "sun", "V_bat", "SoC", "battery",
            "array", "orientation", "voltage", "load shedding",
        ],
        "causal_keywords": [
            "I_sa", "solar array", "battery", "V_bat", "SoC",
            "undervoltage", "safe mode",
        ],
    },
    "ADCS_GYRO_SEU": {
        "root_cause":              "ADCS_GYRO_SEU",
        "subsystem":               "ADCS",
        "confidence":              0.92,
        "risk_level":              "HIGH",
        "requires_human_review":   True,
        "recovery_keywords": [
            "gyro", "gyroscope", "power-cycle", "reset", "reaction wheel",
            "desaturate", "attitude", "star tracker", "SEU",
        ],
        "causal_keywords": [
            "SEU", "gyro", "NaN", "attitude", "reaction wheel",
            "saturation", "safe mode",
        ],
    },
    "OBC_WATCHDOG_OVERFLOW": {
        "root_cause":              "OBC_WATCHDOG_OVERFLOW",
        "subsystem":               "OBC",
        "confidence":              0.90,
        "risk_level":              "MEDIUM",
        "requires_human_review":   False,  # MEDIUM risk + confidence >= 0.70
        "recovery_keywords": [
            "watchdog", "reboot", "reset", "memory", "CPU",
            "leak", "software", "patch", "upload",
        ],
        "causal_keywords": [
            "memory leak", "CPU", "watchdog", "overflow",
            "scheduler", "heap", "reset",
        ],
    },
    "TCS_THERMAL_RUNAWAY": {
        "root_cause":              "TCS_THERMAL_RUNAWAY",
        "subsystem":               "TCS",
        "confidence":              0.88,
        "risk_level":              "HIGH",
        "requires_human_review":   True,
        "recovery_keywords": [
            "heater", "temperature", "thermal", "disable",
            "override", "temp", "component",
        ],
        "causal_keywords": [
            "heater", "temperature", "75", "runaway",
            "thermal", "stuck", "flag",
        ],
    },
    "COMMS_TRANSPONDER_LOSS": {
        "root_cause":              "COMMS_TRANSPONDER_LOSS",
        "subsystem":               "COMMS",
        "confidence":              0.85,
        "risk_level":              "MEDIUM",
        "requires_human_review":   False,
        "recovery_keywords": [
            "transponder", "SNR", "antenna", "lock", "uplink",
            "omni", "pointing", "ground station",
        ],
        "causal_keywords": [
            "transponder", "lock", "SNR", "antenna",
            "attitude", "star tracker", "blind",
        ],
    },
    "MULTI_CASCADE": {
        "root_cause":              "MULTI_CASCADE",
        "subsystem":               "MULTI",
        "confidence":              0.65,
        "risk_level":              "HIGH",
        "requires_human_review":   True,   # LOW confidence + HIGH risk
        "recovery_keywords": [
            "gyro", "gyroscope", "sun", "solar array", "heater",
            "attitude", "battery", "SoC", "cascade", "subsystem",
        ],
        "causal_keywords": [
            "SEU", "gyro", "attitude", "solar array", "I_sa",
            "battery", "heater", "cascade",
        ],
    },
}

# The six canonical demo scenarios (one per fault type, used for metric 8)
DEMO_FAULT_TYPES: list[str] = list(GROUND_TRUTH_REGISTRY.keys())

# Required top-level keys in a SENTINEL response
REQUIRED_RESPONSE_KEYS: set[str] = {
    "hypotheses",
    "recovery_plan",
    "confidence",
    "requires_human_review",
    "reasoning_summary",
}


# ---------------------------------------------------------------------------
# Per-response scoring helpers
# ---------------------------------------------------------------------------

def _is_valid_json(raw: str) -> tuple[bool, Optional[dict]]:
    """Try to parse *raw* as JSON. Returns (valid, parsed_dict)."""
    try:
        obj = json.loads(raw)
        return True, obj
    except (json.JSONDecodeError, TypeError):
        return False, None


def _keyword_coverage(text: str, keywords: list[str]) -> float:
    """Fraction of *keywords* that appear (case-insensitive) in *text*."""
    if not keywords:
        return 1.0
    text_lower = text.lower()
    hit = sum(1 for kw in keywords if kw.lower() in text_lower)
    return hit / len(keywords)


def _recovery_text(parsed: dict) -> str:
    """Concatenate all rationale strings from the recovery_plan."""
    parts: list[str] = []
    for step in parsed.get("recovery_plan", []):
        parts.append(step.get("command", ""))
        parts.append(step.get("rationale", ""))
        parts.append(step.get("verify", ""))
    return " ".join(parts)


def evaluate_response(
    raw_response: str,
    true_fault_type: str,
    latency_ms: Optional[float] = None,
) -> dict[str, Any]:
    """Score a single SENTINEL response string against ground truth.

    Parameters
    ----------
    raw_response : str
        The raw assistant content string (should be JSON).
    true_fault_type : str
        The ground-truth fault type key from GROUND_TRUTH_REGISTRY.
    latency_ms : float, optional
        Inference latency in milliseconds (None if unknown).

    Returns
    -------
    dict
        Per-response score dict with keys:
        ``json_valid``, ``malformed``, ``fault_class_correct``,
        ``confidence_error``, ``review_flag_correct``,
        ``recovery_coverage``, ``latency_ms``,
        ``demo_scenario_pass``, ``parsed`` (the parsed dict or None).
    """
    gt = GROUND_TRUTH_REGISTRY[true_fault_type]

    json_valid, parsed = _is_valid_json(raw_response)
    malformed = not json_valid

    if not json_valid or parsed is None:
        return {
            "json_valid":           False,
            "malformed":            True,
            "fault_class_correct":  False,
            "confidence_error":     1.0,     # worst-case
            "review_flag_correct":  False,
            "recovery_coverage":    0.0,
            "latency_ms":           latency_ms,
            "demo_scenario_pass":   False,
            "parsed":               None,
        }

    # ---- Metric 1: fault class accuracy ----
    hypotheses = parsed.get("hypotheses", [])
    rank1 = next((h for h in hypotheses if h.get("rank") == 1), None)
    if rank1 is None and hypotheses:
        rank1 = hypotheses[0]
    predicted_class = rank1.get("root_cause", "") if rank1 else ""
    fault_class_correct = (predicted_class == gt["root_cause"])

    # ---- Metric 2: confidence calibration ----
    predicted_conf = float(parsed.get("confidence", 0.0))
    confidence_error = abs(predicted_conf - gt["confidence"])

    # ---- Metric 3: requires_human_review correctness ----
    predicted_review = parsed.get("requires_human_review", False)
    # Also accept as correct if confidence < 0.70 triggers review
    expected_review = gt["requires_human_review"] or (predicted_conf < 0.70)
    review_flag_correct = (bool(predicted_review) == bool(expected_review))

    # ---- Metric 4: recovery plan adequacy ----
    recovery_text = _recovery_text(parsed)
    recovery_coverage = _keyword_coverage(recovery_text, gt["recovery_keywords"])

    # ---- Metric 8: demo scenario pass ----
    # A demo scenario passes if: JSON valid, fault class correct, review flag
    # correct, and recovery coverage >= 0.25 (at least some relevant steps).
    demo_pass = (
        json_valid
        and fault_class_correct
        and review_flag_correct
        and recovery_coverage >= 0.25
    )

    return {
        "json_valid":           True,
        "malformed":            False,
        "fault_class_correct":  fault_class_correct,
        "confidence_error":     round(confidence_error, 4),
        "review_flag_correct":  review_flag_correct,
        "recovery_coverage":    round(recovery_coverage, 4),
        "latency_ms":           latency_ms,
        "demo_scenario_pass":   demo_pass,
        "parsed":               parsed,
    }


# ---------------------------------------------------------------------------
# Aggregate evaluator
# ---------------------------------------------------------------------------

class SentinelEvaluator:
    """Evaluates a batch of SENTINEL model responses across all 8 metrics.

    Parameters
    ----------
    candidate_name : str
        Human-readable name for the model/checkpoint being evaluated
        (used in the output report).

    Usage
    -----
    >>> ev = SentinelEvaluator("my_model")
    >>> ev.add(raw_json_str, "EPS_SOLAR_UNDERVOLT", latency_ms=312.0)
    >>> report = ev.report()
    """

    def __init__(self, candidate_name: str) -> None:
        self.candidate_name = candidate_name
        self._scores: list[dict[str, Any]] = []

    def add(
        self,
        raw_response: str,
        true_fault_type: str,
        latency_ms: Optional[float] = None,
    ) -> dict[str, Any]:
        """Score one response and accumulate it.

        Returns the per-response score dict (also stored internally).
        """
        score = evaluate_response(raw_response, true_fault_type, latency_ms)
        score["true_fault_type"] = true_fault_type
        self._scores.append(score)
        return score

    def report(self) -> dict[str, Any]:
        """Compute and return the aggregated 8-metric report.

        Returns
        -------
        dict
            ``{
              "candidate":                     str,
              "n_samples":                     int,
              "fault_class_accuracy":          float,   # metric 1
              "confidence_calibration":        float,   # metric 2 (mean |error|, lower=better)
              "requires_human_review_correct": float,   # metric 3
              "recovery_plan_adequacy":        float,   # metric 4
              "json_validity_rate":            float,   # metric 5
              "retry_malformed_rate":          float,   # metric 6
              "mean_latency_ms":               float | None,  # metric 7
              "demo_scenario_success_rate":    float,   # metric 8
              "per_fault_type":                dict,    # breakdown by fault type
              "raw_scores":                    list,    # all per-response dicts
            }``
        """
        n = len(self._scores)
        if n == 0:
            return {
                "candidate": self.candidate_name,
                "n_samples":  0,
                "fault_class_accuracy":          0.0,
                "confidence_calibration":        1.0,
                "requires_human_review_correct": 0.0,
                "recovery_plan_adequacy":        0.0,
                "json_validity_rate":            0.0,
                "retry_malformed_rate":          1.0,
                "mean_latency_ms":               None,
                "demo_scenario_success_rate":    0.0,
                "per_fault_type":                {},
                "raw_scores":                    [],
            }

        # ---- Aggregate all 8 metrics ----
        m1_correct    = sum(s["fault_class_correct"] for s in self._scores)
        m2_cal_errors = [s["confidence_error"] for s in self._scores]
        m3_correct    = sum(s["review_flag_correct"] for s in self._scores)
        m4_coverages  = [s["recovery_coverage"] for s in self._scores]
        m5_valid      = sum(s["json_valid"] for s in self._scores)
        m6_malformed  = sum(s["malformed"] for s in self._scores)
        latencies     = [s["latency_ms"] for s in self._scores if s["latency_ms"] is not None]

        # Metric 8: demo scenario success rate.
        # For each of the 6 canonical fault types, all responses for that type
        # must pass for the scenario to count as successful.
        demo_pass_by_type: dict[str, list[bool]] = {ft: [] for ft in DEMO_FAULT_TYPES}
        for s in self._scores:
            ft = s["true_fault_type"]
            if ft in demo_pass_by_type:
                demo_pass_by_type[ft].append(s["demo_scenario_pass"])

        demo_scenario_successes = sum(
            1 for ft in DEMO_FAULT_TYPES
            if demo_pass_by_type[ft] and all(demo_pass_by_type[ft])
        )
        demo_scenarios_possible = sum(
            1 for ft in DEMO_FAULT_TYPES if demo_pass_by_type[ft]
        )
        m8_demo_rate = (
            demo_scenario_successes / demo_scenarios_possible
            if demo_scenarios_possible > 0
            else 0.0
        )

        # ---- Per-fault-type breakdown ----
        per_fault: dict[str, dict] = {}
        for ft in DEMO_FAULT_TYPES:
            ft_scores = [s for s in self._scores if s["true_fault_type"] == ft]
            if not ft_scores:
                continue
            nft = len(ft_scores)
            per_fault[ft] = {
                "n":                        nft,
                "fault_class_accuracy":     sum(s["fault_class_correct"] for s in ft_scores) / nft,
                "confidence_calibration":   statistics.mean(s["confidence_error"] for s in ft_scores),
                "review_flag_correct":      sum(s["review_flag_correct"] for s in ft_scores) / nft,
                "recovery_plan_adequacy":   statistics.mean(s["recovery_coverage"] for s in ft_scores),
                "json_validity_rate":       sum(s["json_valid"] for s in ft_scores) / nft,
                "demo_scenario_pass_rate":  sum(s["demo_scenario_pass"] for s in ft_scores) / nft,
            }

        # Strip the heavy "parsed" field from raw_scores to keep JSON compact
        raw_scores_stripped = [
            {k: v for k, v in s.items() if k != "parsed"}
            for s in self._scores
        ]

        return {
            "candidate":                     self.candidate_name,
            "n_samples":                     n,
            "fault_class_accuracy":          round(m1_correct / n, 4),
            "confidence_calibration":        round(statistics.mean(m2_cal_errors), 4),
            "requires_human_review_correct": round(m3_correct / n, 4),
            "recovery_plan_adequacy":        round(statistics.mean(m4_coverages), 4),
            "json_validity_rate":            round(m5_valid / n, 4),
            "retry_malformed_rate":          round(m6_malformed / n, 4),
            "mean_latency_ms":               round(statistics.mean(latencies), 2) if latencies else None,
            "demo_scenario_success_rate":    round(m8_demo_rate, 4),
            "per_fault_type":                per_fault,
            "raw_scores":                    raw_scores_stripped,
        }
