"""dataset_generator.py

Generates a synthetic fine-tuning dataset in JSONL format for training the
SENTINEL LLM agent.  Each line in the output file is a JSON object containing
a ``messages`` list with three turns: system, user (crash dump), and assistant
(ground-truth diagnosis).

Usage
-----
    python dataset_generator.py --samples 600 --output sentinel_training.jsonl --validate

Dependencies: fault_simulator (local), json, random, argparse, os (stdlib only).
"""

from simulation.fault_simulator import SatelliteFaultSimulator
import json
import random
import os
import argparse

# ---------------------------------------------------------------------------
# SENTINEL system prompt — embedded verbatim as a module-level constant
# ---------------------------------------------------------------------------

SENTINEL_SYSTEM_PROMPT = (
    "You are SENTINEL, an autonomous spacecraft fault diagnosis AI. When given a\n"
    "crash dump, you MUST reason step by step and output ONLY valid JSON.\n"
    "\n"
    "SATELLITE SUBSYSTEMS:\n"
    "- ADCS (Attitude): gyroscopes, star trackers, reaction wheels, thrusters\n"
    "- EPS (Power): solar arrays (I_sa, V_bat, V_bus, SoC%), battery packs\n"
    "- OBC (Computer): CPU load, watchdog counter, SEU counter, fault register, memory\n"
    "- COMMS (Radio): transponder lock, signal-to-noise ratio\n"
    "- TCS (Thermal): component temperatures, heater enable flags\n"
    "- PYLD (Payload): instruments, cameras, spectrometers\n"
    "\n"
    "NOMINAL THRESHOLDS:\n"
    "V_bat: 28.0-33.6V | Critical: <22V\n"
    "SoC: 20-100% | Critical: <15%\n"
    "I_sa: 0-12A | Anomaly: sudden drop to 0A in sunlight\n"
    "Gyro Rate: 0-7 deg/s | Anomaly: NaN or constant = sensor failure\n"
    "CPU Load: <70% nominal | Anomaly: sustained 100% = software loop\n"
    "Attitude Error: <0.01 deg | Anomaly: >5 deg sustained\n"
    "SEU Counter: 0 in nominal orbit | Anomaly: spike = cosmic ray hit\n"
    "\n"
    "FAULT SIGNATURES YOU MUST RECOGNIZE:\n"
    "1. SEU signature: sudden SEU_counter spike + anomaly in ONE specific subsystem\n"
    "-> Diagnosis: radiation-induced fault. Recovery: software restart (NOT hardware replacement)\n"
    "2. Power cascade: I_sa drops -> V_bat falls -> V_bus out of range -> safe mode\n"
    "-> Diagnosis: solar array or eclipse error. Recovery: verify sun angle, switch array\n"
    "3. Software loop: CPU=100% + watchdog overflow + memory monotonically increasing\n"
    "-> Near-certain OBC software fault. Recovery: controlled reboot\n"
    "4. ADCS tumble cascade: gyro NaN -> attitude error >5deg -> sun loss -> power loss -> thermal\n"
    "-> Check SEU counter first. Recovery: gyro reset or switch to backup gyro\n"
    "\n"
    "CRITICAL SAFETY RULES — NEVER VIOLATE:\n"
    "- Never command battery discharge below 15% SoC\n"
    "- Never command attitude maneuvers without first verifying gyro health\n"
    "- Never restart OBC without confirming comms lock on low-gain antenna\n"
    "- If any recovery step has risk level HIGH, set requires_human_review: true\n"
    "- If your confidence is below 0.70, set requires_human_review: true\n"
    "\n"
    'OUTPUT FORMAT — STRICT JSON, NO EXCEPTIONS:\n'
    '{"hypotheses": [{"rank": 1, "root_cause": "EPS_SOLAR_UNDERVOLT", "component": "SOLAR_ARRAY_A",\n'
    '"confidence": 0.88, "causal_chain": ["I_sa drops to 0A in sunlight",\n'
    '"battery begins draining", "V_bat falls to 24.1V", "EPS fault flag set",\n'
    '"safe mode triggered"]},\n'
    '{"rank": 2, ...},\n'
    '{"rank": 3, ...}],\n'
    '"recovery_plan": [{"step": 1, "command": "CMD_VERIFY_SUN_ANGLE", "rationale": "...",\n'
    '"wait_seconds": 10, "verify": "sun_sensor_angle < 90 deg", "risk": "LOW"},\n'
    '{"step": 2, "command": "CMD_SOLAR_ARRAY_A_RESET", ...}],\n'
    '"confidence": 0.88,\n'
    '"requires_human_review": false,\n'
    '"reasoning_summary": "..."}'
)

# ---------------------------------------------------------------------------
# All valid fault types (mirrors SatelliteFaultSimulator._VALID_FAULT_TYPES)
# ---------------------------------------------------------------------------

_FAULT_TYPES = [
    "EPS_SOLAR_UNDERVOLT",
    "ADCS_GYRO_SEU",
    "OBC_WATCHDOG_OVERFLOW",
    "TCS_THERMAL_RUNAWAY",
    "COMMS_TRANSPONDER_LOSS",
    "MULTI_CASCADE",
]

# Plausible alternative hypotheses for each fault type (rank-2 and rank-3).
# These represent physically adjacent failure modes that a real analyst might
# consider before ruling them out.
_ALT_HYPOTHESES: dict = {
    "EPS_SOLAR_UNDERVOLT":      ("MULTI_CASCADE", "COMMS_TRANSPONDER_LOSS"),
    "ADCS_GYRO_SEU":            ("MULTI_CASCADE", "OBC_WATCHDOG_OVERFLOW"),
    "OBC_WATCHDOG_OVERFLOW":    ("ADCS_GYRO_SEU",    "MULTI_CASCADE"),
    "TCS_THERMAL_RUNAWAY":      ("MULTI_CASCADE", "EPS_SOLAR_UNDERVOLT"),
    "COMMS_TRANSPONDER_LOSS":   ("ADCS_GYRO_SEU",    "MULTI_CASCADE"),
    "MULTI_CASCADE":            ("ADCS_GYRO_SEU",    "EPS_SOLAR_UNDERVOLT"),
}

# Recovery command prefixes derived from each fault type's subsystem.
_CMD_PREFIX: dict = {
    "EPS_SOLAR_UNDERVOLT":      "CMD_EPS",
    "ADCS_GYRO_SEU":            "CMD_ADCS",
    "OBC_WATCHDOG_OVERFLOW":    "CMD_OBC",
    "TCS_THERMAL_RUNAWAY":      "CMD_TCS",
    "COMMS_TRANSPONDER_LOSS":   "CMD_COMMS",
    "MULTI_CASCADE":            "CMD_MULTI",
}


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def format_crash_dump_as_prompt(crash_dump: dict) -> str:
    """Convert a crash dump dict into a human-readable LLM user message.

    Parameters
    ----------
    crash_dump : dict
        A crash dump dict as returned by ``SatelliteFaultSimulator.generate_crash_dump``.

    Returns
    -------
    str
        A formatted text block suitable as the ``user`` turn in a chat message.
    """
    lines: list[str] = []

    lines.append("CRASH DUMP REPORT")
    lines.append(f"Timestamp: {crash_dump['timestamp']}")
    lines.append(f"Fault Register: {crash_dump['fault_register']}")
    lines.append("")

    # Pre-fault telemetry
    lines.append("Pre-fault Telemetry:")
    for reading in crash_dump.get("pre_fault_telemetry", []):
        offset    = reading.get("timestamp_offset", "T-?s")
        param     = reading.get("parameter", "unknown")
        value     = reading.get("value", "NaN")
        unit      = reading.get("unit", "")
        nom_min   = reading.get("nominal_min", "?")
        nom_max   = reading.get("nominal_max", "?")
        anomalous = reading.get("anomalous", False)

        # Format value: float → 3 decimal places; anything else (e.g. "NaN") as-is
        if isinstance(value, float):
            value_str = f"{value:.3f}"
        else:
            value_str = str(value)

        flag = "[ANOMALOUS]" if anomalous else "[OK]"
        lines.append(
            f"  {offset}: {param} = {value_str} {unit} "
            f"(nominal: {nom_min}-{nom_max}) {flag}"
        )

    lines.append("")

    # Event log
    lines.append("Event Log:")
    for event in crash_dump.get("event_log", []):
        time_offset = event.get("time_offset", "T-?")
        source      = event.get("source", "UNKNOWN")
        message     = event.get("message", "")
        lines.append(f"  {time_offset} {source}: {message}")

    lines.append("")

    # Hardware state
    hw = crash_dump.get("hardware_state", {})
    lines.append("Hardware State:")
    lines.append(f"  Last reset cause: {hw.get('last_reset_cause', 'UNKNOWN')}")
    lines.append(f"  SEU events since boot: {hw.get('SEU_event_count_since_boot', 0)}")
    procs = hw.get("running_processes", [])
    lines.append(f"  Running processes: {', '.join(procs) if procs else 'none'}")
    lines.append(f"  Memory allocated: {hw.get('memory_allocation_MB', 0)} MB")

    lines.append("")

    # Operating context
    ctx = crash_dump.get("operating_context", {})
    lines.append("Operating Context:")
    for key, value in ctx.items():
        lines.append(f"  {key}: {value}")

    return "\n".join(lines)


def format_ground_truth_as_response(ground_truth: dict, crash_dump: dict) -> str:
    """Convert ground truth into a SENTINEL-style JSON response string.

    Produces a valid JSON object that matches the SENTINEL output format,
    with three ranked hypotheses, a 3–5 step recovery plan, a confidence
    score, a ``requires_human_review`` flag, and a brief reasoning summary.

    Parameters
    ----------
    ground_truth : dict
        A ground truth dict as returned by ``SatelliteFaultSimulator.get_ground_truth``.
    crash_dump : dict
        The corresponding crash dump dict (used for contextual detail in
        the recovery plan).

    Returns
    -------
    str
        A JSON string conforming to the SENTINEL output format.
    """
    fault_type  = ground_truth["root_cause_classification"]
    subsystem   = ground_truth["root_cause_subsystem"]
    confidence  = ground_truth["confidence"]
    causal      = ground_truth["causal_chain"]
    recovery    = ground_truth["recovery_action_sequence"]
    risk_level  = ground_truth["risk_level"]

    alt2_type, alt3_type = _ALT_HYPOTHESES[fault_type]

    # ---- Hypotheses ----
    hypotheses = [
        {
            "rank":        1,
            "root_cause":  fault_type,
            "component":   f"{subsystem}_PRIMARY",
            "confidence":  round(confidence, 2),
            "causal_chain": causal,
        },
        {
            "rank":       2,
            "root_cause": alt2_type,
            "component":  f"{alt2_type.split('_')[0]}_SECONDARY",
            "confidence": round(max(0.05, confidence - 0.30), 2),
            "causal_chain": [
                f"Alternative scenario: {alt2_type.replace('_', ' ').lower()} signature present",
                "Secondary indicators overlap with primary fault pattern",
                "Cannot fully rule out without additional telemetry",
            ],
        },
        {
            "rank":       3,
            "root_cause": alt3_type,
            "component":  f"{alt3_type.split('_')[0]}_TERTIARY",
            "confidence": round(max(0.02, confidence - 0.50), 2),
            "causal_chain": [
                f"Low-probability scenario: {alt3_type.replace('_', ' ').lower()}",
                "Minimal supporting evidence in current telemetry window",
            ],
        },
    ]

    # ---- Recovery plan (3–5 steps from recovery_action_sequence) ----
    # Take up to 5 steps; the last step always gets MEDIUM/HIGH risk if
    # the overall risk level is elevated.
    cmd_prefix   = _CMD_PREFIX[fault_type]
    wait_seconds = [15, 30, 20, 10, 60]  # staggered waits per step

    recovery_steps_to_use = recovery[:5]
    recovery_plan = []
    for i, action in enumerate(recovery_steps_to_use):
        step_number = i + 1
        # Derive a short command token from the action text
        cmd_token = action.split()[0].upper().replace(",", "").replace(".", "")
        command   = f"{cmd_prefix}_{cmd_token}"
        # Escalate risk on the last step if overall risk is HIGH
        if risk_level == "HIGH" and step_number == len(recovery_steps_to_use):
            step_risk = "HIGH"
        elif risk_level == "HIGH" and step_number >= len(recovery_steps_to_use) - 1:
            step_risk = "MEDIUM"
        else:
            step_risk = "LOW"

        recovery_plan.append({
            "step":         step_number,
            "command":      command,
            "rationale":    action,
            "wait_seconds": wait_seconds[i % len(wait_seconds)],
            "verify":       f"Telemetry confirms action {step_number} completed successfully",
            "risk":         step_risk,
        })

    # ---- requires_human_review ----
    requires_human_review = (confidence < 0.70) or (risk_level == "HIGH")

    # ---- Reasoning summary ----
    anomalous_params = [
        r["parameter"]
        for r in crash_dump.get("pre_fault_telemetry", [])
        if r.get("anomalous")
    ]
    anomalous_str = (
        ", ".join(anomalous_params[:4]) if anomalous_params else "none flagged"
    )
    reasoning_summary = (
        f"Crash dump analysis for {fault_type}: anomalous parameters detected: "
        f"{anomalous_str}. "
        f"Fault register {crash_dump.get('fault_register', 'N/A')} consistent with "
        f"{subsystem} subsystem failure. "
        f"Causal chain begins with: {causal[0] if causal else 'unknown'}. "
        f"Diagnostic confidence: {confidence:.0%}. "
        f"Risk level: {risk_level}. "
        f"Human review {'required' if requires_human_review else 'not required'}."
    )

    response = {
        "hypotheses":           hypotheses,
        "recovery_plan":        recovery_plan,
        "confidence":           round(confidence, 2),
        "requires_human_review": requires_human_review,
        "reasoning_summary":    reasoning_summary,
    }

    return json.dumps(response)


def generate_dataset(n_samples: int, output_path: str, seed: int = 42) -> None:
    """Generate ``n_samples`` training examples and write them to ``output_path``.

    Distributes samples roughly evenly across the six fault types
    (``n_samples // 6`` each, with the remainder allocated to the first
    fault types in the list).  Each line in the output file is a JSON object::

        {
            "messages": [
                {"role": "system",    "content": <SENTINEL_SYSTEM_PROMPT>},
                {"role": "user",      "content": <crash dump prompt>},
                {"role": "assistant", "content": <ground truth JSON>}
            ]
        }

    Parameters
    ----------
    n_samples : int
        Total number of training examples to generate.
    output_path : str
        Path to the output JSONL file (created or overwritten).
    seed : int, optional
        Random seed forwarded to ``SatelliteFaultSimulator`` (default ``42``).
        Each sample uses an incrementing scenario id so outputs remain unique
        even when the same seed is reused across fault types.
    """
    sim = SatelliteFaultSimulator(seed=seed)

    # Distribute samples evenly across fault types
    base_count  = n_samples // len(_FAULT_TYPES)
    remainder   = n_samples % len(_FAULT_TYPES)
    counts: dict[str, int] = {}
    for i, ft in enumerate(_FAULT_TYPES):
        counts[ft] = base_count + (1 if i < remainder else 0)

    # Build the ordered list of fault types to generate
    fault_sequence: list[str] = []
    for ft in _FAULT_TYPES:
        fault_sequence.extend([ft] * counts[ft])

    # Shuffle for variety while keeping the overall distribution fixed
    rng = random.Random(seed)
    rng.shuffle(fault_sequence)

    written = 0
    type_counts: dict[str, int] = {ft: 0 for ft in _FAULT_TYPES}

    with open(output_path, "w", encoding="utf-8") as fh:
        for scenario_id, fault_type in enumerate(fault_sequence, start=1):
            crash_dump   = sim.generate_crash_dump(fault_type, scenario_id=scenario_id)
            ground_truth = sim.get_ground_truth(fault_type)

            user_content      = format_crash_dump_as_prompt(crash_dump)
            assistant_content = format_ground_truth_as_response(ground_truth, crash_dump)

            record = {
                "messages": [
                    {"role": "system",    "content": SENTINEL_SYSTEM_PROMPT},
                    {"role": "user",      "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ]
            }

            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            type_counts[fault_type] += 1

    print(f"Generated {written} examples to {output_path}")
    print("\nBreakdown by fault type:")
    for ft in _FAULT_TYPES:
        print(f"  {ft}: {type_counts[ft]}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _validate_output(output_path: str) -> None:
    """Re-read ``output_path`` line-by-line and verify every line is valid JSONL.

    Checks that each line:
    * parses as valid JSON,
    * contains a ``"messages"`` key,
    * has exactly 3 message entries with roles ``system``, ``user``, ``assistant``.

    Prints a summary of results to stdout.

    Parameters
    ----------
    output_path : str
        Path to the JSONL file to validate.
    """
    failures: list[str] = []
    total = 0

    with open(output_path, "r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            total += 1
            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                failures.append(f"  Line {line_no}: JSON parse error — {exc}")
                continue

            if "messages" not in obj:
                failures.append(f"  Line {line_no}: missing 'messages' key")
                continue

            messages = obj["messages"]
            if len(messages) != 3:
                failures.append(
                    f"  Line {line_no}: expected 3 messages, got {len(messages)}"
                )
                continue

            expected_roles = ("system", "user", "assistant")
            for idx, (msg, expected_role) in enumerate(zip(messages, expected_roles)):
                if msg.get("role") != expected_role:
                    failures.append(
                        f"  Line {line_no}: message[{idx}] role is "
                        f"{msg.get('role')!r}, expected {expected_role!r}"
                    )
                    break

    if failures:
        print(f"Validation FAILED: {len(failures)}/{total} lines have errors:")
        for f in failures:
            print(f)
    else:
        print(f"Validation passed: {total}/{total} lines valid")


def main() -> None:
    """Parse CLI arguments and run dataset generation (and optional validation)."""
    parser = argparse.ArgumentParser(
        description="Generate a synthetic SENTINEL fine-tuning dataset in JSONL format."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of training examples to generate (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sentinel_training.jsonl",
        help="Output JSONL file path (default: sentinel_training.jsonl)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="After writing, validate every line is parseable JSON with correct structure",
    )

    args = parser.parse_args()

    generate_dataset(
        n_samples=args.samples,
        output_path=args.output,
        seed=args.seed,
    )

    if args.validate:
        _validate_output(args.output)


if __name__ == "__main__":
    main()
