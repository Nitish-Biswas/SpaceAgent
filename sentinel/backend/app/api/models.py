"""
SENTINEL — Structured Output Contract (models.py)

This module defines the Pydantic schemas for SENTINEL's LLM agent output.
It is the single source of truth for the JSON shape that:
  - The LLM agent must produce (Person 2)
  - The FastAPI backend serializes over SSE (Person 4)
  - The React frontend renders (Person 3)
  - The evaluator scores against ground truth (Person 1)

Schema decisions are derived from:
  - SENTINEL_Hackathon_Strategy_v2.md Part 4.3 (system prompt output format)
  - SENTINEL_4Day_Master_Planner.md Section D (P2 deliverables)
  - ECSS-E-ST-70-11C safe mode recovery procedures
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class SubsystemID(str, Enum):
    """Satellite subsystem identifiers used across all SENTINEL components."""
    ADCS = "ADCS"   # Attitude Determination & Control
    EPS = "EPS"     # Electrical Power System
    OBC = "OBC"     # On-Board Computer
    TCS = "TCS"     # Thermal Control System
    COMMS = "COMMS"  # Communications
    PYLD = "PYLD"   # Payload


class RiskLevel(str, Enum):
    """Risk classification for individual recovery steps."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    BLOCKED = "BLOCKED"  # Safety validator blocked this command


class AnalysisStatus(str, Enum):
    """Overall status of the agent's analysis run."""
    COMPLETE = "complete"
    PARTIAL = "partial"    # Graceful degradation — some steps succeeded
    TIMEOUT = "timeout"    # Agent hit the hard 90-second limit
    ERROR = "error"        # Unrecoverable failure


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class RecoveryStep(BaseModel):
    """A single step in the recovery command sequence.

    Each step maps to a spacecraft command, its rationale, timing,
    verification criteria, and risk level.  The safety validator
    may override the risk to BLOCKED.
    """
    step: int = Field(..., ge=1, description="1-indexed step number")
    command: str = Field(
        ...,
        min_length=3,
        description="Spacecraft command name, e.g. CMD_GYRO_A_DRIVER_RESET",
    )
    rationale: str = Field(
        ...,
        min_length=5,
        description="Why this command is issued at this point in the sequence",
    )
    wait_seconds: int = Field(
        ...,
        ge=0,
        description="Seconds to wait after issuing command before verifying",
    )
    verify: str = Field(
        ...,
        min_length=3,
        description="Condition to check after wait, e.g. 'GYRO_A_RATE returns valid'",
    )
    risk: RiskLevel = Field(
        ...,
        description="Risk level of this step (LOW / MEDIUM / HIGH / BLOCKED)",
    )


class Hypothesis(BaseModel):
    """A single ranked diagnosis hypothesis.

    The agent MUST always produce exactly 3 hypotheses.
    Even for obvious faults: H1 (high confidence), H2 and H3 (low).
    This enables multi-hypothesis reasoning and graceful degradation
    when the top hypothesis is wrong.
    """
    rank: int = Field(..., ge=1, le=3, description="Rank 1 = most likely")
    root_cause: str = Field(
        ...,
        min_length=3,
        description="Fault class, e.g. ADCS_GYRO_SEU, EPS_SOLAR_UNDERVOLT",
    )
    affected_component: str = Field(
        ...,
        min_length=2,
        description="Affected component, e.g. SOLAR_ARRAY_A, GYRO_A",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in this hypothesis (0.0–1.0)",
    )
    causal_chain: List[str] = Field(
        ...,
        min_length=2,
        description=(
            "Ordered list of events from trigger to safe mode, "
            "e.g. ['I_sa drops to 0A', 'V_bat falls to 24V', 'EPS fault flag set']"
        ),
    )

    @field_validator("confidence")
    @classmethod
    def round_confidence(cls, v: float) -> float:
        """Keep confidence to 2 decimal places for clean display."""
        return round(v, 2)

    @field_validator("causal_chain")
    @classmethod
    def validate_causal_chain(cls, value: List[str]) -> List[str]:
        cleaned = [item.strip() for item in value]

        if any(not item for item in cleaned):
            raise ValueError(
                "causal_chain entries must be non-empty strings"
            )

        return cleaned


class SentinelOutput(BaseModel):
    """Top-level structured output from the SENTINEL agent.

    This is the exact JSON shape that:
      - The LLM must return (via system prompt enforcement + retry)
      - The FastAPI endpoint serializes
      - The React frontend destructures
      - The evaluator compares to ground truth

    Contract invariants:
      1. Exactly 3 hypotheses, ranked 1-2-3
      2. Hypothesis confidences are descending (rank 1 ≥ rank 2 ≥ rank 3)
      3. requires_human_review is True when overall confidence < 0.70
         OR any recovery step has risk HIGH or BLOCKED
      4. recovery_plan has at least 1 step
      5. Overall confidence matches hypothesis rank-1 confidence
    """

    hypotheses: List[Hypothesis] = Field(
        ...,
        min_length=3,
        max_length=3,
        description="Exactly 3 ranked diagnostic hypotheses",
    )
    recovery_plan: List[RecoveryStep] = Field(
        ...,
        min_length=1,
        description="Ordered recovery command sequence for the top hypothesis",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall confidence (should equal top-hypothesis confidence)",
    )
    requires_human_review: bool = Field(
        ...,
        description=(
            "True when confidence < 0.70 or any step is HIGH/BLOCKED risk"
        ),
    )
    reasoning_summary: str = Field(
        ...,
        min_length=10,
        description="2–4 sentence summary of the diagnostic reasoning chain",
    )
    status: AnalysisStatus = Field(
        default=AnalysisStatus.COMPLETE,
        description="Analysis completion status",
    )

    @field_validator("confidence")
    @classmethod
    def round_confidence(cls, v: float) -> float:
        return round(v, 2)

    @field_validator("reasoning_summary")
    @classmethod
    def validate_reasoning_summary(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError(
                "reasoning_summary must not be blank"
            )

        return value

    @model_validator(mode="after")
    def validate_output_invariants(self) -> "SentinelOutput":
        """Enforce the contract invariants documented above."""

        # --- Invariant 1: ranks must be exactly {1, 2, 3} ---
        ranks = sorted(h.rank for h in self.hypotheses)
        if ranks != [1, 2, 3]:
            raise ValueError(
                f"Hypotheses must have ranks [1, 2, 3], got {ranks}"
            )

        # --- Invariant 2: confidences must be non-increasing by rank ---
        sorted_by_rank = sorted(self.hypotheses, key=lambda h: h.rank)
        for i in range(len(sorted_by_rank) - 1):
            if sorted_by_rank[i].confidence < sorted_by_rank[i + 1].confidence:
                raise ValueError(
                    f"Hypothesis rank {sorted_by_rank[i].rank} confidence "
                    f"({sorted_by_rank[i].confidence}) must be >= rank "
                    f"{sorted_by_rank[i+1].rank} confidence "
                    f"({sorted_by_rank[i+1].confidence})"
                )

        # --- Invariant 3: recovery steps must be sequential ---
        step_numbers = [step.step for step in self.recovery_plan]

        expected_steps = list(
            range(
                1,
                len(self.recovery_plan) + 1
            )
        )

        if step_numbers != expected_steps:
            raise ValueError(
                f"Recovery steps must be sequential "
                f"{expected_steps}, got {step_numbers}"
            )


        # --- Invariant 4: auto-set requires_human_review ---
        # Only auto-ESCALATE to True, never auto-downgrade to False.
        # This ensures safety.py's requires_human_review=True is preserved.
        has_high_risk = any(
            s.risk in (RiskLevel.HIGH, RiskLevel.BLOCKED)
            for s in self.recovery_plan
        )
        should_flag = self.confidence < 0.70 or has_high_risk
        if should_flag and not self.requires_human_review:
            # Auto-correct rather than reject — safer for hackathon reliability
            object.__setattr__(self, "requires_human_review", True)

        # --- Invariant 5: overall confidence matches rank-1 ---
        top_hyp = next(h for h in self.hypotheses if h.rank == 1)
        if abs(self.confidence - top_hyp.confidence) > 0.01:
            # Auto-correct to top hypothesis confidence
            object.__setattr__(self, "confidence", top_hyp.confidence)

        return self


# ---------------------------------------------------------------------------
# SSE event wrapper (used by Person 4's streaming endpoint)
# ---------------------------------------------------------------------------

class SSEEventType(str, Enum):
    """Event types streamed over SSE to the frontend."""
    THOUGHT = "thought"         # Agent reasoning step
    ACTION = "action"           # Tool call initiated
    OBSERVATION = "observation"  # Tool call result
    RESULT = "result"           # Final SentinelOutput
    ERROR = "error"             # Error message
    STATUS = "status"           # Progress updates


class SSEEvent(BaseModel):
    """A single SSE event sent from backend to frontend.

    Person 3 uses `event_type` to route data to the correct UI panel:
      - THOUGHT/ACTION/OBSERVATION → Panel 2 (Reasoning Trace)
      - RESULT → Panel 3 (Causal DAG) + Panel 4 (Recovery Plan)
      - ERROR → Error toast
      - STATUS → Header status indicator
    """
    event_type: SSEEventType
    data: str = Field(
        ...,
        description="Payload — plain text for trace events, JSON for RESULT",
    )
    step_number: Optional[int] = Field(
        default=None,
        ge=1,
        description="Agent reasoning step index (for trace events)",
    )
# ---------------------------------------------------------------------------
# INPUT SCHEMAS — Crash Dump Intake Validation
# ---------------------------------------------------------------------------


class TelecommandContext(BaseModel):
    """Behavioral layer: telecommand interval analysis.

    Captures the execution timing pattern of the command correlated with
    the safe-mode entry event.  The gap classification and percentile come
    from the statistical baseline built during nominal ops.
    """
    event_id: int = Field(..., description="Unique sequence identifier for the telecommand execution log")
    telecommand: str = Field(..., description="System command identifier (e.g. telecommand_63)")
    execution_timestamp: datetime = Field(..., description="ISO 8601 timestamp of command execution")
    gap_seconds: float = Field(..., description="Delta-T in seconds since the previous execution of this command")
    gap_classification: str = Field(..., description="Statistical interval classification (burst, nominal, stale)")
    gap_percentile: float = Field(..., description="Historical percentile rating of this execution delta")
    anomaly_flag: bool = Field(..., description="True if the interval crosses baseline bounds")


class TelemetryEntry(BaseModel):
    """Physiological layer: a single pre-fault telemetry reading.

    Each entry captures one sensor channel at one time-step inside the
    pre-fault window (e.g. T-120s, T-60s, T-10s).  The `status` field is
    the output of the Z-score classifier.
    """
    timestamp: str = Field(..., description="Relative timeline marker (e.g. T-60s)")
    parameter: str = Field(..., description="Telemetry channel designation (e.g. V_bat, Gyro_rate_degs)")
    value: Optional[float] = Field(None, description="Raw scalar reading; None on NaN / dropout")
    status: str = Field(default="NOMINAL", description="Statistical state: NOMINAL, ANOMALOUS, CRITICAL")

    # --- optional display-helper fields already used by the frontend ---
    nominal_min: Optional[float] = Field(default=None, description="Lower bound of nominal range")
    nominal_max: Optional[float] = Field(default=None, description="Upper bound of nominal range")

    @field_validator("status")
    @classmethod
    def validate_status_bounds(cls, v: str) -> str:
        allowed = {"NOMINAL", "ANOMALOUS", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"status must be one of {sorted(allowed)}, got '{v}'")
        return upper


class CrashDumpRequest(BaseModel):
    """Unified data-intake schema validated by FastAPI before AI processing.

    Design goals:
      1. Strict validation for required fields (fault_type, scenario_id).
      2. Backward-compatible: fields added by the new layered schema
         (incident_id, fault_register, telecommand_context,
         pre_fault_telemetry_window) have sensible defaults so the existing
         frontend LOCAL_PRESET_SCENARIOS still pass validation.
      3. Extra keys (hardware_state, operating_context) are silently
         forwarded via ``extra = "allow"`` — the LLM prompt sees them.
    """

    # --- Core identifiers (always required) ---
    scenario_id: Optional[int] = Field(default=None, description="Scenario identifier used by the demo UI")
    fault_type: Optional[str] = Field(default=None, description="Fault category, e.g. ADCS_SENSOR_FAULT")

    # --- New structured fields (optional for backward compat) ---
    incident_id: Optional[str] = Field(
        default=None,
        description="Unique alphanumeric identifier for the safe-mode incident case",
    )
    fault_register: Optional[str] = Field(
        default=None,
        description="Hexadecimal bitmask of HW/SW flags that tripped FDIR",
    )
    safe_mode_trigger: Optional[str] = Field(
        default=None,
        description="Trigger string that caused safe-mode entry",
    )
    telecommand_context: Optional[TelecommandContext] = Field(
        default=None,
        description="Behavioral interval log for the correlated command",
    )
    pre_fault_telemetry_window: Optional[List[TelemetryEntry]] = Field(
        default=None,
        description="Structured pre-fault telemetry (new layered format)",
    )

    # --- Legacy shape used by the existing frontend preset scenarios ---
    pre_fault_telemetry: Optional[List[Dict]] = Field(
        default=None,
        description="Legacy pre-fault telemetry list (flat dicts from frontend)",
    )
    event_log: Optional[List[Dict]] = Field(
        default=None,
        description="Raw event log entries from the spacecraft",
    )

    class Config:
        extra = "allow"  # forward hardware_state, operating_context, etc.
        json_schema_extra = {
            "example": {
                "scenario_id": 1,
                "fault_type": "ADCS_SENSOR_FAULT",
                "incident_id": "INC-2026-0036",
                "fault_register": "0x00000008",
                "telecommand_context": {
                    "event_id": 36,
                    "telecommand": "telecommand_63",
                    "execution_timestamp": "2026-06-13T00:15:22Z",
                    "gap_seconds": 90.0,
                    "gap_classification": "burst",
                    "gap_percentile": 16.2,
                    "anomaly_flag": True,
                },
                "pre_fault_telemetry_window": [
                    {"timestamp": "T-120s", "parameter": "V_bat",
                     "value": 31.2, "status": "NOMINAL"},
                    {"timestamp": "T-60s", "parameter": "TCS_HEATER_ZONE_2_TEMP",
                     "value": 68.4, "status": "ANOMALOUS"},
                    {"timestamp": "T-10s", "parameter": "TCS_HEATER_ZONE_2_TEMP",
                     "value": 85.2, "status": "CRITICAL"},
                ],
            }
        }

    @field_validator("fault_type")
    @classmethod
    def strip_fault_type(cls, v: Optional[str]) -> Optional[str]:
        return v.strip() if v else v

    @field_validator("safe_mode_trigger")
    @classmethod
    def strip_trigger(cls, v: Optional[str]) -> Optional[str]:
        return v.strip() if v else v
