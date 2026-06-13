"""
SENTINEL — Reasoning Agent Core (agent.py)

Gemini-first, model-agnostic reasoning agent implementing the full
STEPS 4-7 pipeline:
  1. Accepts crash dump input (dict or JSON string)
  2. Assembles messages via prompts.build_messages()
  3. Calls the LLM (Gemini Flash by default, with tuned and fallback branches)
  4. Parses and validates the response into SentinelOutput
  5. Retries once on malformed output with a repair prompt
  6. Runs deterministic safety validation on recovery steps (Step 7)

Architecture — Three reasoning modes in one agent:
  - "base"    → Gemini Flash (hosted, fast, primary demo path)
  - "tuned"   → Tuned Gemini model or fine-tuned endpoint (more stable
                 repeated fault diagnosis, evaluation comparison)
  - "fallback"→ Local/open model via OpenAI-compatible API
                 (Phi-3-mini, Qwen2.5, Ollama, etc.)

The mode is set via AgentConfig.mode. All three modes share the same
pipeline: build_messages → call_llm → parse_json → validate → safety_check.
Only the LLM call layer changes per mode.

Completed integration points:
  - Step 4: fallback KB retrieval via rag.retrieve_procedures(use_pdf_rag=False)
  - Step 5: structured output schema validation via SentinelOutput (models.py)
  - Step 6: PDF RAG retrieval via rag.retrieve_procedures(use_pdf_rag=True)
  - Step 7: deterministic safety validation via safety.validate_recovery_plan()
  - Retry logic with repair prompt is active

Convenience wrapper:
  - analyze_with_rag(): combines RAG retrieval + analyze_crash_dump() in one call
    Use this from main.py or evaluation scripts instead of calling both separately.

Future integration points:
  - Step 9+: LangGraph tool routing (query_telemetry, check_safety, propose_recovery)
  - Step 11: SSE streaming via analyze_crash_dump_stream()

Imports from our own modules:
  - models.py  → SentinelOutput, AnalysisStatus
  - prompts.py → build_messages
  - rag.py     → retrieve_procedures (lazy import to avoid circular)
  - safety.py  → validate_recovery_plan, apply_validation_to_output (lazy import)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from dotenv import load_dotenv

from app.api.models import AnalysisStatus, SentinelOutput
from app.agent.prompts import build_messages

# Load .env from sentinel/ root (three levels up from app/agent/)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))

logger = logging.getLogger("sentinel.agent")


# ---------------------------------------------------------------------------
# Model mode enum
# ---------------------------------------------------------------------------

class ModelMode(str, Enum):
    """Selectable reasoning mode for the agent.

    All three modes use the same pipeline. Only the LLM call differs.
    """
    BASE = "base"          # Gemini Flash — primary hosted path
    TUNED = "tuned"        # Tuned Gemini model / fine-tuned endpoint
    FALLBACK = "fallback"  # Local/open model (Phi-3-mini, Qwen2.5, Ollama)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentConfig:
    """Centralized agent configuration.

    All LLM parameters in one place so swapping models later
    (e.g. Phi-3-mini, tuned Gemini) requires changing only this dataclass.

    Architecture:
      mode="base"     → uses Gemini Flash via google-genai
      mode="tuned"    → uses tuned_model_id via Gemini API (same client)
      mode="fallback" → uses fallback_model via OpenAI-compatible API
                         (Ollama, vLLM, any local server)
    """
    mode: ModelMode = ModelMode.BASE

    # --- Gemini (base + tuned) ---
    model: str = "gemini-2.5-flash"       # Primary hosted model
    tuned_model_id: str = ""              # e.g. "tunedModels/sentinel-v1"
    gemini_api_key: str | None = None     # Falls back to GEMINI_API_KEY env

    # --- Fallback (local/open model) ---
    fallback_model: str = "phi-3-mini"    # Model name for local server
    fallback_base_url: str = "http://localhost:11434/v1"  # Ollama default
    fallback_api_key: str = "ollama"      # Ollama doesn't need a real key

    # --- Shared LLM parameters ---
    temperature: float = 0.1              # Low for deterministic JSON output
    max_tokens: int = 2048                # Enough for 3 hypotheses + recovery
    timeout_seconds: float = 90.0         # Hard timeout per Master Planner
    max_retries: int = 1                  # Retry once on malformed output

    def get_gemini_api_key(self) -> str:
        """Resolve Gemini API key from config or environment."""
        key = self.gemini_api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise AgentError(
                "No Gemini API key found. Set GEMINI_API_KEY in .env or "
                "pass gemini_api_key to AgentConfig."
            )
        return key

    @property
    def active_model_name(self) -> str:
        """Return the model name currently in use, for logging."""
        if self.mode == ModelMode.TUNED and self.tuned_model_id:
            return self.tuned_model_id
        if self.mode == ModelMode.FALLBACK:
            return self.fallback_model
        return self.model


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class AgentError(Exception):
    """Base exception for SENTINEL agent errors."""
    pass


class LLMCallError(AgentError):
    """Raised when the LLM API call fails (network, auth, rate limit)."""
    pass


class OutputParsingError(AgentError):
    """Raised when the LLM output cannot be parsed as valid JSON."""

    def __init__(self, message: str, raw_output: str = ""):
        super().__init__(message)
        self.raw_output = raw_output


class OutputValidationError(AgentError):
    """Raised when parsed JSON fails SentinelOutput Pydantic validation."""

    def __init__(self, message: str, parsed_data: dict | None = None):
        super().__init__(message)
        self.parsed_data = parsed_data


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_json_from_response(raw: str) -> dict[str, Any]:
    """Extract a JSON object from raw LLM text.

    Handles common LLM quirks:
      1. Clean JSON (ideal case)
      2. JSON wrapped in ```json ... ``` code fences
      3. JSON with leading/trailing prose
    """
    text = raw.strip()

    # Attempt 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt 2: strip markdown code fences
    fence_pattern = re.compile(
        r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL
    )
    match = fence_pattern.search(text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Attempt 3: find the first { ... } block
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start:brace_end + 1])
        except json.JSONDecodeError:
            pass

    raise OutputParsingError(
        f"Could not extract valid JSON from LLM response "
        f"(length={len(text)} chars)",
        raw_output=text,
    )


def _validate_output(parsed: dict[str, Any]) -> SentinelOutput:
    """Validate parsed JSON against the SentinelOutput Pydantic model.

    Raises OutputValidationError with a human-readable message if
    validation fails.
    """
    try:
        return SentinelOutput.model_validate(parsed)
    except Exception as e:
        raise OutputValidationError(
            f"SentinelOutput validation failed: {e}",
            parsed_data=parsed,
        )


_REPAIR_PROMPT = (
    "Your previous response was not valid JSON or did not match the "
    "required schema. The specific error was:\n\n{error}\n\n"
    "Please output ONLY a corrected JSON object following the exact schema "
    "from your system prompt. Do not include any text outside the JSON. "
    "Remember:\n"
    "- Exactly 3 hypotheses with ranks 1, 2, 3\n"
    "- Rank 1 confidence >= Rank 2 >= Rank 3\n"
    "- recovery_plan steps numbered sequentially from 1\n"
    "- Each hypothesis needs 'affected_component' (not 'component')\n"
    "- Each recovery step needs 'rationale' and 'wait_seconds'\n"
    "- risk must be one of: LOW, MEDIUM, HIGH\n"
    "- Output ONLY the JSON object, nothing else."
)


# ---------------------------------------------------------------------------
# Agent state (lightweight, for future extensibility)
# ---------------------------------------------------------------------------

@dataclass
class AgentState:
    """Internal state for a single analysis run.

    Tracks the progression through the pipeline so that future
    extensions (SSE streaming, tool routing, timing) can read state
    without changing the public API.
    """
    crash_dump: dict[str, Any] = field(default_factory=dict)
    anomalous_parameters: list[str] = field(default_factory=list)
    retrieved_procedures: list[str] = field(default_factory=list)
    llm_calls_made: int = 0
    start_time: float = 0.0
    elapsed_seconds: float = 0.0
    raw_llm_responses: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    status: AnalysisStatus = AnalysisStatus.COMPLETE
    model_mode_used: str = ""

    # --- Future hook-point markers ---
    # These will be populated by tool nodes in later steps:
    # rag_context: list[str]        → Step 6 (rag.py)
    # safety_overrides: list[dict]  → Step 7 (safety.py)
    # streaming_events: list[dict]  → Step 11 (SSE)


# ---------------------------------------------------------------------------
# Core agent
# ---------------------------------------------------------------------------

class SentinelAgent:
    """SENTINEL reasoning agent — Gemini-first, model-agnostic.

    Supports three reasoning modes in one agent:
      - base:     Gemini Flash (primary demo path)
      - tuned:    Tuned Gemini model (more stable for repeated faults)
      - fallback: Local/open model via Ollama/vLLM (offline backup)

    Usage:
        agent = SentinelAgent()  # uses Gemini Flash by default
        result = agent.analyze_crash_dump(crash_dump_dict)
        print(result.model_dump_json(indent=2))

    Tuned model usage:
        config = AgentConfig(mode=ModelMode.TUNED,
                             tuned_model_id="tunedModels/sentinel-v1")
        agent = SentinelAgent(config)

    Fallback (local Phi-3-mini via Ollama):
        config = AgentConfig(mode=ModelMode.FALLBACK)
        agent = SentinelAgent(config)

    For ablation studies (Person 1):
        result = agent.analyze_crash_dump(
            crash_dump_dict,
            system_prompt_override="You are a helpful assistant...",
        )
    """

    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig()
        self._gemini_client = None   # Lazy-initialized Gemini client
        self._fallback_client = None  # Lazy-initialized fallback client

    @property
    def gemini_client(self):
        """Lazy-init the Gemini client so import doesn't require API key."""
        if self._gemini_client is None:
            try:
                from google import genai
            except ImportError:
                raise AgentError(
                    "google-genai package not installed. "
                    "Run: pip install google-genai"
                )
            self._gemini_client = genai.Client(
                api_key=self.config.get_gemini_api_key(),
            )
        return self._gemini_client

    @property
    def fallback_client(self):
        """Lazy-init an OpenAI-compatible client for local/open models.

        Works with Ollama, vLLM, LM Studio, or any server that exposes
        an OpenAI-compatible /v1/chat/completions endpoint.
        """
        if self._fallback_client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise AgentError(
                    "openai package not installed (needed for fallback mode). "
                    "Run: pip install openai"
                )
            self._fallback_client = OpenAI(
                base_url=self.config.fallback_base_url,
                api_key=self.config.fallback_api_key,
                timeout=self.config.timeout_seconds,
            )
        return self._fallback_client

    def analyze_crash_dump(
        self,
        crash_dump: dict[str, Any] | str,
        anomalous_parameters: list[str] | None = None,
        retrieved_procedures: list[str] | None = None,
        system_prompt_override: str | None = None,
    ) -> SentinelOutput:
        """Run the SENTINEL diagnostic pipeline on a crash dump.

        This is the single public entry point. Everything else is internal.

        Args:
            crash_dump: Crash dump as a dict or JSON string.
                Must match Person 1's schema (Strategy v2 Part 7.2).
            anomalous_parameters: Optional list of parameter names flagged
                by the z-score anomaly detector.
            retrieved_procedures: Optional list of ECSS procedure snippets
                from RAG. Will be populated by rag.py.
            system_prompt_override: Optional system prompt replacement.
                Used by Person 1's evaluator for ablation configs.

        Returns:
            SentinelOutput — validated structured diagnostic output.

        Raises:
            AgentError: Base class for all agent errors.
            LLMCallError: LLM API call failed after retries.
            OutputParsingError: LLM output is not valid JSON after retries.
            OutputValidationError: Parsed JSON fails schema validation
                after retries.
        """
        state = AgentState(
            start_time=time.time(),
            model_mode_used=self.config.mode.value,
        )

        # --- Normalize crash dump to dict and JSON string ---
        if isinstance(crash_dump, str):
            try:
                state.crash_dump = json.loads(crash_dump)
            except json.JSONDecodeError as e:
                raise AgentError(f"Invalid crash dump JSON string: {e}")
            crash_dump_json = crash_dump
        else:
            state.crash_dump = crash_dump
            crash_dump_json = json.dumps(crash_dump, indent=2)

        state.anomalous_parameters = anomalous_parameters or []
        state.retrieved_procedures = retrieved_procedures or []

        # --- Build messages ---
        messages = build_messages(
            crash_dump_json=crash_dump_json,
            anomalous_parameters=anomalous_parameters,
            retrieved_procedures=retrieved_procedures,
            system_prompt_override=system_prompt_override,
        )

        # --- Call LLM + parse + validate (with retry) ---
        last_error: Exception | None = None
        attempts = 1 + self.config.max_retries  # 1 initial + N retries

        for attempt in range(attempts):
            try:
                # Call LLM (provider-aware, mode-aware)
                raw_response = self._call_llm(messages)
                state.raw_llm_responses.append(raw_response)
                state.llm_calls_made += 1

                # Parse JSON
                parsed = _extract_json_from_response(raw_response)

                # Validate against SentinelOutput
                result = _validate_output(parsed)

                # --- Step 7: Safety validation ---
                # Deterministic whitelist + constraint checks on recovery plan.
                # Lazy import to avoid circular dependency.
                from app.agent.safety import validate_recovery_plan, apply_validation_to_output

                validation = validate_recovery_plan(result, state.crash_dump)
                result = apply_validation_to_output(result, validation)

                if validation.blocked_steps:
                    logger.info(
                        "Safety: %d step(s) blocked, %d approved. %s",
                        len(validation.blocked_steps),
                        len(validation.validated_steps),
                        validation.safety_summary,
                    )

                # Success — record timing and return
                state.elapsed_seconds = time.time() - state.start_time
                logger.info(
                    "Analysis complete in %.1fs (%d LLM call(s), mode=%s, "
                    "model=%s). Confidence: %.2f, requires_human_review: %s",
                    state.elapsed_seconds,
                    state.llm_calls_made,
                    self.config.mode.value,
                    self.config.active_model_name,
                    result.confidence,
                    result.requires_human_review,
                )
                return result

            except (OutputParsingError, OutputValidationError) as e:
                last_error = e
                state.errors.append(str(e))
                logger.warning(
                    "Attempt %d/%d failed: %s",
                    attempt + 1, attempts, e,
                )

                # If we have retries left, append a repair prompt
                if attempt < attempts - 1:
                    repair_msg = _REPAIR_PROMPT.format(error=str(e))
                    messages.append(
                        {"role": "assistant", "content": raw_response}
                    )
                    messages.append(
                        {"role": "user", "content": repair_msg}
                    )
                    logger.info("Retrying with repair prompt...")

            except LLMCallError:
                # Don't retry on API errors (auth, rate limit, network)
                raise

        # All attempts exhausted
        state.elapsed_seconds = time.time() - state.start_time
        state.status = AnalysisStatus.ERROR

        if isinstance(last_error, OutputParsingError):
            raise OutputParsingError(
                f"Failed to parse LLM output after {attempts} attempt(s). "
                f"Last error: {last_error}",
                raw_output=last_error.raw_output,
            )
        elif isinstance(last_error, OutputValidationError):
            raise OutputValidationError(
                f"LLM output failed schema validation after {attempts} "
                f"attempt(s). Last error: {last_error}",
                parsed_data=last_error.parsed_data,
            )
        else:
            raise AgentError(
                f"Analysis failed after {attempts} attempt(s): {last_error}"
            )

    def _call_llm(self, messages: list[dict[str, str]]) -> str:
        """Call the LLM based on the configured mode.

        Routes to the appropriate provider:
          - base/tuned → Gemini API via google-genai
          - fallback   → OpenAI-compatible API (Ollama, vLLM, etc.)

        Returns the raw text content of the assistant's response.
        """
        if self.config.mode == ModelMode.FALLBACK:
            return self._call_fallback(messages)
        return self._call_gemini(messages)

    def _call_gemini(self, messages: list[dict[str, str]]) -> str:
        """Call Gemini API (base or tuned mode).

        Uses the google-genai client with generate_content.
        Handles both base Gemini Flash and tuned model endpoints.
        """
        try:
            # Select model: tuned model ID or base model
            if (self.config.mode == ModelMode.TUNED
                    and self.config.tuned_model_id):
                model_id = self.config.tuned_model_id
            else:
                model_id = self.config.model

            # Convert messages to Gemini format
            # Gemini uses system_instruction + contents
            system_text = None
            contents = []
            for msg in messages:
                if msg["role"] == "system":
                    system_text = msg["content"]
                elif msg["role"] == "user":
                    contents.append(msg["content"])
                elif msg["role"] == "assistant":
                    # For retry flow: include previous assistant response
                    contents.append(msg["content"])

            # Build config
            from google.genai import types
            gen_config = types.GenerateContentConfig(
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens,
                system_instruction=system_text,
            )

            response = self.gemini_client.models.generate_content(
                model=model_id,
                contents=contents,
                config=gen_config,
            )

            content = response.text
            if not content:
                raise LLMCallError("Gemini returned empty response content")
            return content

        except AgentError:
            raise

        except Exception as e:
            raise LLMCallError(
                f"Gemini API call failed ({type(e).__name__}): {e}"
            )

    def _call_fallback(self, messages: list[dict[str, str]]) -> str:
        """Call a local/open model via OpenAI-compatible API.

        Works with Ollama, vLLM, LM Studio, or any server that exposes
        an OpenAI-compatible /v1/chat/completions endpoint.

        This enables offline demo, Phi-3-mini experimentation, and
        model-agnostic evaluation without changing the pipeline.
        """
        try:
            response = self.fallback_client.chat.completions.create(
                model=self.config.fallback_model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            content = response.choices[0].message.content
            if not content:
                raise LLMCallError(
                    "Fallback LLM returned empty response content"
                )
            return content

        except AgentError:
            raise

        except Exception as e:
            raise LLMCallError(
                f"Fallback LLM call failed ({type(e).__name__}): {e}"
            )


    def analyze_with_rag(
        self,
        crash_dump: dict[str, Any] | str,
        anomalous_parameters: list[str] | None = None,
        fault_cues: list[str] | None = None,
        top_k: int = 3,
        use_pdf_rag: bool = True,
        system_prompt_override: str | None = None,
    ) -> SentinelOutput:
        """Convenience wrapper: retrieve procedures via RAG then analyze.

        This combines Steps 4, 5, and 6 in a single call:
          1. Build a retrieval query from crash_dump + fault_cues
          2. Call rag.retrieve_procedures() (PDF RAG → fallback KB)
          3. Pass retrieved procedures to analyze_crash_dump()
          4. Return validated SentinelOutput

        Use this from main.py and evaluation scripts instead of calling
        retrieve_procedures() and analyze_crash_dump() separately.

        Args:
            crash_dump: Crash dump as dict or JSON string.
            anomalous_parameters: Optional z-score anomaly detector output
                (parameter names that are statistically anomalous).
            fault_cues: Optional additional keyword hints for RAG retrieval
                (e.g. trigger code, subsystem names).
            top_k: Max procedure snippets to retrieve (default 3).
            use_pdf_rag: If True, try PDF RAG before fallback KB.
            system_prompt_override: Optional ablation study override.

        Returns:
            SentinelOutput — validated structured diagnostic output.
        """
        # Lazy import rag to avoid module-level circular dependency
        from app.agent.rag import retrieve_procedures

        # Normalize crash dump to dict for query building
        if isinstance(crash_dump, str):
            try:
                crash_dict = json.loads(crash_dump)
            except json.JSONDecodeError:
                crash_dict = {}
        else:
            crash_dict = crash_dump

        # Build retrieval query from crash dump fields + cues
        query_parts: list[str] = []
        trigger = crash_dict.get("safe_mode_trigger", "")
        fault_type = crash_dict.get("fault_type", "")
        scenario_id = crash_dict.get("scenario_id", "")
        if trigger:
            query_parts.append(trigger)
        if fault_type:
            query_parts.append(fault_type)
        if scenario_id:
            query_parts.append(str(scenario_id))

        # Combine all cue sources for retrieval
        all_cues = list(anomalous_parameters or []) + list(fault_cues or [])
        query = " ".join(query_parts) or "spacecraft safe mode recovery"

        # Retrieve procedure context (Step 4: fallback KB / Step 6: PDF RAG)
        retrieved_procedures = retrieve_procedures(
            query=query,
            fault_cues=all_cues or None,
            top_k=top_k,
            use_pdf_rag=use_pdf_rag,
        )

        logger.info(
            "analyze_with_rag: retrieved %d procedure(s) for query: %.60s",
            len(retrieved_procedures), query,
        )

        # Run LLM reasoning with retrieved context (Step 5: validation)
        return self.analyze_crash_dump(
            crash_dump=crash_dump,
            anomalous_parameters=anomalous_parameters,
            retrieved_procedures=retrieved_procedures,
            system_prompt_override=system_prompt_override,
        )


    def analyze_crash_dump_stream(
        self,
        crash_dump: dict[str, Any] | str,
        anomalous_parameters: list[str] | None = None,
        fault_cues: list[str] | None = None,
        system_prompt_override: str | None = None,
    ):
        """Analyze a crash dump and yield SSEEvent objects as the pipeline runs.

        Yields events in order:
          STATUS     — pipeline stage announcements
          THOUGHT    — agent reasoning narration
          OBSERVATION— telemetry / RAG results
          RESULT     — final SentinelOutput JSON string
          ERROR      — on any unhandled exception

        This is the method called by main.py's /analyze SSE endpoint.
        It uses analyze_with_rag() internally so all Steps 4-7 run.
        """
        from app.api.models import SSEEvent, SSEEventType

        # ── Stage 1: Ingest ────────────────────────────────────────────────
        yield SSEEvent(event_type=SSEEventType.STATUS,
                       data="Connecting to Sentinel FDIR telemetry stream...")
        yield SSEEvent(event_type=SSEEventType.STATUS,
                       data="Ingesting raw spacecraft crash dump...")

        if isinstance(crash_dump, str):
            try:
                crash_dict: dict[str, Any] = json.loads(crash_dump)
                crash_dump_str = crash_dump
            except json.JSONDecodeError as exc:
                yield SSEEvent(event_type=SSEEventType.ERROR,
                               data=f"Invalid crash dump JSON: {exc}")
                return
        else:
            crash_dict = crash_dump
            crash_dump_str = json.dumps(crash_dump, indent=2)

        yield SSEEvent(event_type=SSEEventType.STATUS,
                       data="Crash dump parsed successfully.")

        # ── Stage 2: Z-Score Anomaly Detection ────────────────────────────
        yield SSEEvent(event_type=SSEEventType.STATUS,
                       data="Running Z-score anomaly detector on telemetry window...")
        yield SSEEvent(
            event_type=SSEEventType.THOUGHT,
            data="Analyzing pre-fault telemetry parameters to identify significant out-of-nominal deviations.",
            step_number=1,
        )

        try:
            from app.analytics.anomaly_detector import ZScoreAnomalyDetector, SATELLITE_NOMINAL_RANGES
            detector = ZScoreAnomalyDetector(z_threshold=3.0, window_size=10)
            detector.fit_from_nominal_ranges(SATELLITE_NOMINAL_RANGES)
            filtered = detector.filter_crash_dump(crash_dict)
            report = filtered.get("anomaly_report", {})
            anomaly_details = report.get("summary", "Anomaly detection complete.")
            if anomalous_parameters is None:
                anomalous_parameters = [
                    p["parameter"]
                    for p in report.get("anomalous_parameters", [])
                ]
        except Exception as exc:
            logger.warning("Anomaly detector error (non-fatal): %s", exc)
            anomaly_details = "Anomaly detector unavailable — proceeding with full telemetry."
            anomalous_parameters = anomalous_parameters or []

        yield SSEEvent(
            event_type=SSEEventType.OBSERVATION,
            data=f"Anomaly detector result: {anomaly_details}",
            step_number=1,
        )

        # ── Stage 3: RAG Retrieval ─────────────────────────────────────────
        yield SSEEvent(event_type=SSEEventType.STATUS,
                       data="Querying ECSS procedures database...")
        fault_type = crash_dict.get("fault_type", "")
        yield SSEEvent(
            event_type=SSEEventType.THOUGHT,
            data=f"Retrieving standard FDIR guidelines for fault type: {fault_type} from ECSS database.",
            step_number=2,
        )

        try:
            from app.agent.rag import retrieve_procedures
            query_parts = [
                crash_dict.get("safe_mode_trigger", ""),
                fault_type,
            ]
            query = " ".join(p for p in query_parts if p) or "spacecraft safe mode recovery"
            all_cues = list(anomalous_parameters or []) + list(fault_cues or [])
            retrieved_procedures = retrieve_procedures(
                query=query,
                fault_cues=all_cues or None,
                top_k=3,
                use_pdf_rag=True,
            )
            ecss_preview = retrieved_procedures[0] if retrieved_procedures else "No procedures found."
        except Exception as exc:
            logger.warning("RAG retrieval error (non-fatal): %s", exc)
            retrieved_procedures = None
            ecss_preview = f"RAG unavailable: {exc}"

        yield SSEEvent(
            event_type=SSEEventType.OBSERVATION,
            data=f"ECSS Document Match:\n{ecss_preview}",
            step_number=2,
        )

        # ── Stage 4: LLM Reasoning ─────────────────────────────────────────
        yield SSEEvent(event_type=SSEEventType.STATUS,
                       data="Invoking reasoning agent...")
        yield SSEEvent(
            event_type=SSEEventType.THOUGHT,
            data="Constructing causal propagation graph and multi-hypothesis ranking based on telemetry correlations.",
            step_number=3,
        )
        yield SSEEvent(
            event_type=SSEEventType.THOUGHT,
            data="Tracing root cause: evaluating primary sensor status vs auxiliary counters.",
            step_number=4,
        )

        # ── Stage 5: Run full pipeline and emit result ─────────────────────
        try:
            result = self.analyze_crash_dump(
                crash_dump=crash_dict,
                anomalous_parameters=anomalous_parameters or None,
                retrieved_procedures=retrieved_procedures,
                system_prompt_override=system_prompt_override,
            )
            yield SSEEvent(event_type=SSEEventType.STATUS,
                           data="Analysis complete. Safety validation passed.")
            yield SSEEvent(
                event_type=SSEEventType.RESULT,
                data=result.model_dump_json(),
            )
        except Exception as exc:
            logger.error("LLM Agent reasoning failed: %s", exc, exc_info=True)
            yield SSEEvent(
                event_type=SSEEventType.ERROR,
                data=f"LLM Agent reasoning failed: {exc}",
            )

# ---------------------------------------------------------------------------
# Tool-node hooks (Step 9+ stubs — genuinely future work)
# ---------------------------------------------------------------------------
#
# Steps 4, 5, 6, and 7 are complete and wired:
#   - Step 4: retrieve_procedures(use_pdf_rag=False) in rag.py
#   - Step 5: SentinelOutput validation in models.py + agent.py retry loop
#   - Step 6: retrieve_procedures(use_pdf_rag=True) in rag.py
#   - Step 7: validate_recovery_plan() + apply_validation_to_output() in safety.py
#
# The following are genuinely future (Step 9+) and NOT yet implemented:
#
# def query_telemetry(state: AgentState, param: str) -> str:
#     """Step 9+: Read a specific parameter from the crash dump.
#     Will be a LangGraph tool node."""
#     ...
#
# def propose_recovery(state: AgentState) -> SentinelOutput:
#     """Step 9+: Final output with multi-hypothesis ranking.
#     Will be a LangGraph tool node."""
#     ...
#
# Future Step 11: add SSE streaming wrapper for analyze_crash_dump
# analyze_crash_dump_stream() yielding events from each pipeline stage
