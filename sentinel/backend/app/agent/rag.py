"""
SENTINEL — Retrieval & Knowledge Base (rag.py)

Hybrid retrieval layer with two paths:
  1. FALLBACK_KB  → always available, covers all 6 fault types, zero deps
  2. PDF RAG      → loads ECSS PDFs via LlamaIndex, embeds with
                    sentence-transformers (all-MiniLM-L6-v2, local/free),
                    stores/queries via ChromaDB (local persistent mode)

The public API is:
  retrieve_procedures(query, fault_cues, top_k, use_pdf_rag) → list[str]

Retrieval priority (when use_pdf_rag=True):
  1. Try PDF RAG (ChromaDB similarity search over ECSS chunks)
  2. If PDF RAG fails for ANY reason → fall back to FALLBACK_KB
  3. FALLBACK_KB keyword-matches fault cues → returns top-k entries
  4. If nothing matches → returns MULTI_CASCADE catch-all (never empty)

Graceful degradation guarantees:
  - Missing PDFs                → fallback KB
  - sentence-transformers missing → fallback KB
  - LlamaIndex import error     → fallback KB
  - ChromaDB error              → fallback KB
  - PDF parsing failure         → skip bad PDF, index the rest
  - Zero results from RAG       → fallback KB

References:
  - SENTINEL_Hackathon_Strategy_v2.md Part 4.2 (causal chains)
  - SENTINEL_4Day_Master_Planner.md Section D (P2 H+8–9: fallback KB)
  - SENTINEL_4Day_Master_Planner.md Section F.1 Risk #3 (RAG fallback)
  - ECSS-E-ST-70-11C Rev.1 (safe mode recovery procedures)
  - ECSS-Q-ST-30-02C (dependability requirements)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from dotenv import load_dotenv

# Load .env from sentinel/ root (three levels up from app/agent/)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))

logger = logging.getLogger("sentinel.rag")


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.join(_THIS_DIR, "..", "..")

ECSS_DATA_DIR = os.path.join(_BACKEND_DIR, "data", "ecss")
CHROMA_DB_DIR = os.path.join(_BACKEND_DIR, "data", "chroma_db")
CHROMA_COLLECTION_NAME = "ecss_procedures"

DEFAULT_TOP_K = 3
CHUNK_SIZE = 512        # Tokens per chunk -- good for technical docs
CHUNK_OVERLAP = 50      # Overlap to preserve context across boundaries
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Free, local, no API key needed
GEMINI_EMBED_MODEL = "text-embedding-004"  # Gemini embedding model (free tier)

# -- Memory-saving overrides ------------------------------------------------
#
# Option A: USE_GEMINI_RAG=true  (RECOMMENDED for free hosting)
#   Loads precomputed vectors from data/ecss_gemini_rag.npz (built once locally)
#   1 Gemini embed API call per query. Zero PyTorch. Zero ChromaDB. ~5MB RAM.
#   Run: python sentinel/backend/data_tools/precompute_gemini_embeddings.py
#
# Option B: DISABLE_PDF_RAG=true  (no RAG at all, just fallback KB)
#   Saves ~400MB RAM. Agent still works via Gemini + built-in keyword KB.
#
USE_GEMINI_RAG: bool = os.environ.get("USE_GEMINI_RAG", "false").lower() in ("1", "true", "yes")
DISABLE_PDF_RAG: bool = (
    not USE_GEMINI_RAG
    and os.environ.get("DISABLE_PDF_RAG", "false").lower() in ("1", "true", "yes")
)

_THIS_FILE = os.path.dirname(os.path.abspath(__file__))
GEMINI_RAG_NPZ = os.path.normpath(
    os.path.join(_THIS_FILE, "..", "..", "data", "ecss_gemini_rag.npz")
)

# ── Memory-saving override ──────────────────────────────────────────────────
# Set DISABLE_PDF_RAG=true in your environment (e.g. Render free tier) to skip
# loading sentence-transformers + PyTorch + ChromaDB (~400 MB RAM saved).
# The fallback knowledge base always handles all fault types.
DISABLE_PDF_RAG: bool = os.environ.get("DISABLE_PDF_RAG", "false").lower() in ("1", "true", "yes")
if DISABLE_PDF_RAG:
    logger.info(
        "DISABLE_PDF_RAG=true — skipping sentence-transformers/ChromaDB. "
        "Fallback KB will be used for all retrievals."
    )



# ═══════════════════════════════════════════════════════════════════════════
# RAG STATUS TRACKING
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RAGStatus:
    """Diagnostic metadata for the RAG subsystem.

    Inspect this to know whether retrieval came from PDF RAG or fallback KB,
    and whether initialization succeeded.
    """
    initialized: bool = False
    available: bool = False
    pdf_count: int = 0
    chunk_count: int = 0
    last_source: str = "not_initialized"   # "pdf_rag" | "fallback_kb"
    last_error: str | None = None

    def summary(self) -> str:
        if not self.initialized:
            return "RAG not initialized (using fallback KB)"
        if self.available:
            return (
                f"PDF RAG active: {self.pdf_count} PDFs, "
                f"{self.chunk_count} chunks indexed"
            )
        return f"PDF RAG unavailable: {self.last_error} (using fallback KB)"


# Initialize _rag_status now that RAGStatus is defined
_rag_status = RAGStatus()

if USE_GEMINI_RAG:
    logger.info(
        "USE_GEMINI_RAG=true -- will use precomputed Gemini embeddings "
        "(zero PyTorch/ChromaDB). Query embedding via Gemini API."
    )
elif DISABLE_PDF_RAG:
    logger.info(
        "DISABLE_PDF_RAG=true -- skipping sentence-transformers/ChromaDB. "
        "Fallback KB will be used for all retrievals."
    )


# Module-level state (lazy-initialized after RAGStatus class is defined below)
_rag_status: Any = None   # set to RAGStatus() after class def
_chroma_collection: Any = None
_embedding_fn: Any = None

# Gemini RAG numpy index (loaded once when USE_GEMINI_RAG=true)
_gemini_vectors: Any = None   # np.ndarray shape (N, D), L2-normalised float32
_gemini_chunks: Any = None    # np.ndarray of str
_gemini_sources: Any = None   # np.ndarray of str
_gemini_rag_loaded: bool = False


@dataclass(frozen=True)
class KBEntry:
    """A single knowledge base entry for a fault class.

    Each entry is designed to be self-contained: if the LLM receives only
    this text as retrieved context, it should have enough to diagnose and
    produce a recovery plan for the corresponding fault type.
    """
    fault_class: str                # e.g. "ADCS_GYRO_SEU"
    title: str                      # Human-readable title
    trigger_cues: tuple[str, ...]   # Keywords that indicate this fault
    content: str                    # Full procedure text for the LLM


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — FALLBACK KNOWLEDGE BASE (all 6 fault types)
# ═══════════════════════════════════════════════════════════════════════════

_KB_ADCS_GYRO_SEU = KBEntry(
    fault_class="ADCS_GYRO_SEU",
    title="ADCS Gyroscope Single-Event Upset (SEU) Recovery",
    trigger_cues=(
        "GYRO_A_RATE", "GYRO_B_RATE", "SEU_COUNTER", "SEU",
        "NaN", "attitude_error", "ATTITUDE_ERROR", "ADCS",
        "gyro", "gyroscope", "cosmic_ray", "radiation",
        "ADCS_ERROR_THRESHOLD", "attitude",
        # anomaly_detector.py parameter names
        "Gyro_rate_degs", "Attitude_error_deg", "gyro_rate",
    ),
    content="""\
PROCEDURE: ADCS Gyroscope Single-Event Upset (SEU) Recovery
SOURCE: Based on ECSS-E-ST-70-11C principles for safe mode recovery \
and ECSS-Q-ST-30-02 dependability requirements.

FAULT SIGNATURE:
- SEU_COUNTER spikes from baseline (0) to a non-zero value
- GYRO_A_RATE or GYRO_B_RATE returns NaN or a constant frozen value \
immediately after the SEU event
- ATTITUDE_ERROR grows progressively as ADCS loses attitude knowledge
- Safe mode triggered when ATTITUDE_ERROR exceeds threshold (typically 5 deg)

CAUSAL MECHANISM:
A cosmic ray or trapped particle strikes the gyroscope processor, causing \
a single-event upset (bit flip). This corrupts the rate measurement, making \
the ADCS unable to determine spacecraft orientation. Without valid attitude \
data, the spacecraft cannot maintain sun pointing, and ATTITUDE_ERROR grows \
until the safe mode threshold is exceeded.

KEY DIAGNOSTIC RULE:
Always check SEU_COUNTER first. If it spiked near the time GYRO_RATE went \
anomalous, this is radiation-induced — NOT a hardware failure. Recovery is \
software reset, not hardware replacement.

RECOVERY SEQUENCE:
1. CMD_VERIFY_SEU_COUNTER — Read SEU counter to confirm radiation event. \
Wait 5s. Verify: SEU_COUNTER read successfully. Risk: LOW.
2. CMD_GYRO_A_DRIVER_RESET — Software reset of the gyroscope driver to \
clear corrupted state from SEU. Wait 30s. Verify: GYRO_A_RATE returns a \
valid float value. Risk: LOW.
3. CMD_ATTITUDE_REACQUISITION — Use star tracker and sun sensor to \
re-establish attitude knowledge. Wait 60s. Verify: ATTITUDE_ERROR < 1 deg. \
Risk: MEDIUM.
4. CMD_SAFE_MODE_EXIT — Return spacecraft to nominal operations. Wait 30s. \
Verify: normal_mode_flag = 1. Risk: LOW.

SAFETY NOTES:
- Do NOT attempt attitude maneuvers until gyroscope health is confirmed \
(GYRO_A_RATE returns valid data).
- If SEU_COUNTER continues to increment after driver reset, suspect a \
persistent radiation environment — escalate to human review.
- If GYRO_A fails to recover after reset, switch to GYRO_B backup.""",
)

_KB_EPS_SOLAR_UNDERVOLT = KBEntry(
    fault_class="EPS_SOLAR_UNDERVOLT",
    title="EPS Solar Array Power Loss Recovery",
    trigger_cues=(
        "I_sa", "V_bat", "V_bus", "SoC", "solar", "SOLAR_ARRAY",
        "EPS", "battery", "power", "undervolt", "eclipse",
        "sunlit", "sun_angle", "EPS_FAULT",
    ),
    content="""\
PROCEDURE: EPS Solar Array Undervoltage Recovery
SOURCE: Based on ECSS-E-ST-70-11C power subsystem recovery and \
EPS fault management guidelines.

FAULT SIGNATURE:
- I_sa (solar array current) drops to near 0A while spacecraft is in sunlight
- V_bat drifts downward from nominal (28–33.6V) toward critical (<22V)
- Battery SoC begins falling as load exceeds generation
- EPS fault flag set when V_bus exits safe operating range

CAUSAL MECHANISM:
Solar array current drops to zero while the spacecraft is in a sunlit \
orbital position (not eclipse). This indicates either: (a) solar array \
hardware failure, (b) incorrect sun angle calculation causing the panels \
to point away from the sun, or (c) a deployment mechanism failure in a \
newly launched spacecraft.

KEY DIAGNOSTIC RULE:
Check orbital_position in the crash dump. If the spacecraft is sunlit and \
I_sa ≈ 0, this is NOT an eclipse. It is a genuine solar array or pointing \
fault. Do not dismiss it as an eclipse-related transient.

RECOVERY SEQUENCE:
1. CMD_VERIFY_SUN_ANGLE — Confirm spacecraft has valid sun pointing using \
sun sensor. Wait 10s. Verify: sun_sensor_angle < 90 deg. Risk: LOW.
2. CMD_SOLAR_ARRAY_A_RESET — Attempt to re-initialize the solar array \
drive electronics. Wait 30s. Verify: I_sa > 2A within 30s. Risk: LOW.
3. CMD_SWITCH_SOLAR_ARRAY — If Array A fails, switch to Array B or \
alternative power path. Wait 30s. Verify: I_sa recovery on alternate array. \
Risk: MEDIUM.
4. CMD_BATTERY_CONSERVATION — Shed non-critical loads to preserve remaining \
battery capacity. Wait 10s. Verify: SoC stabilizes. Risk: LOW.
5. CMD_SAFE_MODE_EXIT — Only after power generation is confirmed restored. \
Wait 30s. Verify: V_bat > 27V and rising. Risk: LOW.

SAFETY NOTES:
- NEVER command safe mode exit while battery SoC < 20%. The spacecraft \
could brown out during transition.
- If V_bat has dropped below 22V, this is CRITICAL. Minimize all commands \
until power is restored.
- Monitor V_bat trend after array reset — if not recovering within 60s, \
switch to backup array.""",
)

_KB_OBC_WATCHDOG = KBEntry(
    fault_class="OBC_WATCHDOG_OVERFLOW",
    title="OBC Software Watchdog Overflow Recovery",
    trigger_cues=(
        "CPU_LOAD", "WATCHDOG_COUNTER", "watchdog", "OBC",
        "software", "reboot", "memory", "CPU", "100%",
        "overflow", "loop", "infinite", "hang",
        "OBC_WATCHDOG",
    ),
    content="""\
PROCEDURE: OBC Software Watchdog Timeout Recovery
SOURCE: Based on ECSS-E-ST-70-11C onboard computer recovery procedures \
and software fault management requirements.

FAULT SIGNATURE:
- CPU_LOAD sustained at 100% (software infinite loop)
- Memory usage monotonically increasing (possible memory leak)
- WATCHDOG_COUNTER overflows (timer not refreshed due to stuck software)
- Forced reboot occurs, triggering safe mode

CAUSAL MECHANISM:
A software bug (triggered by unusual input data, timing condition, or \
memory corruption) causes the main flight software to enter an infinite \
loop or deadlock. The CPU saturates at 100%, preventing the watchdog timer \
from being refreshed. When the watchdog overflows, the OBC performs a \
forced hardware reboot, which triggers safe mode entry.

KEY DIAGNOSTIC RULE:
If CPU_LOAD = 100% AND WATCHDOG_COUNTER overflowed, this is near-certain \
an OBC software fault. Check memory usage trend — if monotonically \
increasing, confirms a runaway process. NOT a hardware fault.

RECOVERY SEQUENCE:
1. CMD_CONFIRM_COMMS_LOCK — Verify communications lock on low-gain antenna \
BEFORE any OBC operations. Wait 5s. Verify: TRANSPONDER_LOCK = 1. Risk: LOW.
2. CMD_OBC_CONTROLLED_REBOOT — Perform a clean controlled reboot of the \
onboard computer (not a power cycle). Wait 60s. Verify: CPU_LOAD returns \
to nominal (<70%). Risk: MEDIUM.
3. CMD_VERIFY_MEMORY_STATE — Check that memory usage is stable (not \
monotonically increasing). Wait 10s. Verify: memory_usage stable. Risk: LOW.
4. CMD_SAFE_MODE_EXIT — Return to nominal operations. Wait 30s. \
Verify: normal_mode_flag = 1. Risk: LOW.

SAFETY NOTES:
- CRITICAL: Always confirm comms lock on low-gain antenna BEFORE rebooting \
OBC. An OBC reboot without comms lock risks permanent loss of contact.
- If the same software fault recurs after reboot, suspect a persistent \
trigger — escalate to human review with requires_human_review = true.
- If CPU_LOAD does not drop after reboot, suspect hardware-level fault \
(processor damage). Set risk to HIGH.""",
)

_KB_TCS_THERMAL = KBEntry(
    fault_class="TCS_THERMAL_RUNAWAY",
    title="TCS Thermal Runaway — Heater Stuck ON Recovery",
    trigger_cues=(
        "TEMP_OBC", "TEMP_", "HEATER_ZONE", "thermal", "TCS",
        "temperature", "heater", "overheating", "stuck_on",
        "85", "survival", "runaway",
    ),
    content="""\
PROCEDURE: TCS Thermal Runaway Recovery (Heater Stuck ON)
SOURCE: Based on ECSS-E-ST-70-11C thermal control and ECSS-Q-ST-30-02 \
dependability requirements for thermal protection.

FAULT SIGNATURE:
- HEATER_ZONE_* remains in ON state without cycling OFF
- Component temperature rises continuously beyond operational range
- Critical threshold exceeded (>85°C for most electronics, >125°C for \
survival limit)
- Safe mode triggered by over-temperature protection

CAUSAL MECHANISM:
A heater control relay or thermostat fails in the closed (ON) position, \
causing continuous heating of a spacecraft zone. Without the normal \
on/off cycling, component temperature rises unchecked until it exceeds \
the safe operating range or survival limit.

KEY DIAGNOSTIC RULE:
Thermal runaway is TIME-CRITICAL. Unlike other faults, prolonged \
overheating causes permanent hardware damage. Recovery priority is to \
disable the stuck heater IMMEDIATELY, before other diagnostics.

RECOVERY SEQUENCE:
1. CMD_DISABLE_HEATER_ZONE — Disable the affected heater zone immediately. \
Wait 5s. Verify: HEATER_ZONE status = OFF. Risk: LOW.
2. CMD_MONITOR_TEMPERATURE — Monitor component temperature for cooling \
trend. Wait 120s. Verify: TEMP reading is decreasing. Risk: LOW.
3. CMD_VERIFY_THERMAL_MARGIN — Confirm temperature has returned to safe \
operating range. Wait 300s. Verify: TEMP < 50°C for electronics. Risk: LOW.
4. CMD_SAFE_MODE_EXIT — Only after temperature is within safe range. \
Wait 30s. Verify: normal_mode_flag = 1. Risk: LOW.

SAFETY NOTES:
- Thermal runaway can cause PERMANENT DAMAGE. Recovery urgency is higher \
than other fault types.
- If temperature has already exceeded survival limit (>125°C), the \
component may be permanently damaged. Set requires_human_review = true \
and confidence should be lower due to uncertainty about hardware state.
- Do NOT re-enable the heater zone until the control relay has been \
verified by ground team.""",
)

_KB_COMMS_TRANSPONDER = KBEntry(
    fault_class="COMMS_TRANSPONDER_LOSS",
    title="COMMS Transponder Loss Recovery",
    trigger_cues=(
        "TRANSPONDER_LOCK", "SNR", "COMMS", "transponder",
        "signal", "antenna", "comm", "communication",
        "loss_of_signal", "dB",
    ),
    content="""\
PROCEDURE: Communications Transponder Loss Recovery
SOURCE: Based on ECSS-E-ST-70-11C communications subsystem recovery \
and ground-link management procedures.

FAULT SIGNATURE:
- TRANSPONDER_LOCK drops from 1 to 0 (loss of carrier lock)
- SNR falls below 5 dB (severe signal degradation)
- Ground station loses telemetry and command link
- Safe mode may trigger automatically due to loss of ground contact \
timeout

CAUSAL MECHANISM:
Transponder hardware failure, power supply fault to the transponder, \
antenna pointing error, or frequency drift causes loss of the \
communication link between the spacecraft and ground station.

KEY DIAGNOSTIC RULE:
Without comms, no ground commands can reach the spacecraft. Recovery \
depends on onboard autonomous capability OR a pre-programmed backup \
transponder switch timer. Check if a backup transponder is available.

RECOVERY SEQUENCE:
1. CMD_SWITCH_BACKUP_TRANSPONDER — Switch to redundant transponder unit. \
Wait 30s. Verify: TRANSPONDER_LOCK = 1 on backup unit. Risk: MEDIUM.
2. CMD_VERIFY_SIGNAL_ACQUISITION — Confirm signal quality with ground \
station. Wait 15s. Verify: SNR > 10 dB. Risk: LOW.
3. CMD_CONFIRM_GROUND_CONTACT — Verify two-way communication link is \
established. Wait 10s. Verify: ground station confirms telemetry \
reception. Risk: LOW.
4. CMD_SAFE_MODE_EXIT — Return to nominal operations. Wait 30s. \
Verify: normal_mode_flag = 1. Risk: LOW.

SAFETY NOTES:
- This is a unique fault type: recovery may need to be fully autonomous \
since ground cannot send commands without the transponder.
- If backup transponder also fails, the spacecraft enters a comms blackout. \
Set requires_human_review = true (even though no one can command it — the \
flag is for the log record).
- For deep space missions, comm loss is especially critical due to long \
round-trip light times.""",
)

_KB_MULTI_CASCADE = KBEntry(
    fault_class="MULTI_CASCADE",
    title="Multi-Subsystem Cascade Failure Diagnosis",
    trigger_cues=(
        "cascade", "multi", "multiple", "chain", "propagat",
        "two subsystem", "cross-subsystem", "secondary",
        "downstream", "initiating",
    ),
    content="""\
PROCEDURE: Multi-Subsystem Cascade Failure Diagnosis and Recovery
SOURCE: Based on ECSS-E-ST-70-11C cross-subsystem fault propagation \
analysis and ECSS-Q-ST-30-02 failure mode identification.

FAULT SIGNATURE:
- Anomalies observed in 2 or more subsystems with temporal correlation
- Example cascade: ADCS gyro fault → spacecraft tumbles → solar panels \
lose sun pointing → I_sa drops → battery drains → EPS fault
- The MOST RECENT anomaly is often a SYMPTOM, not the root cause

CAUSAL MECHANISM:
A primary fault in one subsystem propagates to other subsystems through \
physical dependencies. Common cascades:
1. ADCS → EPS: Attitude loss → solar panels misaligned → power loss
2. EPS → TCS: Power loss → heaters disabled → thermal excursion
3. OBC → everything: Software crash → all subsystem control lost

KEY DIAGNOSTIC RULE:
ALWAYS identify the INITIATING fault (earliest anomaly in the timeline), \
not just the most recent symptom. Look at the event_log timestamps and \
pre_fault_telemetry progression to find which parameter went anomalous \
FIRST. Confidence should be LOWER for cascade faults (0.50–0.70) because \
the causal chain is more complex and ambiguous.

RECOVERY STRATEGY:
1. Address the root cause (initiating fault) FIRST.
2. Then recover downstream subsystems in dependency order.
3. Do NOT try to fix symptoms while the root cause is still active.

Example recovery for ADCS → EPS cascade:
1. Fix the gyro/ADCS fault (driver reset, attitude reacquisition)
2. Once attitude is restored, solar panels regain sun pointing
3. Verify power generation is recovering (I_sa rising, V_bat stabilizing)
4. Only then exit safe mode

SAFETY NOTES:
- ALWAYS set requires_human_review = true for cascade faults.
- Confidence should typically be 0.50–0.70 for cascades.
- If you cannot clearly identify the initiating fault, state this \
explicitly in reasoning_summary.
- Recovery order matters: fixing downstream symptoms before the root \
cause can make things worse.""",
)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — KNOWLEDGE BASE REGISTRY
# ═══════════════════════════════════════════════════════════════════════════

FALLBACK_KB: tuple[KBEntry, ...] = (
    _KB_ADCS_GYRO_SEU,
    _KB_EPS_SOLAR_UNDERVOLT,
    _KB_OBC_WATCHDOG,
    _KB_TCS_THERMAL,
    _KB_COMMS_TRANSPONDER,
    _KB_MULTI_CASCADE,
)

# Quick lookup by fault class name
_KB_BY_CLASS: dict[str, KBEntry] = {
    entry.fault_class: entry for entry in FALLBACK_KB
}

# Curated queries for each fault class (used by retrieve_by_fault_class
# when enriching with PDF RAG)
_FAULT_CLASS_QUERIES: dict[str, str] = {
    "ADCS_GYRO_SEU": (
        "single event upset gyroscope attitude sensor recovery "
        "safe mode ADCS SEU radiation"
    ),
    "EPS_SOLAR_UNDERVOLT": (
        "solar array power undervoltage battery voltage recovery "
        "EPS power subsystem safe mode"
    ),
    "OBC_WATCHDOG_OVERFLOW": (
        "onboard computer watchdog timeout software reboot recovery "
        "OBC CPU safe mode"
    ),
    "TCS_THERMAL_RUNAWAY": (
        "thermal control heater stuck overtemperature protection "
        "TCS thermal runaway safe mode"
    ),
    "COMMS_TRANSPONDER_LOSS": (
        "transponder communication loss signal acquisition recovery "
        "COMMS antenna safe mode"
    ),
    "MULTI_CASCADE": (
        "multi-subsystem cascade failure propagation root cause "
        "fault isolation recovery"
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — PDF RAG INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════

def _get_embedding_fn() -> Any:
    """Create a sentence-transformers embedding function for ChromaDB.

    Uses all-MiniLM-L6-v2 (free, local, no API key needed).
    Returns None (triggering fallback KB) if sentence-transformers or
    chromadb embedding utilities are not importable.
    """
    # Try the standard chromadb embedding function import path
    try:
        from chromadb.utils.embedding_functions import (
            SentenceTransformerEmbeddingFunction,
        )
        return SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL,
        )
    except (ImportError, Exception) as e:
        logger.warning("Could not create sentence-transformers embedding fn: %s", e)

    # Fallback import path for older/newer chromadb versions
    try:
        from chromadb.utils import embedding_functions as ef_module
        return ef_module.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL,
        )
    except (ImportError, Exception) as e:
        logger.warning("Fallback embedding import also failed: %s", e)

    return None


def _load_and_chunk_pdfs() -> list[dict[str, str]] | None:
    """Load PDFs from ECSS_DATA_DIR and chunk them.

    Returns a list of dicts with keys:
      - "text": chunk text
      - "source": source filename
      - "page": page label (if available)
      - "chunk_id": unique identifier

    Returns None if loading fails or no PDFs found.
    """
    if not os.path.isdir(ECSS_DATA_DIR):
        logger.info("ECSS_DATA_DIR does not exist: %s", ECSS_DATA_DIR)
        return None

    pdf_files = [
        f for f in os.listdir(ECSS_DATA_DIR)
        if f.lower().endswith(".pdf")
    ]
    if not pdf_files:
        logger.info("No PDF files found in %s", ECSS_DATA_DIR)
        return None

    logger.info("Found %d PDF(s) in %s: %s", len(pdf_files), ECSS_DATA_DIR,
                pdf_files)

    try:
        from llama_index.core import SimpleDirectoryReader
        from llama_index.core.node_parser import SentenceSplitter
    except ImportError as e:
        logger.warning("LlamaIndex not available for PDF loading: %s", e)
        return None

    # Load documents from all PDFs — skip individual failures
    all_documents = []
    loaded_count = 0
    for pdf_file in pdf_files:
        pdf_path = os.path.join(ECSS_DATA_DIR, pdf_file)
        try:
            reader = SimpleDirectoryReader(input_files=[pdf_path])
            docs = reader.load_data()
            all_documents.extend(docs)
            loaded_count += 1
            logger.info("Loaded %s: %d pages/sections", pdf_file, len(docs))
        except Exception as e:
            logger.warning("Failed to load %s (skipping): %s", pdf_file, e)

    if not all_documents:
        logger.warning("No documents loaded from any PDF")
        return None

    # Chunk into manageable pieces
    try:
        splitter = SentenceSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        nodes = splitter.get_nodes_from_documents(all_documents)
        logger.info(
            "Chunked %d documents into %d nodes "
            "(chunk_size=%d, overlap=%d)",
            len(all_documents), len(nodes), CHUNK_SIZE, CHUNK_OVERLAP,
        )
    except Exception as e:
        logger.warning("Chunking failed: %s", e)
        return None

    # Convert to simple dicts
    chunks: list[dict[str, str]] = []
    for i, node in enumerate(nodes):
        text = node.get_content().strip()
        if len(text) < 20:
            continue  # Skip very short chunks (headers, page numbers)

        metadata = getattr(node, "metadata", {}) or {}
        source = metadata.get("file_name", "unknown")
        page = str(metadata.get("page_label", "?"))

        chunks.append({
            "text": text,
            "source": source,
            "page": page,
            "chunk_id": f"ecss_chunk_{i:04d}",
        })

    _rag_status.pdf_count = loaded_count
    logger.info(
        "Prepared %d chunks from %d PDF(s) (skipped %d tiny chunks)",
        len(chunks), loaded_count, len(nodes) - len(chunks),
    )
    return chunks if chunks else None


def initialize_pdf_rag(force_rebuild: bool = False) -> bool:
    """Initialize the PDF RAG pipeline.

    Behavior:
      1. Scan data/ecss/ for PDFs
      2. If already indexed (ChromaDB has data) and not force_rebuild → reuse
      3. Load PDFs with LlamaIndex, chunk, embed, store in ChromaDB
      4. Cache the collection in module globals for reuse

    Args:
        force_rebuild: If True, delete existing collection and rebuild.

    Returns:
        True if PDF RAG is ready, False otherwise (caller should
        fall back to FALLBACK_KB).
    """
    global _chroma_collection, _embedding_fn, _rag_status

    # ── Early exit when PDF RAG is explicitly disabled (saves ~400 MB RAM) ──
    if DISABLE_PDF_RAG:
        _rag_status.initialized = True
        _rag_status.available = False
        _rag_status.last_error = "PDF RAG disabled via DISABLE_PDF_RAG env var"
        return False

    if _rag_status.initialized and _rag_status.available and not force_rebuild:
        return True

    _rag_status.initialized = True

    # --- Step 1: Get embedding function ---
    _embedding_fn = _get_embedding_fn()
    if _embedding_fn is None:
        _rag_status.available = False
        _rag_status.last_error = "sentence-transformers or embedding fn unavailable"
        logger.info("PDF RAG skipped: %s", _rag_status.last_error)
        return False

    # --- Step 2: Connect to ChromaDB (local persistent mode) ---
    try:
        import chromadb
    except ImportError:
        _rag_status.available = False
        _rag_status.last_error = "chromadb not installed"
        return False

    try:
        os.makedirs(CHROMA_DB_DIR, exist_ok=True)
        client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    except Exception as e:
        _rag_status.available = False
        _rag_status.last_error = f"ChromaDB init failed: {e}"
        logger.warning("ChromaDB init failed: %s", e)
        return False

    # --- Step 3: Get or create collection ---
    try:
        if force_rebuild:
            try:
                client.delete_collection(CHROMA_COLLECTION_NAME)
                logger.info("Deleted existing collection for rebuild")
            except Exception:
                pass  # Collection may not exist yet

        collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            embedding_function=_embedding_fn,
        )
    except Exception as e:
        _rag_status.available = False
        _rag_status.last_error = f"Collection create failed: {e}"
        logger.warning("ChromaDB collection error: %s", e)
        return False

    # --- Step 4: Check if already populated ---
    existing_count = collection.count()
    if existing_count > 0 and not force_rebuild:
        _chroma_collection = collection
        _rag_status.available = True
        _rag_status.chunk_count = existing_count
        logger.info(
            "Reusing existing ChromaDB collection with %d chunks",
            existing_count,
        )
        return True

    # --- Step 5: Load, chunk, and index PDFs ---
    chunks = _load_and_chunk_pdfs()
    if not chunks:
        _rag_status.available = False
        _rag_status.last_error = "No PDF chunks produced"
        return False

    # Add chunks to ChromaDB in batches (ChromaDB has batch limits)
    BATCH_SIZE = 100
    try:
        for batch_start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[batch_start:batch_start + BATCH_SIZE]
            collection.add(
                documents=[c["text"] for c in batch],
                metadatas=[
                    {"source": c["source"], "page": c["page"]}
                    for c in batch
                ],
                ids=[c["chunk_id"] for c in batch],
            )
        logger.info("Indexed %d chunks into ChromaDB", len(chunks))
    except Exception as e:
        _rag_status.available = False
        _rag_status.last_error = f"ChromaDB indexing failed: {e}"
        logger.warning("Failed to index chunks: %s", e)
        return False

    _chroma_collection = collection
    _rag_status.available = True
    _rag_status.chunk_count = len(chunks)
    logger.info(
        "PDF RAG initialized: %d PDFs, %d chunks",
        _rag_status.pdf_count, _rag_status.chunk_count,
    )
    return True


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — PDF RAG RETRIEVAL
# ═══════════════════════════════════════════════════════════════════════════

def _try_pdf_rag(query: str, top_k: int = DEFAULT_TOP_K) -> list[str] | None:
    """Run similarity retrieval against the ChromaDB ECSS index.

    Lazy-initializes on first call. Returns None if RAG is unavailable
    or returns no useful results (caller falls back to FALLBACK_KB).

    Args:
        query: Search query combining fault description and parameter names.
        top_k: Number of chunks to retrieve.

    Returns:
        List of formatted procedure snippet strings, or None.
    """
    global _rag_status

    # Lazy initialization
    if not _rag_status.initialized:
        if not initialize_pdf_rag():
            return None

    if not _rag_status.available or _chroma_collection is None:
        return None

    try:
        results = _chroma_collection.query(
            query_texts=[query],
            n_results=min(top_k, _rag_status.chunk_count or top_k),
        )
    except Exception as e:
        logger.warning("ChromaDB query failed: %s", e)
        _rag_status.last_error = f"Query failed: {e}"
        return None

    # Extract and format results
    if not results or not results.get("documents"):
        return None

    documents = results["documents"][0]  # First query's results
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not documents:
        return None

    formatted: list[str] = []
    for i, doc_text in enumerate(documents):
        if not doc_text or len(doc_text.strip()) < 20:
            continue

        # Filter garbled/binary text from scanned PDFs
        # If >30% of chars are non-printable ASCII, skip this chunk
        printable_ratio = sum(
            1 for c in doc_text if 32 <= ord(c) < 127 or c in ('\n', '\t', '\r')
        ) / max(len(doc_text), 1)
        if printable_ratio < 0.70:
            logger.debug(
                "Skipping garbled chunk (printable_ratio=%.2f): %.40r",
                printable_ratio, doc_text
            )
            continue

        # Extract metadata
        meta = metadatas[i] if i < len(metadatas) else {}
        source = meta.get("source", "ECSS standard")
        page = meta.get("page", "?")
        dist = distances[i] if i < len(distances) else None

        # Format as a grounded snippet with provenance
        dist_str = f" (distance: {dist:.3f})" if dist is not None else ""
        header = f"[ECSS Retrieved — {source}, page {page}{dist_str}]"
        formatted.append(f"{header}\n{doc_text.strip()}")

    _rag_status.last_source = "pdf_rag"
    logger.info(
        "PDF RAG returned %d chunks for query: %.60s...",
        len(formatted), query,
    )
    return formatted if formatted else None


# ===========================================================================
# SECTION 5b -- GEMINI RAG (precomputed numpy vectors, zero PyTorch)
# ===========================================================================

def _load_gemini_rag() -> bool:
    """Load precomputed Gemini embeddings from ecss_gemini_rag.npz.

    Returns True on success, False if the file doesn't exist or is invalid.
    Safe to call multiple times (idempotent).
    """
    global _gemini_vectors, _gemini_chunks, _gemini_sources, _gemini_rag_loaded

    if _gemini_rag_loaded:
        return _gemini_vectors is not None

    _gemini_rag_loaded = True  # mark attempted so we don't retry on every call

    if not os.path.exists(GEMINI_RAG_NPZ):
        logger.warning(
            "USE_GEMINI_RAG=true but %s not found. "
            "Run: python sentinel/backend/data_tools/precompute_gemini_embeddings.py",
            GEMINI_RAG_NPZ,
        )
        return False

    try:
        import numpy as np
        data = np.load(GEMINI_RAG_NPZ, allow_pickle=True)
        _gemini_vectors = data["vectors"]   # (N, D) float32, L2-normalised
        _gemini_chunks  = data["chunks"]    # (N,) object array of str
        _gemini_sources = data["sources"]   # (N,) object array of str
        logger.info(
            "Gemini RAG loaded: %d chunks x %d dims from %s",
            _gemini_vectors.shape[0], _gemini_vectors.shape[1], GEMINI_RAG_NPZ,
        )
        return True
    except Exception as e:
        logger.error("Failed to load Gemini RAG file %s: %s", GEMINI_RAG_NPZ, e)
        _gemini_vectors = None
        return False


def _query_gemini_rag(query: str, top_k: int = 3) -> list[str]:
    """Embed query via Gemini API, return top-k chunks by cosine similarity.

    Requires:
      - GEMINI_API_KEY in environment
      - ecss_gemini_rag.npz loaded (call _load_gemini_rag() first)
      - numpy (already a project dependency)

    Raises on API error (caller should catch and fall back to KB).
    """
    import numpy as np

    if _gemini_vectors is None:
        raise RuntimeError("Gemini RAG vectors not loaded")

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    from google import genai
    client = genai.Client(api_key=api_key)

    result = client.models.embed_content(
        model=GEMINI_EMBED_MODEL,
        contents=[query],
    )
    q_vec = np.array(result.embeddings[0].values, dtype=np.float32)
    norm = np.linalg.norm(q_vec)
    if norm > 0:
        q_vec /= norm

    # Cosine similarity = dot product (vectors are already L2-normalised)
    scores = _gemini_vectors @ q_vec          # shape (N,)
    top_idx = np.argsort(scores)[::-1][:top_k]

    chunks = []
    for idx in top_idx:
        source = str(_gemini_sources[idx]) if _gemini_sources is not None else "ECSS"
        text   = str(_gemini_chunks[idx])
        chunks.append(f"[Source: {source}]\n{text}")

    logger.info(
        "Gemini RAG query returned %d chunks (top score=%.3f)",
        len(chunks), float(scores[top_idx[0]]) if len(top_idx) else 0,
    )
    return chunks


# ===========================================================================
# SECTION 6 -- PUBLIC RETRIEVAL API
# ===========================================================================

def retrieve_procedures(

    query: str = "",
    fault_cues: list[str] | None = None,
    top_k: int = DEFAULT_TOP_K,
    use_pdf_rag: bool = True,
) -> list[str]:
    """Retrieve relevant engineering procedure snippets for the agent.

    This is the public API that agent.py calls. It returns a list of
    plain-text procedure strings suitable for passing directly to:
        agent.analyze_crash_dump(retrieved_procedures=result)

    Retrieval strategy (when use_pdf_rag=True):
      1. Build a combined query from query + fault_cues
      2. Try PDF RAG (ChromaDB similarity search)
      3. If PDF RAG returns results → use them
      4. Otherwise → keyword-match against FALLBACK_KB
      5. If nothing matches → return MULTI_CASCADE catch-all

    Args:
        query: Free-text query string. Typically the crash dump scenario_id,
            safe_mode_trigger, or a summary of anomalies.
        fault_cues: Optional list of anomalous parameter names or keywords
            from the crash dump (e.g. ["GYRO_A_RATE", "SEU_COUNTER"]).
        top_k: Maximum number of procedure snippets to return.
        use_pdf_rag: If True, try PDF RAG before falling back to KB.
            Set to False for ablation studies or guaranteed fast responses.

    Returns:
        List of procedure snippet strings, ranked by relevance (best first).
        Never returns an empty list — always returns at least the
        MULTI_CASCADE entry as a catch-all.
    """
    global _rag_status

    # Build combined query for PDF RAG / Gemini RAG
    combined_query = query
    if fault_cues:
        combined_query += " " + " ".join(fault_cues)

    # ── USE_GEMINI_RAG path (precomputed numpy vectors, zero PyTorch) ─────────
    if USE_GEMINI_RAG and use_pdf_rag and combined_query.strip():
        try:
            if _load_gemini_rag():
                results = _query_gemini_rag(combined_query, top_k)
                if results:
                    _rag_status.last_source = "gemini_rag"
                    return results
                logger.info("Gemini RAG returned no results, falling back to KB")
            else:
                logger.warning("Gemini RAG not available, falling back to KB")
        except Exception as exc:
            logger.warning("Gemini RAG query failed (non-fatal): %s", exc)
        _rag_status.last_source = "fallback_kb"
        return _retrieve_from_fallback(query, fault_cues, top_k)

    # ── Standard ChromaDB / PDF RAG path ─────────────────────────────────────
    if use_pdf_rag and combined_query.strip():
        pdf_results = _try_pdf_rag(combined_query, top_k)
        if pdf_results:
            return pdf_results
        if _rag_status.initialized:
            logger.info(
                "PDF RAG returned no results, falling back to FALLBACK_KB"
            )

    # Fall back to keyword-based KB matching
    _rag_status.last_source = "fallback_kb"
    return _retrieve_from_fallback(query, fault_cues, top_k)



def _retrieve_from_fallback(
    query: str,
    fault_cues: list[str] | None,
    top_k: int,
) -> list[str]:
    """Match query and fault cues against FALLBACK_KB entries.

    Scoring:
      - Each trigger cue that appears in (query + fault_cues) adds 1 point
      - Case-insensitive substring matching
      - Entries are ranked by score, top_k returned
      - If fewer than top_k entries match, pads with the next best entries
        (score=0 entries included to meet top_k) so callers always get
        the requested number of entries when KB is large enough.
      - If no entry scores > 0, returns MULTI_CASCADE as the first entry
    """
    combined_text = query.lower()
    if fault_cues:
        combined_text += " " + " ".join(c.lower() for c in fault_cues)

    scored: list[tuple[int, KBEntry]] = []
    for entry in FALLBACK_KB:
        score = sum(
            1 for cue in entry.trigger_cues
            if cue.lower() in combined_text
        )
        scored.append((score, entry))

    # Sort by score descending, preserving original order for equal scores
    scored.sort(key=lambda x: x[0], reverse=True)

    # Find the best-matching entries
    matched_positive = [entry for score, entry in scored if score > 0]

    if not matched_positive:
        # No entries matched — always lead with MULTI_CASCADE catch-all,
        # then pad with other KB entries up to top_k
        logger.info(
            "No KB entries matched query/cues — returning MULTI_CASCADE "
            "as catch-all"
        )
        results: list[KBEntry] = [_KB_MULTI_CASCADE]
        for _score, entry in scored:
            if entry.fault_class != "MULTI_CASCADE" and len(results) < top_k:
                results.append(entry)
        return [entry.content for entry in results[:top_k]]

    # Have some matched entries; pad with next-best if needed
    if len(matched_positive) >= top_k:
        return [entry.content for entry in matched_positive[:top_k]]

    # Fewer matches than requested: add unmatched entries in order
    results_with_data: list[KBEntry] = list(matched_positive)
    matched_classes = {e.fault_class for e in matched_positive}
    for _score, entry in scored:
        if entry.fault_class not in matched_classes and len(results_with_data) < top_k:
            results_with_data.append(entry)
    return [entry.content for entry in results_with_data[:top_k]]


def retrieve_by_fault_class(
    fault_class: str,
    use_pdf_rag: bool = False,
) -> str | None:
    """Retrieve a specific KB entry by exact fault class name.

    Useful for testing, evaluation, and direct lookup when the fault
    type is already known (e.g. from Person 1's ground truth labels).

    Args:
        fault_class: Exact fault class string (e.g. "ADCS_GYRO_SEU").
        use_pdf_rag: If True, also retrieve related ECSS chunks and
            append them to the fallback KB content.

    Returns:
        The procedure content string (optionally enriched with PDF RAG
        context), or None if fault_class not found.
    """
    entry = _KB_BY_CLASS.get(fault_class)
    if entry is None:
        return None

    base_content = entry.content

    # Optionally enrich with PDF RAG
    if use_pdf_rag:
        curated_query = _FAULT_CLASS_QUERIES.get(fault_class, fault_class)
        pdf_results = _try_pdf_rag(curated_query, top_k=2)
        if pdf_results:
            enrichment = "\n\n---\nADDITIONAL ECSS CONTEXT:\n" + "\n\n".join(
                pdf_results
            )
            return base_content + enrichment

    return base_content


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 — UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def list_available_entries() -> list[dict[str, str]]:
    """List all available KB entries with their fault classes and titles.

    Useful for debugging, demo prep, and Person 4's /api/scenarios endpoint.
    """
    return [
        {"fault_class": e.fault_class, "title": e.title}
        for e in FALLBACK_KB
    ]


def get_rag_status() -> RAGStatus:
    """Return the current RAG subsystem status for diagnostics."""
    return _rag_status


def reset_rag_state() -> None:
    """Reset all RAG module state (for testing)."""
    global _chroma_collection, _embedding_fn, _rag_status
    _chroma_collection = None
    _embedding_fn = None
    _rag_status = RAGStatus()
