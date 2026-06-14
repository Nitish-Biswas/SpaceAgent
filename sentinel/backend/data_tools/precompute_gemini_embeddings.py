"""
precompute_gemini_embeddings.py
───────────────────────────────
One-time script: chunks ECSS PDFs with LlamaIndex → embeds each chunk via
Gemini text-embedding-004 API → saves to data/ecss_gemini_rag.npz

Run once locally (needs GEMINI_API_KEY and the ECSS PDFs):
    python sentinel/backend/data_tools/precompute_gemini_embeddings.py

The resulting .npz file (~5 MB) is committed to git so the production server
never has to load sentence-transformers or PyTorch.

Runtime usage (via rag.py with USE_GEMINI_RAG=true):
  - 0 MB for model weights (vectors are pre-baked)
  - 1 Gemini embed call per user query  (~1 API unit, negligible)
  - Cosine similarity via numpy (already a dependency)
"""

import os
import sys
import json
import time
import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
THIS_DIR   = Path(__file__).parent
BACKEND    = THIS_DIR.parent
DATA_DIR   = BACKEND / "data"
ECSS_DIR   = DATA_DIR / "ecss"
OUTPUT_NPZ = DATA_DIR / "ecss_gemini_rag.npz"

CHUNK_SIZE    = 400   # words per chunk
CHUNK_OVERLAP = 50    # word overlap between chunks
EMBED_MODEL   = "text-embedding-004"
BATCH_SIZE    = 50    # Gemini allows up to 100 per batch call
SLEEP_BETWEEN = 0.5   # seconds between batch calls (rate limit safety)

# ── Load env ─────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(BACKEND.parent / ".env")
load_dotenv(BACKEND / ".env")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    log.error("GEMINI_API_KEY not set — export it or add to sentinel/.env")
    sys.exit(1)


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract plain text from a PDF using pypdf (no heavy deps)."""
    try:
        import pypdf
        reader = pypdf.PdfReader(str(pdf_path))
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages.append(text)
        return "\n".join(pages)
    except Exception as e:
        log.warning("pypdf failed for %s: %s", pdf_path.name, e)
        return ""


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping word-level chunks."""
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if len(chunk.strip()) > 50:   # skip tiny chunks
            chunks.append(chunk.strip())
    return chunks


def embed_batch(client, texts: List[str]) -> List[List[float]]:
    """Embed a batch of texts using Gemini text-embedding-004."""
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
    )
    # result.embeddings is a list of ContentEmbedding objects
    return [e.values for e in result.embeddings]


def main():
    # ── 1. Find PDFs ─────────────────────────────────────────────────────────
    pdfs = list(ECSS_DIR.glob("*.pdf"))
    if not pdfs:
        log.error("No PDFs found in %s", ECSS_DIR)
        log.error("Add ECSS PDF files and re-run.")
        sys.exit(1)
    log.info("Found %d PDF(s): %s", len(pdfs), [p.name for p in pdfs])

    # ── 2. Extract + chunk ───────────────────────────────────────────────────
    all_chunks: List[str] = []
    all_sources: List[str] = []

    for pdf_path in pdfs:
        log.info("Extracting text from %s ...", pdf_path.name)
        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            log.warning("  → No text extracted (possibly scanned PDF), skipping.")
            continue
        chunks = chunk_text(text)
        log.info("  → %d chunks", len(chunks))
        all_chunks.extend(chunks)
        all_sources.extend([pdf_path.name] * len(chunks))

    if not all_chunks:
        log.error("No text chunks produced. Cannot create embeddings.")
        sys.exit(1)

    log.info("Total chunks to embed: %d", len(all_chunks))

    # ── 3. Embed with Gemini ─────────────────────────────────────────────────
    from google import genai
    client = genai.Client(api_key=GEMINI_API_KEY)

    all_embeddings: List[List[float]] = []
    total_batches = (len(all_chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(total_batches):
        start = batch_idx * BATCH_SIZE
        end   = min(start + BATCH_SIZE, len(all_chunks))
        batch = all_chunks[start:end]

        log.info("Embedding batch %d/%d (chunks %d–%d)...",
                 batch_idx + 1, total_batches, start, end - 1)
        try:
            vecs = embed_batch(client, batch)
            all_embeddings.extend(vecs)
        except Exception as e:
            log.error("Batch %d failed: %s", batch_idx, e)
            log.error("Partial results: %d embeddings so far.", len(all_embeddings))
            sys.exit(1)

        if batch_idx < total_batches - 1:
            time.sleep(SLEEP_BETWEEN)

    log.info("Embedding complete. Vectors: %d × %d",
             len(all_embeddings), len(all_embeddings[0]))

    # ── 4. Save .npz ─────────────────────────────────────────────────────────
    vectors = np.array(all_embeddings, dtype=np.float32)
    # Normalize for cosine similarity (dot product = cosine sim after L2 norm)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vectors_norm = vectors / norms

    np.savez_compressed(
        OUTPUT_NPZ,
        vectors=vectors_norm,                          # shape: (N, D), float32, L2-normalised
        chunks=np.array(all_chunks, dtype=object),    # shape: (N,), str
        sources=np.array(all_sources, dtype=object),  # shape: (N,), str
        embed_model=np.array([EMBED_MODEL]),           # metadata
    )

    size_mb = OUTPUT_NPZ.stat().st_size / 1024 / 1024
    log.info("Saved %s (%.1f MB)", OUTPUT_NPZ, size_mb)
    log.info("")
    log.info("✅  Done! Now set USE_GEMINI_RAG=true on your hosting platform.")
    log.info("    The precomputed vectors will be loaded instead of sentence-transformers.")


if __name__ == "__main__":
    main()
