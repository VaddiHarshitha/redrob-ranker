"""
Embed all 100K candidates using ``all-MiniLM-L6-v2``.

Outputs
-------
artifacts/candidate_embeddings.npy — float32 array, shape (100000, 384), ~153 MB
artifacts/candidate_ids.txt        — ordered list of candidate IDs (one per line)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from util.candidate_text import build_candidate_text

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_candidate_texts(candidates_file: str) -> tuple[list[str], list[str]]:
    """
    Read a JSONL candidates file and return parallel lists of IDs and embed texts.

    Parameters
    ----------
    candidates_file
        Path to the candidates JSONL file.

    Returns
    -------
    tuple[list[str], list[str]]
        (candidate_ids, candidate_texts) in matching order.
    """
    candidate_ids: list[str] = []
    candidate_texts: list[str] = []

    with open(candidates_file, encoding="utf-8") as fh:
        for line in tqdm(fh, desc="Reading candidates"):
            record = json.loads(line)
            cid = record.get("candidate_id", "")
            text = build_candidate_text(record)
            candidate_ids.append(cid)
            candidate_texts.append(text)

    return candidate_ids, candidate_texts


def embed_texts(texts: list[str], model: SentenceTransformer, batch_size: int = 128) -> np.ndarray:
    """
    Embed a list of texts using a SentenceTransformer.

    Parameters
    ----------
    texts
        List of text strings to embed.
    model
        Loaded SentenceTransformer instance.
    batch_size
        Batch size for encoding (default 128).

    Returns
    -------
    np.ndarray
        float32 array of shape (len(texts), 384), L2-normalised.
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def run(
    candidates_file: str = "data/candidates.jsonl",
    artifacts_dir: str = "artifacts",
    batch_size: int = 128,
) -> None:
    """
    Orchestrate the full candidate embedding pipeline.

    Loads the SentenceTransformer, reads all candidates from the JSONL file,
    embeds them in batches, and saves the artefacts.

    Parameters
    ----------
    candidates_file
        Path to the candidates JSONL file.
    artifacts_dir
        Directory where output files are written.
    batch_size
        Batch size for SentenceTransformer encoding (default 128).
    """
    # 1. Load model
    print("[embed_candidates] Loading SentenceTransformer (all-MiniLM-L6-v2) …")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 2. Read candidates
    candidate_ids, candidate_texts = load_candidate_texts(candidates_file)
    print(f"[embed_candidates] Loaded {len(candidate_ids)} candidates")

    # 3. Embed
    embeddings = embed_texts(candidate_texts, model, batch_size=batch_size)
    print(f"[embed_candidates] Embeddings shape: {embeddings.shape}")

    # 4. Save artefacts
    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = out_dir / "candidate_embeddings.npy"
    np.save(embeddings_path, embeddings)
    print(f"[embed_candidates] Embeddings saved → {embeddings_path}")

    ids_path = out_dir / "candidate_ids.txt"
    ids_path.write_text("\n".join(candidate_ids), encoding="utf-8")
    print(f"[embed_candidates] IDs saved → {ids_path}")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    candidates_file = sys.argv[1] if len(sys.argv) > 1 else "data/candidates.jsonl"
    artifacts_dir = sys.argv[2] if len(sys.argv) > 2 else "artifacts"
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    run(candidates_file=candidates_file, artifacts_dir=artifacts_dir, batch_size=batch_size)