"""
Narrow 100K candidates → 300 via embedding similarity + behavioral score.

Outputs
-------
artifacts/shortlist.jsonl         — top N full candidate records + pre-scores
artifacts/behavioral_scores.json  — {candidate_id: score} for all 100K
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from util.behavioral import compute_behavioral_score

# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

SHORTLIST_EMBEDDING_WEIGHT = 0.75
SHORTLIST_BEHAVIORAL_WEIGHT = 0.25

# ---------------------------------------------------------------------------
# Semantic scoring
# ---------------------------------------------------------------------------

def compute_semantic_scores(embeddings: np.ndarray, jd_embedding: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity (as dot product of L2-normalised vectors) for all candidates.

    Both arrays must already be L2-normalised.
    Maps raw similarity [-1, 1] → [0, 1] via (raw + 1.0) / 2.0.

    Parameters
    ----------
    embeddings
        float32 array of shape (N, 384), L2-normalised.
    jd_embedding
        float32 array of shape (384,), L2-normalised.

    Returns
    -------
    np.ndarray
        float array of shape (N,) with values in [0, 1].
    """
    # embeddings @ jd_embedding = (N, 384) @ (384,) = (N,)
    raw = embeddings @ jd_embedding
    return (raw + 1.0) / 2.0


# ---------------------------------------------------------------------------
# Behavioral scoring
# ---------------------------------------------------------------------------

def compute_all_behavioral_scores(candidates_file: str) -> dict[str, float]:
    """
    Compute behavioral scores for all candidates in one pass.

    Parameters
    ----------
    candidates_file
        Path to the candidates JSONL file.

    Returns
    -------
    dict[str, float]
        Mapping from candidate_id to behavioral score in [0, 1].
    """
    scores: dict[str, float] = {}

    with open(candidates_file, encoding="utf-8") as fh:
        for line in tqdm(fh, desc="Behavioral scoring"):
            record = json.loads(line)
            cid = record.get("candidate_id", "")
            if cid:
                scores[cid] = compute_behavioral_score(record)

    return scores


# ---------------------------------------------------------------------------
# Top-N selection
# ---------------------------------------------------------------------------

def select_top_ids(
    candidate_ids: list[str],
    combined_scores: np.ndarray,
    size: int,
) -> set[str]:
    """
    Return the candidate IDs with the highest combined scores.

    Parameters
    ----------
    candidate_ids
        Ordered list of candidate IDs (index matches ``combined_scores``).
    combined_scores
        float array of shape (N,) with combined scores.
    size
        Number of top IDs to return.

    Returns
    -------
    set[str]
        Set of shortlisted candidate IDs.
    """
    # argsort in descending order
    indices = np.argsort(combined_scores)[::-1][:size]
    return {candidate_ids[i] for i in indices}


# ---------------------------------------------------------------------------
# Shortlist record collection
# ---------------------------------------------------------------------------

def collect_shortlist_records(
    candidates_file: str,
    top_ids: set[str],
    semantic_scores: np.ndarray,
    behavioral_scores: dict[str, float],
    candidate_ids: list[str],
    embedding_weight: float = SHORTLIST_EMBEDDING_WEIGHT,
    behavioral_weight: float = SHORTLIST_BEHAVIORAL_WEIGHT,
) -> list[dict]:
    """
    Second pass through candidates: collect only shortlisted records and attach scores.

    Parameters
    ----------
    candidates_file
        Path to the candidates JSONL file.
    top_ids
        Set of shortlisted candidate IDs.
    semantic_scores
        Array of semantic scores indexed by position in candidate_ids.
    behavioral_scores
        Dict of candidate_id → behavioral score.
    candidate_ids
        Ordered list of candidate IDs (same order as embeddings were saved).
    embedding_weight
        Weight for semantic component.
    behavioral_weight
        Weight for behavioral component.

    Returns
    -------
    list[dict]
        Shortlisted records, each enriched with _semantic_score, _behavioral_score,
        and _combined_score, sorted by combined_score descending.
    """
    # Build a quick lookup: candidate_id → index
    id_to_index = {cid: i for i, cid in enumerate(candidate_ids)}

    shortlist: list[dict] = []

    with open(candidates_file, encoding="utf-8") as fh:
        for line in tqdm(fh, desc="Collecting shortlist records"):
            record = json.loads(line)
            cid = record.get("candidate_id", "")
            if cid not in top_ids:
                continue

            idx = id_to_index[cid]
            sem_score = float(semantic_scores[idx])
            beh_score = float(behavioral_scores.get(cid, 0.0))
            combined = embedding_weight * sem_score + behavioral_weight * beh_score

            enriched = dict(record)
            enriched["_semantic_score"] = sem_score
            enriched["_behavioral_score"] = beh_score
            enriched["_combined_score"] = combined
            shortlist.append(enriched)

    shortlist.sort(key=lambda x: x["_combined_score"], reverse=True)
    return shortlist


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(
    candidates_file: str = "data/candidates.jsonl",
    artifacts_dir: str = "artifacts",
    shortlist_size: int = 300,
) -> None:
    """
    Orchestrate the shortlisting pipeline.

    Loads pre-computed embeddings, scores all candidates on semantic similarity
    and behavioral signals, combines the two scores, selects the top-N, and
    writes the shortlist JSONL and behavioral scores JSON.

    Parameters
    ----------
    candidates_file
        Path to the candidates JSONL file.
    artifacts_dir
        Directory where artefacts are read from / written to.
    shortlist_size
        Number of candidates to include in the shortlist.
    """
    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load artefacts
    embeddings_path = out_dir / "candidate_embeddings.npy"
    ids_path = out_dir / "candidate_ids.txt"
    jd_embedding_path = out_dir / "jd_embedding.npy"

    print(f"[build_shortlist] Loading embeddings from {embeddings_path} …")
    embeddings = np.load(embeddings_path)  # shape (N, 384)
    print(f"[build_shortlist] Embeddings shape: {embeddings.shape}")

    print(f"[build_shortlist] Loading candidate IDs from {ids_path} …")
    candidate_ids = ids_path.read_text(encoding="utf-8").splitlines()
    print(f"[build_shortlist] Candidate IDs: {len(candidate_ids)}")

    print(f"[build_shortlist] Loading JD embedding from {jd_embedding_path} …")
    jd_embedding = np.load(jd_embedding_path)  # shape (384,)
    print(f"[build_shortlist] JD embedding shape: {jd_embedding.shape}")

    # 2. Semantic scores
    print("[build_shortlist] Computing semantic scores …")
    semantic_scores = compute_semantic_scores(embeddings, jd_embedding)
    print(f"[build_shortlist] Semantic scores: min={semantic_scores.min():.4f}, "
          f"max={semantic_scores.max():.4f}, mean={semantic_scores.mean():.4f}")

    # 3. Behavioral scores
    print("[build_shortlist] Computing behavioral scores …")
    behavioral_scores = compute_all_behavioral_scores(candidates_file)
    print(f"[build_shortlist] Behavioral scores computed for {len(behavioral_scores)} candidates")

    # Save behavioral scores for all 100K
    beh_path = out_dir / "behavioral_scores.json"
    beh_path.write_text(json.dumps(behavioral_scores, indent=2), encoding="utf-8")
    print(f"[build_shortlist] Behavioral scores saved → {beh_path}")

    # 4. Combined scores
    beh_array = np.array([behavioral_scores.get(cid, 0.0) for cid in candidate_ids])
    combined_scores = (
        SHORTLIST_EMBEDDING_WEIGHT * semantic_scores
        + SHORTLIST_BEHAVIORAL_WEIGHT * beh_array
    )
    print(f"[build_shortlist] Combined scores: min={combined_scores.min():.4f}, "
          f"max={combined_scores.max():.4f}, mean={combined_scores.mean():.4f}")

    # 5. Select top-N
    top_ids = select_top_ids(candidate_ids, combined_scores, shortlist_size)
    print(f"[build_shortlist] Selected {len(top_ids)} top candidates")

    # 6. Collect and enrich shortlist records
    shortlist = collect_shortlist_records(
        candidates_file,
        top_ids,
        semantic_scores,
        behavioral_scores,
        candidate_ids,
    )
    print(f"[build_shortlist] Collected {len(shortlist)} shortlist records")

    # 7. Save shortlist
    shortlist_path = out_dir / "shortlist.jsonl"
    with open(shortlist_path, "w", encoding="utf-8") as fh:
        for record in shortlist:
            fh.write(json.dumps(record) + "\n")
    print(f"[build_shortlist] Shortlist saved → {shortlist_path}")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    candidates_file = sys.argv[1] if len(sys.argv) > 1 else "data/candidates.jsonl"
    artifacts_dir = sys.argv[2] if len(sys.argv) > 2 else "artifacts"
    shortlist_size = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    run(candidates_file=candidates_file, artifacts_dir=artifacts_dir, shortlist_size=shortlist_size)