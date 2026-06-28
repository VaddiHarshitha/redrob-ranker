"""
Narrow 100K candidates → 300 via a hard-requirement pre-filter, embedding
similarity, and behavioral score.

Pipeline
--------
1. Hard-requirement pre-filter — drop candidates with no positive semantic
   overlap with any of the JD's hard requirements (no LLM calls).
2. Embedding similarity — cosine similarity between candidate and JD vectors.
3. Behavioral score — availability/reachability signals.
4. Combined top-N selection from the survivors.

Outputs
-------
artifacts/shortlist.jsonl         — top N full candidate records + pre-scores
artifacts/behavioral_scores.json  — {candidate_id: score} for all 100K
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from util.behavioral import compute_behavioral_score
from util.candidate_text import build_candidate_text

from pre_computation.config import (
    ARTIFACTS_DIR,
    BEHAVIORAL_SCORES_FILE,
    CANDIDATE_EMBEDDINGS_FILE,
    CANDIDATE_IDS_FILE,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    JD_EMBEDDING_FILE,
    JD_PROFILE_FILE,
    SHORTLIST_FILE,
    SHORTLIST_SIZE,
)

# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

SHORTLIST_EMBEDDING_WEIGHT = 0.75
SHORTLIST_BEHAVIORAL_WEIGHT = 0.25

# Candidates whose hard-requirement score falls below this threshold are
# excluded before top-N selection. Score is mapped from raw cosine via
# (raw + 1.0) / 2.0, so 0.50 ≈ 0.0 raw cosine (no positive semantic overlap
# with ANY hard requirement). Tune UP if too many irrelevant candidates
# slip through; DOWN if too many qualified candidates are filtered out.
HARD_REQUIREMENT_THRESHOLD = 0.50

# ---------------------------------------------------------------------------
# Hard-requirement embedding & scoring
# ---------------------------------------------------------------------------

def embed_hard_requirements(
    jd_profile: dict,
    model: SentenceTransformer,
) -> np.ndarray:
    """
    Embed each hard_requirement string from the JD profile using the same
    model that embedded the candidates. Returns an L2-normalised array of
    shape (H, 768), where H is the number of hard requirements.

    No LLM calls — pure embedding inference.

    Parameters
    ----------
    jd_profile
        Parsed ``jd_profile.json`` (dict). Must contain ``hard_requirements``.
    model
        A loaded ``SentenceTransformer`` instance — must be the same model
        used to embed the candidates and the JD embedding.

    Returns
    -------
    np.ndarray
        float32 array of shape (H, 768), L2-normalised. Shape (0, 768) when
        the JD has no hard requirements.
    """
    requirements = jd_profile.get("hard_requirements", [])
    if not requirements:
        # No hard requirements → empty matrix; the pre-filter passes every
        # candidate (see compute_hard_requirement_scores).
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    vecs = model.encode(
        requirements,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vecs.astype(np.float32, copy=False)


def compute_hard_requirement_scores(
    embeddings: np.ndarray,
    hr_vecs: np.ndarray,
) -> np.ndarray:
    """
    Compute per-candidate hard-requirement evidence scores.

    For each candidate embedding, compute cosine similarity to every
    hard-requirement vector (both sides are L2-normalised, so this is a
    plain dot product), take the maximum across hard requirements, and
    map raw similarity [-1, 1] → [0, 1] via ``(raw + 1.0) / 2.0``.

    Parameters
    ----------
    embeddings
        float32 array of shape (N, 768), L2-normalised.
    hr_vecs
        float32 array of shape (H, 768), L2-normalised. If H == 0 the
        returned scores default to 1.0 (the pre-filter passes everyone).

    Returns
    -------
    np.ndarray
        float array of shape (N,) with values in [0, 1].
    """
    if hr_vecs.shape[0] == 0:
        return np.ones(embeddings.shape[0], dtype=np.float32)
    # embeddings @ hr_vecs.T = (N, 768) @ (768, H) = (N, H)
    raw = embeddings @ hr_vecs.T
    max_per_row = raw.max(axis=1)
    return (max_per_row + 1.0) / 2.0


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
        float32 array of shape (N, 768), L2-normalised.
    jd_embedding
        float32 array of shape (768,), L2-normalised.

    Returns
    -------
    np.ndarray
        float array of shape (N,) with values in [0, 1].
    """
    # embeddings @ jd_embedding = (N, 768) @ (768,) = (N,)
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
    pre_filter_mask: np.ndarray | None = None,
) -> set[str]:
    """
    Return the candidate IDs with the highest combined scores.

    If ``pre_filter_mask`` is provided, only indices where the mask is True
    are eligible for selection — masked-out (False) candidates cannot make
    the shortlist regardless of score. When the mask is None, all indices
    are eligible.

    Parameters
    ----------
    candidate_ids
        Ordered list of candidate IDs (index matches ``combined_scores``).
    combined_scores
        float array of shape (N,) with combined scores.
    size
        Number of top IDs to return.
    pre_filter_mask
        Optional boolean array of shape (N,). True entries are eligible.
        Defaults to None (all candidates are eligible).

    Returns
    -------
    set[str]
        Set of shortlisted candidate IDs (subset of eligible candidates).
    """
    if pre_filter_mask is not None:
        eligible_indices = np.where(pre_filter_mask)[0]
    else:
        eligible_indices = np.arange(len(candidate_ids))
    eligible_scores = combined_scores[eligible_indices]
    top_within_eligible = np.argsort(eligible_scores)[::-1][:size]
    return {candidate_ids[int(eligible_indices[i])] for i in top_within_eligible}


# ---------------------------------------------------------------------------
# Shortlist record collection
# ---------------------------------------------------------------------------

def collect_shortlist_records(
    candidates_file: str,
    top_ids: set[str],
    semantic_scores: np.ndarray,
    behavioral_scores: dict[str, float],
    candidate_ids: list[str],
    hard_requirement_scores: np.ndarray | None = None,
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
    hard_requirement_scores
        Optional array of hard-requirement scores indexed by position in
        candidate_ids. If None, ``_hard_requirement_score`` defaults to 0.0
        on each enriched record.
    embedding_weight
        Weight for semantic component.
    behavioral_weight
        Weight for behavioral component.

    Returns
    -------
    list[dict]
        Shortlisted records, each enriched with _semantic_score,
        _behavioral_score, _hard_requirement_score and _combined_score,
        sorted by combined_score descending.
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
            hr_score = (
                float(hard_requirement_scores[idx])
                if hard_requirement_scores is not None
                else 0.0
            )
            combined = embedding_weight * sem_score + behavioral_weight * beh_score

            enriched = dict(record)
            enriched["_semantic_score"] = sem_score
            enriched["_behavioral_score"] = beh_score
            enriched["_hard_requirement_score"] = hr_score
            enriched["_combined_score"] = combined
            shortlist.append(enriched)

    shortlist.sort(key=lambda x: x["_combined_score"], reverse=True)
    return shortlist


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(
    candidates_file: str = "data/candidates.jsonl",
    artifacts_dir: str = ARTIFACTS_DIR,
    shortlist_size: int = SHORTLIST_SIZE,
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
    embeddings_path = out_dir / CANDIDATE_EMBEDDINGS_FILE
    ids_path = out_dir / CANDIDATE_IDS_FILE
    jd_embedding_path = out_dir / JD_EMBEDDING_FILE
    jd_profile_path = out_dir / JD_PROFILE_FILE

    print(f"[build_shortlist] Loading embeddings from {embeddings_path} …")
    embeddings = np.load(embeddings_path)  # shape (N, 768), L2-normalised
    print(f"[build_shortlist] Embeddings shape: {embeddings.shape}")

    print(f"[build_shortlist] Loading candidate IDs from {ids_path} …")
    candidate_ids = ids_path.read_text(encoding="utf-8").splitlines()
    print(f"[build_shortlist] Candidate IDs: {len(candidate_ids)}")

    print(f"[build_shortlist] Loading JD embedding from {jd_embedding_path} …")
    jd_embedding = np.load(jd_embedding_path)  # shape (768,), L2-normalised
    print(f"[build_shortlist] JD embedding shape: {jd_embedding.shape}")

    print(f"[build_shortlist] Loading JD profile from {jd_profile_path} …")
    with open(jd_profile_path, encoding="utf-8") as fh:
        jd_profile = json.load(fh)
    n_hard_requirements = len(jd_profile.get("hard_requirements", []))
    print(f"[build_shortlist] Hard requirements in JD profile: {n_hard_requirements}")

    # 2. Load embedding model and embed hard requirements
    print(f"[build_shortlist] Loading embedding model ({EMBEDDING_MODEL}) …")
    model = SentenceTransformer(EMBEDDING_MODEL)
    hr_vecs = embed_hard_requirements(jd_profile, model)
    print(f"[build_shortlist] Embedded {hr_vecs.shape[0]} hard requirements "
          f"→ shape {hr_vecs.shape}")

    # 3. Hard-requirement pre-filter
    print("[build_shortlist] Computing hard-requirement scores …")
    hr_scores = compute_hard_requirement_scores(embeddings, hr_vecs)
    pre_filter_mask = hr_scores >= HARD_REQUIREMENT_THRESHOLD
    n_passing = int(pre_filter_mask.sum())
    n_total = len(pre_filter_mask)
    pct = 100.0 * n_passing / n_total if n_total else 0.0
    print(f"[build_shortlist] Hard-requirement pre-filter "
          f"(threshold={HARD_REQUIREMENT_THRESHOLD:.2f}): "
          f"{n_passing} / {n_total} candidates pass ({pct:.1f}%)")

    # 4. Semantic scores
    print("[build_shortlist] Computing semantic scores …")
    semantic_scores = compute_semantic_scores(embeddings, jd_embedding)
    print(f"[build_shortlist] Semantic scores: min={semantic_scores.min():.4f}, "
          f"max={semantic_scores.max():.4f}, mean={semantic_scores.mean():.4f}")

    # 5. Behavioral scores
    print("[build_shortlist] Computing behavioral scores …")
    behavioral_scores = compute_all_behavioral_scores(candidates_file)
    print(f"[build_shortlist] Behavioral scores computed for {len(behavioral_scores)} candidates")

    # Save behavioral scores for all 100K
    beh_path = out_dir / BEHAVIORAL_SCORES_FILE
    beh_path.write_text(json.dumps(behavioral_scores, indent=2), encoding="utf-8")
    print(f"[build_shortlist] Behavioral scores saved → {beh_path}")

    # 6. Combined scores
    beh_array = np.array([behavioral_scores.get(cid, 0.0) for cid in candidate_ids])
    combined_scores = (
        SHORTLIST_EMBEDDING_WEIGHT * semantic_scores
        + SHORTLIST_BEHAVIORAL_WEIGHT * beh_array
    )
    print(f"[build_shortlist] Combined scores: min={combined_scores.min():.4f}, "
          f"max={combined_scores.max():.4f}, mean={combined_scores.mean():.4f}")

    # 7. Select top-N (restricted to candidates passing the hard-requirement filter)
    top_ids = select_top_ids(
        candidate_ids,
        combined_scores,
        shortlist_size,
        pre_filter_mask=pre_filter_mask,
    )
    print(f"[build_shortlist] Selected {len(top_ids)} top candidates")

    if len(top_ids) < shortlist_size:
        print(f"[build_shortlist] WARNING: only {len(top_ids)} candidates passed "
              f"the hard-requirement filter (requested shortlist_size={shortlist_size}). "
              f"Consider lowering HARD_REQUIREMENT_THRESHOLD={HARD_REQUIREMENT_THRESHOLD:.2f} "
              f"or revisiting JD hard_requirements in {jd_profile_path}.")

    # 8. Collect and enrich shortlist records
    shortlist = collect_shortlist_records(
        candidates_file,
        top_ids,
        semantic_scores,
        behavioral_scores,
        candidate_ids,
        hard_requirement_scores=hr_scores,
    )
    print(f"[build_shortlist] Collected {len(shortlist)} shortlist records")

    # 9. Save shortlist
    shortlist_path = out_dir / SHORTLIST_FILE
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
    artifacts_dir = sys.argv[2] if len(sys.argv) > 2 else ARTIFACTS_DIR
    shortlist_size = int(sys.argv[3]) if len(sys.argv) > 3 else SHORTLIST_SIZE
    run(candidates_file=candidates_file, artifacts_dir=artifacts_dir, shortlist_size=shortlist_size)