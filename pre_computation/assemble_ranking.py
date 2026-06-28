"""
Combine llm_score + semantic_score + behavioral_score into final sorted ranking.

Outputs
-------
artifacts/final_ranking.json  — sorted array of candidate records with all scores
artifacts/rank_config.json    — weights used for the ranking
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pre_computation.config import (
    ARTIFACTS_DIR,
    BEHAVIORAL_SCORES_FILE,
    FINAL_RANKING_FILE,
    FINAL_TOP_N,
    LLM_EVALUATIONS_FILE,
    RANK_CONFIG_FILE,
    SHORTLIST_FILE,
)

# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = {
    "llm_weight": 0.65,
    "semantic_weight": 0.20,
    "behavioral_weight": 0.15,
}


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

def compute_final_score(
    llm_score: float,
    semantic_score: float,
    behavioral_score: float,
    weights: dict[str, float] | None = None,
) -> float:
    """
    Compute a weighted final score from three component scores.

    Parameters
    ----------
    llm_score
        LLM evaluation score in [0, 1].
    semantic_score
        Semantic/embedding similarity score in [0, 1].
    behavioral_score
        Behavioral signal score in [0, 1].
    weights
        Optional dict with keys llm_weight, semantic_weight, behavioral_weight.
        Defaults to DEFAULT_WEIGHTS.

    Returns
    -------
    float
        Weighted sum of the three scores, rounded to 6 decimal places.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    w_llm = weights.get("llm_weight", DEFAULT_WEIGHTS["llm_weight"])
    w_sem = weights.get("semantic_weight", DEFAULT_WEIGHTS["semantic_weight"])
    w_beh = weights.get("behavioral_weight", DEFAULT_WEIGHTS["behavioral_weight"])

    return round(w_llm * llm_score + w_sem * semantic_score + w_beh * behavioral_score, 6)


# ---------------------------------------------------------------------------
# Record assembly
# ---------------------------------------------------------------------------

def assemble_records(
    evals: dict[str, dict[str, Any]],
    behavioral_scores: dict[str, float],
    semantic_scores: dict[str, float],
    weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """
    Join the three score sources into unified ranking records.

    Parameters
    ----------
    evals
        Dict of candidate_id → {"score": float, "reasoning": str}.
    behavioral_scores
        Dict of candidate_id → behavioral score float.
    semantic_scores
        Dict of candidate_id → semantic score float (read from shortlist.jsonl
        _semantic_score field).
    weights
        Optional weights dict for score computation.

    Returns
    -------
    list[dict]
        Sorted list of records with candidate_id, llm_score, semantic_score,
        behavioral_score, final_score, and reasoning. Sorted by
        (-final_score, candidate_id) to satisfy tie-break rule V13.
    """
    records: list[dict[str, Any]] = []

    for candidate_id, eval_data in evals.items():
        llm_score = float(eval_data.get("score", 0.0))
        reasoning = str(eval_data.get("reasoning", ""))

        sem_score = float(semantic_scores.get(candidate_id, 0.0))
        beh_score = float(behavioral_scores.get(candidate_id, 0.0))

        final_score = compute_final_score(llm_score, sem_score, beh_score, weights)

        records.append({
            "candidate_id": candidate_id,
            "llm_score": round(llm_score, 6),
            "semantic_score": round(sem_score, 6),
            "behavioral_score": round(beh_score, 6),
            "final_score": final_score,
            "reasoning": reasoning,
        })

    # Sort: descending final_score, then ascending candidate_id for tie-break (V13)
    records.sort(key=lambda r: (-r["final_score"], r["candidate_id"]))

    return records


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(
    artifacts_dir: str = ARTIFACTS_DIR,
    top_n: int = FINAL_TOP_N,
    weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """
    Load all pre-computed artifacts, assemble the final ranking, and write
    ``final_ranking.json`` and ``rank_config.json``.

    Parameters
    ----------
    artifacts_dir
        Directory where all artifacts are stored.
    top_n
        Number of top candidates to include in final_ranking.json (default 300).
    weights
        Optional override for the default scoring weights.

    Returns
    -------
    list[dict]
        The top-N ranking records (same as written to final_ranking.json).
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()

    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load llm_evaluations.json
    evals_path = out_dir / LLM_EVALUATIONS_FILE
    print(f"[assemble_ranking] Loading LLM evaluations from {evals_path} …")
    evals: dict[str, dict[str, Any]] = json.loads(evals_path.read_text(encoding="utf-8"))
    print(f"[assemble_ranking] Loaded {len(evals)} LLM evaluations")

    # 2. Load behavioral_scores.json
    beh_path = out_dir / BEHAVIORAL_SCORES_FILE
    print(f"[assemble_ranking] Loading behavioral scores from {beh_path} …")
    behavioral_scores: dict[str, float] = json.loads(beh_path.read_text(encoding="utf-8"))
    print(f"[assemble_ranking] Loaded {len(behavioral_scores)} behavioral scores")

    # 3. Load semantic scores from shortlist.jsonl (_semantic_score field)
    shortlist_path = out_dir / SHORTLIST_FILE
    print(f"[assemble_ranking] Loading semantic scores from {shortlist_path} …")
    semantic_scores: dict[str, float] = {}
    shortlist_candidates: set[str] = set()
    with open(shortlist_path, encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            cid = record.get("candidate_id", "")
            shortlist_candidates.add(cid)
            semantic_scores[cid] = float(record.get("_semantic_score", 0.0))
    print(f"[assemble_ranking] Loaded semantic scores for {len(semantic_scores)} shortlisted candidates")

    # 4. Assemble all records
    print("[assemble_ranking] Assembling ranking records …")
    all_records = assemble_records(evals, behavioral_scores, semantic_scores, weights)
    print(f"[assemble_ranking] Total assembled records: {len(all_records)}")

    # 5. Trim to top_n
    top_records = all_records[:top_n]

    # 6. Write final_ranking.json
    ranking_path = out_dir / FINAL_RANKING_FILE
    ranking_path.write_text(json.dumps(top_records, indent=2), encoding="utf-8")
    print(f"[assemble_ranking] Wrote {len(top_records)} records → {ranking_path}")

    # 7. Write rank_config.json
    config_path = out_dir / RANK_CONFIG_FILE
    config_path.write_text(json.dumps(weights, indent=2), encoding="utf-8")
    print(f"[assemble_ranking] Wrote rank config → {config_path}")

    return top_records


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    artifacts_dir = sys.argv[1] if len(sys.argv) > 1 else ARTIFACTS_DIR
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else FINAL_TOP_N
    run(artifacts_dir=artifacts_dir, top_n=top_n)