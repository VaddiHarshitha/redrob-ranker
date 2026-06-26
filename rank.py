#!/usr/bin/env python3
"""
Phase B entry point — rank candidates using pre-computed artifacts.

No network access, no LLM calls, no embedding model loading.
Must finish in ≤5 minutes.

Usage
-----
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv --artifacts ./artifacts
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from util.submission import write_submission_csv


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python rank.py",
        description=(
            "Rank candidates using pre-computed LLM, semantic, and behavioral scores. "
            "Produces a submission CSV with the top 100 candidates."
        ),
    )
    parser.add_argument(
        "--candidates",
        required=True,
        help="Path to the candidates JSONL file (for validation; not loaded into memory).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output path for the submission CSV.",
    )
    parser.add_argument(
        "--artifacts",
        default="artifacts",
        help="Directory containing pre-computed artifacts (default: artifacts).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Artifact validation
# ---------------------------------------------------------------------------

def validate_artifacts(artifacts_dir: Path) -> None:
    """
    Verify that required artifact files exist; print a clear error and exit if not.
    """
    required = ["final_ranking.json", "rank_config.json"]
    missing = [fn for fn in required if not (artifacts_dir / fn).exists()]
    if missing:
        print(f"[rank] ERROR: Missing required artifact(s): {missing}")
        print(f"[rank] Expected in: {artifacts_dir}")
        print(f"[rank] Run Phase A first: python -m pre_computation.pipeline")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Ranking logic
# ---------------------------------------------------------------------------

def compute_score(
    record: dict[str, Any],
    weights: dict[str, float],
) -> float:
    """
    Recompute the final score from component scores using the given weights.

    Parameters
    ----------
    record
        A ranking record with llm_score, semantic_score, behavioral_score.
    weights
        Dict with llm_weight, semantic_weight, behavioral_weight.

    Returns
    -------
    float
        Weighted sum of the three component scores.
    """
    w_llm = weights.get("llm_weight", 0.65)
    w_sem = weights.get("semantic_weight", 0.20)
    w_beh = weights.get("behavioral_weight", 0.15)

    return (
        w_llm * record["llm_score"]
        + w_sem * record["semantic_score"]
        + w_beh * record["behavioral_score"]
    )


def load_and_sort(
    artifacts_dir: Path,
    weights: dict[str, float],
) -> list[dict[str, Any]]:
    """
    Load final_ranking.json, recompute scores with given weights, sort by
    (-score, candidate_id) for V13 tie-breaking, and return the top 100.

    Parameters
    ----------
    artifacts_dir
        Path to the artifacts directory.
    weights
        Scoring weights (allows retuning without re-running Phase A).

    Returns
    -------
    list[dict]
        Top-100 records sorted by (-final_score, candidate_id).
    """
    ranking_path = artifacts_dir / "final_ranking.json"
    records: list[dict[str, Any]] = json.loads(ranking_path.read_text(encoding="utf-8"))

    # Recompute scores with provided weights
    for record in records:
        record["final_score"] = compute_score(record, weights)

    # Sort: descending score, then ascending candidate_id (V13 tie-break)
    records.sort(key=lambda r: (-r["final_score"], r["candidate_id"]))

    return records[:100]


# ---------------------------------------------------------------------------
# Row building
# ---------------------------------------------------------------------------

def build_rows(top_100: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Build the list of 100 dicts ready for CSV writing.

    Parameters
    ----------
    top_100
        Sorted list of the top 100 ranking records.

    Returns
    -------
    list[dict]
        List of dicts with candidate_id, rank (native int!), score, reasoning.
    """
    rows = []
    for rank_position, record in enumerate(top_100, start=1):
        rows.append({
            "candidate_id": record["candidate_id"],
            "rank": rank_position,  # native Python int — satisfies V9
            "score": record["final_score"],
            "reasoning": record["reasoning"],
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    start_time = time.time()

    args = parse_args()
    artifacts_dir = Path(args.artifacts)

    # 1. Validate artifact presence
    validate_artifacts(artifacts_dir)

    # 2. Load rank config
    config_path = artifacts_dir / "rank_config.json"
    weights: dict[str, float] = json.loads(config_path.read_text(encoding="utf-8"))
    print(f"[rank] Loaded weights: {weights}")

    # 3. Load, sort, and trim ranking
    top_100 = load_and_sort(artifacts_dir, weights)
    print(f"[rank] Loaded and sorted {len(top_100)} candidates")

    # 4. Build rows
    rows = build_rows(top_100)

    # 5. Write CSV
    write_submission_csv(rows, args.out)
    elapsed = time.time() - start_time
    print(f"[rank] Wrote {args.out} in {elapsed:.1f}s")

    # 6. Warn if approaching 5-minute limit
    if elapsed > 270:
        print(f"[rank] WARNING: Elapsed time {elapsed:.1f}s is approaching the 5-minute limit.")

    print("[rank] Done.")


if __name__ == "__main__":
    main()