"""
CSV formatter enforcing V1–V14 validator rules for the submission file.

Each call to ``write_submission_csv`` validates the provided rows and
writes a UTF-8 CSV with the exact header:
    candidate_id,rank,score,reasoning

Assertions (fail-fast on violation):
  V6  — exactly 100 data rows
  V9  — rank is a native Python int (not float, not str)
  V10 — ranks are precisely [1, 2, ..., 100] (each appearing once)
  V12 — scores are non-increasing as rank increases
  V13 — equal scores are tie-broken by ascending candidate_id
  V8  — candidate_id matches ^CAND_[0-9]{7}$ (after validation)
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from typing import Any


# Compiled once at import time
_CANDIDATE_ID_RE = re.compile(r"^CAND_\d{7}$")


def write_submission_csv(rows: list[dict[str, Any]], output_path: str) -> None:
    """
    Validate and write the ranked-candidate CSV.

    Parameters
    ----------
    rows
        List of exactly 100 dicts, each with keys ``candidate_id``,
        ``rank`` (int), ``score`` (float), and ``reasoning`` (str).
    output_path
        Destination file path (will be overwritten if it exists).

    Raises
    ------
    AssertionError
        If any V1–V14 rule is violated.
    """
    # ── V6: exactly 100 rows ──────────────────────────────────────────────
    assert len(rows) == 100, f"V6 violation: expected 100 rows, got {len(rows)}"

    # ── V9: rank must be a native Python int ──────────────────────────────
    ranks_seen: list[int] = []
    scores: list[float] = []
    ids_seen: list[str] = []

    for row in rows:
        rank = row["rank"]
        assert type(rank) is int, (
            f"V9 violation: rank {rank!r} for {row.get('candidate_id')} "
            f"is not a native int (type={type(rank).__name__})"
        )
        assert 1 <= rank <= 100, (
            f"V9/V10 violation: rank {rank} out of 1-100 range"
        )
        ranks_seen.append(rank)
        scores.append(float(row["score"]))
        ids_seen.append(str(row["candidate_id"]))

    # ── V10: each rank 1-100 appears exactly once ────────────────────────
    assert ranks_seen == list(range(1, 101)), (
        f"V10 violation: ranks are not exactly 1-100 — missing or duplicate: "
        f"{set(range(1, 101)) - set(ranks_seen)}"
    )

    # ── V12: scores non-increasing by rank ───────────────────────────────
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], (
            f"V12 violation at rank {i+1}→{i+2}: "
            f"score {scores[i]} < {scores[i+1]}"
        )

    # ── V13: equal scores → candidate_id ascending ──────────────────────
    for i in range(len(scores) - 1):
        if scores[i] == scores[i + 1]:
            assert ids_seen[i] <= ids_seen[i + 1], (
                f"V13 violation at rank {i+1} vs {i+2}: "
                f"equal score {scores[i]} but "
                f"candidate_id {ids_seen[i]} > {ids_seen[i+1]}"
            )

    # ── V8: candidate_id format and no duplicates ────────────────────────
    assert len(set(ids_seen)) == len(ids_seen), "V8 violation: duplicate candidate_id"
    for cid in ids_seen:
        assert _CANDIDATE_ID_RE.match(cid), (
            f"V8 violation: candidate_id {cid!r} does not match ^CAND_\\d{{7}}$"
        )

    # ── Write CSV ─────────────────────────────────────────────────────────
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["candidate_id", "rank", "score", "reasoning"]

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "candidate_id": row["candidate_id"],
                "rank": row["rank"],          # already validated as int
                "score": row["score"],        # will be float; csv writer handles str()
                "reasoning": row["reasoning"],
            })