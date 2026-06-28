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

The error-collecting helpers ``validate_submission_header`` and
``validate_submission_rows`` return lists of human-readable error
strings instead of asserting. They are reused by ``validate_submission.py``.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Shared constants and patterns
# ---------------------------------------------------------------------------

REQUIRED_HEADER: list[str] = ["candidate_id", "rank", "score", "reasoning"]
EXPECTED_DATA_ROWS: int = 100

# Compiled once at import time
_CANDIDATE_ID_RE = re.compile(r"^CAND_\d{7}$")


# ---------------------------------------------------------------------------
# Error-collecting validators (reused by validate_submission.py)
# ---------------------------------------------------------------------------

def validate_submission_header(header: list[str]) -> list[str]:
    """
    Validate the CSV header row.

    Parameters
    ----------
    header
        The list of column names read from row 1 of the CSV.

    Returns
    -------
    list[str]
        A list of human-readable error strings. Empty list means the
        header is valid.
    """
    if list(header) != REQUIRED_HEADER:
        return [
            "Row 1 (header) must be exactly:\n"
            f"  {','.join(REQUIRED_HEADER)}\n"
            "Found:\n"
            f"  {','.join(header)}"
        ]
    return []


def validate_submission_rows(rows: list[dict[str, Any]]) -> list[str]:
    """
    Validate the data rows of a submission CSV.

    Each row dict must contain the keys ``candidate_id``, ``rank``,
    ``score``, and ``reasoning``. Values may be either strings (as
    read from a CSV) or already-typed (``int`` rank, ``float`` score).

    Parameters
    ----------
    rows
        The list of data-row dicts to validate.

    Returns
    -------
    list[str]
        A list of human-readable error strings. Empty list means the
        rows are valid.
    """
    errors: list[str] = []

    n = len(rows)
    if n != EXPECTED_DATA_ROWS:
        errors.append(
            f"After the header (row 1), there must be exactly {EXPECTED_DATA_ROWS} "
            f"data rows (rows 2–{1 + EXPECTED_DATA_ROWS}); "
            f"found {n}."
        )

    seen_ids: set[str] = set()
    seen_ranks: set[int] = set()
    by_rank: list[tuple[int, float, str]] = []

    for i, row in enumerate(rows):
        row_num = i + 2  # data rows start at row 2

        cid_raw = row.get("candidate_id", "")
        cid = cid_raw.strip() if isinstance(cid_raw, str) else str(cid_raw)

        rank_raw = row.get("rank", "")
        rank_str = rank_raw.strip() if isinstance(rank_raw, str) else str(rank_raw)

        score_raw = row.get("score", "")
        score_str = score_raw.strip() if isinstance(score_raw, str) else str(score_raw)

        if not cid:
            errors.append(f"Row {row_num}: candidate_id is required.")
        elif not _CANDIDATE_ID_RE.match(cid):
            errors.append(
                f"Row {row_num}: candidate_id must be CAND_XXXXXXX (7 digits)."
            )
        elif cid in seen_ids:
            errors.append(f"Row {row_num}: duplicate candidate_id '{cid}'.")
        else:
            seen_ids.add(cid)

        rank: int | None = None
        try:
            rank = int(rank_str)
            if str(rank) != rank_str:
                raise ValueError
            if not 1 <= rank <= EXPECTED_DATA_ROWS:
                errors.append(f"Row {row_num}: rank must be between 1 and {EXPECTED_DATA_ROWS}.")
            elif rank in seen_ranks:
                errors.append(f"Row {row_num}: duplicate rank {rank}.")
            else:
                seen_ranks.add(rank)
        except (ValueError, TypeError):
            errors.append(f"Row {row_num}: rank must be an integer (1–{EXPECTED_DATA_ROWS}).")
            rank = None

        score: float | None = None
        try:
            score = float(score_str)
        except (ValueError, TypeError):
            errors.append(f"Row {row_num}: score must be a float.")
            score = None

        if rank is not None and score is not None and cid:
            by_rank.append((rank, score, cid))

    missing = set(range(1, EXPECTED_DATA_ROWS + 1)) - seen_ranks
    if missing:
        errors.append(
            f"Each rank 1–{EXPECTED_DATA_ROWS} must appear exactly once; missing: {sorted(missing)}"
        )

    by_rank.sort(key=lambda x: x[0])

    for i in range(len(by_rank) - 1):
        r1, s1, _ = by_rank[i]
        r2, s2, _ = by_rank[i + 1]
        if s1 < s2:
            errors.append(
                f"score must be non-increasing by rank: "
                f"rank {r1} ({s1}) < rank {r2} ({s2})."
            )

    for i in range(len(by_rank) - 1):
        r1, s1, c1 = by_rank[i]
        r2, s2, c2 = by_rank[i + 1]
        if s1 == s2 and c1 > c2:
            errors.append(
                f"Equal scores at ranks {r1} and {r2}: "
                f"tie-break requires candidate_id ascending "
                f"({c1!r} > {c2!r})."
            )

    return errors


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

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
    for row in rows:
        rank = row["rank"]
        assert type(rank) is int, (
            f"V9 violation: rank {rank!r} for {row.get('candidate_id')} "
            f"is not a native int (type={type(rank).__name__})"
        )

    # ── Shared row-level validation (header-independent rules) ────────────
    row_errors = validate_submission_rows(rows)
    assert not row_errors, (
        "Submission row validation failed:\n - " + "\n - ".join(row_errors)
    )

    # ── Write CSV ─────────────────────────────────────────────────────────
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = REQUIRED_HEADER

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
