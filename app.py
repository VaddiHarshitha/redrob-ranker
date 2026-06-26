#!/usr/bin/env python3
"""
Redrob Candidate Ranker — Streamlit sandbox app.

Allows demo/testing with up to 100 candidates.
- Loads pre-computed ranking from artifacts/final_ranking.json if available.
- For candidates in the pre-computed ranking: displays their final_score and reasoning.
- For candidates NOT in the pre-computed ranking: falls back to behavioral-only
  scoring via compute_behavioral_score with simple reasoning "Not in pre-computed ranking."

Usage
-----
    streamlit run app.py
"""

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Any

import streamlit as st

from util.behavioral import compute_behavioral_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARTIFACTS_DIR = Path("artifacts")
RANKING_FILE = ARTIFACTS_DIR / "final_ranking.json"
CONFIG_FILE = ARTIFACTS_DIR / "rank_config.json"
MAX_CANDIDATES = 100

# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_ranking_data() -> dict[str, dict[str, Any]]:
    """Load and cache the pre-computed ranking lookup by candidate_id."""
    if not RANKING_FILE.exists():
        return {}
    records: list[dict[str, Any]] = json.loads(RANKING_FILE.read_text(encoding="utf-8"))
    return {rec["candidate_id"]: rec for rec in records}


@st.cache_data
def load_weights() -> dict[str, float]:
    """Load scoring weights from rank_config.json."""
    if not CONFIG_FILE.exists():
        return {"llm_weight": 0.65, "semantic_weight": 0.20, "behavioral_weight": 0.15}
    return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))


def recompute_score(record: dict[str, Any], weights: dict[str, float]) -> float:
    """Recompute final score from component scores using given weights."""
    w_llm = weights.get("llm_weight", 0.65)
    w_sem = weights.get("semantic_weight", 0.20)
    w_beh = weights.get("behavioral_weight", 0.15)
    return w_llm * record.get("llm_score", 0.0) + w_sem * record.get("semantic_score", 0.0) + w_beh * record.get("behavioral_score", 0.0)


# ---------------------------------------------------------------------------
# File parsing
# ---------------------------------------------------------------------------

def parse_candidates(content: str) -> list[dict[str, Any]]:
    """
    Parse uploaded file content as either a JSON list or JSONL.

    Parameters
    ----------
    content
        Raw text from the uploaded file.

    Returns
    -------
    list[dict]
        List of candidate dicts.
    """
    content = content.strip()
    if content.startswith("["):
        # JSON array
        return json.loads(content)
    else:
        # JSONL — one JSON object per line
        records = []
        for line in content.splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records


# ---------------------------------------------------------------------------
# Ranking logic
# ---------------------------------------------------------------------------

def rank_candidates(
    candidates: list[dict[str, Any]],
    ranking_lookup: dict[str, dict[str, Any]],
    weights: dict[str, float],
) -> list[dict[str, Any]]:
    """
    Rank a list of candidates.

    For candidates found in ranking_lookup: use pre-computed score + reasoning.
    For missing candidates: use behavioral score with reduced weight (0.3) to
    indicate lower confidence, with reasoning "Not in pre-computed ranking."

    Returns a sorted list of dicts ready for display.
    """
    records = []
    for candidate in candidates:
        cid = candidate.get("candidate_id", "unknown")
        if cid in ranking_lookup:
            rec = ranking_lookup[cid].copy()
            rec["final_score"] = recompute_score(rec, weights)
            rec["reasoning"] = rec.get("reasoning", "")
        else:
            beh_score = compute_behavioral_score(candidate)
            # Apply reduced weight (0.3) to indicate lower confidence
            adjusted_score = beh_score * 0.3
            rec = {
                "candidate_id": cid,
                "title": candidate.get("title", ""),
                "llm_score": 0.0,
                "semantic_score": 0.0,
                "behavioral_score": beh_score,
                "final_score": adjusted_score,
                "reasoning": "Not in pre-computed ranking.",
            }
        rec["title"] = candidate.get("title", rec.get("title", ""))
        records.append(rec)

    # Sort: descending score, then ascending candidate_id for tie-breaking
    records.sort(key=lambda r: (-r["final_score"], r["candidate_id"]))

    return records


def build_display_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build rows for DataFrame display with rank column."""
    rows = []
    for rank_pos, rec in enumerate(records, start=1):
        rows.append({
            "rank": rank_pos,
            "candidate_id": rec["candidate_id"],
            "title": rec.get("title", ""),
            "score": round(rec["final_score"], 6),
            "reasoning": rec.get("reasoning", ""),
        })
    return rows


def build_csv(rows: list[dict[str, Any]]) -> str:
    """Convert display rows to CSV string."""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["rank", "candidate_id", "title", "score", "reasoning"])
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Redrob Candidate Ranker", page_icon="🏆", layout="wide")
st.title("🏆 Redrob Candidate Ranker")
st.markdown(
    "Upload a JSON or JSONL file with ≤100 candidates to see their rankings. "
    "Pre-computed rankings from `artifacts/final_ranking.json` are used when available; "
    "missing candidates fall back to behavioral-only scoring."
)

# Load artifacts
ranking_lookup = load_ranking_data()
weights = load_weights()

if ranking_lookup:
    st.success(f"Loaded pre-computed rankings for {len(ranking_lookup)} candidates.")
else:
    st.warning(
        "`artifacts/final_ranking.json` not found — running in sandbox mode "
        "(behavioral score only, weight 1.0)."
    )

# File uploader
uploaded_file = st.file_uploader(
    "Upload candidates (JSON or JSONL, max 100)",
    type=["json", "jsonl"],
    help="Upload a JSON array or JSONL file with candidate records.",
)

if uploaded_file is not None:
    content = uploaded_file.read().decode("utf-8")
    try:
        candidates = parse_candidates(content)
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse JSON: {e}")
        st.stop()

    if len(candidates) > MAX_CANDIDATES:
        st.error(f"Too many candidates: {len(candidates)} (max {MAX_CANDIDATES}).")
        st.stop()

    st.info(f"Parsed {len(candidates)} candidate(s).")

    # Rank candidates
    ranked = rank_candidates(candidates, ranking_lookup, weights)
    rows = build_display_rows(ranked)
    csv_out = build_csv(rows)

    # Display sortable DataFrame
    st.dataframe(
        rows,
        column_config={
            "rank": st.column_config.NumberColumn("Rank", width="small"),
            "candidate_id": st.column_config.TextColumn("Candidate ID"),
            "title": st.column_config.TextColumn("Title"),
            "score": st.column_config.NumberColumn("Score", format="%.6f"),
            "reasoning": st.column_config.TextColumn("Reasoning"),
        },
        use_container_width=True,
        hide_index=True,
    )

    # Download button
    st.download_button(
        "📥 Download Ranked CSV",
        csv_out,
        file_name="ranked.csv",
        mime="text/csv",
        help="Download the ranked candidates as a CSV file.",
    )
else:
    st.info("👆 Upload a candidate file to get started.")

# Footer
st.divider()
st.caption(
    "Redrob Candidate Ranker · Phase A pre-computation required for full scoring · "
    "Sandbox mode uses behavioral signals only for missing candidates."
)