#!/usr/bin/env python3
"""
Redrob Candidate Ranker — Streamlit demo app.

Demonstration UI that ranks the top candidates from the full candidate
pool using pre-computed scores produced by the offline scoring pipeline.

Usage
-----
    streamlit run app.py
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
RANKING_FILE = ARTIFACTS_DIR / "final_ranking.json"
CONFIG_FILE = ARTIFACTS_DIR / "rank_config.json"
MAX_CANDIDATES = 100
HARD_FLOOR_THRESHOLD = 0.15  # must match util/scoring.py — disqualifying band

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
    """
    Recompute final score from component scores using given weights.
    Mirrors util/scoring.compute_final_score() (and therefore rank.py):
    an llm_score at or below the disqualifying band is authoritative on
    its own and is never blended with semantic/behavioral signals.
    """
    llm_score = record.get("llm_score", 0.0)
    if llm_score <= HARD_FLOOR_THRESHOLD:
        return llm_score

    w_llm = weights.get("llm_weight", 0.65)
    w_sem = weights.get("semantic_weight", 0.20)
    w_beh = weights.get("behavioral_weight", 0.15)
    return (
        w_llm * llm_score
        + w_sem * record.get("semantic_score", 0.0)
        + w_beh * record.get("behavioral_score", 0.0)
    )


# ---------------------------------------------------------------------------
# Row building
# ---------------------------------------------------------------------------

def build_display_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build rows for DataFrame display with rank, candidate_id, score, reasoning."""
    rows = []
    for rank_pos, rec in enumerate(records, start=1):
        rows.append({
            "rank": rank_pos,
            "candidate_id": rec["candidate_id"],
            "score": round(rec["final_score"], 6),
            "reasoning": rec.get("reasoning", ""),
        })
    return rows


def build_csv(rows: list[dict[str, Any]]) -> str:
    """Convert display rows to CSV string."""
    output = io.StringIO()
    writer = csv.DictWriter(
        output, fieldnames=["rank", "candidate_id", "score", "reasoning"]
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Redrob Candidate Ranker", page_icon="🏆", layout="wide")
st.title("🏆 Redrob Candidate Ranker")
st.markdown(
    "An AI-driven ranking system that reasons about candidate fit against a "
    "role — combining semantic understanding, LLM judgment, and availability "
    "signals into a single ranked shortlist. Click **Run Ranking** to see the "
    "top matches for this role."
)

# Load artifacts
ranking_lookup = load_ranking_data()
weights = load_weights()

if not ranking_lookup:
    st.error(
        "No ranking results are available yet. Run the scoring pipeline first "
        "to generate a candidate ranking before using this app."
    )
    st.stop()

st.success("Candidate ranking is ready.")

with st.expander("How candidates are scored"):
    st.markdown(
        "Each candidate is evaluated on three signals: how well an LLM judges "
        "their actual career history against the role's requirements, how "
        "semantically similar their profile is to the ideal candidate "
        "description, and how reachable/available they currently are. "
        "A candidate the LLM identifies as a clear mismatch is not allowed to "
        "be rescued by the other two signals — that judgment is treated as "
        "final rather than just one vote among three."
    )

# Run Ranking button
if st.button("▶️ Run Ranking", type="primary"):
    # Recompute final scores with the active weights (mirrors rank.py)
    records: list[dict[str, Any]] = []
    for rec in ranking_lookup.values():
        rec_copy = dict(rec)
        rec_copy["final_score"] = recompute_score(rec_copy, weights)
        records.append(rec_copy)

    # Sort: descending score, then ascending candidate_id (same tie-break as rank.py)
    records.sort(key=lambda r: (-r["final_score"], r["candidate_id"]))

    # Cap display at MAX_CANDIDATES
    top = records[:MAX_CANDIDATES]

    st.info(f"Showing the top {len(top)} ranked candidates for this role.")

    rows = build_display_rows(top)
    csv_out = build_csv(rows)

    # Display sortable DataFrame
    st.dataframe(
        rows,
        column_config={
            "rank": st.column_config.NumberColumn("Rank", width="small"),
            "candidate_id": st.column_config.TextColumn("Candidate ID"),
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

# Footer
st.divider()
st.caption("Redrob Candidate Ranker · AI-driven candidate ranking, not keyword matching.")