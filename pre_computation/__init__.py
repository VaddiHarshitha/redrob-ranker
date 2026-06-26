"""
Phase A pre-computation pipeline for the Redrob Candidate Ranker.

This package implements the five-step offline pipeline that runs with the
network available and produces artefacts consumed by the Phase B ranker.

Step 1 — analyze_jd
    Reads the job-description document, extracts structured profile
    information (required skills, nice-to-have skills, role context), and
    embeds the JD using the same sentence-transformer model used for
    candidates.  Produces: jd_profile.json, jd_embedding.npy.

Step 2 — embed_candidates
    Loads all candidate records from candidates.jsonl, converts each one to
    a dense text representation (see util.candidate_text), and encodes every
    candidate with the all-MiniLM-L6-v2 model.  Produces:
    candidate_embeddings.npy, candidate_ids.txt.

Step 3 — build_shortlist
    Loads the candidate embeddings and JD embedding, computes cosine
    similarity across all candidates, selects the top-K most relevant
    candidates, and enriches each shortlist entry with a behavioural score
    (see util.behavioural).  Produces: shortlist.jsonl, behavioural_scores.json.

Step 4 — evaluate_candidates
    Loads the shortlist and JD profile, then uses the Groq LLM (llama-3.1-8b-
    instant) to score each candidate on a 0-1 scale with reasoning.  Batches
    8 candidates per API call to manage rate limits.  Produces:
    llm_evaluations.json.

Step 5 — assemble_ranking
    Loads all intermediate artefacts (llm_evaluations.json, behavioural_
    scores.json, shortlist.jsonl), computes the weighted final score
    (0.65 × llm + 0.20 × semantic + 0.15 × behavioural), sorts candidates,
    and writes final_ranking.json and rank_config.json for the Phase B
    ranker to consume.

No file in this package directly produces the submission CSV; that is the
responsibility of the top-level rank.py script.
"""

__all__ = [
    "analyze_jd",
    "embed_candidates",
    "build_shortlist",
    "evaluate_candidates",
    "assemble_ranking",
]