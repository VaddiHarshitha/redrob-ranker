"""
Extract a semantic JD profile via Groq LLM and embed it.

Outputs
-------
artifacts/jd_profile.json  — structured semantic context
artifacts/jd_embedding.npy — 768-dim vector (L2-normalised)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from docx import Document as DocxDocument
from sentence_transformers import SentenceTransformer

from util.llm_client import build_groq_llm, parse_json_response

from pre_computation.config import (
    ARTIFACTS_DIR,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    GROQ_MAX_TOKENS,
    GROQ_MODEL,
    JD_EMBEDDING_FILE,
    JD_PROFILE_FILE,
)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

JD_EXTRACTION_PROMPT = """You are analyzing a job description to power an AI candidate ranking system.

Read this job description carefully — including any sections addressed to hackathon participants or systems being built to evaluate it. Pay special attention to what the role explicitly says it does NOT want, and what it says about how candidates should be evaluated.

CRITICAL DISTINCTION — hard vs. nice-to-have:
The JD typically signals three tiers of requirements using different language:
  - HARD (required / must have / X+ years / "we need" / "you will"): missing these disqualifies the candidate
  - NICE-TO-HAVE (preferred / "good to have" / "bonus" / "would be a plus" / "ideally"): boosts fit but not required
  - DISQUALIFIERS (avoid / "we don't want" / red flags): presence of these disqualifies

You MUST separate hard_requirements from nice_to_have_requirements. A capability that appears in the JD only inside a "nice to have" or "bonus" section must NOT appear in hard_requirements.

Extract the following and return as a single valid JSON object. No other text.

{{
  "embedding_text": "<200-250 word dense prose written in the PRESENT TENSE describing what an ideal candidate HAS ACTUALLY SHIPPED and DELIVERED in this role. Write it from the candidate's perspective — as if it is the ideal candidate's career summary. CRITICAL: describe DEMONSTRATED OUTCOMES, not vocabulary. Say 'designed and shipped a production embedding retrieval pipeline that served 50M daily queries with sub-100ms p99 latency and was measured against NDCG@10 in offline evaluation' — NOT 'experienced with embeddings and vector databases'. The embedding model must be able to discriminate a candidate who actually built these systems from one who merely mentions the same terms in passing. Optimize for semantic vector comparison against real candidate career descriptions.>",

  "role_summary": "<2-3 sentence plain-English description of what this person does day to day in this role — useful for explaining the role to an LLM evaluator who has never seen the JD>",

  "hard_requirements": [
    "<Specific technical capability the candidate MUST have DEMONSTRATED — describe the actual work, not just a technology name. E.g. 'Shipped a production embedding-based retrieval system that served real user queries at scale' rather than 'knows sentence-transformers'. These are items the JD signals as required / must-have / X+ years / 'we need'.>",
    ...list all HARD requirements this way. If the JD says something only in a 'nice to have' or 'bonus' section, it goes in nice_to_have_requirements instead, NOT here....
  ],

  "nice_to_have_requirements": [
    "<Preferred-but-not-required capability the JD signals with language like 'nice to have', 'good to have', 'bonus', 'would be a plus', 'ideally'. Describe the actual work, not just a technology name. E.g. 'Experience mentoring junior engineers' or 'Familiarity with reinforcement learning from user feedback'. These BOOST a candidate's fit but their absence does NOT disqualify.>",
    ...list all nice-to-have items this way. Empty array [] if the JD has no such section....
  ],

  "disqualifier_patterns": [
    "<Career pattern that makes a candidate unsuitable — describe the pattern, not the technology. E.g. 'Entire career at IT outsourcing/consulting firms with no product company experience' or 'Current AI experience is limited to calling hosted LLM APIs via LangChain without any pre-LLM production ML depth'>",
    ...list all disqualifiers this way...
  ],

  "preferred_location": "<City/region preferences from the JD, as a plain string>",

  "experience_years": {{"min": <integer>, "max": <integer>}},

  "notice_preference": "<What the JD says about notice period, as a plain string>",

  "evaluation_guidance": "<2-3 sentences of guidance for the LLM that will score individual candidates — summarizing the spirit of what this JD is really looking for, including any warnings about traps or anti-patterns to avoid when scoring. Note: nice-to-have items should INCREASE a candidate's score only if their hard requirements are already met — they should never rescue a candidate who is missing hard requirements.>"
}}

Job Description:
{jd_text}"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_jd(path: str) -> str:
    """
    Read a job description from a file.

    Supports .docx (via python-docx), .txt, and .md files.

    Parameters
    ----------
    path
        Path to the job description file.

    Returns
    -------
    str
        Plain-text content of the JD.

    Raises
    ------
    ValueError
        If the file extension is unsupported.
    """
    p = Path(path)
    ext = p.suffix.lower()

    if ext == ".docx":
        doc = DocxDocument(str(p))
        paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
        return "\n".join(paragraphs)
    elif ext in (".txt", ".md"):
        return p.read_text(encoding="utf-8")
    else:
        raise ValueError(
            f"Unsupported JD file format: {ext!r}. "
            "Supported formats: .docx, .txt, .md"
        )


def extract_jd_profile(jd_text: str, llm) -> dict:
    """
    Extract a structured JD profile using one LLM call.

    Parameters
    ----------
    jd_text
        Plain-text job description.
    llm
        A LangChain ChatGroq (or compatible) instance.

    Returns
    -------
    dict
        Parsed JSON object conforming to the schema defined in JD_EXTRACTION_PROMPT.
    """
    prompt = f"{JD_EXTRACTION_PROMPT}\n\n--- JOB DESCRIPTION ---\n{jd_text}"
    raw = llm.invoke(prompt)
    return parse_json_response(raw.content)


def embed_jd_profile(jd_profile: dict, embedder) -> np.ndarray:
    """
    Embed the ``embedding_text`` field of a JD profile.

    Parameters
    ----------
    jd_profile
        The structured profile dict returned by ``extract_jd_profile``.
    embedder
        A SentenceTransformer instance (already loaded).

    Returns
    -------
    np.ndarray
        768-dim L2-normalised embedding vector (float32).
    """
    embedding_text = jd_profile.get("embedding_text", "")
    if not embedding_text:
        raise ValueError("jd_profile['embedding_text'] is empty; cannot embed.")

    vector = embedder.encode(
        embedding_text,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vector.astype(np.float32)


def run(jd_path: str = "data/job_description.docx", artifacts_dir: str = ARTIFACTS_DIR) -> dict:
    """
    Orchestrate the full JD analysis pipeline.

    Reads the JD file, extracts a semantic profile via Groq LLM,
    embeds the profile text, and saves both artefacts.

    Parameters
    ----------
    jd_path
        Path to the job description file.
    artifacts_dir
        Directory where output files are written.

    Returns
    -------
    dict
        The extracted JD profile (also saved to ``<artifacts_dir>/jd_profile.json``).
    """
    # 1. Read JD
    jd_text = read_jd(jd_path)
    print(f"[analyze_jd] Read JD from {jd_path!r} ({len(jd_text)} chars)")

    # 2. Extract profile via LLM
    llm = build_groq_llm(model=GROQ_MODEL, max_tokens=GROQ_MAX_TOKENS)
    jd_profile = extract_jd_profile(jd_text, llm)
    print("[analyze_jd] JD profile extracted successfully")
    print(f"  Hard requirements:     {len(jd_profile.get('hard_requirements', []))}")
    print(f"  Nice-to-have:          {len(jd_profile.get('nice_to_have_requirements', []))}")
    print(f"  Disqualifiers:         {len(jd_profile.get('disqualifier_patterns', []))}")

    # 3. Embed
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    embedding = embed_jd_profile(jd_profile, embedder)
    print(f"[analyze_jd] Embedding shape: {embedding.shape}")

    # 4. Save artefacts
    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    profile_path = out_dir / JD_PROFILE_FILE
    profile_path.write_text(json.dumps(jd_profile, indent=2), encoding="utf-8")
    print(f"[analyze_jd] Profile saved → {profile_path}")

    embedding_path = out_dir / JD_EMBEDDING_FILE
    np.save(embedding_path, embedding)
    print(f"[analyze_jd] Embedding saved → {embedding_path}")

    return jd_profile


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    jd_path = sys.argv[1] if len(sys.argv) > 1 else "data/job_description.docx"
    artifacts_dir = sys.argv[2] if len(sys.argv) > 2 else ARTIFACTS_DIR
    run(jd_path=jd_path, artifacts_dir=artifacts_dir)