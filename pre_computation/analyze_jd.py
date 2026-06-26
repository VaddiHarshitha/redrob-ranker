"""
Extract a semantic JD profile via Groq LLM and embed it.

Outputs
-------
artifacts/jd_profile.json  — structured semantic context
artifacts/jd_embedding.npy — 384-dim vector (L2-normalised)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from docx import Document as DocxDocument
from sentence_transformers import SentenceTransformer

from util.llm_client import build_groq_llm, parse_json_response

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

JD_EXTRACTION_PROMPT = """You are an expert technical recruiter analysing a job description.

Return a JSON object (no markdown fences, no extra commentary) with exactly these fields:

{
  "embedding_text": "200-250 word present-tense prose from the perspective of the ideal candidate, describing the work they do, the problems they solve, and the impact they have in this role. Write as if describing a LinkedIn 'About' section for this ideal person. Focus on demonstrable capabilities, not wish-list keywords.",
  "role_summary": "2-3 sentence plain-English description of what the daily work involves in this role.",
  "hard_requirements": ["list of prose descriptions (NOT keyword lists) of technical capabilities the candidate must have DEMONSTRATED through work experience — be specific about depth and context, e.g. '3+ years building and shipping production REST APIs at scale' rather than just 'REST APIs'"],
  "disqualifier_patterns": ["list of prose descriptions of career patterns that would make a candidate unsuitable — e.g. 'Has spent the majority of the last 5 years in a non-technical management role with no hands-on coding' or 'Has never worked on a team that shipped to production'"],
  "preferred_location": "city/region preference if stated, otherwise 'Any'",
  "experience_years": {"min": integer minimum years, "max": integer maximum years or null},
  "notice_preference": "notice period preference as a prose string, e.g. '30 days or less' or 'immediate' or 'any'",
  "evaluation_guidance": "2-3 sentences summarising what this JD is really looking for, including any common pitfalls or traps to watch out for (e.g. requiring '5 years of Kubernetes' when the JD only mentions 2 years experience total, or red flags like excessive on-call expectations not reflected in compensation)."
}

Guidelines
----------
- hard_requirements: Each item must be a full prose sentence describing a capability, not a bullet-point keyword list.
- disqualifier_patterns: Be specific enough that an LLM evaluator could use these as guidance.
- Do NOT add any field not listed above.
- Return ONLY the JSON object — no preamble, no explanation.
"""


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
        384-dim L2-normalised embedding vector (float32).
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


def run(jd_path: str = "data/job_description.docx", artifacts_dir: str = "artifacts") -> dict:
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
    llm = build_groq_llm(model="llama-3.3-70b-versatile", max_tokens=4096)
    jd_profile = extract_jd_profile(jd_text, llm)
    print("[analyze_jd] JD profile extracted successfully")

    # 3. Embed
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embedding = embed_jd_profile(jd_profile, embedder)
    print(f"[analyze_jd] Embedding shape: {embedding.shape}")

    # 4. Save artefacts
    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    profile_path = out_dir / "jd_profile.json"
    profile_path.write_text(json.dumps(jd_profile, indent=2), encoding="utf-8")
    print(f"[analyze_jd] Profile saved → {profile_path}")

    embedding_path = out_dir / "jd_embedding.npy"
    np.save(embedding_path, embedding)
    print(f"[analyze_jd] Embedding saved → {embedding_path}")

    return jd_profile


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    jd_path = sys.argv[1] if len(sys.argv) > 1 else "data/job_description.docx"
    artifacts_dir = sys.argv[2] if len(sys.argv) > 2 else "artifacts"
    run(jd_path=jd_path, artifacts_dir=artifacts_dir)