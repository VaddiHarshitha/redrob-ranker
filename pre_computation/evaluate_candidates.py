"""
Groq LLM batch evaluation of shortlisted candidates against JD profile.

Outputs
-------
artifacts/llm_evaluations.json — {candidate_id: {"score": float, "reasoning": str}}
"""

from __future__ import annotations

import groq
import json
import re
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from util.llm_client import build_groq_llm, parse_json_response

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EVAL_MODEL = "llama-3.1-8b-instant"
DEFAULT_BATCH_SIZE = 1
FULL_DETAIL_ROLES = 1
MAX_TEXT_CHARS = 100
MAX_RETRIES = 10
BATCH_SLEEP_SECONDS = 20.0

# ---------------------------------------------------------------------------
# Evaluation prompt
# ---------------------------------------------------------------------------

EVAL_PROMPT = """Evaluate candidates for this role.

{RoleContext}

{WhatThisRoleRequires}

{DisqualifyingCareerPatterns}

{EvaluationGuidance}

{AdditionalContext}

Scoring rubric (0.00–1.00):
- 0.85–1.00: Clearly demonstrates requirements at product companies.
- 0.65–0.84: Most requirements met; minor gaps.
- 0.40–0.64: Relevant but significant gaps.
- 0.20–0.39: Tangential only.
- 0.00–0.19: Wrong domain or disqualifier present.

For EACH candidate, write 1–2 sentences citing SPECIFIC FACTS from their career descriptions. No generic praise.

## Output
JSON array, no markdown fences:
[{"candidate_id":"...","score":0.XX,"reasoning":"..."},...]
"""


# ---------------------------------------------------------------------------
# Truncation helper
# ---------------------------------------------------------------------------

def _truncate(text: str, max_chars: int) -> str:
    """
    Truncate ``text`` to ``max_chars`` characters at a word boundary.

    If the text is already within the limit, returns it unchanged.
    Otherwise, finds the last space before or at max_chars and appends '…'.

    Parameters
    ----------
    text
        Input string.
    max_chars
        Maximum allowed characters.

    Returns
    -------
    str
        Truncated string ending with '…' if it was cut, otherwise unchanged.
    """
    if len(text) <= max_chars:
        return text
    # Find last space within the limit
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]
    return truncated + "…"


# ---------------------------------------------------------------------------
# Candidate summary builder
# ---------------------------------------------------------------------------

def build_candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    """
    Build a compact summary dict from a full candidate record for LLM evaluation.

    Fields: candidate_id, current_title, years_of_experience, location, headline,
    summary (120 chars), career_history (1 recent role with desc, older roles title/company/duration only),
    education (degree, field, institution, tier).

    Parameters
    ----------
    candidate
        A full candidate record dict.

    Returns
    -------
    dict
        Compact summary dict.
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])

    # Only 1 recent role gets description; older roles are title/company only
    career_history: list[dict[str, Any]] = []
    for i, role in enumerate(career):
        is_recent = i == 0
        entry: dict[str, Any] = {
            "title": role.get("title", ""),
            "company": role.get("company", ""),
            "duration_months": role.get("duration_months", 0),
        }
        if is_recent:
            entry["description"] = _truncate(role.get("description", ""), MAX_TEXT_CHARS)
        career_history.append(entry)

    # Compact education: degree + field only
    education: list[dict[str, Any]] = []
    for edu in candidate.get("education", [])[:2]:
        education.append({
            "d": edu.get("degree", ""),
            "f": edu.get("field_of_study", ""),
        })

    return {
        "id": candidate.get("candidate_id", ""),
        "title": profile.get("current_title", ""),
        "yrs": profile.get("years_of_experience", 0),
        "loc": profile.get("location", ""),
        "hl": profile.get("headline", ""),
        "sum": _truncate(profile.get("summary", ""), MAX_TEXT_CHARS),
        "ch": career_history,
        "ed": education,
    }


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_eval_prompt(batch: list[dict], jd_profile: dict[str, Any]) -> str:
    """
    Format the evaluation prompt for a batch of candidate summaries.

    Parameters
    ----------
    batch
        List of candidate summary dicts.
    jd_profile
        The structured JD profile dict.

    Returns
    -------
    str
        Formatted prompt string.
    """
    # Truncate all JD fields aggressively to control token budget
    role_summary = _truncate(jd_profile.get("role_summary", ""), 200)
    evaluation_guidance = _truncate(jd_profile.get("evaluation_guidance", ""), 200)
    hard_requirements = jd_profile.get("hard_requirements", [])
    disqualifier_patterns = jd_profile.get("disqualifier_patterns", [])
    preferred_location = jd_profile.get("preferred_location", "Any")
    notice_preference = jd_profile.get("notice_preference", "any")
    exp_years = jd_profile.get("experience_years", {})

    role_context = role_summary

    requires_block = "## What This Role Requires\n"
    if hard_requirements:
        for item in hard_requirements:
            requires_block += f"- {item}\n"
    else:
        requires_block += "(not specified)\n"

    disqual_block = "## Disqualifying Career Patterns\n"
    if disqualifier_patterns:
        for item in disqualifier_patterns:
            disqual_block += f"- {item}\n"
    else:
        disqual_block += "(none specified)\n"

    guidance_block = f"## Evaluation Guidance\n{evaluation_guidance}"

    exp_str = ""
    if exp_years:
        exp_min = exp_years.get("min")
        exp_max = exp_years.get("max")
        if exp_min is not None and exp_max is not None:
            exp_str = f"Experience: {exp_min}–{exp_max} years"
        elif exp_min is not None:
            exp_str = f"Experience: {exp_min}+ years"
        elif exp_max is not None:
            exp_str = f"Experience: up to {exp_max} years"

    additional_block = "## Additional Context\n"
    parts = [f"Location preference: {preferred_location}"]
    if exp_str:
        parts.append(exp_str)
    if notice_preference:
        parts.append(f"Notice period: {notice_preference}")
    additional_block += "\n".join(f"- {p}" for p in parts)

    # Inject candidates as compact single-line JSON to save tokens
    candidates_block = "## Candidates to Evaluate\n"
    for c in batch:
        candidates_block += json.dumps(c, separators=(",", ":")) + "\n"

    # Dynamic output format based on batch size
    n = len(batch)
    if n == 1:
        output_instruction = (
            "## Output\n"
            "Return ONE JSON object, no markdown fences:\n"
            '{"candidate_id":"...","score":0.XX,"reasoning":"..."}'
        )
    else:
        output_instruction = (
            f"## Output\n"
            f"Return a JSON array with {n} objects, no markdown fences:\n"
            '[{"candidate_id":"...","score":0.XX,"reasoning":"..."},...]'
        )

    return (
        EVAL_PROMPT
        .replace("{RoleContext}", f"## Role Context\n{role_context}")
        .replace("{WhatThisRoleRequires}", requires_block)
        .replace("{DisqualifyingCareerPatterns}", disqual_block)
        .replace("{EvaluationGuidance}", guidance_block)
        .replace("{AdditionalContext}", additional_block)
        + "\n" + candidates_block
        + "\n" + output_instruction
    )


# ---------------------------------------------------------------------------
# Single-candidate fallback
# ---------------------------------------------------------------------------

def evaluate_one_with_fallback(
    candidate: dict[str, Any],
    jd_profile: dict[str, Any],
    llm,
    max_retries: int = MAX_RETRIES,
) -> dict[str, Any]:
    """
    Evaluate a single candidate with per-candidate retry on rate-limit errors.

    Used when a batch evaluation fails after all retries.

    Parameters
    ----------
    candidate
        Candidate summary dict.
    jd_profile
        JD profile dict.
    llm
        LangChain ChatGroq instance.
    max_retries
        Maximum retry attempts.

    Returns
    -------
    dict
        {"candidate_id": str, "score": float, "reasoning": str}
    """
    prompt = build_eval_prompt([candidate], jd_profile)
    attempt = 0

    for attempt in range(max_retries):
        try:
            raw = llm.invoke(prompt)
            data = parse_json_response(raw.content)
            # data should be a list
            if isinstance(data, list) and len(data) > 0:
                return data[0]
            elif isinstance(data, dict) and "candidate_id" in data:
                return data
            # If the parsed JSON is unexpected, raise to trigger retry
            raise ValueError(f"Unexpected response shape: {type(data)}")
        except groq.RateLimitError as e:
            msg = str(e)
            wait = _suggested_wait_seconds(msg)
            wait = wait if wait else 2 ** attempt + 5
            if wait > 90:
                raise RuntimeError(
                    "Rate limit retry-after suggests daily token cap (>90s). "
                    "Aborting evaluation."
                ) from e
            print(f"    [evaluate_one] Rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})")
            time.sleep(wait)
        except Exception as exc:
            msg = str(exc)
            if "rate" in msg.lower() or "429" in msg or "limit" in msg.lower():
                wait = _suggested_wait_seconds(msg)
                wait = wait if wait else 2 ** attempt + 5
                if wait > 90:
                    raise RuntimeError(
                        "Rate limit retry-after suggests daily token cap (>90s). "
                        "Aborting evaluation."
                    ) from exc
                print(f"    [evaluate_one] Rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise

    # Should not reach here
    return {
        "candidate_id": candidate.get("candidate_id", "?"),
        "score": 0.0,
        "reasoning": f"Evaluation failed after {max_retries} retries.",
    }


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_batch(
    batch: list[dict],
    jd_profile: dict[str, Any],
    llm,
    max_retries: int = MAX_RETRIES,
) -> list[dict[str, Any]]:
    """
    Evaluate a batch of candidates with a single LLM call.

    Retries with exponential backoff on rate-limit errors.

    Parameters
    ----------
    batch
        List of candidate summary dicts.
    jd_profile
        JD profile dict.
    llm
        LangChain ChatGroq instance.
    max_retries
        Maximum retry attempts.

    Returns
    -------
    list[dict]
        List of {"candidate_id": str, "score": float, "reasoning": str}.
    """
    prompt = build_eval_prompt(batch, jd_profile)
    for attempt in range(max_retries):
        try:
            raw = llm.invoke(prompt)
            data = parse_json_response(raw.content)

            if not isinstance(data, list):
                raise ValueError(
                    f"Expected a JSON array but got {type(data).__name__}: "
                    f"{str(data)[:200]}"
                )
            return data

        except groq.RateLimitError as e:
            # Groq-specific rate limit
            msg = str(e)
            wait = _suggested_wait_seconds(msg)
            wait = wait if wait else 2 ** attempt + 5
            print(f"  [RateLimitError] waiting {wait:.1f}s (attempt {attempt+1}/{max_retries}):")
            time.sleep(wait)
        except Exception as e:
            msg = str(e)
            if "rate" in msg.lower() or "429" in msg or "limit" in msg.lower():
                wait = _suggested_wait_seconds(msg)
                wait = wait if wait else 2 ** attempt + 5
                print(f"  [Rate limit] waiting {wait:.1f}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                print(f"  [Non-rate error] {type(e).__name__}: {msg[:200]}")
                raise

    raise RuntimeError(f"evaluate_batch failed after {max_retries} attempts")


def _suggested_wait_seconds(error_message: str) -> float | None:
    """
    Parse a retry-after suggestion from a Groq rate-limit error message.

    Looks for patterns like "retry after 42", "try again in 3.5 seconds", "1.26s".

    Parameters
    ----------
    error_message
        The stringified exception message.

    Returns
    -------
    float or None
        Seconds to wait, or None if no retry-after could be parsed.
    """
    text = error_message.lower()
    _RETRY_AFTER_RE = re.compile(r"(?:try again in|retry after)\s*([\d.]+)\s*s?", re.IGNORECASE)
    m = _RETRY_AFTER_RE.search(text)
    if m:
        return float(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(
    artifacts_dir: str = "artifacts",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """
    Orchestrate the LLM evaluation pipeline for the shortlisted candidates.

    Loads the shortlist and JD profile, batches candidates, evaluates via Groq,
    and writes results to ``llm_evaluations.json``.

    Supports resume: if ``llm_evaluations.json`` already exists, already-evaluated
    candidates are skipped.

    Parameters
    ----------
    artifacts_dir
        Directory where artefacts are read from / written to.
    batch_size
        Number of candidates per LLM API call (default 8).

    Returns
    -------
    dict
        The full evaluations dict {candidate_id: {"score": float, "reasoning": str}}.
    """
    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load artefacts
    shortlist_path = out_dir / "shortlist.jsonl"
    jd_profile_path = out_dir / "jd_profile.json"
    evals_path = out_dir / "llm_evaluations.json"

    print(f"[evaluate_candidates] Loading shortlist from {shortlist_path} …")
    shortlist: list[dict] = []
    with open(shortlist_path, encoding="utf-8") as fh:
        for line in fh:
            shortlist.append(json.loads(line))
    print(f"[evaluate_candidates] Shortlist size: {len(shortlist)}")

    print(f"[evaluate_candidates] Loading JD profile from {jd_profile_path} …")
    jd_profile = json.loads(jd_profile_path.read_text(encoding="utf-8"))

    # 2. Load existing evals for resume support
    existing_evals: dict[str, dict[str, Any]] = {}
    if evals_path.exists():
        existing_evals = json.loads(evals_path.read_text(encoding="utf-8"))
        print(f"[evaluate_candidates] Resuming with {len(existing_evals)} existing evaluations")

    # 3. Build candidate summaries
    summaries = [build_candidate_summary(c) for c in shortlist]

    # Filter to only candidates not yet evaluated
    pending = [
        (c, s)
        for c, s in zip(shortlist, summaries)
        if c.get("candidate_id") not in existing_evals
    ]

    if not pending:
        print("[evaluate_candidates] All candidates already evaluated.")
        return existing_evals

    print(f"[evaluate_candidates] {len(pending)} candidates pending evaluation "
          f"({len(existing_evals)} already done)")

    # 4. Build LLM client
    # Dynamic max_tokens based on batch size: 300 tokens/candidate + 300 base
    max_tokens = 300 * batch_size + 300
    llm = build_groq_llm(model=EVAL_MODEL, max_tokens=max_tokens, temperature=0.0)

    # 5. Batch evaluation loop
    all_evals = dict(existing_evals)
    pending_candidates = [c for c, _ in pending]
    pending_summaries = [s for _, s in pending]

    for batch_start in tqdm(range(0, len(pending_summaries), batch_size), desc="Evaluating batches"):
        batch_end = min(batch_start + batch_size, len(pending_summaries))
        batch_cands = pending_candidates[batch_start:batch_end]
        batch_sums = pending_summaries[batch_start:batch_end]

        try:
            results = evaluate_batch(batch_sums, jd_profile, llm, max_retries=MAX_RETRIES)
        except RuntimeError as exc:
            if "daily token cap" in str(exc):
                raise
            print(f"    [evaluate_candidates] Batch failed, falling back to single-candidate: {exc}")
            results = []
            for c, s in zip(batch_cands, batch_sums):
                result = evaluate_one_with_fallback(s, jd_profile, llm)
                results.append(result)
                # Save checkpoint after each single-candidate fallback
                cid = result.get("candidate_id", "")
                all_evals[cid] = {"score": result.get("score", 0.0), "reasoning": result.get("reasoning", "")}
                evals_path.write_text(json.dumps(all_evals, indent=2), encoding="utf-8")

        # Merge results into all_evals
        for result in results:
            cid = result.get("candidate_id", "")
            if cid:
                all_evals[cid] = {
                    "score": float(result.get("score", 0.0)),
                    "reasoning": str(result.get("reasoning", "")),
                }

        # Checkpoint after EVERY batch
        evals_path.write_text(json.dumps(all_evals, indent=2), encoding="utf-8")
        print(f"    [evaluate_candidates] Checkpoint saved ({len(all_evals)}/{len(pending_summaries)} evals)")

        # Sleep between batches
        time.sleep(BATCH_SLEEP_SECONDS)

    print(f"[evaluate_candidates] Evaluation complete. "
          f"{len(all_evals)} total evaluations → {evals_path}")
    return all_evals


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    artifacts_dir = sys.argv[1] if len(sys.argv) > 1 else "artifacts"
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_BATCH_SIZE
    run(artifacts_dir=artifacts_dir, batch_size=batch_size)