"""
Phase A - Step 9: LLM batch evaluation of shortlisted candidates.

Public entrypoint: run(...) -> dict (results, also saved to llm_evaluations.json)

Uses Groq's llama-3.3-70b-versatile via util.llm_client — same GROQ_API_KEY as
analyze_jd.py, and the same model. The earlier choice of 8B here was a
token-budget optimization that turned out to be the wrong trade: at only ~300
candidates (post hard-requirement pre-filter in Step 8), token budget is not
binding, but reasoning quality IS the dominant signal in the final CSV — and
8B produces visibly templated output with no specific facts cited. 70B
produces the structured, fact-specific reasoning the submission format requires.

REASONING FORMAT: each candidate's `reasoning` field must follow exactly:
  "<current_title> with <years_of_experience> yrs; <ai_core_skill_count> AI core skills; response rate <response_rate>."
The prompt injects `current_title`, `years_of_experience`, `ai_core_skill_count`
(pre-computed from the candidate's evidenced AI/ML skills), and `response_rate`
(recruiter_response_rate) so the LLM fills in values rather than inventing them.

Outputs
-------
artifacts/llm_evaluations.json — {candidate_id: {"score": float, "reasoning": str}}
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from util.llm_client import build_groq_llm, parse_json_response

from pre_computation.config import (
    ARTIFACTS_DIR,
    GROQ_MODEL,
    JD_PROFILE_FILE,
    LLM_EVALUATIONS_FILE,
    SHORTLIST_FILE,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH_SIZE = 2                    # per-call candidate count; small bursts avoid TPM spikes
SLEEP_BETWEEN_BATCHES = 4.0       # seconds — courtesy floor between calls
MAX_RETRIES = 5
LONG_WAIT_THRESHOLD_S = 60        # wait longer than this = likely daily cap, not per-minute
TARGET_RPM = 25                   # conservative under the 30 RPM free-tier limit
GROQ_TPM_LIMIT = 6000             # Groq free-tier tokens-per-minute ceiling
TPM_SAFETY_FACTOR = 0.7           # only use 70% of TPM; absorbs our rough token estimator
EFFECTIVE_TPM = int(GROQ_TPM_LIMIT * TPM_SAFETY_FACTOR)
MAX_PROMPT_TOKENS = 4000          # prompt-only cap; total per call is kept under EFFECTIVE_TPM by pacing

# career_history is the single biggest token cost in the prompt by a wide margin.
FULL_DETAIL_ROLES = 2             # most recent N roles get full description text
MAX_TEXT_CHARS = 400              # truncation length for descriptions/summary — preserves concrete facts

# Skill names (lowercased substrings) that count as "AI/ML core skills" for the
# reasoning format. Used to pre-compute the <N> AI core skills count shown in
# each candidate's reasoning string.
AI_CORE_SKILL_KEYWORDS = {
    "machine learning", "deep learning", "neural network", "neural net",
    "artificial intelligence", "nlp", "natural language processing",
    "computer vision", "embedding", "retrieval", "ranking", "recommendation",
    "recommender", "llm", "large language model", "transformer", "bert", "gpt",
    "rag", "vector search", "vector database", "pinecone", "weaviate",
    "elasticsearch", "faiss", "sentence-transformer", "tensorflow", "pytorch",
    "scikit-learn", "sklearn", "huggingface", "langchain", "mlops", "mlflow",
    "kubeflow", "feature store", "knowledge graph", "search relevance",
    "learning to rank", "ltr", "ndcg", "mrr", "map", "semantic search",
    "information retrieval", "ir", "text classification", "named entity",
    "question answering", "qa system", "chatbot", "speech recognition",
    "reinforcement learning", "rlhf", "prompt engineering",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_ai_core_skills(skills: list) -> int:
    """
    Count the candidate's AI/ML core skills: skills from their skills list whose
    name matches one of AI_CORE_SKILL_KEYWORDS AND that have evidence
    (endorsements > 0 OR duration_months > 0). Same evidence filter used in
    util.candidate_text.build_candidate_text — no zero-evidence keyword stuffing.
    """
    count = 0
    for s in skills or []:
        name = (s.get("name") or "").strip().lower()
        if not name:
            continue
        endorsements = s.get("endorsements", 0) or 0
        duration = s.get("duration_months", 0) or 0
        if endorsements <= 0 and duration <= 0:
            continue
        if any(kw in name for kw in AI_CORE_SKILL_KEYWORDS):
            count += 1
    return count


def _truncate(text: str, max_chars: int) -> str:
    """Hard truncates at a word boundary."""
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "\u2026"


# ---------------------------------------------------------------------------
# Token estimation and token-aware batching
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Rough token estimate using a 4-char-per-token heuristic.

    Good enough for prompt sizing against Groq's per-minute token budget
    (6,000 TPM on the free tier) — we only need to avoid blowing past it,
    not hit it exactly. Rounding up means we slightly under-batch, which is
    the safe direction.
    """
    return (len(text) + 3) // 4


def _estimate_total_tokens_for_batch(
    batch: list[dict[str, Any]],
    jd_profile: dict[str, Any],
) -> int:
    """Estimated input + output tokens for one batch call."""
    prompt = build_eval_prompt(batch, jd_profile)
    prompt_tokens = _estimate_tokens(prompt)
    output_tokens = estimate_max_tokens(len(batch))
    return prompt_tokens + output_tokens


def _batch_by_token_budget(
    candidates: list[dict[str, Any]],
    jd_profile: dict[str, Any],
    max_prompt_tokens: int,
    max_candidates_per_call: int,
) -> list[list[dict[str, Any]]]:
    """
    Build batches whose estimated prompt tokens stay within ``max_prompt_tokens``
    and whose candidate count stays within ``max_candidates_per_call``.

    Pure — no I/O, no API call. Builds the prompt once with an empty candidate
    list to measure the fixed overhead (role context, JD requirements, etc.),
    then greedily adds candidates until adding the next one would exceed either
    budget, at which point it starts a new batch.

    Parameters
    ----------
    candidates
        Candidate records to partition into batches. Order is preserved.
    jd_profile
        The structured JD profile dict — used to measure prompt overhead.
    max_prompt_tokens
        Hard cap on the estimated prompt token count for any single batch.
    max_candidates_per_call
        Hard cap on the number of candidates in any single batch.

    Returns
    -------
    list[list[dict]]
        Ordered list of batches. Empty input yields ``[]``.
    """
    if not candidates:
        return []

    # Measure the fixed prompt overhead by building the prompt with no
    # candidates. ``build_eval_prompt`` accepts an empty list and serialises
    # it as "[]".
    base_prompt = build_eval_prompt([], jd_profile)
    base_tokens = _estimate_tokens(base_prompt)

    # Defensive fallback: if the fixed overhead alone already blows the
    # budget (e.g. a giant JD), degrade to one candidate per call so we still
    # make progress instead of returning empty batches.
    if base_tokens >= max_prompt_tokens:
        return [[c] for c in candidates]

    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_tokens = base_tokens

    for c in candidates:
        # Each candidate adds its serialised summary plus a few separator
        # tokens (",", newline, indentation). 5 is a safe upper bound on
        # those structural tokens for the indented JSON array.
        c_tokens = _estimate_tokens(json.dumps(build_candidate_summary(c), indent=2)) + 5

        if current and (
            current_tokens + c_tokens > max_prompt_tokens
            or len(current) >= max_candidates_per_call
        ):
            batches.append(current)
            current = []
            current_tokens = base_tokens

        current.append(c)
        current_tokens += c_tokens

    if current:
        batches.append(current)

    return batches


# ---------------------------------------------------------------------------
# Candidate summary builder
# ---------------------------------------------------------------------------

def build_candidate_summary(c: dict[str, Any]) -> dict[str, Any]:
    """
    Compact candidate representation for the LLM evaluator. Pure — no I/O.

    Token design:
    - Full description text only for the FULL_DETAIL_ROLES most recent roles.
      Older roles still contribute title/company/duration_months for pattern
      detection (title-chasing, tenure) — just not full prose.
    - recruiter_response_rate and ai_core_skill_count are included so the LLM
      can produce the structured reasoning format. These are tiny — a few
      tokens per candidate.
    - No full behavioral_signals block. Those signals are already scored for
      free and deterministically in util.behavioral.compute_behavioral_score().
    """
    profile = c.get("profile", {})
    career = c.get("career_history", [])
    signals = c.get("redrob_signals", {}) or {}

    career_summary = []
    for i, job in enumerate(career):
        entry = {
            "title":           job.get("title", ""),
            "company":         job.get("company", ""),
            "duration_months": job.get("duration_months", 0),
            "industry":        job.get("industry", ""),
        }
        if i < FULL_DETAIL_ROLES:
            entry["description"] = _truncate(job.get("description", ""), MAX_TEXT_CHARS)
        career_summary.append(entry)

    return {
        "candidate_id":        c.get("candidate_id", ""),
        "current_title":       profile.get("current_title", ""),
        "years_of_experience": profile.get("years_of_experience", 0),
        "location":            f"{profile.get('location', '')}, {profile.get('country', '')}",
        "headline":            profile.get("headline", ""),
        "summary":             _truncate(profile.get("summary", ""), MAX_TEXT_CHARS),
        "career_history":      career_summary,
        "education": [
            {
                "degree":      edu.get("degree", ""),
                "field":       edu.get("field_of_study", ""),
                "institution": edu.get("institution", ""),
                "tier":        edu.get("tier", "unknown"),
            }
            for edu in c.get("education", [])
        ],
        # ── Small fields used by the structured reasoning format ──
        "response_rate":       round(float(signals.get("recruiter_response_rate", 0.0)), 2),
        "ai_core_skill_count": _count_ai_core_skills(c.get("skills", [])),
    }


# ---------------------------------------------------------------------------
# Evaluation prompt
# ---------------------------------------------------------------------------

EVAL_PROMPT = """You are a senior technical recruiter scoring candidates for a specific role.

ROLE CONTEXT:
{role_summary}

WHAT THIS ROLE REQUIRES (describe DONE work, not keywords):
{hard_requirements}

NICE-TO-HAVE (these can BOOST a candidate's score only if hard requirements are already met; never rescue a candidate missing hard requirements):
{nice_to_have}

DISQUALIFYING CAREER PATTERNS (these make a candidate unsuitable):
{disqualifier_patterns}

EVALUATION GUIDANCE:
{evaluation_guidance}

ADDITIONAL CONTEXT: Location preference: {preferred_location}. Experience: {exp_min}\u2013{exp_max} years. Notice period: {notice_preference}.

Scoring rubric (0.00\u20131.00):
- 0.85\u20131.00: Career descriptions clearly demonstrate the required experience at product companies.
- 0.65\u20130.84: Most hard requirements met; minor gaps or concerns.
- 0.40\u20130.64: Relevant experience exists but significant gaps present in hard requirements.
- 0.20\u20130.39: Tangential or adjacent experience only \u2014 missing core hard requirements.
- 0.00\u20130.19: Wrong domain, or a disqualifying career pattern is present.

=== REASONING FORMAT (MANDATORY \u2014 match exactly) ===
For each candidate, write the reasoning string in EXACTLY this format:

  "<current_title> with <years_of_experience> yrs; <ai_core_skill_count> AI core skills; response rate <response_rate>."

Examples of CORRECT format:
  "Senior ML Engineer with 6.5 yrs; 12 AI core skills; response rate 0.85."
  "Data Scientist with 3.2 yrs; 4 AI core skills; response rate 0.40."
  "HR Manager with 6.1 yrs; 0 AI core skills; response rate 0.76."

Rules:
- Use the EXACT values from the candidate data (current_title, years_of_experience, ai_core_skill_count, response_rate).
- Keep all four components in the same order, separated by '; ' and ending with '.'.
- Do NOT write free-form prose like "Strong match" or "Extensive experience with...".
- Do NOT include explanations, caveats, or additional sentences.
- One line per candidate, period at the end.

Return ONLY a JSON array. No other text:
[
  {{
    "candidate_id": "CAND_XXXXXXX",
    "score": <float 0.0 to 1.0>,
    "reasoning": "<current_title> with <X.X> yrs; <N> AI core skills; response rate <R.RR>."
  }},
  ...
]

Candidates to evaluate:
{candidates_json}"""


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_eval_prompt(batch: list[dict[str, Any]], jd_profile: dict[str, Any]) -> str:
    """
    Format the evaluation prompt for a batch of candidate summaries.

    Pure — no I/O, no API call. Reads ``nice_to_have_requirements``,
    ``hard_requirements``, ``disqualifier_patterns`` and the standard JD
    fields out of ``jd_profile`` and injects them into ``EVAL_PROMPT``.

    Parameters
    ----------
    batch
        List of full candidate record dicts (will be summarized).
    jd_profile
        The structured JD profile dict. Expected to contain:
        ``role_summary``, ``hard_requirements``, ``nice_to_have_requirements``,
        ``disqualifier_patterns``, ``evaluation_guidance``,
        ``preferred_location``, ``experience_years``, ``notice_preference``.

    Returns
    -------
    str
        Formatted prompt string.
    """
    summaries = [build_candidate_summary(c) for c in batch]
    exp = jd_profile.get("experience_years", {"min": 5, "max": 9})
    nice_to_have_items = jd_profile.get("nice_to_have_requirements", [])
    nice_to_have_block = (
        "\n".join(f"- {r}" for r in nice_to_have_items)
        if nice_to_have_items else "(none specified)"
    )
    hard_reqs = jd_profile.get("hard_requirements", []) or []
    hard_reqs_block = (
        "\n".join(f"- {r}" for r in hard_reqs) if hard_reqs else "(not specified)"
    )
    disqual = jd_profile.get("disqualifier_patterns", []) or []
    disqual_block = (
        "\n".join(f"- {d}" for d in disqual) if disqual else "(none specified)"
    )
    return EVAL_PROMPT.format(
        role_summary          = jd_profile.get("role_summary", ""),
        hard_requirements     = hard_reqs_block,
        nice_to_have          = nice_to_have_block,
        disqualifier_patterns = disqual_block,
        evaluation_guidance   = jd_profile.get("evaluation_guidance", ""),
        preferred_location    = jd_profile.get("preferred_location", "India"),
        exp_min               = exp.get("min", 5),
        exp_max               = exp.get("max", 9),
        notice_preference     = jd_profile.get("notice_preference", "under 30 days preferred"),
        candidates_json       = json.dumps(summaries, indent=2),
    )


# ---------------------------------------------------------------------------
# Token sizing + retry-after parsing
# ---------------------------------------------------------------------------

def estimate_max_tokens(batch_size: int) -> int:
    """Scales max_tokens with actual batch size. ~120 tokens per output entry + buffer."""
    return 120 * batch_size + 250


_RETRY_AFTER_RE = re.compile(r"(?:try again in|retry after)\s*([\d.]+)\s*s?", re.IGNORECASE)


def _suggested_wait_seconds(error_message: str) -> float | None:
    """Pulls suggested wait time out of a Groq rate-limit error message."""
    match = _RETRY_AFTER_RE.search(error_message)
    return float(match.group(1)) if match else None


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_batch(
    batch: list[dict[str, Any]],
    jd_profile: dict[str, Any],
    llm,
    max_retries: int = MAX_RETRIES,
) -> tuple[list[dict[str, Any]], float]:
    """
    One LLM call for a batch, with retry/backoff on per-minute rate-limit errors.

    Distinguishes per-minute limits (fixable by waiting) from daily caps (not
    fixable — surfaces them immediately so you can resume later rather than
    burning retries).

    Parameters
    ----------
    batch
        List of full candidate record dicts.
    jd_profile
        JD profile dict.
    llm
        LangChain ChatGroq instance.
    max_retries
        Maximum retry attempts.

    Returns
    -------
    tuple[list[dict], float]
        ``(parsed_results, total_rate_limit_wait_seconds)``. The wait total
        lets ``run()`` skip redundant pacing sleep when a rate-limit backoff
        already kept us idle longer than ``SLEEP_BETWEEN_BATCHES``.
    """
    prompt = build_eval_prompt(batch, jd_profile)
    last_err: Exception | None = None
    total_rate_limit_wait = 0.0
    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt)
            return parse_json_response(response.content), total_rate_limit_wait
        except Exception as e:
            last_err = e
            msg = str(e)
            if "rate" in msg.lower() or "429" in msg:
                wait = _suggested_wait_seconds(msg)
                # Daily-cap detection: very long suggested wait OR message
                # literally mentions a daily limit. Either way we won't fix
                # it by retrying inside this minute — surface it so the run
                # halts cleanly and can be resumed from the checkpoint.
                if (wait is not None and wait > LONG_WAIT_THRESHOLD_S) or "daily" in msg.lower():
                    wait_desc = f"{wait:.0f}s" if wait is not None else "an extended period"
                    raise RuntimeError(
                        f"Groq suggests waiting {wait_desc} — likely a daily token cap, "
                        f"not a per-minute one. Resume later with: "
                        f"python -m pre_computation.pipeline --from 4"
                    ) from e
                # Per-minute TPM/RPM limit: respect Groq's suggested wait
                # when present, otherwise exponential backoff.
                backoff = wait if wait is not None else 2 ** attempt
                total_rate_limit_wait += backoff
                print(f"  Rate limited, waiting {backoff:.0f}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(backoff)
                continue
            raise
    raise last_err  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Single-candidate fallback
# ---------------------------------------------------------------------------

def evaluate_one_with_fallback(
    c: dict[str, Any],
    jd_profile: dict[str, Any],
    llm,
) -> dict[str, Any]:
    """Single-candidate retry used when a batch fails — isolates the broken candidate."""
    try:
        time.sleep(1)
        results, _ = evaluate_batch([c], jd_profile, llm)
        result = results[0]
        return {"score": float(result["score"]), "reasoning": result["reasoning"]}
    except Exception as e:
        print(f"  Individual eval failed for {c.get('candidate_id', '?')}: {e}")
        return {
            "score":     c.get("_semantic_score", 0.0),
            "reasoning": "Automated scoring only — LLM evaluation failed for this candidate.",
        }


# ---------------------------------------------------------------------------
# Checkpoint helper
# ---------------------------------------------------------------------------

def _save_results(results: dict[str, Any], artifacts_dir: str) -> None:
    """Writes the current results dict to disk. Called after every batch, not
    just at the end — this is the actual checkpoint. A crash, Ctrl-C, or a
    daily-token-cap RuntimeError after this point loses at most one batch's
    worth of work, not the whole run."""
    out_path = Path(artifacts_dir) / LLM_EVALUATIONS_FILE
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(artifacts_dir: str = ARTIFACTS_DIR, batch_size: int = BATCH_SIZE) -> dict[str, Any]:
    """
    Orchestrate the LLM evaluation pipeline for the shortlisted candidates.

    Loads the shortlist and JD profile, batches candidates, evaluates via Groq
    using ``GROQ_MODEL``, and writes results to ``llm_evaluations.json``.

    Supports resume: if ``llm_evaluations.json`` already exists, already-
    evaluated candidates are skipped.

    Parameters
    ----------
    artifacts_dir
        Directory where artefacts are read from / written to.
    batch_size
        Maximum number of candidates per LLM API call. Actual batch sizes are
        smaller when the token budget (``MAX_PROMPT_TOKENS``) would otherwise
        be exceeded (see ``_batch_by_token_budget``).

    Returns
    -------
    dict
        The full evaluations dict {candidate_id: {"score": float, "reasoning": str}}.
    """
    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dynamic max_tokens based on batch size: ~120 tokens/candidate + 250 base
    max_tokens = estimate_max_tokens(batch_size)
    llm = build_groq_llm(model=GROQ_MODEL, max_tokens=max_tokens, temperature=0.0)
    print(f"[INFO] Groq LLM initialized: {GROQ_MODEL} (max_tokens={max_tokens})")

    # 1. Load artefacts
    jd_profile_path = out_dir / JD_PROFILE_FILE
    shortlist_path = out_dir / SHORTLIST_FILE
    evals_path = out_dir / LLM_EVALUATIONS_FILE

    print(f"[evaluate_candidates] Loading JD profile from {jd_profile_path} …")
    jd_profile = json.loads(jd_profile_path.read_text(encoding="utf-8"))

    candidates: list[dict[str, Any]] = []
    with open(shortlist_path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                candidates.append(json.loads(line))
    print(f"[evaluate_candidates] Shortlist size: {len(candidates)}")

    # ── Resume support: load any partial results from a prior interrupted run ──
    # This is what makes "resume later" (the advice given when a daily-cap
    # RuntimeError fires in evaluate_batch) actually work, instead of just
    # being a suggestion that re-spends tokens on candidates already scored.
    results: dict[str, dict[str, Any]] = {}
    if evals_path.exists():
        try:
            results = json.loads(evals_path.read_text(encoding="utf-8"))
            if results:
                print(f"[INFO] Resuming — {len(results)} candidates already evaluated in a prior run.")
        except json.JSONDecodeError:
            pass

    remaining = [c for c in candidates if c.get("candidate_id") not in results]
    skipped = len(candidates) - len(remaining)
    if skipped:
        print(f"[INFO] Skipping {skipped} already-evaluated candidates.")

    if not remaining:
        print("[evaluate_candidates] All candidates already evaluated.")
        return results

    print(f"[evaluate_candidates] Evaluating {len(remaining)} candidates "
          f"with token-aware batching (max {batch_size}/call)...")

    # Build batches using token-aware sizing: respects MAX_PROMPT_TOKENS AND
    # the caller's batch_size cap. Counts and sizes are logged below.
    batches = _batch_by_token_budget(
        remaining,
        jd_profile,
        max_prompt_tokens=MAX_PROMPT_TOKENS,
        max_candidates_per_call=batch_size,
    )
    print(f"[evaluate_candidates] {len(batches)} batches planned "
          f"(target \u2264 {TARGET_RPM} RPM, \u2264 {EFFECTIVE_TPM} total tokens/min, "
          f"\u2264 {MAX_PROMPT_TOKENS} prompt tokens/call).")

    # 5. Batch evaluation loop
    failed: list[str] = []
    # Adaptive pacing: respect both RPM and TPM limits. For each batch we
    # estimate total tokens (prompt + expected output) and ensure we do not
    # exceed EFFECTIVE_TPM per minute. We also stay under TARGET_RPM. If a
    # rate-limit backoff inside ``evaluate_batch`` already kept us idle
    # longer than the computed interval, we skip redundant extra sleep.
    last_call_time = time.monotonic()
    rpm_interval = 60.0 / TARGET_RPM

    for batch in tqdm(batches, desc="LLM evaluation"):
        rate_limit_wait = 0.0
        try:
            batch_results, rate_limit_wait = evaluate_batch(batch, jd_profile, llm)
            for r in batch_results:
                cid = r.get("candidate_id", "")
                if cid:
                    results[cid] = {
                        "score": float(r.get("score", 0.0)),
                        "reasoning": str(r.get("reasoning", "")),
                    }
        except Exception as e:
            print(f"\nBatch failed ({e}). Retrying individually...")
            for c in batch:
                outcome = evaluate_one_with_fallback(c, jd_profile, llm)
                cid = c.get("candidate_id", "")
                if cid:
                    results[cid] = outcome
                if outcome["reasoning"].startswith("Automated scoring only"):
                    failed.append(cid)

        # Checkpoint after every batch — not just once at the end.
        _save_results(results, artifacts_dir)
        print(f"    [evaluate_candidates] Checkpoint saved ({len(results)} evals)")

        # Adaptive sleep: respect RPM and TPM budgets. Larger calls wait longer.
        elapsed = time.monotonic() - last_call_time
        batch_tokens = _estimate_total_tokens_for_batch(batch, jd_profile)
        tpm_interval = (batch_tokens / EFFECTIVE_TPM) * 60.0
        min_interval = max(rpm_interval, tpm_interval)
        needed = min_interval - elapsed
        if rate_limit_wait < SLEEP_BETWEEN_BATCHES:
            sleep_for = max(SLEEP_BETWEEN_BATCHES, needed)
            if sleep_for > 0:
                time.sleep(sleep_for)
        last_call_time = time.monotonic()

    print(f"[evaluate_candidates] llm_evaluations.json saved "
          f"({len(results)} total candidates)")
    scored_this_run = len(remaining) - len(failed)
    print(f"  Scored this run: {scored_this_run} | Failed: {len(failed)}")
    if failed:
        print(f"  Failed IDs: {failed}")
    return results


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    artifacts_dir = sys.argv[1] if len(sys.argv) > 1 else ARTIFACTS_DIR
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else BATCH_SIZE
    run(artifacts_dir=artifacts_dir, batch_size=batch_size)
