# Optimization Plan: `pre_computation/evaluate_candidates.py`

## Goal
Reduce Groq API rate-limit failures while preserving the recent quality improvements (70B model, structured reasoning, resume/checkpoint).

## Problem
The recent refactor increased per-call token volume dramatically:
- `batch_size` 1 → 8
- `FULL_DETAIL_ROLES` 1 → 4
- `MAX_TEXT_CHARS` 100 → 400
- Sleep between calls 20 s → 2 s

Groq free-tier limits for `llama-3.3-70b-versatile` are 30 RPM / 6,000 TPM / ~1,000 RPD. A single 8-candidate batch can consume most of the 6,000 TPM budget, and the 2-second sleep fires the next large call before the TPM window resets.

## Solution
Token-aware batching + adaptive pacing + better rate-limit classification.

## Files Changed
- `pre_computation/evaluate_candidates.py`

## Detailed Changes

### 1. Token estimation and token-aware batch sizing
- Add `_estimate_tokens(text: str) -> int` using a simple 4-char-per-token heuristic (good enough for prompt sizing).
- Add `_batch_by_token_budget(
      candidates: list[dict[str, Any]],
      jd_profile: dict[str, Any],
      max_prompt_tokens: int,
      max_candidates_per_call: int,
  ) -> list[list[dict[str, Any]]]`
  - Builds batches where prompt tokens ≤ `max_prompt_tokens` AND count ≤ `max_candidates_per_call`.
- Replace the fixed `remaining[i:i + batch_size]` slicing with `_batch_by_token_budget(...)`.

### 2. Conservative defaults
- `BATCH_SIZE = 3` (down from 8).
- `SLEEP_BETWEEN_BATCHES = 4.0` (up from 2.0).
- `FULL_DETAIL_ROLES = 2` (down from 4).
- `MAX_TEXT_CHARS = 200` (down from 400).
- `MAX_PROMPT_TOKENS = 3500` (new) — leaves headroom for 6,000 TPM free-tier limit plus ~1,200 output tokens.

### 3. Adaptive sleep in `run()`
- Track `_last_call_time`.
- After each batch, compute `min_interval = 60.0 / TARGET_RPM` with `TARGET_RPM = 25` (conservative under 30 RPM free tier).
- Sleep `max(SLEEP_BETWEEN_BATCHES, min_interval - elapsed)`.
- If a rate-limit wait was longer than `SLEEP_BETWEEN_BATCHES`, do not add extra sleep on top.

### 4. Better rate-limit classification in `evaluate_batch()`
- Keep existing retry loop.
- When a 429/rate error arrives:
  - Parse `Retry-After` / suggested wait from the message.
  - If wait > 60 s or message contains "daily", raise the daily-cap `RuntimeError` immediately (existing behavior).
  - If wait between ~2–60 s, treat as TPM/RPM and sleep that duration.
  - Otherwise use exponential backoff `2 ** attempt`.
- Return the actual wait time so `run()` can skip redundant pacing sleep.

### 5. Preserve existing quality features
- Keep 70B model via `GROQ_MODEL`.
- Keep structured reasoning format and `_count_ai_core_skills`.
- Keep resume/checkpoint logic.
- Keep single-candidate fallback on batch failure.

## Complexity
**complex** — rate-limit handling, retries, adaptive pacing, token budgeting. Requires careful reasoning about edge cases (empty candidate list, daily cap, malformed rate-limit messages, resume state).

## Acceptance Criteria
1. `python -m py_compile pre_computation/evaluate_candidates.py` exits 0.
2. `python -c "from pre_computation import evaluate_candidates; print(evaluate_candidates.BATCH_SIZE)"` exits 0 and prints `3`.
3. `_batch_by_token_budget` splits a list of dummy candidates into multiple batches when the token budget is exceeded.
4. `_suggested_wait_seconds` still parses retry-after text correctly.
5. No `TODO` or placeholder code remains.

## Out of Scope
- Unit test files (explicitly requested not to add).
- Changing the LLM model.
- Modifying `util/llm_client.py` or other modules.
