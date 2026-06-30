# README.md Review

## Verdict: NEEDS_FIXES

## Issues

1. **Misleading LLM model name**
   - File: `/workspaces/redrob-ranker/README.md`
   - Line: 3
   - Problem: The README states the system "uses Groq (Llama 3.3 70B)" for JD analysis and candidate evaluation. However, `pre_computation/config.py` sets `GROQ_MODEL = "openai/gpt-oss-120b"` (line 30), and `pre_computation/evaluate_candidates.py` initializes the LLM with that model. This will confuse Stage 4 reviewers and anyone trying to reproduce the pre-computation step.
   - Suggested fix: Update line 3 to say "Groq (openai/gpt-oss-120b)" or, if Llama 3.3 70B is the intended model, change `GROQ_MODEL` in `pre_computation/config.py` to `"llama-3.3-70b-versatile"` and keep the README text.

2. **"Score all candidates" with LLM is inaccurate**
   - File: `/workspaces/redrob-ranker/README.md`
   - Line: 9
   - Problem: The Phase A bullet says Groq is used "to analyze the job description and score all candidates." In reality, embeddings and behavioral scores are computed for all candidates, but the LLM only evaluates the shortlist (`SHORTLIST_SIZE = 300` in `pre_computation/config.py`).
   - Suggested fix: Change the wording to "analyze the job description and score the shortlist of top candidates (e.g., 300) with Groq, while computing embeddings and behavioral signals for the full pool."

3. **Batch-call count is outdated**
   - File: `/workspaces/redrob-ranker/README.md`
   - Line: 36
   - Problem: The README claims "~38 batch calls for typical datasets." `pre_computation/evaluate_candidates.py` uses `BATCH_SIZE = 1` with token-aware batching (`_batch_by_token_budget`) over the 300-candidate shortlist. Actual call counts are driven by the token budget and are likely closer to 75–300 calls, not 38.
   - Suggested fix: Remove the specific number or replace it with a qualified range such as "token-aware batches over the shortlisted candidates; count depends on prompt size."

4. **Phase B command does not name the new default output file**
   - File: `/workspaces/redrob-ranker/README.md`
   - Lines: 43–47, 63–67
   - Problem: After the default output filename changed to `AI Builders.csv`, the README still only says "Produce the submission CSV" and gives `python rank.py --candidates ./candidates.jsonl`. A reader will not know what file was created until they reach the Validation section.
   - Suggested fix: Add the output filename to the Phase B description, e.g., "Produces `AI Builders.csv` by default" and show the optional `--out` form: `python rank.py --candidates ./candidates.jsonl --out "AI Builders.csv"`.

5. **Single-step isolation commands use wrong paths and flags**
   - File: `/workspaces/redrob-ranker/README.md`
   - Lines: 76–89
   - Problem: The listed commands (`python embed_candidates.py --candidates ./candidates.jsonl`, `python build_shortlist.py ...`, etc.) will fail because those modules live under `pre_computation/` and their CLI entrypoints take positional arguments, not `--candidates`. Verified: `python embed_candidates.py` raises "can't open file"; `python pre_computation/embed_candidates.py --candidates ./candidates.jsonl` raises `ModuleNotFoundError: No module named 'util'` because relative imports break when run as a script.
   - Suggested fix: Update the block to use module execution with positional arguments:
     ```bash
     python -m pre_computation.analyze_jd data/job_description.docx
     python -m pre_computation.embed_candidates data/candidates.jsonl
     python -m pre_computation.build_shortlist data/candidates.jsonl
     python -m pre_computation.evaluate_candidates
     python -m pre_computation.assemble_ranking
     ```

6. **Sandbox candidate limit is wrong**
   - File: `/workspaces/redrob-ranker/README.md`
   - Line: 102
   - Problem: The README says the Streamlit sandbox is "max 100 candidates," but `app.py` defines `MAX_CANDIDATES = 300` (line 29).
   - Suggested fix: Change "max 100 candidates" to "max 300 candidates" or update `app.py` to enforce 100 if that is the intended limit.

7. **Sandbox instructions describe a file upload that no longer exists**
   - File: `/workspaces/redrob-ranker/README.md`
   - Lines: 107–108
   - Problem: The README says "Upload a JSON or JSONL file with candidate records." `app.py` was simplified to a single "Run Ranking" button that always loads the built-in `data/sample_candidates.json`. There is no `st.file_uploader` or upload flow.
   - Suggested fix: Replace the sentence with: "Click **Run Ranking** to rank the bundled `data/sample_candidates.json`. Pre-computed rankings are used when available; missing candidates fall back to behavioral-only scoring. Results can be downloaded as `ranked.csv`."

8. **Formatting error at end of file**
   - File: `/workspaces/redrob-ranker/README.md`
   - Line: 108
   - Problem: The last sentence runs into the repository title: "missing candidates fall back to behavioral-only scoring.# redrob-ranker". The `# redrob-ranker` heading is missing a leading newline.
   - Suggested fix: Insert a blank line before `# redrob-ranker` or remove the stray title line.

9. **Setup instructions omit cache environment variables**
   - File: `/workspaces/redrob-ranker/README.md`
   - Line: 22
   - Problem: The `.env` example only sets `GROQ_API_KEY`. The repository's actual `.env` also sets `HF_HOME=.cache` and `HUGGINGFACE_HUB_CACHE=.cache/hub`, which controls where sentence-transformers downloads models. Omitting these can cause model re-downloads and unexpected disk usage during Stage 3 reproduction.
   - Suggested fix: Expand the `.env` example to:
     ```bash
     echo "GROQ_API_KEY=gsk_..." > .env
     echo "HF_HOME=.cache" >> .env
     echo "HUGGINGFACE_HUB_CACHE=.cache/hub" >> .env
     ```

10. **Submission metadata still references old filename**
    - File: `/workspaces/redrob-ranker/submission_metadata.yaml`
    - Lines: 27, 69
    - Problem: Although this is not `README.md`, it is a Stage 3/Stage 4 artifact and still lists the reproduce command as `python rank.py --candidates ./candidates.jsonl --out ./submission.csv` and the methodology summary mentions `submission.csv`. Since the new default output is `AI Builders.csv`, this metadata is inconsistent with the code.
    - Suggested fix: Update the `reproduce_command` to `python rank.py --candidates ./candidates.jsonl --out "AI Builders.csv"` and update the methodology summary to reference `AI Builders.csv`.
