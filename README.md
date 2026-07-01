# Redrob Hackathon — AI-Driven Candidate Ranker

An offline-capable, two-phase candidate ranking system that uses Groq (openai/gpt-oss-120b) for JD analysis and candidate evaluation during pre-computation, then runs completely network-free during the ranking phase.

## Sandbox / Demo

Try it online with no install required:

**Live demo:** https://redrob-airanker.streamlit.app/

The sandbox exposes a single **Run Ranking** button that scores the bundled candidate pool and displays the top-100 ranking directly in your browser.

To run the same sandbox locally:

```bash
streamlit run app.py
```

Pre-computed rankings are used when available; missing candidates fall back to behavioral-only scoring. Results can be downloaded as `ranked.csv`.

## Quick start

Generate the top-100 submission CSV locally from the bundled candidates and pre-computed artifacts. **No network access required**, finishes in **under 5 minutes on CPU**:

```bash
python rank.py --candidates ./candidates.jsonl
```

This writes **`AI Builders.csv`** to the repository root — that is the default output filename used by `rank.py`. The command is fully offline because Phase A artifacts are already cached under `artifacts/`.

To specify an explicit output path:

```bash
python rank.py --candidates ./candidates.jsonl --out "AI Builders.csv"
```

## Repository overview

The repository is organised around the two phases of the pipeline:

- **`rank.py`** — Phase B entry point. Loads pre-computed artifacts and produces the ranked submission CSV (`AI Builders.csv` by default).
- **`app.py`** — Streamlit sandbox / demo UI (see the *Sandbox / Demo* section above).
- **`pre_computation/`** — Phase A pipeline:
  - `analyze_jd.py` — extract key requirements from the job description (Groq, 1 LLM call)
  - `embed_candidates.py` — generate candidate embeddings (sentence-transformers)
  - `build_shortlist.py` — shortlist candidates based on embedding similarity
  - `evaluate_candidates.py` — LLM-based scoring of shortlisted candidates
  - `assemble_ranking.py` — assemble the final ranking from component scores
  - `pipeline.py` — wires the steps above together as the single `python -m pre_computation.pipeline` entry point
- **`artifacts/`** — pre-computed files required by `rank.py` and `app.py` (JD profile and embedding, LLM evaluations, behavioral scores, shortlist, final ranking, rank config).
- **`data/`** — released job description (`job_description.docx`), full candidate pool (`candidates.jsonl`), submission spec (`submission_spec.docx`), and the smaller sample used by the sandbox (`sample_candidates.json`).
- **`util/`** — shared helpers (LLM client, behavioral scoring, candidate text builder, submission utilities).

## Architecture

The system follows a **two-phase design**:

- **Phase A — Pre-computation**: Network-dependent. Uses Groq LLM API and sentence-transformers to analyze the job description and score the shortlist of top candidates (e.g., 300) with Groq, while computing embeddings and behavioral signals for the full pool. Produces artifacts (LLM scores, semantic embeddings, behavioral signals). Takes ~45–60 minutes for large candidate pools.
- **Phase B — Ranking**: Network-free. Loads pre-computed artifacts and produces a ranked submission CSV in under 5 minutes on CPU.

## Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

echo "GROQ_API_KEY=gsk" > .env
echo "HF_HOME=.cache" >> .env
echo "HUGGINGFACE_HUB_CACHE=.cache/hub" >> .env
```

## Phase A — Pre-computation

Analyze the job description and score all candidates (requires network access):

```bash
python -m pre_computation.pipeline --jd data/job_description.docx
```

This step:

- Extracts key requirements from the job description using Groq (1 LLM call)
- Generates candidate embeddings via sentence-transformers
- Evaluates shortlisted candidates with Groq in token-aware batches over the shortlisted candidates; count depends on prompt size
- Produces `artifacts/final_ranking.json` and `artifacts/rank_config.json`

**Time:** ~45–60 minutes for large candidate pools (network ON required)

## Phase B — Ranking step

Produces `AI Builders.csv` by default from pre-computed artifacts (network OFF):

```bash
python rank.py --candidates ./candidates.jsonl
```

To specify an explicit output path:

```bash
python rank.py --candidates ./candidates.jsonl --out "AI Builders.csv"
```

**Time:** Under 5 minutes on CPU with 16 GB RAM (no network access)

## Weight Tuning

After Phase A, edit `artifacts/rank_config.json` to adjust scoring weights:

```json
{
  "llm_weight": 0.65,
  "semantic_weight": 0.20,
  "behavioral_weight": 0.15
}
```

Then re-run Phase B:

```bash
python rank.py --candidates ./candidates.jsonl
```

Scores are recomputed from component scores using the new weights — no need to re-run Phase A.

## Running a single step in isolation

Each pre-computation step can be run independently using module execution with positional arguments:

```bash
# Step 1: Analyze job description
python -m pre_computation.analyze_jd data/job_description.docx

# Step 2: Generate candidate embeddings
python -m pre_computation.embed_candidates data/candidates.jsonl

# Step 3: Build shortlist based on embeddings
python -m pre_computation.build_shortlist data/candidates.jsonl

# Step 4: Evaluate candidates with LLM
python -m pre_computation.evaluate_candidates

# Step 5: Assemble final ranking
python -m pre_computation.assemble_ranking
```

## Validation

Validate your submission CSV:

```bash
python validate_submission.py "AI Builders.csv"
```
