# Redrob Hackathon — AI-Driven Candidate Ranker

An offline-capable, two-phase candidate ranking system that uses Groq (Llama 3.3 70B) for JD analysis and candidate evaluation during pre-computation, then runs completely network-free during the ranking phase.

## Architecture

The system follows a **two-phase design**:

- **Phase A — Pre-computation**: Network-dependent. Uses Groq LLM API and sentence-transformers to analyze the job description and score all candidates. Produces artifacts (LLM scores, semantic embeddings, behavioral signals). Takes ~45–60 minutes for large candidate pools.
- **Phase B — Ranking**: Network-free. Loads pre-computed artifacts and produces a ranked submission CSV in under 5 minutes on CPU.

## Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

echo "GROQ_API_KEY=gsk" > .env
```

## Phase A — Pre-computation

Analyze the job description and score all candidates (requires network access):

```bash
python -m pre_computation.pipeline --jd data/job_description.docx
```

This step:
- Extracts key requirements from the job description using Groq (1 LLM call)
- Generates candidate embeddings via sentence-transformers
- Evaluates each candidate with Groq (~38 batch calls for typical datasets)
- Produces `artifacts/final_ranking.json` and `artifacts/rank_config.json`

**Time:** ~45–60 minutes for large candidate pools (network ON required)

## Phase B — Ranking step

Produce the submission CSV from pre-computed artifacts (network OFF):

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
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
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

Scores are recomputed from component scores using the new weights — no need to re-run Phase A.

## Running a single step in isolation

Each pre-computation step can be run independently:

```bash
# Step 1: Analyze job description
python -m pre_computation.analyze_jd --jd data/job_description.docx

# Step 2: Generate candidate embeddings
python embed_candidates.py --candidates ./candidates.jsonl

# Step 3: Build shortlist based on embeddings
python build_shortlist.py --candidates ./candidates.jsonl

# Step 4: Evaluate candidates with LLM
python evaluate_candidates.py --candidates ./candidates.jsonl

# Step 5: Assemble final ranking
python assemble_ranking.py
```

## Validation

Validate your submission CSV:

```bash
python validate_submission.py submission.csv
```

## Sandbox

Run the interactive Streamlit sandbox for demo and testing (max 100 candidates):

```bash
streamlit run app.py
```

Upload a JSON or JSONL file with candidate records. Candidates found in the pre-computed ranking display their full scores and reasoning; missing candidates fall back to behavioral-only scoring.# redrob-ranker
