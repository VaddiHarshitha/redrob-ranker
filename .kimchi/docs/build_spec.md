# Redrob Hackathon — Build Specification

## Project Root
/Users/saichandini/redrobai/

## Data Files
- `/Users/saichandini/redrobai/candidates.jsonl` — 100K candidate records (JSONL format)
- `/Users/saichandini/redrobai/job_description.docx` — Job description
- `/Users/saichandini/redrobai/sample_candidates.json` — 5 sample candidates
- `/Users/saichandini/redrobai/sample_submission.csv` — Sample submission format
- `/Users/saichandini/redrobai/validate_submission.py` — Official validator (DO NOT MODIFY)

## Virtual Environment
Activated at `/Users/saichandini/redrobai/.venv/`
Installed packages: langchain-groq, python-dotenv, sentence-transformers, numpy, pandas, tqdm, pyyaml, python-docx, torch, transformers, groq

## Candidate Data Schema (Top-Level Keys)
`candidate_id`, `profile`, `career_history`, `education`, `skills`, `certifications`, `languages`, `redrob_signals`

## Profile Keys
`anonymized_name`, `headline`, `summary`, `location`, `country`, `years_of_experience`, `current_title`, `current_company`, `current_company_size`, `current_industry`

## Career History Entry Keys
`company`, `title`, `start_date`, `end_date`, `duration_months`, `is_current`, `industry`, `company_size`, `description`

## Education Entry Keys
`institution`, `degree`, `field_of_study`, `start_year`, `end_year`, `grade`, `tier`

## Skills Entry Keys
`name`, `proficiency`, `endorsements`, `duration_months`

## redrob_signals Keys
`profile_completeness_score`, `signup_date`, `last_active_date`, `open_to_work_flag`, `profile_views_received_30d`, `applications_submitted_30d`, `recruiter_response_rate`, `avg_response_time_hours`, `skill_assessment_scores`, `connection_count`, `endorsements_received`, `notice_period_days`, `expected_salary_range_inr_lpa`, `preferred_work_mode`, `willing_to_relocate`, `github_activity_score`, `search_appearance_30d`, `saved_by_recruiters_30d`, `interview_completion_rate`, `offer_acceptance_rate`, `verified_email`, `verified_phone`, `linkedin_connected`

## Validator Rules (V1-V14)
- V1: `.csv` extension
- V2: non-empty filename stem
- V3: UTF-8 encoding
- V4: header exactly `candidate_id,rank,score,reasoning`
- V5: blank rows skipped
- V6: exactly 100 data rows
- V7: exactly 4 columns
- V8: candidate_id matches `^CAND_[0-9]{7}$`, no duplicates
- V9: rank is clean integer (1-100), `str(int(rank_s)) == rank_s`
- V10: ranks 1-100 each exactly once
- V11: score parses as float
- V12: scores non-increasing by rank
- V13: equal scores → candidate_id ascending (lexicographic)
- V14: NOT checked by validator (candidate existence in pool)

## Compute Constraints
- Phase A: network ON, unlimited time
- Phase B: network OFF, ≤5 min, ≤16GB RAM, CPU only
- rank.py must be at project root
- reproduce command: `python rank.py --candidates ./candidates.jsonl --out ./submission.csv`

## Scoring Weights
```
final_score = 0.65 × llm_score + 0.20 × semantic_score + 0.15 × behavioral_score
```

## File Dependency Graph
```
util/llm_client.py       -> used by analyze_jd.py, evaluate_candidates.py
util/candidate_text.py   -> used by embed_candidates.py, evaluate_candidates.py
util/behavioral.py       -> used by build_shortlist.py
util/submission.py       -> used by rank.py

pre_computation/analyze_jd.py      -> produces: jd_profile.json, jd_embedding.npy
pre_computation/embed_candidates.py -> produces: candidate_embeddings.npy, candidate_ids.txt
pre_computation/build_shortlist.py  -> reads: embeddings + ids + jd_embedding; produces: shortlist.jsonl, behavioral_scores.json
pre_computation/evaluate_candidates.py -> reads: shortlist.jsonl + jd_profile.json; produces: llm_evaluations.json
pre_computation/assemble_ranking.py -> reads: llm_evaluations.json + behavioral_scores.json + shortlist.jsonl; produces: final_ranking.json, rank_config.json
pre_computation/pipeline.py -> imports and calls all above in sequence
rank.py -> reads: final_ranking.json + rank_config.json; produces: submission.csv
```

## Groq Configuration
- API key from `.env` file (GROQ_API_KEY)
- JD analysis model: `llama-3.3-70b-versatile` (runs once, quality matters)
- Candidate eval model: `llama-3.1-8b-instant` (volume step, needs higher token budget)
- Batch size: 8 candidates per API call
- Sleep between batches: 2 seconds
- Max retries: 5

## Embedding Configuration
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Output dim: 384
- Normalize embeddings: True (L2)
- Batch size: 128 (CPU)

## Build Chunks
1. **Chunk 1:** Project utilities + scaffolding (`util/`, `scripts/`, `.gitignore`, `requirements.txt`)
2. **Chunk 2:** `pre_computation/analyze_jd.py` + `pre_computation/embed_candidates.py`
3. **Chunk 3:** `pre_computation/build_shortlist.py` + `pre_computation/evaluate_candidates.py`
4. **Chunk 4:** `pre_computation/assemble_ranking.py` + `pre_computation/pipeline.py` + `rank.py`
5. **Chunk 5:** `app.py` + `README.md` + `submission_metadata.yaml`
