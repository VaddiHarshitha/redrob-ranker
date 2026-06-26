# Redrob Hackathon — Consolidated Requirements Reference

> Single source of truth synthesized from: `job_description.docx`, `submission_spec.docx`,
> `redrob_signals_doc.docx`, `candidate_schema.json`, `submission_metadata_template.yaml`,
> `sample_submission.csv`, and `validate_submission.py`.

---

## 1. Problem Statement

Build a candidate ranking system that reads a Job Description and ranks the **top 100 candidates** (out of 100,000) from a provided pool. The system must reason about *semantic fit* — not keyword matching. A candidate with the right career history but wrong buzzwords should outscore a keyword-stuffer who has never shipped production code.

| | |
|---|---|
| **Input** | `job_description.docx` + `candidates.jsonl` (100K candidates) |
| **Output** | A CSV of the top 100 candidates, ranked by fit |

---

## 2. The Job Description — What You're Ranking Against

**Role:** Senior AI Engineer — Founding Team  
**Company:** Redrob AI (Series A, AI-native talent intelligence platform)  
**Location:** Pune / Noida (Hybrid) — also accepts Hyderabad, Mumbai, Delhi NCR; outside India is case-by-case, no visa sponsorship  
**Type:** Full-time | **Experience Band:** 5–9 years (soft — outliers considered if other signals are strong)

### 2.1 Hard Requirements (Missing any → Disqualify)

| Requirement | Detail |
|---|---|
| Production embeddings experience | Sentence-transformers, OpenAI embeddings, BGE, E5, etc. — must have handled drift, index refresh, retrieval-quality regression in production |
| Production vector DB / hybrid search | Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS, or equivalent |
| Strong Python | Code quality matters |
| Ranking evaluation frameworks | NDCG, MRR, MAP, offline-to-online correlation, A/B test design — must have used these in practice |

### 2.2 Nice-to-Have Skills (Presence boosts rank; absence does not disqualify)

- LLM fine-tuning (LoRA, QLoRA, PEFT)
- Learning-to-rank models (XGBoost-based or neural)
- Prior exposure to HR-tech, recruiting, or marketplace products
- Distributed systems or large-scale inference optimization
- Open-source AI/ML contributions

### 2.3 Explicit Disqualifiers (Profile patterns to heavily penalize)

| Pattern | Reason |
|---|---|
| Pure research career (academic labs, no production deployments) | Tried twice; didn't work |
| AI experience is <12 months, primarily LangChain + hosted LLM wrappers | Needs pre-LLM era ML production depth |
| Senior engineer who hasn't written production code in 18+ months | This role writes code |
| Title-chaser (company-hopping every ~1.5 years for Senior→Staff→Principal) | Needs 3+ year commitment |
| Entire career at consulting firms only (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, etc.) | Proven bad fit (if they have prior product-company experience, they are fine) |
| Primarily CV / speech / robotics background with minimal NLP/IR exposure | Would be re-learning fundamentals |
| 5+ years exclusively on closed-source proprietary systems, no external validation | No way to assess how they think |
| "Framework enthusiast" — GitHub full of LangChain/hot-framework tutorials | Needs systems thinkers, not framework users |

### 2.4 Location & Notice Period Signals

- **Preferred locations:** Pune, Noida (office exists); also Hyderabad, Mumbai, Delhi NCR
- **Notice period ideal:** sub-30 days (company can buy out up to 30 days)  
- **30+ day notice:** still in scope but bar is higher; factor it into ranking

---

## 3. Candidate Data Schema

Each candidate record (`CAND_XXXXXXX`, 7-digit ID) has these top-level sections:

### 3.1 `profile` — Static Identity Fields

| Field | Type | Notes |
|---|---|---|
| `anonymized_name` | string | |
| `headline` | string | One-line professional headline |
| `summary` | string | Multi-sentence professional summary |
| `location` / `country` | string | City + region; country |
| `years_of_experience` | number (0–50) | |
| `current_title` / `current_company` | string | |
| `current_company_size` | enum | `1-10`, `11-50`, `51-200`, `201-500`, `501-1000`, `1001-5000`, `5001-10000`, `10001+` |
| `current_industry` | string | |

### 3.2 `career_history` — Array (1–10 entries)

Each entry: `company`, `title`, `start_date`, `end_date` (null if current), `duration_months`, `is_current` (bool), `industry`, `company_size`, `description` (role responsibilities + achievements)

### 3.3 `education` — Array (0–5 entries)

Each entry: `institution`, `degree`, `field_of_study`, `start_year`, `end_year`, `grade` (nullable), `tier` (`tier_1`–`tier_4` or `unknown`)

### 3.4 `skills` — Array

Each entry: `name`, `proficiency` (`beginner` / `intermediate` / `advanced` / `expert`), `endorsements` (int), `duration_months` (int)

> **Watch out:** Skills listed with 0 endorsements and 0 duration months are a keyword-stuffing red flag.

### 3.5 `certifications` (optional)

Each entry: `name`, `issuer`, `year`

### 3.6 `languages` (optional)

Each entry: `language`, `proficiency` (`basic` / `conversational` / `professional` / `native`)

---

## 4. Redrob Behavioral Signals (23 signals in `redrob_signals`)

These signals are often *more predictive* than the static profile. Use them as a multiplier/modifier on top of skill-match scores.

| # | Signal | Range / Type | What to Use It For |
|---|---|---|---|
| 1 | `profile_completeness_score` | 0–100 | Low completeness → reduce trust in profile |
| 2 | `signup_date` | date | Context for recency |
| 3 | `last_active_date` | date | **Key signal** — inactive for 6+ months → practically unavailable |
| 4 | `open_to_work_flag` | bool | Direct availability signal |
| 5 | `profile_views_received_30d` | int ≥ 0 | Recruiter market demand signal |
| 6 | `applications_submitted_30d` | int ≥ 0 | Active job search indicator |
| 7 | `recruiter_response_rate` | 0.0–1.0 | **Key signal** — low rate (e.g., 5%) → not reachable in practice |
| 8 | `avg_response_time_hours` | number ≥ 0 | Responsiveness; very high = red flag |
| 9 | `skill_assessment_scores` | dict (skill → 0–100) | Platform-verified skill proficiency |
| 10 | `connection_count` | int ≥ 0 | Platform engagement proxy |
| 11 | `endorsements_received` | int ≥ 0 | Social validation of skills |
| 12 | `notice_period_days` | 0–180 | Important for this JD (<30 preferred) |
| 13 | `expected_salary_range_inr_lpa` | `{min, max}` | Salary expectations in INR LPA |
| 14 | `preferred_work_mode` | `onsite/hybrid/remote/flexible` | Match against JD (hybrid role) |
| 15 | `willing_to_relocate` | bool | Relevant for Pune/Noida requirement |
| 16 | `github_activity_score` | −1 to 100 | -1 = no GitHub linked; 0–100 = activity score (last 12 months) |
| 17 | `search_appearance_30d` | int ≥ 0 | How often profile appears in recruiter searches |
| 18 | `saved_by_recruiters_30d` | int ≥ 0 | Market validation — others are interested |
| 19 | `interview_completion_rate` | 0.0–1.0 | Reliability signal; low = ghosting risk |
| 20 | `offer_acceptance_rate` | −1 to 1.0 | −1 = no offer history; 0–1 = acceptance rate |
| 21 | `verified_email` | bool | Profile authenticity |
| 22 | `verified_phone` | bool | Profile authenticity |
| 23 | `linkedin_connected` | bool | Profile authenticity / cross-verification |

---

## 5. Output File — Submission CSV

**Filename:** `{team_registered_id}.csv` | **Encoding:** UTF-8

### 5.1 Required Columns (in this order)

| Column | Type | Required | Description |
|---|---|---|---|
| `candidate_id` | string | ✅ | `CAND_XXXXXXX` from candidates.jsonl |
| `rank` | int (1–100) | ✅ | Rank position; each integer 1–100 exactly once |
| `score` | float | ✅ | Model score; must be **monotonically non-increasing** as rank increases |
| `reasoning` | string | ⚠ Strongly recommended | 1–2 sentence justification for the rank |

### 5.2 Format Rules (violation = auto-rejection)

> Confirmed directly against the `validate_submission.py` source — not just the spec doc's prose description.

- File extension must be `.csv`; filename stem must be your registered team ID (non-empty)
- UTF-8 encoding
- Header row must be **exactly** `candidate_id,rank,score,reasoning` — no reordering, no extra columns
- Exactly **100 data rows** (+ 1 header row); blank rows are silently skipped and don't count toward the 100
- `candidate_id` must match `^CAND_[0-9]{7}$`, non-empty, no duplicates
- `rank` must be a clean integer string 1–100 — **`"01"`, `"1.0"`, and `"1."` all fail validation**; each value 1–100 must appear exactly once
- `score` must parse as a float
- Scores must be **non-increasing by rank** (strict — any increase between consecutive ranks is an error)
- **Tie-break rule (precise):** when two consecutive ranks have equal scores, the `candidate_id` at the lower rank must be lexicographically **≤** the one at the higher rank (ascending order). This is enforced exactly, not left to "deterministic" interpretation.
- The validator does **not** check that `candidate_id` values actually exist in `candidates.jsonl` — that's a format-stage gap, not a guarantee. A fabricated or wrong ID passes Stage 1 but scores as irrelevant at Stage 2.

### 5.3 Sample Row Structure

```
candidate_id,rank,score,reasoning
CAND_0042871,1,0.987,"Senior AI Engineer with 7 years building RAG systems at product companies; strong recent engagement and Bangalore-based."
CAND_0019884,2,0.973,"6 years applied ML; previously shipped vector search at scale; matches the 'product over research' profile in the JD."
```

---

## 6. Compute Constraints (Hard Limits for Ranking Step)

| Constraint | Limit |
|---|---|
| Wall-clock runtime | ≤ 5 minutes |
| RAM | ≤ 16 GB |
| Compute | **CPU only — no GPU** |
| Network | **Off** — no external API calls (OpenAI, Anthropic, Cohere, Gemini, etc.) |
| Disk (intermediate state) | ≤ 5 GB |

> Pre-computation (embeddings, indexes, model training) may exceed the 5-minute window — only the *ranking step that produces the CSV* must finish within 5 minutes.

---

## 7. Submission Rules

- **3 submissions maximum** total. Your **last valid submission** is your final entry.
- **No live leaderboard** — scores revealed only when final results are announced. Validate locally.
- Submission closes at competition deadline; scoring runs once afterward.

---

## 8. Scoring Metrics

### 8.1 Composite Score Formula

```
Final = 0.50 × NDCG@10 + 0.30 × NDCG@50 + 0.15 × MAP + 0.05 × P@10
```

| Metric | Weight | What It Measures |
|---|---|---|
| NDCG@10 | 50% | Quality of your top-10 picks (highest priority) |
| NDCG@50 | 30% | Quality of your top-50 picks |
| MAP | 15% | Precision across all relevance levels |
| P@10 | 5% | Fraction of top-10 that are "relevant" (tier 3+) |

### 8.2 Tiebreakers (in order)

1. Higher P@5
2. Higher P@10
3. Earlier submission timestamp

---

## 9. Evaluation Pipeline (5 Stages)

| Stage | What Happens | Elimination Criteria |
|---|---|---|
| **1 — Format Validation** | Auto-validator on every submission | Any spec violation in Section 5 |
| **2 — Scoring** | Composite computed once against hidden ground truth | Score below advancement cutoff |
| **3 — Code Reproduction + Honeypot Check** | Full repo requested; ranking step reproduced in sandboxed Docker (5 min, 16 GB, no GPU, no network); honeypot rate computed | Cannot reproduce; honeypot rate >10% in top 100; missing/fabricated repo |
| **4 — Manual Review** | Reasoning quality (6 checks), methodology coherence, Git history authenticity, code quality | Failed reasoning checks; flat Git history; codebase is just LLM API calls |
| **5 — Defend-Your-Work Interview** | 30-min video call: walk through architecture, defend choices, show familiarity with own code | Cannot explain architecture; contradicts code; clearly didn't build it |

### 9.1 Reasoning Quality Checks (Stage 4 — 10 rows sampled)

| Check | What Reviewers Look For |
|---|---|
| Specific facts | References actual profile data (years, title, skill names, signal values) |
| JD connection | Links to specific JD requirements, not generic praise |
| Honest concerns | Acknowledges candidate gaps where they exist |
| No hallucination | Every claim corresponds to something actually in the profile |
| Variation | 10 sampled reasonings are substantively different (not templated) |
| Rank consistency | Tone matches rank position (no glowing reasoning for rank-95 candidates) |

**Penalized:** Empty reasoning, identical strings, templated fill-in-the-name reasoning, hallucinated skills/employers, reasoning contradicting the rank.

---

## 10. Honeypot Awareness

The dataset contains ~80 **honeypot candidates** with subtly impossible profiles (e.g., 8 years at a company founded 3 years ago; "expert" in 10 skills with 0 duration months). They are forced to **relevance tier 0** in ground truth.

- **Threshold:** Honeypot rate >10% in your top 100 → **disqualified at Stage 3**
- **Detection:** A system that reads profiles semantically will naturally avoid them. No special-casing needed. Treat very high endorsements for a skill with 0 duration, date inconsistencies, and title-vs-summary mismatches as red flags.

---

## 11. What to Submit (Full Picture)

Three deliverables, all required:

### 11.1 CSV File
Top-100 ranking as specified in Sections 5–6.

### 11.2 Portal Metadata (entered at upload)

| Field | Required |
|---|---|
| Team name | ✅ |
| Primary contact: name, email, phone | ✅ |
| GitHub repository URL | ✅ (reachable; private OK if you grant organizer access at Stage 3) |
| Sandbox / demo link | ✅ (see Section 11.4) |
| AI tools declared (multi-select) | ✅ (transparent, not penalized) |
| Compute environment summary (one line) | ✅ |
| Team member list (name + email) | ✅ |
| Methodology summary (≤200 words) | Optional but strongly recommended |

### 11.3 Code Repository (GitHub)

Must include:
- `README.md` — setup instructions + **single reproduce command** (e.g., `python rank.py --candidates ./candidates.jsonl --out ./submission.csv`)
- Full source code that produced the CSV (no hidden steps, no manual edits post-run)
- Pre-computed artifacts (embeddings, indexes, model weights), or a script to generate them
- `requirements.txt` / `pyproject.toml` — all dependencies with versions
- `submission_metadata.yaml` at repo root (fill in the provided template)

### 11.4 Sandbox / Demo Link

A hosted environment where the ranker can be run on a small candidate sample (≤100 candidates). Acceptable platforms:

- HuggingFace Spaces, Streamlit Cloud, Replit, Google Colab, Docker (public registry), Binder

Requirements for the sandbox:
- Accept ≤100 candidates as input (upload or pre-loaded)
- Run ranking end-to-end and produce a ranked CSV
- Complete within 5-minute CPU budget

---

## 12. Key Design Guidance (From JD Hackathon Note)

> These are direct signals from the organizers about what will score well:

1. **Don't keyword-match.** Reason about what the JD *means*, not just the words it uses.
2. **Career history > skills list.** A candidate who built a recommendation system at a product company ranks above one with all the AI keywords but a "Marketing Manager" title.
3. **Behavioral signals are a multiplier.** A great-on-paper candidate who is inactive (6+ months), has low recruiter response rate, and has not completed assessments is *not actually hirable*. Down-weight them.
4. **The consulting firm signal is decisive.** Candidates whose *entire* career is at TCS, Infosys, Wipro, Accenture, etc. are an explicit disqualifier per the JD. Apply it.
5. **NDCG@10 is 50% of your score.** Getting the top 10 right matters most. Be precise at the top.
