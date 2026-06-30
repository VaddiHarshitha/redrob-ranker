# Redrob Hackathon — Pre-Submission Action Items

> Extracted from `data/submission_spec.docx` (Redrob Hackathon v4) and current repo state.
> Last updated: 2026-06-30

---

## 1. Output CSV format (Section 2 & 3 of spec)

### File naming
- **Rule:** filename must be your **registered participant ID** with `.csv` extension, e.g. `team_xxx.csv`.
- **Current state:** repo contains `submission.csv`.
- **Action:** rename the final file to your exact registered participant ID. Uploading `submission.csv` will be auto-rejected.

### Required format
| Field | Type | Required? | Notes |
|-------|------|-----------|-------|
| `candidate_id` | string | ✅ Yes | Must match `CAND_XXXXXXX` IDs from `candidates.jsonl` |
| `rank` | int (1–100) | ✅ Yes | Each integer 1–100 used exactly once |
| `score` | float | ✅ Yes | Non-increasing with rank (rank 1 score ≥ rank 2 score ≥ …) |
| `reasoning` | string | ⚠️ Optional but strongly recommended | 1–2 sentence justification used at Stage 4 manual review |

- **Encoding:** UTF-8.
- **Header row:** exactly `candidate_id,rank,score,reasoning` (in this order).
- **Data rows:** exactly 100 rows after the header.
- **Tie-breaking:** if two candidates have the same score, break ties deterministically (current code breaks by `candidate_id` ascending, which satisfies the spec).

### Validation command
```bash
python validate_submission.py <participant_id>.csv
```
Expected output: `Submission is valid.`

---

## 2. Portal metadata (Section 10.2 of spec)

Have the following ready before starting the upload:

| Field | Required? | Notes / Current status |
|-------|-----------|------------------------|
| Team name | ✅ Yes | Must match registered participant ID |
| Primary contact name | ✅ Yes | Currently placeholder `"Redrob Team"` — replace with real name |
| Primary contact email | ✅ Yes | Currently `team@redrob.ai` — replace with real email |
| Primary contact phone | ✅ Yes | Currently `+1-555-000-0000` — replace with real phone |
| GitHub repository URL | ✅ Yes | Currently `https://github.com/VaddiHarshitha/redrob-ranker` — verify it is public/reachable |
| Sandbox / demo link | ✅ Yes | Currently `https://redrob-airanker.streamlit.app/` — verify it loads and runs |
| AI tools declared | ✅ Yes | Multi-select honest declaration |
| Compute environment summary | ✅ Yes | e.g. "MacBook Pro M2, 16GB RAM, Python 3.11" |
| Team member list | ✅ Yes | Name + email for each member |
| Methodology summary | Optional but recommended | ≤200 words explaining the two-phase approach |

### Action items
- [ ] Update `submission_metadata.yaml` with real team name, contact details, and team members.
- [ ] Confirm `github_repo` URL is correct and reachable.
- [ ] Confirm `sandbox_link` loads and can run a small sample end-to-end.
- [ ] Ensure `submission_metadata.yaml` is at repo root and mirrors portal metadata.

---

## 3. Code repository requirements (Section 10.3 of spec)

Repo must contain:
- [ ] `README.md` with setup instructions and exact command to reproduce submission CSV.
- [ ] Full source code that produced the CSV (no hidden steps, no manual edits).
- [ ] Pre-computed artifacts the code depends on, or a script that produces them.
- [ ] `requirements.txt` (or `pyproject.toml`) with all dependencies and versions.
- [ ] `submission_metadata.yaml` at repo root.

### Reproduction command
The README must indicate a single command that produces the submission CSV:
```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

### Pre-computation note
If pre-computation is required (e.g. embeddings, LLM scores), document it clearly. Pre-computation may exceed 5 minutes, but the **ranking step** must complete within 5 minutes.

---

## 4. Sandbox / demo link (Section 10.5 of spec)

The sandbox must:
- [ ] Accept a small candidate sample (≤100 candidates) as input, or have one pre-loaded.
- [ ] Run the ranking system end-to-end and produce a ranked CSV.
- [ ] Complete within ≤5 minutes on CPU.

Acceptable platforms: Hugging Face Spaces, Streamlit Cloud, Replit, Google Colab, Docker pull/run link, Binder.

### Action items
- [ ] Verify the deployed sandbox link loads.
- [ ] Test with a small sample and confirm it produces a downloadable CSV.
- [ ] Confirm it does not time out or exceed CPU/memory limits.
- [ ] If no hosted sandbox is possible, provide a self-contained Dockerfile in the repo.

---

## 5. Compute constraints (Section 3 of spec)

| Constraint | Limit |
|------------|-------|
| Total runtime | ≤ 5 minutes wall-clock |
| Memory | ≤ 16 GB RAM |
| Compute | CPU only — no GPU during ranking |
| Network | Off — no external API calls during ranking |
| Disk | ≤ 5 GB intermediate state |

### Action items
- [ ] Confirm `rank.py` runs in under 5 minutes on a 16 GB CPU-only machine.
- [ ] Confirm no network calls happen during ranking.
- [ ] Confirm no GPU is used during ranking.
- [ ] Document these constraints in `README.md` or `submission_metadata.yaml`.

---

## 6. Reasoning quality checks (Section 3 — Stage 4)

At Stage 4, organizers sample 10 random rows and score reasoning against:
- [ ] **Specific facts:** references years of experience, current title, named skills, signal values.
- [ ] **JD connection:** connects to specific JD requirements, not generic praise.
- [ ] **Honest concerns:** acknowledges gaps where present.
- [ ] **No hallucination:** every claim appears in the candidate profile.
- [ ] **Variation:** reasonings are substantively different, not templated.
- [ ] **Rank consistency:** tone matches the rank (top ranks are strong, bottom ranks are weaker/filler).

### Penalized
- Empty reasoning
- All-identical reasoning strings
- Templated reasoning
- Hallucinated skills/employers/experience
- Reasoning that contradicts the rank

### Action items
- [ ] Sample ~10 rows from `submission.csv` and manually verify the above criteria.
- [ ] Check that reasoning strings are not all the same format.
- [ ] Verify claimed facts exist in `data/candidates.jsonl`.

---

## 7. Honeypot audit (Section 7 of spec)

- **Rule:** honeypot rate in the top 100 must be ≤10%.
- Honeypots have subtly impossible profiles (e.g. 8 years at a 3-year-old company, "expert" in 10 skills with 0 years used).

### Action items
- [ ] Inspect top 100 candidates for impossible profiles.
- [ ] If honeypots appear in top 100, adjust the ranker or manually review reasoning.
- [ ] The spec says a good system should naturally avoid them; avoid special-casing unless necessary.

---

## 8. Scoring metrics (Section 4 of spec)

Submissions are scored against hidden ground truth:

| Metric | Weight |
|--------|--------|
| NDCG@10 | 0.50 |
| NDCG@50 | 0.30 |
| MAP | 0.15 |
| P@10 | 0.05 |

Tiebreaks: higher P@5 → higher P@10 → earlier submission timestamp.

### Action items
- [ ] Optimize for NDCG@10 above all (it carries 50% weight).
- [ ] Ensure top 10 and top 50 are highest-quality candidates.
- [ ] Submit early in case of composite ties.

---

## 9. Three-submission cap (Section 3 of spec)

- At most 3 submissions during the competition window.
- Final entry = last valid submission.
- Earlier submissions are not preserved.

### Action items
- [ ] Treat the first upload as the final, validated version if possible.
- [ ] Do not submit iterative tweaks unless they are meaningful.
- [ ] Validate locally before any upload.

---

## 10. Presentation / slide deck for Stage 5 interview

The spec does **not** require a slide deck for upload, but Stage 5 is a **30-minute defend-your-work interview**. Prepare a short deck (~10 slides) covering:

1. **Problem framing** — what the JD emphasizes and why ranking matters.
2. **Approach overview** — two-phase design (pre-compute expensive features, fast CPU-only ranker).
3. **Feature engineering** — LLM scores, semantic similarity, behavioral signals.
4. **Scoring formula** — 65% LLM + 20% semantic + 15% behavioral, plus tie-breaking by `candidate_id`.
5. **Compute/latency story** — why the design fits the 5-minute CPU-only constraint.
6. **Validation** — local validator output, honeypot handling, reasoning quality.
7. **Reproducibility** — single command, sandbox link, dependencies.
8. **AI tool declaration** — how Groq and sentence-transformers were used; no LLM calls at ranking time.
9. **Demo** — screenshots or link to the working sandbox.
10. **Limitations / future work** — e.g. no explicit anti-honeypot filter, weights hand-tuned.

### Action items
- [ ] Prepare the 10-slide deck (or one-pager) for the Stage 5 interview.
- [ ] Practice walking through architecture and defending design choices.
- [ ] Be ready to demonstrate familiarity with your own code.

---

## Pre-upload checklist

- [ ] CSV renamed to registered participant ID, e.g. `<team_id>.csv`
- [ ] `python validate_submission.py <team_id>.csv` passes
- [ ] All `candidate_id` values in the CSV exist in `data/candidates.jsonl`
- [ ] Exactly 100 data rows; ranks 1–100 used exactly once
- [ ] Scores are non-increasing with rank; ties broken deterministically
- [ ] `submission_metadata.yaml` uses real names, emails, phone, repo URL, and working sandbox link
- [ ] README clearly shows the one-line reproduce command
- [ ] `rank.py` reproduces the CSV from a fresh clone in <5 minutes on CPU
- [ ] Sandbox link loads and runs with a small sample
- [ ] Honeypot scan of top 100 completed
- [ ] Reasoning strings spot-checked for hallucination and relevance
- [ ] `requirements.txt` / `requirements-freeze.txt` are current and pinned
- [ ] `artifacts/final_ranking.json` and `artifacts/rank_config.json` are committed and pushed
- [ ] 10-slide deck prepared for Stage 5 interview
