"""
Centralised configuration constants for the pre-computation pipeline.

Single source of truth for all shared constants used by the pipeline step
modules in ``pre_computation/*.py``. Each step imports the constants it needs
from this module instead of duplicating literal values.

Grouped sections:
- Embedding configuration (model name, dim, batch size)
- LLM configuration (Groq model + max tokens)
- Pipeline I/O (artefacts directory)
- Artefact filenames (per-step outputs)
- Shortlist / ranking sizing (top-N values)
"""

# ---------------------------------------------------------------------------
# Embedding configuration
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIM = 768
EMBEDDING_BATCH_SIZE = 128

# ---------------------------------------------------------------------------
# LLM configuration
# ---------------------------------------------------------------------------

GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_MAX_TOKENS = 4096

# ---------------------------------------------------------------------------
# Pipeline I/O
# ---------------------------------------------------------------------------

ARTIFACTS_DIR = "artifacts"

# ---------------------------------------------------------------------------
# Artefact filenames
# ---------------------------------------------------------------------------

JD_PROFILE_FILE = "jd_profile.json"
JD_EMBEDDING_FILE = "jd_embedding.npy"
CANDIDATE_EMBEDDINGS_FILE = "candidate_embeddings.npy"
CANDIDATE_IDS_FILE = "candidate_ids.txt"
SHORTLIST_FILE = "shortlist.jsonl"
BEHAVIORAL_SCORES_FILE = "behavioral_scores.json"
LLM_EVALUATIONS_FILE = "llm_evaluations.json"
FINAL_RANKING_FILE = "final_ranking.json"
RANK_CONFIG_FILE = "rank_config.json"

# ---------------------------------------------------------------------------
# Shortlist / ranking sizing
# ---------------------------------------------------------------------------

SHORTLIST_SIZE = 300
FINAL_TOP_N = 300
