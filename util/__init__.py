"""
Utility package for the Redrob Candidate Ranker.

Provides shared, stateless helper modules used across both Phase A
(pre-computation) and Phase B (ranking):

    util.llm_client    — Groq LLM client builder and JSON-fence parser.
    util.candidate_text — Candidate record → embedding text conversion.
    util.behavioral    — Data-driven behavioural signal scoring.
    util.submission    — CSV formatter with V1-V14 validator assertions.

All utilities are pure functions with no side-effects; they accept data as
arguments and return results.  I/O (file reads/writes, API calls) is
handled by the caller.
"""

from util import llm_client
from util import candidate_text
from util import behavioral
from util import submission

__all__ = ["llm_client", "candidate_text", "behavioral", "submission"]