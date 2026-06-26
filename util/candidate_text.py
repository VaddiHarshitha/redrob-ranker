"""
Convert a candidate record into a dense text string for embedding.

The generated text is used as input to the ``all-MiniLM-L6-v2`` sentence-
transformer model.  The text prioritises career history as the primary
signal, uses headline and summary for context, and includes skills only
when they have meaningful evidence (endorsements or duration).

No keyword matching or JD-specific logic lives here — this is a generic,
data-driven representation.
"""

from __future__ import annotations

from typing import Any


def build_candidate_text(candidate: dict[str, Any]) -> str:
    """
    Build a dense text representation of a candidate.

    The text is structured as follows (sections separated by newlines):

    1. Headline and summary (context)
    2. Career history — each role as "Title at Company: description"
       (primary signal, included in full)
    3. Skills that have endorsements > 0 OR duration_months > 0,
       formatted as "Skill (proficiency, N endorsements, M months)"

    Parameters
    ----------
    candidate
        A single candidate record conforming to the schema described in
        the build spec (top-level keys: profile, career_history, skills,
        redrob_signals, ...).

    Returns
    -------
    str
        A single text string ready for embedding.
    """
    parts: list[str] = []

    # ── Profile context ────────────────────────────────────────────────────
    profile = candidate.get("profile") or {}
    headline = profile.get("headline") or ""
    summary = profile.get("summary") or ""

    if headline:
        parts.append(f"Headline: {headline}")
    if summary:
        parts.append(f"Summary: {summary}")

    # ── Career history (primary signal) ────────────────────────────────────
    career_history = candidate.get("career_history") or []
    if career_history:
        parts.append("Career History:")
        for job in career_history:
            title = job.get("title") or "Unknown Title"
            company = job.get("company") or "Unknown Company"
            description = (job.get("description") or "").strip()
            entry = f"- {title} at {company}"
            if description:
                entry += f": {description}"
            parts.append(entry)

    # ── Skills (only evidenced ones) ───────────────────────────────────────
    skills = candidate.get("skills") or []
    evidenced_skills: list[str] = []
    for skill in skills:
        endorsements = skill.get("endorsements") or 0
        duration = skill.get("duration_months") or 0
        if endorsements > 0 or duration > 0:
            name = skill.get("name") or "Unknown Skill"
            proficiency = skill.get("proficiency") or "unknown"
            endorsements_label = (
                f"{endorsements} endorsement{'s' if endorsements != 1 else ''}"
            )
            duration_label = (
                f"{duration} month{'s' if duration != 1 else ''}"
                if duration > 0
                else "no duration recorded"
            )
            evidenced_skills.append(
                f"{name} ({proficiency}, {endorsements_label}, {duration_label})"
            )

    if evidenced_skills:
        parts.append("Skills:")
        for s in evidenced_skills:
            parts.append(f"- {s}")

    return "\n".join(parts)