"""
Data-driven behavioral scorer using redrob_signals.

Each signal is converted to a multiplicative factor in the range (0, 1].
All factors are multiplied together, then the product is clamped to [0, 1].

Signal rules (applied in order):
  last_active_date          → <45 d  : ×1.00
                              45-90  : ×0.85
                              90-180 : ×0.65
                              >180   : ×0.30
  open_to_work_flag         → false  : ×0.75
  recruiter_response_rate   → <0.20  : ×0.50
                              <0.40  : ×0.80
  interview_completion_rate → <0.50  : ×0.80
  notice_period_days        → ≤30    : ×1.00
                              30-45  : ×0.90
                              45-60  : ×0.80
                              >60    : ×0.65
  profile_completeness_score→ 0.70 + 0.30 × (score / 100)
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any


def compute_behavioral_score(candidate: dict[str, Any]) -> float:
    """
    Compute the behavioral score for a candidate.

    Parameters
    ----------
    candidate
        A single candidate record. Must contain a ``redrob_signals`` key
        with the sub-fields documented in the build spec.

    Returns
    -------
    float
        A score in the interval [0.0, 1.0].
    """
    signals: dict[str, Any] = candidate.get("redrob_signals") or {}

    def get(key: str, default: Any = None) -> Any:
        return signals.get(key, default)

    # ── last_active_date ──────────────────────────────────────────────────
    last_active_raw = get("last_active_date")
    last_active_multiplier = 1.0
    if last_active_raw:
        try:
            last_active = (
                date.fromisoformat(last_active_raw)
                if isinstance(last_active_raw, str)
                else last_active_raw
            )
            reference = date(2026, 6, 24)
            days_inactive = (reference - last_active).days
            if days_inactive > 180:
                last_active_multiplier = 0.30
            elif days_inactive > 90:
                last_active_multiplier = 0.65
            elif days_inactive > 45:
                last_active_multiplier = 0.85
        except (ValueError, TypeError):
            pass

    # ── open_to_work_flag ─────────────────────────────────────────────────
    open_to_work = get("open_to_work_flag")
    open_to_work_multiplier = 1.0 if open_to_work else 0.75

    # ── recruiter_response_rate ───────────────────────────────────────────
    response_rate = get("recruiter_response_rate")
    if response_rate is None or response_rate < 0.20:
        response_multiplier = 0.50
    elif response_rate < 0.40:
        response_multiplier = 0.80
    else:
        response_multiplier = 1.0

    # ── interview_completion_rate ─────────────────────────────────────────
    interview_rate = get("interview_completion_rate")
    interview_multiplier = 1.0 if (interview_rate is not None and interview_rate >= 0.50) else 0.80

    # ── notice_period_days ────────────────────────────────────────────────
    notice_days = get("notice_period_days")
    notice_multiplier = 1.0
    if notice_days is not None:
        if notice_days > 60:
            notice_multiplier = 0.65
        elif notice_days > 45:
            notice_multiplier = 0.80
        elif notice_days > 30:
            notice_multiplier = 0.90

    # ── profile_completeness_score ────────────────────────────────────────
    completeness = get("profile_completeness_score")
    if completeness is not None:
        completeness_multiplier = 0.70 + 0.30 * (completeness / 100.0)
    else:
        completeness_multiplier = 0.70

    # ── Multiply all factors ───────────────────────────────────────────────
    product = (
        last_active_multiplier
        * open_to_work_multiplier
        * response_multiplier
        * interview_multiplier
        * notice_multiplier
        * completeness_multiplier
    )

    return float(min(max(product, 0.0), 1.0))