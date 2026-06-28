"""
Orchestrate Steps 1–5 of the pre-computation pipeline in order.

Usage
-----
    python -m pre_computation.pipeline                    # run all steps
    python -m pre_computation.pipeline --from 3           # resume from step 3
    python -m pre_computation.pipeline --only 4           # run just step 4
    python -m pre_computation.pipeline --force            # re-run everything
    python -m pre_computation.pipeline --jd path/to/docx  # specify JD path
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, Any

# Import steps — each must expose a `run(artifacts_dir=..., **kwargs)` function
from pre_computation import analyze_jd
from pre_computation import embed_candidates
from pre_computation import build_shortlist
from pre_computation import evaluate_candidates
from pre_computation import assemble_ranking

from pre_computation.config import (
    ARTIFACTS_DIR,
    BEHAVIORAL_SCORES_FILE,
    CANDIDATE_EMBEDDINGS_FILE,
    CANDIDATE_IDS_FILE,
    FINAL_RANKING_FILE,
    JD_EMBEDDING_FILE,
    JD_PROFILE_FILE,
    LLM_EVALUATIONS_FILE,
    RANK_CONFIG_FILE,
    SHORTLIST_FILE,
)


# ---------------------------------------------------------------------------
# Pipeline definition
# ---------------------------------------------------------------------------

PIPELINE = [
    (
        1,
        analyze_jd.run,
        "JD analysis (Groq LLM)",
        [JD_PROFILE_FILE, JD_EMBEDDING_FILE],
    ),
    (
        2,
        embed_candidates.run,
        "Embed all 100K candidates",
        [CANDIDATE_EMBEDDINGS_FILE, CANDIDATE_IDS_FILE],
    ),
    (
        3,
        build_shortlist.run,
        "Shortlist top N candidates",
        [SHORTLIST_FILE, BEHAVIORAL_SCORES_FILE],
    ),
    (
        4,
        evaluate_candidates.run,
        "LLM evaluate shortlist",
        [LLM_EVALUATIONS_FILE],
    ),
    (
        5,
        assemble_ranking.run,
        "Assemble final ranking",
        [FINAL_RANKING_FILE, RANK_CONFIG_FILE],
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def artifacts_exist(artifacts_dir: str, filenames: list[str]) -> bool:
    """Return True only if every filename in ``filenames`` exists in artifacts_dir."""
    artifacts_path = Path(artifacts_dir)
    return all((artifacts_path / fn).exists() for fn in filenames)


def run_step(
    step_num: int,
    step_fn: Callable[..., Any],
    label: str,
    artifacts_dir: str,
    output_files: list[str],
    force: bool,
    jd_path: str | None,
    total_steps: int,
) -> bool:
    """
    Run a single pipeline step, optionally skipping if outputs exist.

    Parameters
    ----------
    step_num
        1-based step number.
    step_fn
        The callable to invoke for this step.
    label
        Human-readable step label.
    artifacts_dir
        Path to the artifacts directory.
    output_files
        Files this step is expected to produce.
    force
        If True, skip artifact-existence check and always run.
    jd_path
        Optional path to the job description file.
    total_steps
        Total number of steps in the pipeline (for banner display).

    Returns
    -------
    bool
        True if the step ran (or was skipped), False on error.
    """
    banner = f"STEP {step_num}/{total_steps} — {label}"
    sep = "=" * len(banner)
    print(f"\n{sep}")
    print(banner)
    print(sep)

    if not force and artifacts_exist(artifacts_dir, output_files):
        print(f"  [skip] All output files already exist — skipping.")
        print(f"  [skip] Use --force to re-run this step.")
        return True

    try:
        start = time.time()

        # Build kwargs per step
        kwargs: dict[str, Any] = {"artifacts_dir": artifacts_dir}
        if step_num == 1 and jd_path is not None:
            kwargs["jd_path"] = jd_path

        step_fn(**kwargs)

        elapsed = time.time() - start
        print(f"\n  [done] {label} completed in {elapsed:.1f}s")

        return True

    except Exception as exc:
        print(f"\n  [ERROR] {label} failed: {exc}")
        print(f"\n  Resume with: python -m pre_computation.pipeline --from {step_num}")
        raise  # Re-raise so Python shows the full traceback


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m pre_computation.pipeline",
        description="Run the pre-computation pipeline (Steps 1–5).",
    )
    parser.add_argument(
        "--jd",
        type=str,
        default="job_description.docx",
        help="Path to the job description file (default: job_description.docx).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run all steps even if output artifacts already exist.",
    )
    parser.add_argument(
        "--from",
        dest="from_step",
        type=int,
        default=None,
        help="Resume pipeline from the given step number (1–5).",
    )
    parser.add_argument(
        "--only",
        type=int,
        default=None,
        help="Run only the specified step number (1–5).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    artifacts_dir = ARTIFACTS_DIR
    total_steps = len(PIPELINE)

    # Determine step range
    if args.only is not None:
        start_step = args.only
        end_step = args.only
    elif args.from_step is not None:
        start_step = args.from_step
        end_step = total_steps
    else:
        start_step = 1
        end_step = total_steps

    print(f"=== Redrob Pre-Computation Pipeline ===")
    print(f"  artifacts directory : {artifacts_dir}")
    print(f"  job description     : {args.jd}")
    print(f"  force               : {args.force}")
    print(f"  steps               : {start_step}–{end_step} of {total_steps}")

    for step_num, step_fn, label, output_files in PIPELINE:
        if step_num < start_step or step_num > end_step:
            continue

        ok = run_step(
            step_num=step_num,
            step_fn=step_fn,
            label=label,
            artifacts_dir=artifacts_dir,
            output_files=output_files,
            force=args.force,
            jd_path=args.jd if step_num == 1 else None,
            total_steps=total_steps,
        )

        if not ok:
            sys.exit(1)

    print(f"\n=== Pipeline complete ===")


if __name__ == "__main__":
    main()