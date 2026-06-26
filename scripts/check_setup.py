#!/usr/bin/env python3
"""
One-off environment verification script.

Checks:
  1. sentence-transformers model downloads and produces (1, 384) shape output.
  2. Groq API is reachable via a lightweight "Reply with: OK" probe.

Prints a green checkmark (PASS) or red X (FAIL) for each check.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path so 'util' and 'pre_computation' packages resolve
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def check_embedding_model() -> bool:
    """Verify sentence-transformers model and shape."""
    try:
        from sentence_transformers import SentenceTransformer

        print("Loading sentence-transformers model (all-MiniLM-L6-v2) ...", end=" ", flush=True)
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embedding = model.encode(["Test sentence for shape verification."])
        shape = embedding.shape

        expected = (1, 384)
        if shape == expected:
            print(f"PASS  shape={shape}")
            return True
        else:
            print(f"FAIL  shape={shape} (expected {expected})")
            return False
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL  {type(exc).__name__}: {exc}")
        return False


def check_groq_api() -> bool:
    """Verify Groq API key is present and the API is reachable."""
    try:
        from dotenv import load_dotenv
        import os

        load_dotenv()
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key:
            print("GROQ_API_KEY not found in .env file — SKIPPED")
            return False

        from util.llm_client import build_groq_llm

        print("Connecting to Groq API ...", end=" ", flush=True)
        llm = build_groq_llm(model="llama-3.1-8b-instant", max_tokens=20)
        response = llm.invoke("Reply with exactly: OK")
        content = str(response.content).strip()

        if content == "OK":
            print("PASS  API reachable, received 'OK'")
            return True
        else:
            print(f"FAIL  expected 'OK', got {content!r}")
            return False
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL  {type(exc).__name__}: {exc}")
        return False


def main() -> None:
    print("=" * 60)
    print("Redrob Candidate Ranker — Environment Setup Check")
    print("=" * 60)
    print()

    results: list[tuple[str, bool]] = []

    # Check 1 — sentence-transformers
    print("[1/2] Sentence-Transformer embedding model:")
    results.append(("Embedding model", check_embedding_model()))
    print()

    # Check 2 — Groq API
    print("[2/2] Groq API connectivity:")
    results.append(("Groq API", check_groq_api()))
    print()

    # Summary
    print("=" * 60)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print(f"Result: {passed}/{total} checks passed")

    if passed == total:
        print("All checks PASSED — environment is ready.")
        sys.exit(0)
    else:
        print("Some checks FAILED — review the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()