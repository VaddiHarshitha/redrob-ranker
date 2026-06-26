"""
Groq LLM client builder and JSON response parser.

Usage
-----
    from util.llm_client import build_groq_llm, parse_json_response, DEFAULT_GROQ_MODEL

    llm = build_groq_llm(model=DEFAULT_GROQ_MODEL, max_tokens=2048)
    raw = llm.invoke("...")
    data = parse_json_response(raw.content)
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

# ---------------------------------------------------------------------------
# Client builder
# ---------------------------------------------------------------------------

def build_groq_llm(
    model: str = DEFAULT_GROQ_MODEL,
    max_tokens: int = 2048,
    temperature: float = 0.0,
) -> ChatGroq:
    """
    Build and return a ChatGroq client.

    Loads ``GROQ_API_KEY`` from a ``.env`` file in the project root.
    Raises a clear error if the key is missing or empty.

    Parameters
    ----------
    model
        Groq model identifier (e.g. ``"llama-3.3-70b-versatile"``).
    max_tokens
        Maximum number of tokens the model may generate.
    temperature
        Sampling temperature; 0.0 (default) gives deterministic output.

    Returns
    -------
    ChatGroq
        Configured LangChain Groq chat model instance.
    """
    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set in the .env file. "
            "Create a .env file in the project root with: GROQ_API_KEY=your_key"
        )
    return ChatGroq(
        model=model,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
    )


# ---------------------------------------------------------------------------
# JSON parser
# ---------------------------------------------------------------------------

_RETRY_AFTER_RE = re.compile(r"(?:try again in|retry after)\s*([\d.]+)\s*s?", re.IGNORECASE)


def parse_json_response(raw: str | Any) -> dict[str, Any]:
    """
    Parse a JSON object from an LLM text response.

    Handles common LLM output patterns:
    - Markdown fenced blocks: `````json ... ``` ``
    - Bare JSON object without fences

    Parameters
    ----------
    raw
        The raw string response from the LLM. Will be coerced to str.

    Returns
    -------
    dict
        Parsed JSON object.

    Raises
    ------
    ValueError
        If the content cannot be parsed as JSON after stripping fences.
    """
    text = str(raw).strip()

    # Strip markdown code fences (with optional language tag)
    if text.startswith("```"):
        # Find first opening ``` and skip to after the line
        first_fence = text.find("```")
        first_nl = text.find("\n", first_fence)
        if first_nl != -1:
            text = text[first_nl + 1:]
        else:
            text = text[first_fence + 3:]
        # Remove trailing closing ```
        last_fence = text.rfind("```")
        if last_fence != -1:
            text = text[:last_fence]

    text = text.strip()

    # Try to extract JSON substring if there's surrounding prose
    first_open = min(text.find("["), text.find("{"))
    last_close = max(text.rfind("]"), text.rfind("}"))
    if first_open != -1 and last_close != -1 and last_close >= first_open:
        extracted = text[first_open:last_close + 1]
    else:
        extracted = text

    try:
        return json.loads(extracted)
    except json.JSONDecodeError:
        # Fall back to original stripped text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try partial recovery: find largest balanced JSON prefix
            first_open = text.find("{")
            if first_open != -1:
                # Try progressively longer prefixes from first { to last }
                last_brace = text.rfind("}")
                if last_brace >= first_open:
                    for end in range(last_brace, first_open - 1, -1):
                        try:
                            return json.loads(text[first_open:end + 1])
                        except json.JSONDecodeError:
                            continue
            raise ValueError(
                f"Failed to parse JSON response after partial recovery attempts.\n"
                f"Response text (first 500 chars): {text[:500]}"
            )