"""OpenAI-compatible API client wrapper."""

import re
from typing import Any

from openai import OpenAI

from tow.config import load_settings


_client: OpenAI | None = None
_settings: dict[str, Any] | None = None


def get_client(settings: dict | None = None) -> OpenAI:
    """Get or create a singleton OpenAI client."""
    global _client, _settings
    if settings is None:
        settings = load_settings()
    if _client is None or _settings is not settings:
        api_cfg = settings["api"]
        _client = OpenAI(
            base_url=api_cfg["base_url"],
            api_key=api_cfg["api_key"],
            timeout=api_cfg.get("timeout", 120),
        )
        _settings = settings
    return _client


def chat_completion(prompt: str, settings: dict | None = None) -> str:
    """Send a single user message and return the assistant's reply."""
    if settings is None:
        settings = load_settings()
    client = get_client(settings)
    api_cfg = settings["api"]

    response = client.chat.completions.create(
        model=api_cfg["model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=api_cfg.get("temperature", 0.0),
        max_tokens=api_cfg.get("max_tokens", 4096),
    )
    return response.choices[0].message.content or ""


def extract_score(text: str) -> int | None:
    """Extract a score from [[X]] pattern in LLM output."""
    match = re.search(r"\[\[(\d+)\]\]", text)
    if match:
        return int(match.group(1))
    return None
