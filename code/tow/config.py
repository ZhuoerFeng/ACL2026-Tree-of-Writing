"""Configuration management for Tree-of-Writing evaluation."""

import os
from pathlib import Path
from typing import Any

import yaml


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge *override* into *base* recursively."""
    merged = dict(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_settings(extra_path: str | None = None) -> dict[str, Any]:
    """Load settings from config/settings.yaml, with env-var overrides.

    Environment variables take precedence:
      TOW_BASE_URL  -> api.base_url
      TOW_API_KEY   -> api.api_key
      TOW_MODEL     -> api.model
    """
    with open(_CONFIG_DIR / "settings.yaml", encoding="utf-8") as f:
        settings = yaml.safe_load(f)

    if extra_path:
        with open(extra_path, encoding="utf-8") as f:
            extra = yaml.safe_load(f) or {}
        settings = _deep_merge(settings, extra)

    # Environment overrides
    env_map = {
        "TOW_BASE_URL": ("api", "base_url"),
        "TOW_API_KEY": ("api", "api_key"),
        "TOW_MODEL": ("api", "model"),
    }
    for env_var, keys in env_map.items():
        val = os.environ.get(env_var)
        if val:
            node = settings
            for k in keys[:-1]:
                node = node.setdefault(k, {})
            node[keys[-1]] = val

    return settings


def load_prompts() -> dict[str, str]:
    """Load all prompt macros from config/prompts.yaml."""
    with open(_CONFIG_DIR / "prompts.yaml", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)
    return prompts


def get_data_root(settings: dict | None = None) -> Path:
    """Return the absolute path to the data directory."""
    if settings is None:
        settings = load_settings()
    rel = settings.get("data", {}).get("root", "../data")
    return (_PROJECT_ROOT / rel).resolve()


def render_prompt(template: str, **kwargs: str) -> str:
    """Render a prompt template by substituting {key} placeholders."""
    result = template
    for k, v in kwargs.items():
        result = result.replace("{" + k + "}", str(v))
    return result
