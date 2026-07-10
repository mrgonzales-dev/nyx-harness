"""Configuration loading with sensible defaults."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULTS = {
    "model": "qwen2.5-coder:1.5b",
    "system_prompt": (
        "You are Nyx — a documentation guide, not an agent.\n"
        "You do not run commands, edit files, or perform actions.\n\n"
        "Rules:\n"
        "- Be direct. No pleasantries, no recapping the question, no filler.\n"
        "- Keep responses to 3-4 sentences max. One sentence if one suffices.\n"
        "- If the user asks for detail or says 'explain more', expand freely.\n"
        "- Only show code when the user asks for it or it is essential. "
        "Prefer describing the approach in words.\n"
        "- When showing code, use fenced code blocks. No explanation unless asked.\n"
        "- If the question is vague or lacks context, ask ONE clarifying question "
        "before answering.\n"
        "- You guide. The user does the work."
    ),
    "max_history_turns": 12,
    "temperature": 0.4,
    "context_limit": 4096,
    "compact_to": 1500,
    "keep_recent_turns": 4,
    "request_timeout": 300,
}


@dataclass
class Config:
    model: str = DEFAULTS["model"]
    system_prompt: str = DEFAULTS["system_prompt"]
    max_history_turns: int = DEFAULTS["max_history_turns"]
    temperature: float = DEFAULTS["temperature"]
    context_limit: int = DEFAULTS["context_limit"]
    compact_to: int = DEFAULTS["compact_to"]
    keep_recent_turns: int = DEFAULTS["keep_recent_turns"]
    request_timeout: int = DEFAULTS["request_timeout"]
    # Mutable runtime state — not persisted to config.toml.
    system_override: str | None = field(default=None, repr=False)

    @property
    def effective_system(self) -> str:
        return self.system_override or self.system_prompt


def load_config(path: Path | None = None) -> Config:
    """Load config from *path* (defaults to ./config.toml), falling back to DEFAULTS."""
    if path is None:
        path = Path.cwd() / "config.toml"
    data: dict = {}
    if path.exists():
        with path.open("rb") as f:
            data = tomllib.load(f)
    return Config(
        model=data.get("model", DEFAULTS["model"]),
        system_prompt=data.get("system_prompt", DEFAULTS["system_prompt"]),
        max_history_turns=data.get("max_history_turns", DEFAULTS["max_history_turns"]),
        temperature=data.get("temperature", DEFAULTS["temperature"]),
        context_limit=data.get("context_limit", DEFAULTS["context_limit"]),
        compact_to=data.get("compact_to", DEFAULTS["compact_to"]),
        keep_recent_turns=data.get("keep_recent_turns", DEFAULTS["keep_recent_turns"]),
        request_timeout=data.get("request_timeout", DEFAULTS["request_timeout"]),
    )
