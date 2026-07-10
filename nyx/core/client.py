"""Thin wrapper over the Ollama Python SDK with streaming support."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import httpx
import ollama

from nyx.core.config import Config


def _close_stream(stream) -> None:
    """Close an ollama SDK stream iterator if it supports closing."""
    close = getattr(stream, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


@dataclass
class ChatResult:
    """Metadata from a completed chat call."""
    prompt_tokens: int = 0   # tokens in the input messages
    response_tokens: int = 0  # tokens in the model's response


@dataclass
class StreamToken:
    """A token from the stream, either thinking or content."""
    kind: Literal["thinking", "content"]
    text: str


class OllamaClient:
    """Minimal client: chat with streaming, list available models."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._client = ollama.Client(
            timeout=httpx.Timeout(
                connect=5,
                read=config.request_timeout,
                write=10,
                pool=5,
            ),
        )
        self._num_ctx: int | None = None

    def _chat_options(self, **extra) -> dict:
        """Build options dict with num_ctx if known."""
        opts = dict(extra)
        if self._num_ctx:
            opts["num_ctx"] = self._num_ctx
        return opts

    def set_context_length(self, ctx: int | None) -> None:
        """Set the num_ctx to pass to Ollama for chat calls."""
        self._num_ctx = ctx

    def stream_chat(self, messages: list[dict]) -> tuple[Iterator[str], ChatResult]:
        """Return (token_iterator, result_holder).

        The iterator yields content tokens as they arrive. After the iterator
        is exhausted, *result_holder* is populated with token counts from the
        final chunk.
        """
        result = ChatResult()
        stream = self._client.chat(
            model=self.config.model,
            messages=messages,
            stream=True,
            options=self._chat_options(temperature=self.config.temperature),
        )

        def _gen() -> Iterator[str]:
            try:
                for chunk in stream:
                    piece = chunk.get("message", {}).get("content")
                    if piece:
                        yield piece
                    # Capture token counts from the final chunk.
                    if chunk.get("done"):
                        result.prompt_tokens = chunk.get("prompt_eval_count", 0) or 0
                        result.response_tokens = chunk.get("eval_count", 0) or 0
            finally:
                _close_stream(stream)

        return _gen(), result

    def stream_chat_with_thinking(
        self, messages: list[dict]
    ) -> tuple[Iterator[StreamToken], ChatResult]:
        """Return (token_iterator, result_holder) with thinking support.

        The iterator yields StreamToken objects that indicate whether the
        token is thinking or content. This handles models that output
        thinking via the 'thinking' field (like lfm2.5-thinking).
        """
        result = ChatResult()
        stream = self._client.chat(
            model=self.config.model,
            messages=messages,
            stream=True,
            options=self._chat_options(temperature=self.config.temperature),
        )

        def _gen() -> Iterator[StreamToken]:
            try:
                for chunk in stream:
                    msg = chunk.get("message", {})

                    # Check for thinking content.
                    thinking = msg.get("thinking")
                    if thinking:
                        yield StreamToken(kind="thinking", text=thinking)

                    # Check for regular content.
                    content = msg.get("content")
                    if content:
                        yield StreamToken(kind="content", text=content)

                    # Capture token counts from the final chunk.
                    if chunk.get("done"):
                        result.prompt_tokens = chunk.get("prompt_eval_count", 0) or 0
                        result.response_tokens = chunk.get("eval_count", 0) or 0
            finally:
                _close_stream(stream)

        return _gen(), result

    def summarize(self, messages: list[dict]) -> str:
        """Non-streaming call to summarize old conversation turns."""
        resp = self._client.chat(
            model=self.config.model,
            messages=messages,
            stream=False,
            options=self._chat_options(temperature=0.3),
        )
        return resp.get("message", {}).get("content", "") if isinstance(resp, dict) else resp.message.content

    def list_models(self) -> list[str]:
        """Return names of locally available models."""
        resp = self._client.list()
        models = resp.get("models", []) if isinstance(resp, dict) else resp.models
        return [m["model"] if isinstance(m, dict) else m.model for m in models]

    def get_context_length(self, model: str | None = None) -> int | None:
        """Return the model's native context window size, or None if unknown."""
        model = model or self.config.model
        try:
            info = self._client.show(model)
            mi = info.modelinfo
            for key, val in mi.items():
                if "context_length" in key.lower():
                    return int(val)
        except Exception:
            return None
        return None
