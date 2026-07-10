"""Conversation state: message history with token tracking and auto-compaction."""

from __future__ import annotations

from nyx.core.client import OllamaClient
from nyx.core.config import Config

# Rough heuristic: ~3.5 chars per token for English text + code.
_CHARS_PER_TOKEN = 3.5

SUMMARY_PROMPT = (
    "Summarize the following conversation in 200 words or less. "
    "Preserve key facts, decisions, code references, and context "
    "needed to continue the conversation. Be concise."
)


def estimate_tokens(text: str) -> int:
    """Quick heuristic token estimate (no network call)."""
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def estimate_messages_tokens(messages: list[dict]) -> int:
    """Estimate total tokens for a message list (includes role overhead)."""
    total = 0
    for m in messages:
        total += estimate_tokens(m.get("content", "")) + 4  # ~4 tokens for role tags
    return total


class Conversation:
    """Holds the running message list sent to the model.

    Tracks token usage and auto-compacts when the context limit is exceeded.
    Compaction summarizes old turns into a single message, preserving recent
    turns verbatim.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.messages: list[dict] = [
            {"role": "system", "content": config.effective_system}
        ]
        # Exact token count from the last model response.
        self.last_prompt_tokens: int = 0
        # Heuristic estimate for pre-send display.
        self.estimated_tokens: int = estimate_messages_tokens(self.messages)
        # Whether the last for_request() call triggered a compaction.
        self.compacted_last: bool = False

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": self.config.effective_system}]
        self.last_prompt_tokens = 0
        self.estimated_tokens = estimate_messages_tokens(self.messages)
        self.compacted_last = False

    def set_system(self, text: str) -> None:
        self.config.system_override = text
        if self.messages and self.messages[0]["role"] == "system":
            self.messages[0]["content"] = text
        else:
            self.messages.insert(0, {"role": "system", "content": text})
        self.estimated_tokens = estimate_messages_tokens(self.messages)

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})
        self.estimated_tokens += estimate_tokens(content) + 4

    def add_assistant(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})
        self.estimated_tokens += estimate_tokens(content) + 4

    def update_exact_tokens(self, prompt_tokens: int) -> None:
        """Replace the heuristic estimate with the exact count from the model."""
        self.last_prompt_tokens = prompt_tokens
        self.estimated_tokens = prompt_tokens

    def add_doc_context(self, content: str) -> None:
        """Append a documentation context system message and refresh estimates."""
        self.messages.append({"role": "system", "content": content})
        self.estimated_tokens = estimate_messages_tokens(self.messages)
        self.last_prompt_tokens = 0  # force re-estimation on next send

    def remove_doc_context(self) -> None:
        """Remove all doc-context system messages (prefix ``[docs context:``)."""
        self.messages = [
            m for m in self.messages
            if not (
                m["role"] == "system"
                and isinstance(m.get("content"), str)
                and m["content"].startswith("[docs context:")
            )
        ]
        self.estimated_tokens = estimate_messages_tokens(self.messages)
        self.last_prompt_tokens = 0  # force re-estimation on next send

    @property
    def token_count(self) -> int:
        """Best available token count (exact if we have one, else estimate).

        Returns the *higher* of the exact count and the heuristic estimate so
        that newly added messages (which only bump the estimate) are never
        invisible to the context-limit check.
        """
        return max(self.last_prompt_tokens, self.estimated_tokens)

    @property
    def context_usage(self) -> float:
        """Fraction of context limit used (0.0 to 1.0+)."""
        return self.token_count / self.config.context_limit if self.config.context_limit else 0.0

    @property
    def turn_count(self) -> int:
        return sum(1 for m in self.messages if m["role"] == "user")

    def _split_for_compaction(self) -> tuple[list[dict], list[dict]]:
        """Split messages into (old_turns, recent_turns) based on token budget.

        System messages are not included in either — they're always kept
        separately. Recent turns are kept up to compact_to tokens; everything
        older becomes old_turns for summarization.
        """
        system_msgs = [m for m in self.messages if m["role"] == "system"]
        turns = [m for m in self.messages if m["role"] != "system"]

        if not turns:
            return [], []

        # Walk backwards, accumulating recent turns until we hit compact_to.
        budget = self.config.compact_to
        recent_tokens = 0
        split_idx = len(turns)
        for i in range(len(turns) - 1, -1, -1):
            msg_tokens = estimate_tokens(turns[i].get("content", "")) + 4
            if recent_tokens + msg_tokens > budget:
                break
            recent_tokens += msg_tokens
            split_idx = i

        # Keep at least 1 recent turn.
        if split_idx >= len(turns):
            split_idx = len(turns) - 1

        # Also enforce keep_recent_turns as a minimum floor (user + assistant
        # each count as one message, so multiply by 2).
        min_recent = self.config.keep_recent_turns * 2
        if len(turns) - split_idx < min_recent:
            split_idx = max(0, len(turns) - min_recent)

        old_turns = turns[:split_idx]
        recent_turns = turns[split_idx:]
        return old_turns, recent_turns

    def compact(self, client: OllamaClient) -> bool:
        """Summarize old turns and replace them with a compact summary.

        Returns True if compaction happened, False if there was nothing to compact.
        """
        old_turns, recent_turns = self._split_for_compaction()
        if not old_turns:
            return False

        # Build a summary request: system instruction + old conversation.
        summary_messages = [
            {"role": "system", "content": SUMMARY_PROMPT},
            {"role": "user", "content": self._turns_to_text(old_turns)},
        ]
        summary = client.summarize(summary_messages)

        # Rebuild messages: all original system messages + summary + recent turns.
        system_msgs = [m for m in self.messages if m["role"] == "system"]
        if not system_msgs:
            system_msgs = [{"role": "system", "content": self.config.effective_system}]
        summary_msg = {
            "role": "system",
            "content": f"Summary of earlier conversation:\n{summary}",
        }
        self.messages = system_msgs + [summary_msg] + recent_turns
        self.estimated_tokens = estimate_messages_tokens(self.messages)
        self.last_prompt_tokens = 0  # Reset exact count; will update after next call.
        self.compacted_last = True
        return True

    @staticmethod
    def _turns_to_text(turns: list[dict]) -> str:
        """Render conversation turns as plain text for summarization."""
        lines = []
        for m in turns:
            role = m["role"].upper()
            lines.append(f"{role}: {m['content']}")
        return "\n\n".join(lines)

    def for_request(self, client: OllamaClient | None = None, context_limit: int | None = None) -> list[dict]:
        """Return messages ready to send, auto-compacting if needed.

        If *client* is provided and token count exceeds *context_limit*
        (or config.context_limit if not given), compaction runs before
        returning. If no client, falls back to simple turn-count trimming.
        """
        self.compacted_last = False
        limit = context_limit or self.config.context_limit
        if client and self.token_count >= limit:
            self.compact(client)

        # Fallback: simple trim if still over max_history_turns.
        max_msgs = self.config.max_history_turns * 2
        non_system = [m for m in self.messages if m["role"] != "system"]
        if len(non_system) > max_msgs:
            system_msgs = [m for m in self.messages if m["role"] == "system"]
            self.messages = system_msgs + non_system[-(max_msgs):]
            self.estimated_tokens = estimate_messages_tokens(self.messages)

        return self.messages
