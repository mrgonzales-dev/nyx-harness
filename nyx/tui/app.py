"""Nyx Textual chat app — full-feature TUI.

Streaming chat with local Ollama models, slash commands,
conversation compaction, and autocomplete.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import httpx
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, VerticalScroll
from textual.screen import ModalScreen
from textual.suggester import Suggester
from textual.theme import Theme
from textual.widgets import Input as _Input, ListView, ListItem, Static

from nyx.core.chat import Conversation, estimate_messages_tokens, estimate_tokens
from nyx.core.client import OllamaClient
from nyx.core.config import load_config
from nyx.tui.config_modal import ConfigModal
from nyx.tui.doc_browser import DocBrowser
from nyx.tui.markdown_renderer import render_markdown
from nyx.tools import docs as docset_manager


# ── command registry ────────────────────────────────────────
# (name, description, method_suffix)
COMMANDS: list[tuple[str, str, str]] = [
    ("model",       "switch model",                 "cmd_model"),
    ("models",      "list available models",         "cmd_models"),
    ("system",      "set system prompt",             "cmd_system"),
    ("code",        "toggle code-only mode",         "cmd_code"),
    ("docs",        "browse/install docsets",        "cmd_docs"),
    ("followup",    "re-inject last doc and ask",    "cmd_followup"),
    ("clear",       "clear conversation history",    "cmd_clear"),
    ("compact",     "manually compact conversation", "cmd_compact"),
    ("context",     "show context usage breakdown",  "cmd_context"),
    ("config",      "adjust settings",               "cmd_config"),
    ("status",      "show current config",           "cmd_status"),
    ("help",        "show available commands",       "cmd_help"),
    ("quit",        "exit (or Ctrl+D)",              "cmd_quit"),
]

# Models that use thinking/CoT output.
THINKING_KWS = frozenset({"thinking", "deepseek-r1", "qwq", "reasoner"})

# System prompt appended when /code mode is active.
CODE_MODE_PROMPT = (
    "\n\n[code mode]\n"
    "Output ONLY code. No explanations, no commentary, no markdown wrapping "
    "around the code. Use markdown code fences (```) to delimit code blocks. "
    "If there are multiple blocks, output each in its own fence."
)


class ModelSwitchModal(ModalScreen[str]):
    """Modal screen for switching models."""

    CSS = """
    ModelSwitchModal {
        align: center middle;
        background: $background 60%;
    }

    ModelSwitchModal > Container {
        width: 50;
        max-height: 18;
        background: $surface;
        border: solid cyan;
        padding: 1 2;
    }

    ModelSwitchModal > Container > #modal-header {
        color: cyan;
        text-style: bold;
        text-align: center;
        width: 100%;
        margin-bottom: 1;
    }

    ModelSwitchModal > Container > #current-model {
        color: #888888;
        margin-bottom: 1;
    }

    ModelSwitchModal > Container > ListView {
        height: auto;
        max-height: 12;
    }

    ModelSwitchModal > Container > ListItem {
        padding: 0 1;
    }
    """

    def __init__(self, models: list[str], current: str) -> None:
        super().__init__()
        self._models = [m for m in models if m != current]
        self._current = current

    def compose(self) -> ComposeResult:
        with Container():
            yield Static("Nyx's Brain", id="modal-header")
            yield Static(f"current: {self._current}", id="current-model")
            items = []
            for m in self._models:
                li = ListItem(Static(m))
                li.mark = m
                items.append(li)
            yield ListView(*items, id="model-list")

    def on_mount(self) -> None:
        self.query_one("#model-list", ListView).focus()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        model_name = getattr(event.item, "mark", None)
        if model_name:
            self.dismiss(model_name)

    def key_escape(self) -> None:
        self.dismiss(None)


class NyxSuggester(Suggester):
    """Ghost-text autocomplete for commands and model names."""

    def __init__(
        self,
        get_models_fn,
        *,
        case_sensitive: bool = False,
    ) -> None:
        super().__init__(case_sensitive=case_sensitive)
        self._get_models = get_models_fn

    async def get_suggestion(self, value: str) -> str | None:
        if not value.startswith("/"):
            return None

        parts = value[1:].split(" ", 1)
        cmd = parts[0]
        arg = parts[1] if len(parts) > 1 else ""

        # Still typing the command name.
        if len(parts) == 1:
            for c, _desc, _suffix in COMMANDS:
                if c.startswith(cmd) and c != cmd:
                    return "/" + c
            return None

        # Model name completion for /model.
        if cmd == "model" and arg:
            models = self._get_models()
            for m in models:
                if m.lower().startswith(arg.lower()):
                    return value[: len(value) - len(arg)] + m
            return None

        # Docset slug completion for /docs (but not for subcommands).
        if cmd == "docs" and arg and not arg.startswith(("install ", "uninstall ", "available ", "list", "done", "cat ")):
            from nyx.tools import docs as _dm
            for d in _dm.list_installed():
                if d.slug.startswith(arg):
                    return value[: len(value) - len(arg)] + d.slug
            return None

        return None


_PASTE_MARKER = re.compile(r"\[paste: (\d+) lines\]")


class NyxInput(_Input):
    """Input that collapses multi-line pastes into inline markers.

    Single-line paste → normal behaviour.
    Multi-line paste  → stores full text, inserts ``[paste: N lines]`` at cursor.
    On submit, markers are expanded back to the original content.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._paste_chunks: list[str] = []

    # ── public API ────────────────────────────────────────────

    def expand_pastes(self) -> str:
        """Replace every ``[paste: N lines]`` marker with stored content.

        Returns the expanded string.  Markers without a matching chunk
        are left as-is; orphaned chunks (marker deleted by user) are dropped.
        """
        value = self.value
        chunks = list(self._paste_chunks)
        idx = 0

        def _repl(m: re.Match) -> str:
            nonlocal idx
            if idx < len(chunks):
                chunk = chunks[idx]
                idx += 1
                return chunk
            return m.group(0)

        return _PASTE_MARKER.sub(_repl, value)

    def clear_pastes(self) -> None:
        self._paste_chunks.clear()

    @property
    def paste_count(self) -> int:
        return len(self._paste_chunks)

    @property
    def paste_line_count(self) -> int:
        return sum(c.count("\n") + 1 for c in self._paste_chunks)

    # ── internals ─────────────────────────────────────────────

    def _on_paste(self, event) -> None:
        text = event.text
        if not text or "\n" not in text:
            # Single-line — defer to parent.
            super()._on_paste(event)
            return

        # Multi-line — collapse to marker.
        event.stop()
        event.prevent_default()

        # Strip trailing newlines so a trailing \n doesn't inflate the count.
        clean = text.rstrip("\n")
        n_lines = clean.count("\n") + 1
        marker = f"[paste: {n_lines} lines]"
        self._paste_chunks.append(clean)

        selection = self.selection
        if selection.is_empty:
            self.insert_text_at_cursor(marker)
        else:
            start, end = selection
            self.replace(marker, start, end)

        self._notify_indicator()

    def clear(self) -> None:
        """Override clear to also wipe paste chunks."""
        super().clear()
        self.clear_pastes()
        self._notify_indicator()

    def _notify_indicator(self) -> None:
        """Tell the app to refresh the paste indicator."""
        app = self.app
        if isinstance(app, NyxApp):
            app._update_paste_indicator()

    def _on_key(self, event) -> None:
        """Intercept up/down/enter/escape for the doc popup."""
        app = self.app
        if isinstance(app, NyxApp):
            popup = app.query_one("#doc-popup", ListView)
            if popup.has_class("visible"):
                if event.key == "down":
                    popup.action_cursor_down()
                    event.stop()
                    event.prevent_default()
                    return
                elif event.key == "up":
                    popup.action_cursor_up()
                    event.stop()
                    event.prevent_default()
                    return
                elif event.key == "escape":
                    popup.remove_class("visible")
                    event.stop()
                    event.prevent_default()
                    return
                elif event.key == "enter":
                    app._select_doc_slug(popup)
                    event.stop()
                    event.prevent_default()
                    return
        super()._on_key(event)


class NyxApp(App):
    """Chat with a local Ollama model through a Textual TUI."""

    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [
        Binding("ctrl+p", "open_model_modal", "Switch model", show=False),
        Binding("ctrl+c", "copy_selected", show=False, priority=True),
        Binding("ctrl+q", "quit", "Quit", show=False, priority=True),
        Binding("escape", "interrupt", "Interrupt", show=True),
    ]

    CSS = """
    Screen {
        layout: vertical;
        background: #000000;
    }

    #chat-history {
        height: 1fr;
        overflow-y: auto;
        overflow-x: hidden;
        scrollbar-size: 0 0;
        padding: 0 1;
    }

    #logo-container {
        height: 1fr;
        align: center middle;
        layout: vertical;
    }

    #logo-text {
        color: cyan;
        text-style: bold;
        width: 100%;
        text-align: center;
    }

    #logo-subtext {
        color: #666666;
        width: 100%;
        text-align: center;
        margin-top: 1;
    }

    #chat-input {
        width: 100%;
        margin: 1 1 2 1;
        height: 3;
        padding: 1 1;
        border: none;
        border-left: solid cyan;
    }

    #chat-input:focus {
        border: none;
        border-left: solid cyan;
        outline: none;
    }

    #paste-indicator {
        width: 100%;
        padding: 0 1;
        height: 1;
        color: #4a9999;
        text-style: italic;
        display: none;
    }

    ToastRack {
        margin-bottom: 7;
    }

    #status-bar {
        width: 100%;
        padding: 0 1;
        height: 1;
        background: #0a0a0a;
    }

    #status-bar > .rich-text {
        color: #555555;
    }

    #doc-popup {
        max-height: 8;
        height: 0;
        border: solid cyan;
        background: #0a0a0a;
        margin: 0 1;
        display: none;
    }

    #doc-popup.visible {
        height: auto;
        display: block;
    }

    #doc-popup > ListItem {
        padding: 0 1;
    }

    #doc-popup > ListItem > Widget:hover {
        background: #1a1a2a;
    }

    .user-message {
        border-left: solid cyan;
        padding: 0 1;
        margin: 1 0;
        height: auto;
        background: $surface;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_theme(Theme(
            name="nyx-dark",
            primary="#0178D4",
            secondary="#004578",
            accent="#00cccc",
            foreground="#cccccc",
            background="#000000",
            surface="#0a0a0a",
            dark=True,
        ))
        self.theme = "nyx-dark"
        self.config = load_config()
        self.client = OllamaClient(self.config)
        self.convo = Conversation(self.config)
        self._stream_widget: Static | None = None
        self._stream_gen: int = 0          # increments on each new stream worker
        self._active_gen: int | None = None  # generation of the currently active stream
        self._compacting: bool = False      # True while /compact worker is running
        self._thinking = False
        self._auto_scroll = True

        # Code mode — strips response to code blocks only.
        self._code_mode: bool = False
        self._original_temperature: float = self.config.temperature
        self._code_system_saved: str | None = None

        # Fetch the model's actual context window from Ollama.
        self._model_ctx: int | None = None
        self._refresh_model_ctx()

        # Doc popup — skip one changed event after selection.
        self._doc_popup_skip = False

        # Doc context — single doc at a time.
        # Doc stays in context for one question, then is auto-removed.
        # /followup re-injects it. /docs done wipes it entirely.
        self._doc_slug: str | None = None
        self._doc_entry_name: str | None = None
        self._doc_full_markdown: str = ""
        self._doc_full_tokens: int = 0
        self._doc_in_context: bool = False  # is the doc currently in the convo?
        self._doc_pending_removal: bool = False  # remove doc after this response?

        # Suggester callbacks
        self._model_cache: list[str] | None = None
        self._spinner_idx = 0
        self._spinner_chars = "◐◓◑◒"

    def _get_models(self) -> list[str]:
        if self._model_cache is None:
            try:
                self._model_cache = self.client.list_models()
            except Exception:
                self._model_cache = []
        return self._model_cache

    def _refresh_models(self) -> None:
        self._model_cache = None
        self._refresh_model_ctx()

    def _refresh_model_ctx(self) -> None:
        """Fetch the current model's context window from Ollama."""
        try:
            self._model_ctx = self.client.get_context_length()
            self.client.set_context_length(self._model_ctx)
        except Exception:
            self._model_ctx = None
            self.client.set_context_length(None)

    @property
    def _effective_context_limit(self) -> int:
        """The real context limit: model's native window, or config fallback."""
        return self._model_ctx or self.config.context_limit

    @property
    def _is_streaming(self) -> bool:
        """True when a stream worker is active."""
        return self._active_gen is not None

    def action_open_model_modal(self) -> None:
        """Ctrl+P — open the model switch modal."""
        if self._is_streaming:
            return
        models = self._get_models()
        self.push_screen(ModelSwitchModal(models, self.config.model), self._on_model_selected)

    def action_copy_selected(self) -> None:
        """Ctrl+C — copy dragged selection, or toast if nothing selected."""
        selection = self.screen.get_selected_text()
        if selection is not None and selection.strip():
            self.copy_to_clipboard(selection)
            self.notify("Copied to clipboard", timeout=1)
        else:
            # No selection — check if Input has its own selection
            focused = self.focused
            if focused and hasattr(focused, 'selected_text'):
                inp_text = focused.selected_text
                if inp_text:
                    self.copy_to_clipboard(inp_text)
                    self.notify("Copied to clipboard", timeout=1)
                    return
            self.notify("Nothing selected — press Ctrl+Q to quit", timeout=2)

    def action_interrupt(self) -> None:
        """Esc — interrupt streaming, or clear selection if active."""
        # Close doc popup first if open.
        popup = self.query_one("#doc-popup", ListView)
        if popup.has_class("visible"):
            popup.remove_class("visible")
            return
        if self.screen.get_selected_text():
            self.screen.clear_selection()
            return
        if self._is_streaming:
            self._active_gen = None  # invalidate the active worker
            self._auto_scroll = False
            if self._stream_widget is not None:
                self._stream_widget = None
            self.notify("Interrupted", timeout=1)
            self._update_status()
            self.query_one("#chat-input", NyxInput).focus()

    # ── compose / lifecycle ──────────────────────────────────

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="chat-history")
        with Container(id="logo-container"):
            yield Static(id="logo-text")
            yield Static(id="logo-subtext")
        yield ListView(id="doc-popup")
        yield NyxInput(
            id="chat-input",
            placeholder="",
            suggester=NyxSuggester(self._get_models),
        )
        yield Static(id="paste-indicator")
        yield Static(id="status-bar")

    def on_mount(self) -> None:
        """Set window title, load logo, focus the input."""
        self.title = f"nyx — {self.config.model}"
        # Load ASCII logo.
        logo_path = Path(__file__).parent.parent / "assets" / "logo.txt"
        if logo_path.exists():
            logo_text = logo_path.read_text(encoding="utf-8")
            self.query_one("#logo-text", Static).update(
                Text(logo_text, style="bold cyan", justify="center"),
            )
            self.query_one("#logo-subtext", Static).update(
                Text("Nope not your agent, YOU CODE it yourself, use me as your guide.", style="dim", justify="center"),
            )
        self._update_status()
        self.query_one("#chat-input", NyxInput).focus()
        self.query_one("#chat-history", VerticalScroll).can_focus = False
        self.set_interval(0.12, self._tick_spinner)
        # Pre-populate model cache in a worker so the suggester doesn't block.
        self.run_worker(self._warm_model_cache(), name="warm-models")

    async def _warm_model_cache(self) -> None:
        """Fetch model list in the background to avoid suggester blocking."""
        try:
            self._model_cache = await asyncio.to_thread(self.client.list_models)
        except Exception:
            self._model_cache = []

    # ── input handling ───────────────────────────────────────

    def on_mouse_scroll_up(self, event) -> None:
        """User scrolled up — interrupt auto-scroll."""
        self._auto_scroll = False

    def on_key(self, event) -> None:
        """Up/PageUp interrupts auto-scroll when streaming."""
        if event.key in ("up", "pageup") and self._is_streaming:
            self._auto_scroll = False

    def on_input_changed(self, event: NyxInput.Changed) -> None:
        """Watch input for @-triggered doc slug popup."""
        if self._doc_popup_skip:
            self._doc_popup_skip = False
            return
        self._check_doc_popup(event.value)

    def _check_doc_popup(self, value: str) -> None:
        """Show/hide the doc slug popup based on input content."""
        popup = self.query_one("#doc-popup", ListView)

        if not value.startswith("/docs "):
            popup.remove_class("visible")
            return

        # Find the last @ — the popup triggers on @<partial> at the end.
        at_idx = value.rfind("@")
        if at_idx == -1:
            popup.remove_class("visible")
            return

        after_at = value[at_idx + 1:]
        if " " in after_at:
            popup.remove_class("visible")
            return

        # @ must be right after /docs, /docs install, or /docs uninstall.
        before_at = value[:at_idx]
        if not before_at.endswith(("/docs ", "/docs install ", "/docs uninstall ")):
            popup.remove_class("visible")
            return

        # Populate with matching slugs.
        from nyx.tools import docs as _dm
        slugs = [d.slug for d in _dm.list_installed() if d.slug.startswith(after_at)]

        popup.clear()
        if not slugs:
            popup.remove_class("visible")
            return

        for slug in slugs:
            li = ListItem(Static(slug))
            li.slug = slug  # type: ignore[attr-defined]
            popup.append(li)
        popup.index = 0
        popup.add_class("visible")

    def _select_doc_slug(self, popup: ListView) -> None:
        """Insert the selected slug into the input, replacing @<partial>."""
        if popup.index is None:
            return
        items = list(popup.query(ListItem))
        if popup.index >= len(items):
            return
        item = items[popup.index]
        slug = getattr(item, "slug", None)
        if not slug:
            return

        inp = self.query_one("#chat-input", NyxInput)
        value = inp.value
        at_idx = value.rfind("@")
        if at_idx == -1:
            return

        new_value = value[:at_idx] + "@" + slug
        self._doc_popup_skip = True
        inp.value = new_value
        inp.cursor_position = len(new_value)
        popup.remove_class("visible")

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle click selection in the doc popup."""
        if event.list_view.id == "doc-popup":
            self._select_doc_slug(event.list_view)

    def on_input_submitted(self, event: NyxInput.Submitted) -> None:
        if self._is_streaming:
            return

        # Expand paste markers to full multi-line content.
        text = event.input.expand_pastes().strip()
        if not text:
            return
        event.input.clear()

        # First message hides the logo.
        logo = self.query_one("#logo-container")
        if logo.display is not False:
            logo.display = False
            self.query_one("#chat-history").display = True

        # Slash command?
        if text.startswith("/"):
            self._dispatch_command(text)
            return

        # ── normal chat turn ──
        self._add_user_message(text)
        self.convo.add_user(text)

        # If doc is in context, mark it for removal after the response.
        if self._doc_in_context:
            self._doc_pending_removal = True

        # Start streaming.
        self.run_worker(self._stream(), name="stream")

    def _dispatch_command(self, text: str) -> None:
        cmd_name, _, arg = text[1:].partition(" ")
        arg = arg.strip()

        # Find the matching command suffix.
        suffix = None
        for c, _desc, s in COMMANDS:
            if c == cmd_name:
                suffix = s
                break

        if suffix is None:
            self._add_system(f"unknown command: /{cmd_name}  (try /help)")
            return

        handler = getattr(self, suffix, None)
        if handler:
            handler(arg)

    # ── command handlers ─────────────────────────────────────

    def cmd_help(self, _arg: str) -> None:
        table = Table(
            title="available commands",
            show_header=False,
            border_style="cyan",
            padding=(0, 2),
        )
        table.add_column("command", style="cyan", no_wrap=True)
        table.add_column("description", style="dim")
        for c, desc, _s in COMMANDS:
            table.add_row(f"/{c}", desc)
        self._add_system(table)

    def cmd_model(self, arg: str) -> None:
        if not arg:
            models = self._get_models()
            self.push_screen(ModelSwitchModal(models, self.config.model), self._on_model_selected)
            return
        self.config.model = arg
        self._refresh_models()
        self.title = f"nyx — {self.config.model}"
        self._update_status()
        self._add_system(f"model → {arg}")

    def _on_model_selected(self, model: str | None) -> None:
        """Callback when a model is selected from the modal."""
        if model is None:
            return
        self.config.model = model
        self._refresh_models()
        self.title = f"nyx — {self.config.model}"
        self._update_status()

        # Check if conversation fits the new model's context window.
        new_limit = self._effective_context_limit
        total = sum(estimate_messages_tokens([m]) for m in self.convo.messages)
        if total > new_limit:
            # Try auto-compacting conversation turns.
            sys_tokens = sum(
                estimate_messages_tokens([m])
                for m in self.convo.messages
                if m["role"] == "system"
            )
            if sys_tokens > new_limit * 0.8:
                # System messages alone exceed the limit (e.g. doc context).
                self._add_system(Text(
                    f"model → {model}  ({new_limit} ctx)\n"
                    f"context is {total} tokens — system messages alone are {sys_tokens}.\n"
                    f"use /docs done or /clear before chatting.",
                    style="yellow",
                ))
            else:
                self._add_system(Text(
                    f"model → {model}  ({new_limit} ctx)\n"
                    f"context is {total} tokens — auto-compacting...",
                    style="yellow",
                ))
                self.run_worker(self._auto_compact_for_model(model), name="auto-compact")
        else:
            self._add_system(f"model → {model}")

        self.query_one("#chat-input", NyxInput).focus()

    async def _auto_compact_for_model(self, model: str) -> None:
        """Auto-compact conversation after switching to a smaller model."""
        try:
            ok = await asyncio.to_thread(self.convo.compact, self.client)
            if ok:
                remaining = sum(estimate_messages_tokens([m]) for m in self.convo.messages)
                limit = self._effective_context_limit
                if remaining > limit:
                    self._add_system(Text(
                        f"still over limit after compaction ({remaining}/{limit})\n"
                        f"use /docs done or /clear to free more context.",
                        style="yellow",
                    ))
                else:
                    self._add_system(Text(
                        f"compacted — {self.convo.turn_count} turns, "
                        f"~{remaining} tokens",
                        style="cyan",
                    ))
            else:
                remaining = sum(estimate_messages_tokens([m]) for m in self.convo.messages)
                self._add_system(Text(
                    f"nothing to compact ({remaining} tokens, limit {self._effective_context_limit})\n"
                    f"use /docs done or /clear to free context.",
                    style="yellow",
                ))
        except Exception as e:
            self._add_system(Text(f"auto-compaction failed: {e}", style="bold red"))
        self._update_status()

    def cmd_models(self, _arg: str) -> None:
        self.run_worker(self._fetch_and_show_models(), name="fetch-models")

    async def _fetch_and_show_models(self) -> None:
        self._add_system(Text("listing models...", style="dim"))
        try:
            models = await asyncio.to_thread(self.client.list_models)
            self._model_cache = models
            self._show_models_table(models, self.config.model)
        except Exception as e:
            self._add_system(Text(f"could not list models: {e}", style="bold red"))

    def _show_models_table(self, models: list[str], current: str) -> None:
        table = Table(show_header=False, border_style="cyan", padding=(0, 2))
        table.add_column("model", style="cyan", no_wrap=True)
        table.add_column("status", style="dim")
        for m in models:
            marker = "active" if m == current else ""
            table.add_row(m, marker)
        self._add_system(table)

    def cmd_system(self, arg: str) -> None:
        if not arg:
            self._add_system("usage: /system <prompt text>")
            return
        self.convo.set_system(arg)
        self._code_system_saved = arg if self._code_mode else self._code_system_saved
        if self._code_mode:
            self._apply_code_system()
        self._add_system("system prompt updated")

    def _code_on(self) -> None:
        """Activate code mode."""
        self._code_mode = True
        # Save state for clean restore.
        self._code_system_saved = self.config.effective_system
        self._original_temperature = self.config.temperature
        # Apply code-mode system prompt.
        self._apply_code_system()
        # Force deterministic output.
        self.config.temperature = 0.0
        self.notify("code mode ON — temperature set to 0.0", timeout=2)
        self._update_status()

    def _code_off(self) -> None:
        """Deactivate code mode."""
        self._code_mode = False
        # Restore original system prompt.
        if self._code_system_saved is not None:
            self.convo.set_system(self._code_system_saved)
            self._code_system_saved = None
        # Restore original temperature.
        self.config.temperature = self._original_temperature
        self.notify("code mode OFF", timeout=2)
        self._update_status()

    def _apply_code_system(self) -> None:
        """Layer the code-mode instruction on top of the saved system prompt."""
        base = self._code_system_saved or self.config.effective_system
        self.convo.set_system(base + CODE_MODE_PROMPT)

    def cmd_code(self, arg: str) -> None:
        """Toggle code-only mode."""
        if self._is_streaming:
            return
        if self._code_mode:
            self._code_off()
        else:
            self._code_on()

    def cmd_docs(self, arg: str) -> None:
        """Browse and manage docsets."""
        if self._is_streaming:
            return

        parts = arg.split(None, 1)
        sub = parts[0] if parts else ""
        rest = parts[1].strip() if len(parts) > 1 else ""

        if sub == "install" and rest:
            self.run_worker(self._docs_install(rest.lstrip("@")), name="docs-install")
        elif sub == "uninstall" and rest:
            ok, msg = docset_manager.uninstall(rest.lstrip("@"))
            style = "cyan" if ok else "bold red"
            self._add_system(Text(msg, style=style))
        elif sub == "list":
            self._docs_list()
        elif sub == "available":
            self.run_worker(self._docs_available(rest), name="docs-available")
        elif sub == "done":
            self._docs_done()
        elif sub == "cat":
            self._docs_cat()
        elif sub and not rest:
            # /docs <slug> — open the browser for an installed docset.
            slug = sub.lstrip("@")
            if not docset_manager.is_installed(slug):
                self._add_system(
                    Text(
                        f"'{slug}' is not installed — try /docs install {slug}",
                        style="bold red",
                    )
                )
                return
            self.run_worker(self._docs_open(slug), name="docs-open")
        else:
            self._add_system(
                "usage:\n"
                "  /docs <slug>            browse an installed docset\n"
                "  /docs install <name>    download a docset\n"
                "  /docs uninstall <name>  remove a docset\n"
                "  /docs list              show installed docsets\n"
                "  /docs available [query] show available docsets\n"
                "  /docs cat               print the current doc page in chat\n"
                "  /docs done              wipe doc from memory (no /followup after)\n"
                "  /followup <question>    re-inject last doc and ask a question"
            )

    def cmd_followup(self, arg: str) -> None:
        """Re-inject the last doc and ask a question about it."""
        if self._is_streaming:
            return

        if self._doc_slug is None or not self._doc_full_markdown:
            self._add_system(
                Text("no doc in memory — browse a docset with /docs <slug>", style="bold red")
            )
            return

        question = arg.strip()
        if not question:
            self._add_system(Text("usage: /followup <question>", style="dim"))
            return

        # Re-inject the doc into context.
        self._remove_doc_from_convo()
        self.convo.add_doc_context(
            f"[docs context: {self._doc_entry_name}]\n{self._doc_full_markdown}"
        )
        self._doc_in_context = True

        # Show the question in chat like a normal message.
        self._add_user_message(question)
        self.convo.add_user(question)

        # Mark for removal after response.
        self._doc_pending_removal = True
        self._update_status()

        # Start streaming.
        self.run_worker(self._stream(), name="stream")

    async def _docs_install(self, slug: str) -> None:
        self._add_system(Text(f"downloading {slug}...", style="dim"))
        try:
            ok, msg = await asyncio.to_thread(docset_manager.install, slug)
        except Exception as e:
            ok, msg = False, f"install error: {e}"
        style = "cyan" if ok else "bold red"
        self._add_system(Text(msg, style=style))

    def _docs_list(self) -> None:
        installed = docset_manager.list_installed()
        if not installed:
            self._add_system(
                "no docsets installed — try /docs available to browse, "
                "then /docs install <name>"
            )
            return
        table = Table(show_header=True, border_style="cyan", padding=(0, 2))
        table.add_column("name", style="cyan", no_wrap=True)
        table.add_column("slug", style="dim")
        table.add_column("version", style="dim")
        table.add_column("entries", justify="right")
        table.add_column("size", justify="right", style="dim")
        for d in installed:
            size_mb = d.db_size / 1024 / 1024
            table.add_row(d.name, d.slug, d.version, str(d.entry_count), f"{size_mb:.1f}MB")
        self._add_system(table)

    async def _docs_available(self, query: str) -> None:
        self._add_system(Text("fetching catalog...", style="dim"))
        try:
            available = await asyncio.to_thread(docset_manager.list_available, query)
        except Exception as e:
            self._add_system(Text(f"could not fetch catalog: {e}", style="bold red"))
            return
        if not available:
            self._add_system("no available docsets found")
            return
        table = Table(show_header=True, border_style="cyan", padding=(0, 2))
        table.add_column("slug", style="cyan", no_wrap=True)
        table.add_column("name", style="dim")
        table.add_column("version", style="dim")
        for d in available[:30]:
            table.add_row(
                d["slug"],
                d.get("name", ""),
                str(d.get("version", "")),
            )
        if len(available) > 30:
            self._add_system(table)
            self._add_system(Text(f"...and {len(available) - 30} more", style="dim"))
        else:
            self._add_system(table)

    async def _docs_open(self, slug: str) -> None:
        """Load docset from cache and open the browser modal."""
        self._add_system(Text(f"loading {slug}...", style="dim"))
        try:
            result = await asyncio.to_thread(docset_manager.open_docset, slug)
        except Exception as e:
            self._add_system(Text(f"could not load docset: {e}", style="bold red"))
            return
        if result is None:
            self._add_system(
                Text(f"'{slug}' could not be loaded", style="bold red")
            )
            return
        index, db = result
        self.push_screen(
            DocBrowser(slug, index, db),
            self._on_doc_page_selected,
        )

    def _on_doc_page_selected(self, result) -> None:
        """Callback when user feeds selected doc text to the AI."""
        if result is None:
            # User closed without feeding.
            self.query_one("#chat-input", _Input).focus()
            return

        slug, entry_name, selected_text = result

        # Remove any existing doc context first (single doc at a time).
        self._remove_doc_from_convo()

        # Store the selected text for /followup re-injection.
        self._doc_slug = slug
        self._doc_entry_name = entry_name
        self._doc_full_markdown = selected_text
        self._doc_full_tokens = estimate_tokens(selected_text)
        self._doc_in_context = True

        # Inject into conversation as a system message.
        self.convo.add_doc_context(
            f"[docs context: {entry_name}]\n{selected_text}"
        )

        # Show in chat.
        self._add_system(
            Text(
                f"fed to AI: {entry_name} ({self._doc_full_tokens} tokens) — ask a question "
                f"(doc auto-removes after answer, /followup to query again)",
                style="cyan",
            )
        )
        self._update_status()
        self.query_one("#chat-input", _Input).focus()

    def _docs_cat(self) -> None:
        """Print the current doc page in the chat."""
        if self._doc_slug is None:
            self._add_system("no doc page read yet — browse a docset with /docs <slug>")
            return

        chat = self.query_one("#chat-history", VerticalScroll)
        width = chat.content_size.width

        label = self._doc_entry_name or ""
        if not self._doc_in_context:
            label += " (not in context — use /followup)"
        self._add_system(Text(f"── {label} ({self._doc_slug}) ──", style="cyan"))
        self._add_message(render_markdown(self._doc_full_markdown, max(width, 40)))

    def _docs_done(self) -> None:
        """Wipe doc context entirely — no /followup after this."""
        if self._doc_slug is None:
            self._add_system("no doc context active")
            return
        self._wipe_doc()
        self._add_system("doc context wiped — /followup will not work until you read a new doc")
        self._update_status()

    def _remove_doc_from_convo(self) -> None:
        """Remove the doc system message from conversation, keep metadata for /followup."""
        self.convo.remove_doc_context()
        self._doc_in_context = False

    def _wipe_doc(self) -> None:
        """Wipe all doc metadata — /followup won't work after this."""
        self._remove_doc_from_convo()
        self._doc_slug = None
        self._doc_entry_name = None
        self._doc_full_markdown = ""
        self._doc_full_tokens = 0

    def cmd_clear(self, _arg: str) -> None:
        self.convo.reset()
        self._wipe_doc()
        self._update_status()
        self._add_system("conversation cleared")

    def cmd_compact(self, _arg: str) -> None:
        self.run_worker(self._run_compact(), name="compact")

    async def _run_compact(self) -> None:
        self._compacting = True
        self._add_system(Text("compacting...", style="dim"))
        try:
            ok = await asyncio.to_thread(self.convo.compact, self.client)
            if ok:
                self._add_system(
                    f"compacted — {self.convo.turn_count} turns, "
                    f"~{self.convo.estimated_tokens} tokens",
                )
            else:
                self._add_system("nothing to compact (context is small enough)")
        except httpx.TimeoutException:
            self._add_system(Text("compaction timed out", style="bold red"))
        except Exception as e:
            self._add_system(Text(f"compaction failed: {e}", style="bold red"))
        finally:
            self._compacting = False
        self._update_status()

    def cmd_context(self, _arg: str) -> None:
        system_tokens = sum(
            estimate_messages_tokens([m])
            for m in self.convo.messages
            if m["role"] == "system"
        )
        convo_tokens = sum(
            estimate_messages_tokens([m])
            for m in self.convo.messages
            if m["role"] != "system"
        )
        total = self.convo.token_count
        limit = self._effective_context_limit
        pct = (total / limit * 100) if limit else 0

        lines = [
            f"system messages:  ~{system_tokens} tokens",
            f"conversation:     ~{convo_tokens} tokens",
            f"total:           ~{total} / {limit} tokens ({pct:.0f}%)",
            f"model ctx window: {limit} tokens" + (" (from Ollama)" if self._model_ctx else " (config fallback)"),
            f"compaction at:   {limit} tokens",
            f"target after:    {self.config.compact_to} tokens",
            f"keep recent:     {self.config.keep_recent_turns} turns",
        ]
        self._add_system(Panel(Text("\n".join(lines)), title="context usage", border_style="cyan"))

    def cmd_config(self, _arg: str) -> None:
        """Open the config modal to adjust settings."""
        if self._is_streaming:
            return
        self.push_screen(ConfigModal(self.config.temperature), self._on_config_saved)

    def _on_config_saved(self, result) -> None:
        """Callback when config modal is saved."""
        if result is None:
            self.query_one("#chat-input", _Input).focus()
            return
        changed = []
        if "temperature" in result:
            self.config.temperature = result["temperature"]
            changed.append(f"temperature → {result['temperature']:.1f}")
        if changed:
            self._add_system(Text("  ·  ".join(changed), style="cyan"))
        self._update_status()
        self.query_one("#chat-input", _Input).focus()

    def cmd_status(self, _arg: str) -> None:
        limit = self._effective_context_limit
        desc = f"  model: {self.config.model}  |  turns: {self.convo.turn_count}  |  ctx: {self.convo.token_count}/{limit}"
        system = self.config.effective_system
        if len(system) > 80:
            system = system[:80] + "..."
        self._add_system(Panel(Text(f"{desc}\n  system: {system}"), title="status", border_style="cyan"))
        self._update_status()

    def cmd_quit(self, _arg: str) -> None:
        self.exit()

    # ── streaming ────────────────────────────────────────────

    def _add_message(self, content) -> Static:
        """Append a widget to the chat history and scroll to bottom."""
        chat = self.query_one("#chat-history", VerticalScroll)
        widget = Static(content)
        chat.mount(widget)
        chat.scroll_end(animate=False)
        return widget

    def _add_user_message(self, text: str) -> None:
        """Render user input as a read-only field styled like the chat input."""
        chat = self.query_one("#chat-history", VerticalScroll)
        widget = Static(text, classes="user-message")
        chat.mount(widget)
        chat.scroll_end(animate=False)

    def _add_system(self, content) -> None:
        """Add a dim system-style message to the chat."""
        if isinstance(content, str):
            self._add_message(Text(content, style="dim"))
        else:
            self._add_message(content)

    def _update_paste_indicator(self) -> None:
        """Show/hide the paste buffer indicator below the input."""
        try:
            indicator = self.query_one("#paste-indicator", Static)
        except Exception:
            return
        inp = self.query_one("#chat-input", NyxInput)
        if inp.paste_count == 0:
            indicator.update("")
            indicator.display = False
            return
        indicator.display = True
        n = inp.paste_count
        lines = inp.paste_line_count
        word = "paste" if n == 1 else "pastes"
        indicator.update(
            Text.assemble(
                Text(f"  {n} {word} · {lines} lines", style="cyan dim italic"),
            )
        )

    def _update_status(self) -> None:
        bar = self.query_one("#status-bar", Static)
        model = self.config.model
        limit = self._effective_context_limit

        sys_tokens = sum(
            estimate_messages_tokens([m])
            for m in self.convo.messages
            if m["role"] == "system"
        )
        convo_tokens = sum(
            estimate_messages_tokens([m])
            for m in self.convo.messages
            if m["role"] != "system"
        )
        total = sys_tokens + convo_tokens
        pct = total / limit if limit else 0.0

        # Stacked bar: system (magenta, non-compactable) + convo (color by usage).
        bar_width = 20
        sys_filled = max(0, min(bar_width, int(bar_width * sys_tokens / limit))) if limit else 0
        convo_filled = max(0, min(bar_width - sys_filled, int(bar_width * convo_tokens / limit))) if limit else 0
        empty = bar_width - sys_filled - convo_filled

        if pct >= 0.9:
            convo_color = "red"
        elif pct >= 0.7:
            convo_color = "yellow"
        else:
            convo_color = "cyan"

        bar_chars = (
            f"[magenta]{'█' * sys_filled}[/magenta]"
            f"[{convo_color}]{'█' * convo_filled}[/{convo_color}]"
            f"[dim]{'░' * empty}[/dim]"
        )

        # Escape model name to prevent markup injection if it contains brackets.
        safe_model = model.replace("[", "\\[")
        status = Text.from_markup(
            f"[dim]model:[/dim] [cyan]{safe_model}[/cyan]  "
            f"[dim]ctx[/dim] {bar_chars} {total}/{limit}",
        )
        if self._doc_in_context:
            doc_part = Text(f"  docs: {self._doc_full_tokens}", style="magenta")
            status = Text.assemble(status, doc_part)
        if self._code_mode:
            status = Text.assemble(status, Text("  code", style="bold cyan"))
        dots = self._spinner_chars[self._spinner_idx]
        status = Text.assemble(
            Text(f"{dots}  ", style="cyan"),
            status,
        )
        if self._is_streaming:
            bar_w = self.query_one("#status-bar", Static).content_size.width
            right = Text("(esc) interrupt", style="dim cyan")
            pad_len = max(0, bar_w - status.cell_len - right.cell_len)
            status = Text.assemble(status, Text(" " * pad_len), right)
        bar.update(status)

    def _tick_spinner(self) -> None:
        if not self._is_streaming:
            return
        self._spinner_idx = (self._spinner_idx + 1) % len(self._spinner_chars)
        self._update_status()
        # Update thinking indicator only while waiting for first token.
        if self._thinking and self._stream_widget is not None:
            self._stream_widget.update(
                Text.assemble(
                    Text("\n"),
                    Text("☾ ", style="cyan"),
                    Text("NYX", style="bold cyan"),
                    Text("\n"),
                    Text("  thinking ", style="dim"),
                    Text(self._spinner_chars[self._spinner_idx], style="cyan"),
                )
            )

    async def _stream(self) -> None:
        """Run the Ollama stream in a thread and push tokens to the UI."""
        # Assign a new generation so we can detect interruption.
        self._stream_gen += 1
        my_gen = self._stream_gen
        self._active_gen = my_gen
        self._auto_scroll = True

        # Show thinking indicator.
        self._thinking = True
        self._stream_widget = self._add_message(
            Text.assemble(
                Text("\n"),
                Text("☾ ", style="cyan"),
                Text("NYX", style="bold cyan"),
                Text("\n"),
                Text("  thinking ", style="dim"),
                Text(self._spinner_chars[self._spinner_idx], style="cyan"),
            )
        )

        is_thinking = any(kw in self.config.model.lower() for kw in THINKING_KWS)

        # BUG-002 fix: for_request may trigger compaction (a sync HTTP call),
        # so run it in a thread to avoid freezing the event loop.
        messages = await asyncio.to_thread(
            self.convo.for_request, self.client, self._effective_context_limit
        )

        current_text = ""
        result_holder: list = [None]

        def _run() -> None:
            nonlocal current_text
            if is_thinking:
                token_iter, result = self.client.stream_chat_with_thinking(messages)
                result_holder[0] = result
                for token in token_iter:
                    if self._active_gen is not my_gen:
                        break
                    if token.kind == "content":
                        current_text += token.text
                        self.call_from_thread(self._update_response, current_text, my_gen)
            else:
                token_iter, result = self.client.stream_chat(messages)
                result_holder[0] = result
                for token in token_iter:
                    if self._active_gen is not my_gen:
                        break
                    current_text += token
                    self.call_from_thread(self._update_response, current_text, my_gen)

        try:
            await asyncio.to_thread(_run)
            await asyncio.sleep(0)  # drain any pending call_from_thread callbacks
        except httpx.TimeoutException:
            self._error_response(
                f"request timed out — is ollama responding? "
                f"(timeout: {self.config.request_timeout}s)",
                my_gen,
            )
            return
        except Exception as e:
            self._error_response(f"model error: {e}", my_gen)
            return

        # Interrupted — don't finalize, keep partial response as-is.
        if self._active_gen is not my_gen:
            return

        full_text = current_text
        result = result_holder[0]

        # In code mode, strip to code blocks for context and display.
        store_text = self._extract_code_blocks(full_text) if self._code_mode else full_text

        # Update conversation state.
        self.convo.add_assistant(store_text)
        if result:
            self.convo.update_exact_tokens(result.prompt_tokens)

        self._finalize_response(store_text, my_gen)
        self._update_status()

        # Auto-remove doc from context after the response (keep metadata for /followup).
        if self._doc_pending_removal:
            self._doc_pending_removal = False
            if self._doc_in_context:
                self._remove_doc_from_convo()
                self._add_system(
                    Text("doc removed — use /followup <question> to query the doc again", style="dim")
                )
                self._update_status()

    @staticmethod
    def _extract_code_blocks(text: str) -> str:
        """Extract all markdown fenced code blocks (including fences).

        Handles partial fences during streaming — an opening fence without
        a matching close is captured as partial content.
        """
        parts: list[str] = []
        i = 0
        while True:
            start = text.find("```", i)
            if start == -1:
                break
            nl_after = text.find("\n", start)
            if nl_after == -1:
                # Opening fence line not complete — still accumulating.
                break
            close = text.find("```", nl_after + 1)
            if close == -1:
                # No closing fence yet — capture partial (live streaming).
                parts.append(text[start:])
                break
            parts.append(text[start : close + 3])
            i = close + 3
        return "\n\n".join(parts)

    def _assistant_code_text(self, text: str) -> Text:
        """Render code-blocks-only text during streaming (no markdown)."""
        return Text.assemble(
            Text("\n"),
            Text("☾ ", style="cyan"),
            Text("NYX", style="bold cyan"),
            Text(" [code]", style="dim"),
            Text("\n\n"),
            Text(text, style="dim"),
            Text("\n"),
        )

    def _assistant_text(self, text: str) -> Text:
        """Build the assistant message content: header + markdown body."""
        chat = self.query_one("#chat-history", VerticalScroll)
        width = chat.content_size.width or 80
        header_suffix = " [code]" if self._code_mode else ""
        return Text.assemble(
            Text("\n"),
            Text("☾ ", style="cyan"),
            Text("NYX", style="bold cyan"),
            Text(header_suffix, style="dim"),
            Text("\n\n"),
            render_markdown(text, width),
            Text("\n"),
        )

    def _update_response(self, text: str, gen: int | None = None) -> None:
        if gen is not None and self._active_gen is not gen:
            return
        if self._stream_widget is not None:
            if self._code_mode:
                code_text = self._extract_code_blocks(text)
                if not code_text:
                    return  # Keep thinking indicator until code arrives.
                self._thinking = False
                self._stream_widget.update(self._assistant_code_text(code_text))
            else:
                self._thinking = False
                self._stream_widget.update(self._assistant_text(text))
            if self._auto_scroll:
                chat = self.query_one("#chat-history", VerticalScroll)
                chat.scroll_end(animate=False)

    def _error_response(self, message: str, gen: int | None = None) -> None:
        if gen is not None and self._active_gen is not gen:
            return
        self._thinking = False
        if self._stream_widget is not None:
            self._stream_widget.update(
                Panel(Text(message, style="bold red"), title="error", border_style="red"),
            )
            self._stream_widget = None
        self._active_gen = None
        self.query_one("#chat-input", NyxInput).focus()

    def _finalize_response(self, text: str, gen: int | None = None) -> None:
        if gen is not None and self._active_gen is not gen:
            return
        if self._stream_widget is not None:
            self._stream_widget.update(self._assistant_text(text))
            self._stream_widget = None
        self._active_gen = None
        self.query_one("#chat-input", NyxInput).focus()
