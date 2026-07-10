"""Documentation browser modal — search and read DevDocs docsets.

Opened by ``/docs <slug>`` from the chat app.  Provides:
  - fuzzy search over the docset's index entries
  - page rendering (HTML → markdown → rich Text)
  - "Feed to AI" — user mouse-selects text on the page, presses 'a',
    and the selected text is returned to the caller for context injection.
    If nothing is selected, a notification guides the user.
"""

from __future__ import annotations

from dataclasses import dataclass

from markdownify import markdownify as md
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Input, ListView, ListItem, Static

from nyx.core.chat import estimate_tokens
from nyx.tui.doc_page_view import DocPageView
from nyx.tui.markdown_renderer import render_markdown


# ── fuzzy search ─────────────────────────────────────────────


@dataclass
class ScoredEntry:
    name: str
    path: str
    type: str
    score: int


def _filter_entries(query: str, entries: list[dict]) -> list[ScoredEntry]:
    """Fuzzy-filter index entries by query. Returns scored, sorted list."""
    q = query.lower().strip()
    if not q:
        # No query — return first 50 alphabetically.
        sorted_entries = sorted(entries, key=lambda e: e["name"])
        scored = [
            ScoredEntry(e["name"], e["path"], e.get("type", ""), 0)
            for e in sorted_entries[:50]
        ]
        return scored

    results: list[ScoredEntry] = []
    for e in entries:
        name = e["name"]
        lower = name.lower()
        if q == lower:
            score = 100
        elif lower.startswith(q):
            score = 80
        elif any(word.startswith(q) for word in lower.split()):
            score = 60
        elif q in lower:
            score = 40
        else:
            continue
        results.append(ScoredEntry(name, e["path"], e.get("type", ""), score))

    # Sort by score desc, then by name length asc (shorter = more relevant).
    results.sort(key=lambda s: (-s.score, len(s.name)))
    return results[:50]


def _page_key(path: str) -> str:
    """Extract the db.json key from an index entry path.

    Index paths look like ``net/http/index#HandleFunc``.
    The db key is the part before ``#``.
    """
    return path.split("#")[0]


# ── modal ────────────────────────────────────────────────────


class DocBrowser(ModalScreen):
    """Full-screen documentation browser modal.

    Two states:
      - ``search``: input focused, results list visible
      - ``reading``: page content shown, input dimmed

    Dismisses with:
      - ``None`` — user closed without feeding
      - ``(slug, entry_name, selected_text)`` — user pressed 'a' with
        mouse-selected text on the page
    """

    BINDINGS = [
        Binding("a", "ask_ai", "Feed to AI", show=False),
        Binding("q", "close", "Close", show=False),
        Binding("escape", "back_or_close", "Back/Close", show=False),
        Binding("backspace", "back_or_close", "Back/Close", show=False),
    ]

    CSS = """
    DocBrowser {
        align: center middle;
        background: $background 90%;
    }

    DocBrowser > Container {
        width: 100%;
        height: 100%;
        background: $background;
        padding: 0 1;
    }

    #doc-search {
        width: 100%;
        margin: 1 0 0 0;
        height: 3;
        padding: 1 1;
        border: none;
        border-left: solid cyan;
    }

    #doc-search:focus {
        border: none;
        border-left: solid cyan;
        outline: none;
    }

    #doc-search.reading {
        opacity: 0.4;
    }

    #doc-results {
        height: 1fr;
        border: solid #333333;
        margin: 0 0 0 0;
        min-height: 0;
    }

    #doc-results.hidden {
        display: none;
    }

    #doc-results > ListItem {
        padding: 0 1;
    }

    #doc-page {
        height: 1fr;
        border: solid #333333;
        margin: 0 0 0 0;
        padding: 1 1;
        min-height: 0;
    }

    #doc-page.hidden {
        display: none;
    }

    #doc-hints {
        height: 1;
        color: #555555;
        padding: 0 1;
        margin: 0 0 1 0;
    }
    """

    def __init__(self, slug: str, index: dict, db: dict) -> None:
        super().__init__()
        self._slug = slug
        self._entries: list[dict] = index.get("entries", [])
        self._db: dict = db
        self._state = "search"
        self._current_entry: ScoredEntry | None = None
        self._current_markdown: str = ""

    def compose(self) -> ComposeResult:
        with Container():
            yield Input(id="doc-search", placeholder=f"search {self._slug} docs...")
            yield ListView(id="doc-results")
            yield DocPageView(id="doc-page", classes="hidden")
            yield Static(Text("[enter] open  [esc] close", style="dim"), id="doc-hints")

    def on_mount(self) -> None:
        self._update_hints()
        self._do_search("")
        self.query_one("#doc-search", Input).focus()

    # ── search state ─────────────────────────────────────────

    def on_input_changed(self, event: Input.Changed) -> None:
        if self._state != "search":
            return
        self._do_search(event.value)

    def _do_search(self, query: str) -> None:
        results = _filter_entries(query, self._entries)
        lv = self.query_one("#doc-results", ListView)
        lv.clear()
        if not results:
            lv.append(ListItem(Static(Text("  no matches", style="dim"))))
            return
        for r in results:
            label = r.name
            if r.type:
                label = f"  {r.name}  [{r.type}]"
            else:
                label = f"  {r.name}"
            li = ListItem(Static(label))
            li.entry = r  # type: ignore[attr-defined]
            lv.append(li)
        lv.index = 0

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if self._state != "search":
            return
        entry = getattr(event.item, "entry", None)
        if entry is None:
            return
        self._open_page(entry)

    # ── reading state ────────────────────────────────────────

    def _open_page(self, entry: ScoredEntry) -> None:
        key = _page_key(entry.path)
        html = self._db.get(key, "")
        page = self.query_one("#doc-page", DocPageView)
        if not html:
            page.update(Text(f"page not found: {key}", style="bold red"))
        else:
            # Convert HTML → markdown for both rendering and AI context.
            self._current_markdown = md(html)
            width = self.size.width - 4  # account for padding/border
            text = render_markdown(self._current_markdown, max(width, 40))
            page.update(text)

        self._current_entry = entry
        self._state = "reading"

        # Switch UI.
        self.query_one("#doc-results", ListView).add_class("hidden")
        page.remove_class("hidden")
        page.focus()

        # Dim and disable the search input so it doesn't consume keys.
        search = self.query_one("#doc-search", Input)
        search.add_class("reading")
        search.disabled = True
        self._update_hints()

    def _back_to_search(self) -> None:
        self._state = "search"
        self.query_one("#doc-page", DocPageView).add_class("hidden")
        self.query_one("#doc-results", ListView).remove_class("hidden")
        search = self.query_one("#doc-search", Input)
        search.remove_class("reading")
        search.disabled = False
        search.focus()
        self._update_hints()

    # ── key handling ─────────────────────────────────────────

    def _update_hints(self) -> None:
        hints = self.query_one("#doc-hints", Static)
        if self._state == "search":
            hints.update(Text("[enter] open  [esc] close", style="dim"))
        else:
            tokens = estimate_tokens(self._current_markdown) if self._current_markdown else 0
            hints.update(Text(
                "[mouse] select text  [a] Feed to AI  [backspace/esc] back  [q] close",
                style="dim",
            ))

    def action_ask_ai(self) -> None:
        """'a' key — feed mouse-selected text to the AI.

        If no text is selected, notify the user to select first.
        """
        if self._state != "reading":
            return

        # Try to get mouse-selected text from the screen.
        selection = self.screen.get_selected_text()
        if not selection or not selection.strip():
            self.notify(
                "Select text with your mouse first, then press [a] to feed it to the AI.",
                timeout=4,
            )
            return

        if self._current_entry:
            self.dismiss((
                self._slug,
                self._current_entry.name,
                selection.strip(),
            ))

    def action_close(self) -> None:
        """'q' key — close the modal."""
        self.dismiss(None)

    def action_back_or_close(self) -> None:
        """Escape/Backspace — back to search, or close if already searching."""
        if self._state == "reading":
            self._back_to_search()
        else:
            self.dismiss(None)
