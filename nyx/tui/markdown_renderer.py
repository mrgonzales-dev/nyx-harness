"""Markdown-to-Rich-Text renderer that preserves Textual selection.

Uses markdown-it-py (already a Textual dependency) to parse, then converts
to a ``rich.text.Text`` so ``Static`` widgets keep native drag-selection
and Ctrl+C copy.  No box-drawing artifacts in copied text.
"""

from __future__ import annotations

from markdown_it import MarkdownIt
from rich.syntax import Syntax
from rich.text import Text

_md = MarkdownIt("commonmark", {"html": False})
try:
    _md.enable(["strikethrough", "table"])
except Exception:
    pass

_HEADING_STYLES = {
    1: "bold cyan",
    2: "bold cyan",
    3: "bold",
    4: "bold dim",
    5: "bold dim",
    6: "dim",
}
_INLINE_STYLES = {
    "strong": "bold",
    "em": "italic",
    "s": "strike",
    "a": "underline cyan",
}

_SYNTAX_THEME = "github-dark"


def _highlight_code(code: str, lang: str) -> Text:
    """Syntax-highlight *code* using Pygments via Rich's ``Syntax``.

    Falls back to plain dim text if the language is unknown.
    Returns a ``Text`` so selection stays intact.
    """
    if lang:
        try:
            syn = Syntax(
                code, lexer=lang, theme=_SYNTAX_THEME,
                background_color="default",
            )
            return syn.highlight(code)
        except Exception:
            pass
    return Text(code, style="dim")


def render_markdown(text: str, width: int = 80) -> Text:
    """Parse markdown *text* and return a styled Rich ``Text``."""
    if not text or not text.strip():
        return Text(text)
    tokens = _md.parse(text)
    r = _Renderer(width)
    r.render(tokens, 0, len(tokens))
    # Strip trailing newlines.
    while r.text.plain.endswith("\n"):
        r.text = r.text[:-1]
    return r.text


def _render_inline(children) -> Text:
    """Convert inline tokens to a styled ``Text``."""
    result = Text()
    stack: list[str] = []
    for child in children:
        t = child.type
        if t == "text":
            result.append(child.content, style=" ".join(stack))
        elif t in ("softbreak", "hardbreak"):
            result.append("\n")
        elif t == "code_inline":
            result.append(child.content, style=" ".join(stack + ["cyan on #111111"]))
        elif t == "image":
            pass
        elif child.nesting == 1:
            s = _INLINE_STYLES.get(child.tag)
            if s:
                stack.append(s)
        elif child.nesting == -1 and stack:
            stack.pop()
    return result


class _Renderer:
    """Walks block-level markdown-it tokens and builds a ``Text``."""

    def __init__(self, width: int = 80) -> None:
        self.width = width
        self.text = Text()
        self._list_stack: list[list] = []  # each: [ordered: bool, counter: int]
        self._prefix = ""  # blockquote prefix, applied at line start
        self._at_line_start = True
        self._list_first_para = True

    def render(self, tokens: list, start: int, end: int) -> None:
        i = start
        while i < end:
            t = tokens[i]
            handler = getattr(self, f"_t_{t.type}", None)
            if handler:
                i = handler(tokens, i, end)
            else:
                i += 1

    def _add(self, content, style: str = "") -> None:
        if self._at_line_start and self._prefix:
            self.text.append(self._prefix, style="cyan dim")
        if isinstance(content, str):
            self.text.append(content, style=style or None)
        elif isinstance(content, Text):
            self.text.append(content)
        self._at_line_start = False

    def _nl(self) -> None:
        self.text.append("\n")
        self._at_line_start = True

    # ── block handlers ───────────────────────────────────────

    def _t_heading_open(self, tokens, i, end):
        level = int(tokens[i].tag[1])
        j = i + 1
        parts: list[Text] = []
        while j < end and tokens[j].type != "heading_close":
            if tokens[j].type == "inline":
                parts.append(_render_inline(tokens[j].children))
            j += 1
        heading = Text.assemble(*parts) if parts else Text()
        heading.stylize(_HEADING_STYLES.get(level, "bold"))
        self._add(heading)
        self._nl()
        if level <= 2:
            self._add(Text("─" * min(self.width, 60), style="cyan dim"))
            self._nl()
        return j + 1 if j < end else j

    def _t_paragraph_open(self, tokens, i, end):
        j = i + 1
        parts: list[Text] = []
        while j < end and tokens[j].type != "paragraph_close":
            if tokens[j].type == "inline":
                parts.append(_render_inline(tokens[j].children))
            j += 1
        content = Text.assemble(*parts) if parts else Text()
        if self._list_stack and self._list_first_para:
            ordered, counter = self._list_stack[-1]
            indent = "  " * (len(self._list_stack) - 1)
            bullet = f"{counter}. " if ordered else "• "
            self._add(Text(indent + bullet, style="cyan"))
            self._list_first_para = False
        elif self._list_stack:
            self._add("  " * len(self._list_stack))
        self._add(content)
        self._nl()
        return j + 1 if j < end else j

    def _t_fence(self, tokens, i, end):
        self._add_code_block(tokens[i].content.rstrip("\n"), tokens[i].info.strip())
        return i + 1

    def _t_code_block(self, tokens, i, end):
        self._add_code_block(tokens[i].content.rstrip("\n"), "")
        return i + 1

    def _add_code_block(self, code: str, lang: str) -> None:
        # Margin above.
        if not self._at_line_start:
            self._nl()
        self._nl()
        if lang:
            self._add(Text(f"  {lang}", style="cyan dim"))
            self._nl()
        highlighted = _highlight_code(code, lang)
        for line in highlighted.split("\n"):
            self._add(Text("  "))
            self._add(line)
            self._nl()
        # Margin below.
        self._nl()

    def _t_bullet_list_open(self, tokens, i, end):
        self._list_stack.append([False, 0])
        return i + 1

    def _t_ordered_list_open(self, tokens, i, end):
        start = 1
        attrs = tokens[i].attrs
        if attrs and "start" in attrs:
            try:
                start = int(attrs["start"])
            except (ValueError, TypeError):
                pass
        self._list_stack.append([True, start - 1])
        return i + 1

    def _t_bullet_list_close(self, tokens, i, end):
        self._list_stack.pop()
        if not self._list_stack:
            self._nl()
        return i + 1

    def _t_ordered_list_close(self, tokens, i, end):
        return self._t_bullet_list_close(tokens, i, end)

    def _t_list_item_open(self, tokens, i, end):
        if self._list_stack:
            self._list_stack[-1][1] += 1
        self._list_first_para = True
        return i + 1

    def _t_list_item_close(self, tokens, i, end):
        return i + 1

    def _t_blockquote_open(self, tokens, i, end):
        depth = 1
        j = i + 1
        while j < end and depth > 0:
            if tokens[j].type == "blockquote_open":
                depth += 1
            elif tokens[j].type == "blockquote_close":
                depth -= 1
            if depth == 0:
                break
            j += 1
        # Render inner content in a sub-renderer, then prefix each line.
        inner = _Renderer(self.width)
        inner.render(tokens, i + 1, j)
        while inner.text.plain.endswith("\n"):
            inner.text = inner.text[:-1]
        prefix = self._prefix + "│ "
        if not self._at_line_start:
            self._nl()
        for line in inner.text.split("\n"):
            self.text.append(prefix, style="cyan dim")
            self.text.append(line)
            self._nl()
        return j + 1 if j < end else j

    def _t_blockquote_close(self, tokens, i, end):
        return i + 1

    # ── tables ───────────────────────────────────────────────

    def _t_table_open(self, tokens, i, end):
        """Collect the whole table block, then render aligned with borders."""
        depth = 1
        j = i + 1
        while j < end and depth > 0:
            if tokens[j].type == "table_open":
                depth += 1
            elif tokens[j].type == "table_close":
                depth -= 1
            if depth == 0:
                break
            j += 1
        rows, aligns = self._collect_table(tokens, i + 1, j)
        self._render_table(rows, aligns)
        return j + 1 if j < end else j

    def _t_table_close(self, tokens, i, end):
        return i + 1

    @staticmethod
    def _align_of(token) -> str:
        style = (token.attrs or {}).get("style", "")
        for part in style.split(";"):
            part = part.strip()
            if part.startswith("text-align:"):
                return part.split(":", 1)[1].strip()
        return "left"

    @staticmethod
    def _cell_text(tokens, i, end) -> Text:
        parts: list[Text] = []
        j = i + 1
        while j < end and tokens[j].type not in ("th_close", "td_close"):
            if tokens[j].type == "inline":
                parts.append(_render_inline(tokens[j].children))
            j += 1
        return Text.assemble(*parts) if parts else Text()

    def _collect_table(self, tokens, start, end):
        """Return (rows, aligns). rows = list of list[Text]; aligns = list[str]."""
        rows: list[list[Text]] = []
        aligns: list[str] = []
        i = start
        row: list[Text] = []
        while i < end:
            t = tokens[i]
            if t.type in ("th_open", "td_open"):
                align = self._align_of(t)
                cell = self._cell_text(tokens, i, end)
                row.append(cell)
                if not aligns or len(row) > len(aligns):
                    aligns.append(align)
                # advance past the cell close
                while i < end and tokens[i].type not in ("th_close", "td_close"):
                    i += 1
                i += 1
                continue
            if t.type == "tr_close":
                if row:
                    rows.append(row)
                    row = []
            i += 1
        return rows, aligns

    def _render_table(self, rows: list[list[Text]], aligns: list[str]) -> None:
        if not rows:
            return
        ncols = max(len(r) for r in rows)
        aligns = (aligns + ["left"] * ncols)[:ncols]
        # Column widths from cell plain-text length, capped to keep tables sane.
        widths = [0] * ncols
        for r in rows:
            for c in range(ncols):
                cell = r[c] if c < len(r) else Text()
                widths[c] = max(widths[c], len(cell.plain))
        # Cap each column to a fraction of total width so wide tables wrap-ish.
        cap = max(8, self.width // max(ncols, 1) - 3)
        widths = [min(w, cap) for w in widths]

        def pad(cell: Text, w: int, align: str) -> Text:
            s = cell.plain
            if len(s) > w:
                s = s[: w - 1] + "…"
                cell = Text(s, style=cell.style)
                if align == "right":
                    return Text(" " * (w - len(s))) + cell
                return cell + Text(" " * (w - len(s)))
            gap = w - len(s)
            if align == "right":
                return Text(" " * gap) + cell
            if align == "center":
                left = gap // 2
                return Text(" " * left) + cell + Text(" " * (gap - left))
            return cell + Text(" " * gap)

        border = "cyan dim"
        if not self._at_line_start:
            self._nl()
        self._nl()

        def border_row(left, mid, right, fill="─"):
            segs = [fill * (w + 2) for w in widths]
            self._add(Text(left + mid.join(segs) + right, style=border))
            self._nl()

        border_row("┌", "┬", "┐")
        for ri, r in enumerate(rows):
            cells = []
            for c in range(ncols):
                cell = r[c] if c < len(r) else Text()
                cells.append(pad(cell, widths[c], aligns[c]))
            line = Text("│ ", style=border)
            for c, cell in enumerate(cells):
                if c:
                    line.append(" │ ", style=border)
                line.append(cell)
            line.append(" │", style=border)
            self._add(line)
            self._nl()
            if ri == 0 and len(rows) > 1:
                border_row("├", "┼", "┤")
        border_row("└", "┴", "┘")
        self._nl()

    def _t_hr(self, tokens, i, end):
        self._add(Text("─" * min(self.width, 60), style="cyan dim"))
        self._nl()
        return i + 1

    def _t_inline(self, tokens, i, end):
        self._add(_render_inline(tokens[i].children))
        return i + 1
