"""Virtualized document page widget using Textual's Line API.

Renders only visible lines instead of the entire page, eliminating
the lag that ``Static`` causes on large doc pages (1000+ lines).

Stores per-line ``Text`` objects (preserving rich styles from
``render_markdown``) and pre-wraps them to the widget's content width.
``render_line(y)`` converts the appropriate ``Text`` to a ``Strip`` on
demand, with an LRU cache.  Mouse-drag text selection is supported via
``get_selection()`` / ``selection_updated()``.
"""

from __future__ import annotations

from rich.cells import cell_len
from rich.segment import Segment
from rich.style import Style as RichStyle
from rich.text import Text

from textual.cache import LRUCache
from textual.geometry import Size
from textual.scroll_view import ScrollView
from textual.selection import Selection
from textual.strip import Strip


class DocPageView(ScrollView, can_focus=True):
    """Line-API widget that renders a rich Text document virtualized.

    Only the visible lines are rendered — O(viewport_height) per frame,
    not O(total_lines).  Scrollbar support and scroll keybindings are
    inherited from ``ScrollView``.

    Mouse-drag selection works via ``get_selection()`` which returns the
    full plain text for ``Selection.extract`` to slice.
    """

    ALLOW_SELECT = True
    DEFAULT_CSS = """
    DocPageView {
        background: $surface;
        color: $text;
        overflow-y: auto;
        overflow-x: auto;
        scrollbar-size: 1 0;
        &:focus {
            background-tint: $foreground 5%;
        }
    }
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Per-line Text objects (logical lines, not wrapped).
        self._line_texts: list[Text] = []
        # Pre-wrapped display lines: each is (source_index, Text).
        self._display_lines: list[tuple[int, Text]] = []
        # Plain text per display line for selection extraction.
        self._plain_lines: list[str] = []
        # The original source Text, kept for re-wrapping on resize.
        self._source_text: Text | None = None
        # Content width used for the current wrapping.
        self._content_width: int = 0
        # LRU cache for rendered strips (cleared on scroll/selection/resize).
        self._strip_cache: LRUCache[int, Strip] = LRUCache(1024)

    # ── public API ──────────────────────────────────────────

    def update(self, text: Text) -> None:
        """Set new content from a rendered Text and re-wrap to content width."""
        self._source_text = text
        self._wrap_and_store()
        self.scroll_home(animate=False)
        self.refresh()

    def clear(self) -> None:
        """Remove all content."""
        self._line_texts.clear()
        self._display_lines.clear()
        self._plain_lines.clear()
        self._source_text = None
        self._strip_cache.clear()
        self.virtual_size = Size(0, 0)
        self.refresh()

    # ── wrapping ────────────────────────────────────────────

    def _wrap_and_store(self) -> None:
        """Split source Text into logical lines and pre-wrap to content width."""
        if self._source_text is None:
            return

        width = self.scrollable_content_region.width or self.size.width or 80
        width = max(width, 20)
        self._content_width = width

        # Split into logical lines (preserves spans per line).
        logical_lines = self._source_text.split("\n")
        self._line_texts = logical_lines

        # Wrap each logical line to the content width.
        console = self.app.console
        opts = console.options.update_width(width)

        display_lines: list[tuple[int, Text]] = []
        plain_lines: list[str] = []

        for src_idx, line_text in enumerate(logical_lines):
            if not line_text.plain:
                # Empty line — no wrapping needed.
                display_lines.append((src_idx, Text()))
                plain_lines.append("")
                continue

            # Render this line through the console to get wrapped segments.
            segments = list(console.render(line_text, opts))
            for seg_line in Segment.split_lines(segments):
                # Reconstruct a Text from the segments of this wrapped line.
                parts: list[str] = []
                for seg in seg_line:
                    parts.append(seg.text)
                wrapped_plain = "".join(parts)
                # We store a Text for style application during selection.
                # Re-render the wrapped line as a Text by joining segments.
                wrapped_text = Text(wrapped_plain)
                # Copy styles from segments onto the Text.
                pos = 0
                for seg in seg_line:
                    seg_len = len(seg.text)
                    if seg.style:
                        wrapped_text.stylize(seg.style, pos, pos + seg_len)
                    pos += seg_len
                display_lines.append((src_idx, wrapped_text))
                plain_lines.append(wrapped_plain)

        self._display_lines = display_lines
        self._plain_lines = plain_lines

        max_width = max((cell_len(p) for p in plain_lines), default=0)
        self.virtual_size = Size(max_width, len(display_lines))
        self._strip_cache.clear()

    # ── resize ──────────────────────────────────────────────

    def on_resize(self, event) -> None:
        new_width = self.scrollable_content_region.width or self.size.width
        if new_width and new_width != self._content_width and self._source_text is not None:
            self._wrap_and_store()

    # ── style updates ───────────────────────────────────────

    def notify_style_update(self) -> None:
        super().notify_style_update()
        self._strip_cache.clear()

    # ── selection ───────────────────────────────────────────

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        """Return the selected text from the full plain-text buffer."""
        text = "\n".join(self._plain_lines)
        return selection.extract(text), "\n"

    def selection_updated(self, selection: Selection | None) -> None:
        """Called when the selection changes — clear cache and refresh."""
        self._strip_cache.clear()
        self.refresh()

    # ── Line API ────────────────────────────────────────────

    def render_line(self, y: int) -> Strip:
        scroll_x, scroll_y = self.scroll_offset
        return self._render_line(scroll_y + y, scroll_x, self.size.width)

    def _render_line(self, y: int, scroll_x: int, width: int) -> Strip:
        rich_style = self.rich_style

        if y >= len(self._display_lines):
            return Strip.blank(width, rich_style)

        selection = self.text_selection
        if selection is None and y in self._strip_cache:
            cached = self._strip_cache[y]
            return cached.crop_extend(scroll_x, scroll_x + width, rich_style)

        # Copy the stored Text so stylize() doesn't mutate the original.
        _src_idx, stored_text = self._display_lines[y]
        line_text = stored_text.copy()
        line_text.stylize(rich_style)

        # Apply selection highlight if this line is in the selection range.
        # We use only the *background* from the selection style, not the
        # foreground.  The selection CSS sets ``color: transparent`` (alpha 0),
        # but Rich styles don't support alpha — Textual's conversion composites
        # transparent-fg over the bg, yielding ``color == bgcolor``, which
        # makes the text invisible under the highlight.  By applying only
        # ``bgcolor`` we preserve the original text colours and attributes.
        if selection is not None:
            if (span := selection.get_span(y)) is not None:
                start, end = span
                if end == -1:
                    end = len(line_text.plain)
                comp_styles = self.screen.get_component_styles("screen--selection")
                selection_bg = RichStyle(bgcolor=comp_styles.background.rich_color)
                line_text.stylize(selection_bg, start, end)

        # Render Text → Strip.
        plain = line_text.plain
        strip = Strip(line_text.render(self.app.console), cell_len(plain))

        # Cache the strip BEFORE crop/offsets so horizontal scroll works.
        if selection is None:
            self._strip_cache[y] = strip

        strip = strip.crop_extend(scroll_x, scroll_x + width, rich_style)
        strip = strip.apply_offsets(scroll_x, y)
        return strip
