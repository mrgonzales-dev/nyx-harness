"""Configuration modal — adjust settings with visual sliders."""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Static


# Temperature presets: (label, value)
TEMP_LEVELS = [
    ("strict",      0.0),
    ("concise",     0.2),
    ("balanced",    0.4),
    ("flexible",    0.6),
    ("creative",    0.8),
    ("wild",        1.0),
]


def _nearest_temp_level(value: float) -> int:
    """Find the closest preset index for a temperature value."""
    best_idx = 0
    best_diff = float("inf")
    for i, (_, v) in enumerate(TEMP_LEVELS):
        diff = abs(v - value)
        if diff < best_diff:
            best_diff = diff
            best_idx = i
    return best_idx


class ConfigModal(ModalScreen):
    """Modal for adjusting Nyx configuration with visual levels.

    Dismisses with a dict of changed settings, or None if cancelled.
    """

    BINDINGS = [
        Binding("left", "dec", "Decrease", show=False),
        Binding("right", "inc", "Increase", show=False),
        Binding("up", "prev_setting", "Previous", show=False),
        Binding("down", "next_setting", "Next", show=False),
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("enter", "save", "Save", show=False),
    ]

    CSS = """
    ConfigModal {
        align: center middle;
        background: $background 60%;
    }

    ConfigModal > Container {
        width: 80;
        max-height: 10;
        background: $surface;
        border: solid cyan;
        padding: 1 2;
    }

    ConfigModal > Container > #modal-header {
        color: cyan;
        text-style: bold;
        text-align: center;
        width: 100%;
        margin-bottom: 1;
    }

    ConfigModal > Container > #config-body {
        height: auto;
        margin-bottom: 1;
    }

    ConfigModal > Container > #config-footer {
        color: #555555;
        text-align: center;
        width: 100%;
    }

    .config-row {
        height: 1;
        margin: 0 0 1 0;
        padding: 0 1;
    }

    .config-row.active {
        background: #1a1a2a;
    }

    .config-label {
        color: #888888;
        height: 1;
    }

    .config-bar {
        height: 1;
    }

    .config-value {
        color: cyan;
        height: 1;
    }
    """

    def __init__(self, temperature: float) -> None:
        super().__init__()
        self._temp_idx = _nearest_temp_level(temperature)
        self._settings = ["temperature"]
        self._cursor = 0  # which setting row is active

    def compose(self) -> ComposeResult:
        with Container():
            yield Static("Nyx Config", id="modal-header")
            with Container(id="config-body"):
                yield Static("", id="row-0", classes="config-row")
            yield Static(
                Text("[←/→] adjust  [↑/↓] switch  [enter] save  [esc] cancel", style="dim"),
                id="config-footer",
            )

    def on_mount(self) -> None:
        self._render_rows()

    def _render_rows(self) -> None:
        """Render all setting rows."""
        row = self.query_one("#row-0", Static)
        active = self._cursor == 0

        label, value = TEMP_LEVELS[self._temp_idx]

        bar_width = 40
        filled = int(bar_width * (self._temp_idx / (len(TEMP_LEVELS) - 1)))
        bar = Text()
        bar.append("  temperature  ", style="dim")
        bar.append("█" * filled, style="cyan")
        bar.append("░" * (bar_width - filled), style="dim")
        bar.append(f"  {label} ({value:.1f})", style="cyan bold" if active else "cyan")

        row.update(bar)
        row.set_class(active, "active")

    @property
    def _temp_value(self) -> float:
        return TEMP_LEVELS[self._temp_idx][1]

    # ── actions ──────────────────────────────────────────────

    def action_inc(self) -> None:
        if self._cursor == 0:
            self._temp_idx = min(len(TEMP_LEVELS) - 1, self._temp_idx + 1)
            self._render_rows()

    def action_dec(self) -> None:
        if self._cursor == 0:
            self._temp_idx = max(0, self._temp_idx - 1)
            self._render_rows()

    def action_prev_setting(self) -> None:
        self._cursor = max(0, self._cursor - 1)
        self._render_rows()

    def action_next_setting(self) -> None:
        self._cursor = min(len(self._settings) - 1, self._cursor + 1)
        self._render_rows()

    def action_save(self) -> None:
        self.dismiss({"temperature": self._temp_value})

    def action_cancel(self) -> None:
        self.dismiss(None)
