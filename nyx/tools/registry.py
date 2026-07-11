"""Tool registry — central place for tool definitions and dispatch.

Each tool module exports a ``TOOL_DEF`` dict (Ollama/OpenAI format) and
a function matching the tool name.  This registry collects them, provides
the list for the Ollama API call, and dispatches incoming tool calls to
the right function.
"""

from __future__ import annotations

import json

from nyx.tools import calculator
from nyx.tools import doc_search

# Map tool name -> callable.
TOOL_FUNCS: dict[str, callable] = {
    "calculate": calculator.calculate,
    "search_docs": doc_search.search_docs,
}

# Map tool name -> Ollama tool definition dict.
TOOL_DEFS: list[dict] = [
    calculator.TOOL_DEF,
    doc_search.TOOL_DEF,
]


def get_tool_definitions() -> list[dict]:
    """Return the list of tool definitions to pass to Ollama."""
    return TOOL_DEFS


def execute_tool(name: str, arguments: dict | str) -> str:
    """Look up a tool by name and call it with the given arguments.

    *arguments* may be a dict or a JSON string (some models emit strings).
    Returns the tool result as a string.
    """
    fn = TOOL_FUNCS.get(name)
    if fn is None:
        return f"Error: unknown tool '{name}'"

    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return f"Error: could not parse arguments as JSON: {arguments}"

    try:
        return fn(**arguments)
    except TypeError as e:
        return f"Error: bad arguments for '{name}': {e}"
