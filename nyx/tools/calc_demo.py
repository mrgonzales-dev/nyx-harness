"""MiniCPM tool-calling demo: a basic calculator.

Run:  python -m nyx.tools.calc_demo

Demonstrates the full tool-call loop with a 1B model:
  1. User asks a math question
  2. Model decides to call the `calculate` tool
  3. We execute the tool locally
  4. We feed the result back to the model
  5. Model gives the final answer

No constrained decoding needed — Ollama's native tool-call parsing
handles the format. The model puts its reasoning in the `thinking`
field and emits a structured `tool_calls` array.

This demo uses the shared tool registry, so any tools registered there
are available.
"""

from __future__ import annotations

import json
import sys

import ollama

from nyx.tools import registry

MODEL = "minicpm-v4.6:1b"


# ── the chat loop ───────────────────────────────────────────────

def run_tool_loop(client: ollama.Client, user_query: str, max_turns: int = 5) -> None:
    """Run the full tool-calling conversation loop."""
    tools = registry.get_tool_definitions()

    messages: list[dict] = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant with access to tools. "
                "For any arithmetic, use the calculate tool instead of "
                "doing math in your head. Give clear, concise answers."
            ),
        },
        {"role": "user", "content": user_query},
    ]

    print(f"\n{'='*60}")
    print(f"USER: {user_query}")
    print(f"{'='*60}")

    for turn in range(max_turns):
        resp = client.chat(
            model=MODEL,
            messages=messages,
            tools=tools,
            stream=False,
        )

        if not isinstance(resp, dict):
            resp = resp.model_dump()

        msg = resp.get("message", {})

        thinking = msg.get("thinking", "")
        if thinking:
            print(f"\n[thinking] {thinking.strip()}")

        content = msg.get("content", "")
        if content:
            print(f"\n[assistant] {content}")

        tool_calls = msg.get("tool_calls", [])
        if not tool_calls:
            print(f"\n{'='*60}")
            print("DONE — no more tool calls.")
            return

        messages.append(msg)

        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            fn_args = tc["function"]["arguments"]
            if isinstance(fn_args, str):
                fn_args = json.loads(fn_args)

            print(f"\n[tool call] {fn_name}({fn_args})")

            result = registry.execute_tool(fn_name, fn_args)
            print(f"[tool result] {result}")

            messages.append({
                "role": "tool",
                "name": fn_name,
                "content": result,
            })

    print(f"\n{'='*60}")
    print("Stopped — max turns reached.")


# ── entry point ─────────────────────────────────────────────────

DEMO_QUERIES = [
    "What is 25 * 17?",
    "Calculate (3 + 4) * 2 - 10",
    "If I have 144 eggs and divide them into cartons of 12, how many cartons do I get?",
]


def main() -> None:
    if len(sys.argv) > 1:
        queries = [" ".join(sys.argv[1:])]
    else:
        queries = DEMO_QUERIES

    client = ollama.Client()
    for q in queries:
        run_tool_loop(client, q)

    print("\n" + "=" * 60)
    print("All demos complete.")


if __name__ == "__main__":
    main()
