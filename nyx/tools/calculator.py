"""Calculator tool — safe arithmetic expression evaluator.

Used by the tool-calling system to let models perform calculations
instead of doing math in their heads (especially useful for small
models that struggle with arithmetic).
"""

from __future__ import annotations

import ast
import operator

# Only allow these AST node types and operators.
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float | int:
    """Recursively evaluate an AST node with whitelisted operators only."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):  # numbers
        if isinstance(node.value, int | float):
            return node.value
        raise ValueError(f"unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPS:
            raise ValueError(f"unsupported operator: {op_type.__name__}")
        return _SAFE_OPS[op_type](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPS:
            raise ValueError(f"unsupported unary op: {op_type.__name__}")
        return _SAFE_OPS[op_type](_safe_eval(node.operand))
    raise ValueError(f"unsupported expression: {ast.dump(node)}")


def calculate(expression: str) -> str:
    """Safely evaluate a math expression. Returns the result as a string."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree)
        # Pretty-print: drop trailing .0 for integers.
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# Ollama/OpenAI tool definition.
TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": (
            "Evaluate a basic arithmetic expression. "
            "Supports +, -, *, /, parentheses, and integers. "
            "Use this for any math calculation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A math expression, e.g. '25 * 17' or '(3 + 4) * 2'",
                }
            },
            "required": ["expression"],
        },
    },
}
