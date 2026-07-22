# coding=utf-8
# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The `calc` tool — exact arithmetic so the model never computes in its head.

A safe AST evaluator over numbers, the standard operators, and a small whitelist
of functions. LLMs are reliable at reasoning and unreliable at arithmetic (a
recurring faithfulness failure: "$37B is a 123% increase from $13B"); offloading
every growth rate, ratio, and percentage to this tool keeps the numbers exact
and
auditable.
"""

from __future__ import annotations

import ast
import math
import operator

from financeharness.runtime.tool_registry import ToolError, ToolResponse, ToolSpec
from pydantic import BaseModel, Field

_DESCRIPTION = (
    "Evaluate an arithmetic expression exactly and return the result — use it"
    " for growth rates, ratios, percentages, and sums so the figures are exact"
    " and auditable, e.g. '(37-13)/13*100' for a percent change. Supports + - *"
    " / ** % // , parentheses, and abs/round/min/max/sqrt/log/exp."
)

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "log": math.log,
    "exp": math.exp,
}


def _eval(node):
  """Recursively evaluate a parsed expression, allowing only safe nodes."""
  if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
    return node.value
  if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
    return _BIN_OPS[type(node.op)](_eval(node.left), _eval(node.right))
  if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
    return _UNARY_OPS[type(node.op)](_eval(node.operand))
  if (
      isinstance(node, ast.Call)
      and isinstance(node.func, ast.Name)
      and node.func.id in _FUNCS
      and not node.keywords
  ):
    return _FUNCS[node.func.id](*(_eval(a) for a in node.args))
  raise ToolError(
      "unsupported expression — numbers, + - * / ** % //, () and "
      "abs/round/min/max/sqrt/log/exp only"
  )


def safe_eval(expression):
  """Evaluate an arithmetic expression safely (no names, calls, or attribute access

  beyond the whitelisted functions). Raises ValueError on anything unsupported.
  """
  tree = ast.parse(expression, mode="eval")
  return _eval(tree.body)


class CalcRequest(BaseModel):
  """Input for the exact arithmetic tool."""

  expression: str = Field(
      Ellipsis, description="Arithmetic expression, e.g. '(37-13)/13*100'."
  )


async def _handler(req):
  result = safe_eval(req.expression)
  # tidy float display without lying about precision
  shown = f"{result:.6g}"
  return ToolResponse(
      markdown=f"`{req.expression}` = **{shown}**",
      structured={"expression": req.expression, "result": result},
      meta={},
  )


SPEC = ToolSpec(
    name="calc",
    display_name="calc",
    tier="core",  # always visible — arithmetic comes up in every mode
    description=_DESCRIPTION,
    request_schema=CalcRequest,
    handler=_handler,
    tags=("compute", "arithmetic"),
)
