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

"""Tool dispatch — the never-raise bridge from a tool_call to a tool_result.

Every tool invocation flows through :func:`dispatch`: coerce args → resolve
reference chaining → validate against the Pydantic schema → run the handler →
format the :class:`ToolResponse`. It **always** returns a
:class:`DispatchResult` and never
raises; every error becomes an actionable ``ok=False`` markdown the model
self-corrects from on the next turn.
"""

from __future__ import annotations

import json
import time
from typing import Any

from financeharness.runtime.arg_coercion import coerce_stringified_json, coerce_wrapped_ref
from financeharness.runtime.chaining import format_chain_errors, resolve_references
from financeharness.runtime.tool_events import ToolEmit, tool_event_scope
from financeharness.runtime.tool_registry import (
    ToolError,
    ToolRegistry,
    ToolResponse,
    ToolSessionState,
)
from pydantic import ValidationError

_MAX_VALIDATION_ISSUES = 5


def _append_call_id_footer(markdown, call_id):
  """Soft footer so the model can reference this call's structured payload

  downstream via the chaining syntax. No-op without a call_id.
  """
  if not call_id:
    return markdown
  return (
      markdown
      + f"\n\n_call_id: {call_id} — reference downstream via"
      f" prev:{call_id}.<path>_"
  )


def _format_validation_error(err):
  """Compact, path-aware validation message the model can act on."""
  issues = err.errors()
  lines = []
  for issue in issues[:_MAX_VALIDATION_ISSUES]:
    loc = ".".join(str(p) for p in issue.get("loc", ())) or "(root)"
    msg = issue.get("msg", "invalid")
    kind = issue.get("type", "")
    lines.append(f"  {loc}: {msg}" + (f" [{kind}]" if kind else ""))
  if len(issues) > _MAX_VALIDATION_ISSUES:
    lines.append(f"  …and {len(issues) - _MAX_VALIDATION_ISSUES} more")
  return "schema validation failed:\n" + "\n".join(lines)


class DispatchResult:
  """Carries the model-facing markdown plus trajectory-side structured fields."""

  def __init__(
      self,
      *,
      name,
      args,
      ok,
      markdown,
      structured,
      meta,
  ):
    self.name = name
    self.args = args
    self.ok = ok
    self.markdown = markdown
    self.structured = structured
    self.meta = meta

  def for_model(self):
    """The tool_result content the model sees."""
    return self.markdown

  def to_log(self):
    """The trajectory tool_log entry."""
    return {
        "name": self.name,
        "args": self.args,
        "ok": self.ok,
        "markdown_chars": len(self.markdown),
        "structured": self.structured,
        "meta": self.meta,
    }


def _error_result(
    name, args, markdown, error, **meta
):
  return DispatchResult(
      name=name,
      args=args,
      ok=False,
      markdown=markdown,
      structured={"error": error},
      meta=meta,
  )


async def dispatch(
    name,
    args,
    *,
    registry,
    session_state = None,
    call_id = None,
    emit = None,
):
  """Execute a registered tool by name. Always returns a DispatchResult.

  ``session_state``/``call_id`` drive reference chaining and the call-id footer.
  """
  spec = registry.get(name)
  if spec is None:
    return _error_result(
        name,
        args,
        (
            f"Unknown tool: '{name}'. Available core tools: "
            f"{sorted(s.name for s in registry.core_tools())}. "
            "For deferred tools, call load_tool with the names you need first."
        ),
        "unknown_tool",
    )

  # 1. Argument coercion (schema-driven; repairs stringified containers and
  #    wrapped lone refs). Runs before validation.
  coerced = coerce_stringified_json(args, spec.request_schema)
  coerced = coerce_wrapped_ref(coerced, spec.request_schema)

  # 2. Reference-chaining: resolve prev:<call_id>.<path> against prior
  #    structured payloads, before validation, so the schema sees real values.
  if session_state is not None:
    resolved, chain_errors = resolve_references(
        coerced, session_state.tool_results
    )
    if chain_errors:
      return DispatchResult(
          name=name,
          args=args,
          ok=False,
          markdown=format_chain_errors(chain_errors),
          structured={
              "error": "chain_resolution",
              "failures": [
                  {"reference": e.reference, "reason": e.reason}
                  for e in chain_errors
              ],
          },
          meta={},
      )
  else:
    resolved = coerced

  # 3. Schema validation.
  try:
    validated = spec.request_schema.model_validate(resolved)
  except ValidationError as err:
    return _error_result(
        name, args, _format_validation_error(err), "schema_validation"
    )

  # 4. Handler execution (the handler owns provider-specific retry; the
  #    dispatcher is the never-raise backstop).
  t0 = time.time()
  try:
    with tool_event_scope(emit, name, call_id):
      response: ToolResponse = await spec.handler(validated)
  except ValidationError as err:
    return _error_result(
        name,
        args,
        f"tool handler returned an invalid response: {err}",
        "handler_response_invalid",
        elapsed_ms=int((time.time() - t0) * 1000),
    )
  except (
      ToolError
  ) as err:  # expected, actionable failure → clean message, no prefix
    return _error_result(
        name,
        args,
        str(err),
        "tool_error",
        elapsed_ms=int((time.time() - t0) * 1000),
    )
  except Exception as err:  # noqa: BLE001 — never-raise: surface to the model
    return _error_result(
        name,
        args,
        f"{type(err).__name__}: {err}",
        "handler_exception",
        exception=type(err).__name__,
        elapsed_ms=int((time.time() - t0) * 1000),
    )

  meta = dict(response.meta)
  meta.setdefault("elapsed_ms", int((time.time() - t0) * 1000))

  # Record this call's structured payload so later calls can chain to it.
  if session_state is not None and call_id:
    session_state.tool_results[call_id] = response.structured

  return DispatchResult(
      name=name,
      args=args,
      ok=True,
      markdown=_append_call_id_footer(response.markdown, call_id),
      structured=response.structured,
      meta=meta,
  )


async def dispatch_json_args(
    name,
    args_json,
    *,
    registry,
    session_state = None,
    call_id = None,
    emit = None,
):
  """Dispatch a tool whose args arrive as a JSON string (the form the

  function-calling parser produces).
  """
  try:
    args = json.loads(args_json) if args_json else {}
  except json.JSONDecodeError as err:
    return _error_result(
        name,
        {"_raw": args_json},
        f"could not parse arguments as JSON: {err}",
        "args_not_json",
    )
  return await dispatch(
      name,
      args,
      registry=registry,
      session_state=session_state,
      call_id=call_id,
      emit=emit,
  )
