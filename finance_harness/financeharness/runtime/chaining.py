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

"""Reference-based chaining — resolve ``prev:<call_id>.<path>`` arg strings.

When a tool argument is ``prev:<call_id>.<path>``, the runtime resolves it
against the structured payload of a prior call (recorded in
``ToolSessionState.tool_results``) *before* the consumer's schema validates — so
a compute tool can consume the full series from a prior data call without
round-tripping bulk data through the LLM's context.

Path syntax (JSONPath-lite): ``prev:<id>`` (whole dict), ``.field`` (dict key),
``[N]`` (index), ``[*]`` (map over a list); tokens compose. Filters, recursive
descent, and nested ``[*]`` are deliberately unsupported (YAGNI). Never raises —
failures return a structured :class:`ChainError` the dispatcher turns into an
``ok=False`` result the model self-corrects on.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

_REF_PREFIX = "prev:"


@dataclass(frozen=True)
class ChainError:
  """A single resolution failure with enough context for the model to fix it."""

  reference: str
  reason: str


def is_reference(value):
  """True when ``value`` is a reference-chaining string."""

  return isinstance(value, str) and value.startswith(_REF_PREFIX)


_TOKEN_RE = re.compile(r"^(?P<key>[^.\[\]]+)(?P<indexers>(?:\[[^\]]*\])*)$")
_INDEXER_RE = re.compile(r"\[([^\]]*)\]")


@dataclass(frozen=True)
class _Step:
  key: str
  indexers: tuple[str, Ellipsis]  # each "*" or a decimal int as str


@dataclass(frozen=True)
class _ParsedRef:
  call_id: str
  steps: tuple[_Step, Ellipsis]
  raw: str


def _parse_reference(raw):
  if not raw.startswith(_REF_PREFIX):
    return ChainError(raw, "reference must start with 'prev:'")
  body = raw[len(_REF_PREFIX) :]
  if not body:
    return ChainError(raw, "missing call_id after 'prev:'")
  call_id, path = body.split(".", 1) if "." in body else (body, "")
  if not call_id:
    return ChainError(raw, "empty call_id between 'prev:' and '.'")
  if not path:
    return _ParsedRef(call_id=call_id, steps=(), raw=raw)

  steps: list[_Step] = []
  for tok in path.split("."):
    if not tok:
      return ChainError(raw, "empty path segment (stray '.'?)")
    m = _TOKEN_RE.match(tok)
    if not m:
      return ChainError(raw, f"could not parse path segment {tok!r}")
    indexers: list[str] = []
    for im in _INDEXER_RE.finditer(m.group("indexers") or ""):
      inner = im.group(1)
      if inner == "*":
        indexers.append("*")
      elif inner.lstrip("-").isdigit():
        indexers.append(inner)
      else:
        return ChainError(
            raw,
            f"unsupported indexer '[{inner}]' (only [N] index and [*] wildcard"
            " supported)",
        )
    steps.append(_Step(key=m.group("key"), indexers=tuple(indexers)))
  return _ParsedRef(call_id=call_id, steps=tuple(steps), raw=raw)


def _list_dict_keys(obj):
  return sorted(map(str, obj.keys())) if isinstance(obj, dict) else []


def _descend_key(cur, key, raw):
  if not isinstance(cur, dict):
    return ChainError(
        raw, f"field '{key}' — expected dict, got {type(cur).__name__}"
    )
  if key in cur:
    return cur[key]
  if key.lstrip("-").isdigit() and int(key) in cur:  # int-typed dict keys
    return cur[int(key)]
  return ChainError(
      raw,
      f"no field '{key}' on resolved object (available:"
      f" {_list_dict_keys(cur)})",
  )


def _index(cur, ix, key, raw):
  if not isinstance(cur, list):
    return ChainError(
        raw, f"'[{ix}]' on '{key}' requires a list, got {type(cur).__name__}"
    )
  idx = int(ix)
  if idx < -len(cur) or idx >= len(cur):
    return ChainError(
        raw, f"'[{ix}]' on '{key}' — index out of range (length {len(cur)})"
    )
  return cur[idx]


def _walk(obj, steps, raw):
  cur: Any = obj
  for step_i, step in enumerate(steps):
    cur = _descend_key(cur, step.key, raw)
    if isinstance(cur, ChainError):
      return cur
    for pos, ix in enumerate(step.indexers):
      if ix == "*":
        if not isinstance(cur, list):
          return ChainError(
              raw,
              f"'[*]' on '{step.key}' requires a list, got"
              f" {type(cur).__name__}",
          )
        remaining = step.indexers[pos + 1 :]
        if "*" in remaining:
          return ChainError(raw, "nested '[*]' wildcards are not supported")
        tail = (_Step(key="__tail__", indexers=remaining), *steps[step_i + 1 :])
        out: list[Any] = []
        for elem in cur:
          sub = _apply_tail(elem, tail, raw)
          if isinstance(sub, ChainError):
            return sub
          out.append(sub)
        return out
      cur = _index(cur, ix, step.key, raw)
      if isinstance(cur, ChainError):
        return cur
  return cur


def _apply_tail(
    item, tail_steps, raw
):
  cur: Any = item
  head, *rest = tail_steps
  for ix in head.indexers:
    if ix == "*":
      return ChainError(raw, "nested '[*]' wildcards are not supported")
    cur = _index(cur, ix, "[*]", raw)
    if isinstance(cur, ChainError):
      return cur
  return _walk(cur, tuple(rest), raw) if rest else cur


def resolve_one(
    ref, tool_results
):
  """Resolve a single ``prev:`` reference against the chain map."""
  parsed = _parse_reference(ref)
  if isinstance(parsed, ChainError):
    return parsed
  if parsed.call_id not in tool_results:
    return ChainError(
        ref,
        f"no prior tool call with id '{parsed.call_id}' in this session "
        f"(known: {sorted(tool_results.keys())})",
    )
  return _walk(tool_results[parsed.call_id], parsed.steps, ref)


def resolve_references(
    args, tool_results
):
  """Recursively substitute every ``prev:`` string in ``args``.

  Returns ``(resolved_args, errors)``; a non-empty ``errors`` means the
  dispatcher should short-circuit before validation.
  """
  errors: list[ChainError] = []

  def _walk_args(node):
    if is_reference(node):
      result = resolve_one(node, tool_results)
      if isinstance(result, ChainError):
        errors.append(result)
        return node
      return result
    if isinstance(node, dict):
      return {k: _walk_args(v) for k, v in node.items()}
    if isinstance(node, list):
      return [_walk_args(v) for v in node]
    return node

  return _walk_args(args), errors


def format_chain_errors(errors):
  """Render ChainErrors as one actionable markdown block for an ok=False result."""
  if not errors:
    return ""
  lines = ["chaining failed — could not resolve reference(s):"]
  lines.extend(f"  {e.reference}: {e.reason}" for e in errors)
  return "\n".join(lines)
