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

"""Argument coercion — repair model-emitted args before validation.

A boundary concern at the dispatcher: some models emit a *structured* argument
(list/dict/nested model) as a JSON-encoded **string**, or wrap a lone ``prev:``
reference in a 1-element list. Both repairs here are **schema-driven** (the
request schema is the source of truth for which fields are structured) and run
before chaining + Pydantic validation. Neither raises — an unrepairable value is
left as-is and Pydantic surfaces the path-aware error the model self-corrects
on.

(The aggressive flat-list/URL coercions for the no-schema research tools live
with those tools, not here — this module is the registry path only.)
"""

from __future__ import annotations

import json
import types
import typing
from typing import Any, Union

from pydantic import BaseModel

_CONTAINER_ORIGINS = (list, dict, tuple, set, frozenset)
# `X | None` → types.UnionType; `Optional[X]`/`Union[...]` → typing.Union.
_UNION_ORIGINS = tuple(
    o for o in (Union, getattr(types, "UnionType", None)) if o is not None
)
_REF_PREFIX = "prev:"


def _expects_container(annotation):
  """True when a field's declared type is a structured container the model

  might have JSON-stringified — list/dict/tuple/set (optionally
  Optional/Union-wrapped) or a nested Pydantic model.
  """
  origin = typing.get_origin(annotation)
  if origin in _UNION_ORIGINS:
    return any(
        _expects_container(arg)
        for arg in typing.get_args(annotation)
        if arg is not type(None)
    )
  if origin in _CONTAINER_ORIGINS or annotation in _CONTAINER_ORIGINS:
    return True
  return isinstance(annotation, type) and issubclass(annotation, BaseModel)


def _is_list_of_scalar(annotation):
  """True when ``annotation`` is ``list[<scalar>]`` (NOT ``list[list[...]]`` /

  ``list[dict[...]]`` / ``list[Model]``), including Optional/Union wrappers.
  Decides whether a ``[<prev: ref>]`` wrapper should be unwrapped.
  """
  origin = typing.get_origin(annotation)
  if origin in _UNION_ORIGINS:
    return any(
        _is_list_of_scalar(arg)
        for arg in typing.get_args(annotation)
        if arg is not type(None)
    )
  if not (origin is list or annotation is list):
    return False
  inner_args = typing.get_args(annotation)
  if not inner_args:
    return False  # bare `list` — can't introspect; be conservative
  inner = inner_args[0]
  inner_origin = typing.get_origin(inner)
  if inner_origin in _CONTAINER_ORIGINS or inner in _CONTAINER_ORIGINS:
    return False
  if isinstance(inner, type) and issubclass(inner, BaseModel):
    return False
  if inner_origin in _UNION_ORIGINS:
    return all(
        not (
            typing.get_origin(a) in _CONTAINER_ORIGINS
            or a in _CONTAINER_ORIGINS
        )
        for a in typing.get_args(inner)
        if a is not type(None)
    )
  return True


def coerce_stringified_json(args, schema):
  """Parse stringified-JSON structured fields back to containers, driven by

  ``schema``. Only string values destined for container-typed fields and
  starting with ``[`` or ``{`` are candidates; scalars and non-JSON strings
  pass through untouched. Returns a copy.
  """
  if not isinstance(args, dict):
    return args
  fields = getattr(schema, "model_fields", None)
  if not fields:
    return args
  out = dict(args)
  for name, field_info in fields.items():
    value = out.get(name)
    if name not in out or not isinstance(value, str):
      continue
    if not _expects_container(field_info.annotation):
      continue
    s = value.strip()
    if not s or s[0] not in "[{":
      continue
    try:
      out[name] = json.loads(s)
    except (json.JSONDecodeError, ValueError):
      continue  # leave as-is; Pydantic surfaces the error
  return out


def coerce_wrapped_ref(args, schema):
  """Unwrap ``[<single prev: ref>]`` to the bare ref for ``list[<scalar>]``

  fields, so the chaining resolver expands it once into a ``list[float]``
  instead of element-by-element into ``list[list[float]]``. Schema-aware: a
  length-1 list under ``list[list[X]]`` stays wrapped. Returns a copy.
  """
  if not isinstance(args, dict):
    return args
  fields = getattr(schema, "model_fields", None)
  if not fields:
    return args
  out = dict(args)
  for name, field_info in fields.items():
    value = out.get(name)
    if name not in out or not isinstance(value, list) or len(value) != 1:
      continue
    elem = value[0]
    if not (isinstance(elem, str) and elem.startswith(_REF_PREFIX)):
      continue
    if _is_list_of_scalar(field_info.annotation):
      out[name] = elem
  return out
