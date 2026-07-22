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

"""Quirk adapters — per-model transforms of the chat-completion kwargs.

Most backbones need none (the vanilla kwargs work). An adapter exists for the
handful of dialects that differ: some open-weight models served via vLLM want
``enable_thinking`` via ``chat_template_kwargs`` and a ``top_k``.

An adapter is ``(kwargs: dict) -> dict`` and must not mutate its input. Selected
by ``ModelProfile.adapter``; ``None`` → identity.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from financeharness.providers.adapters.vllm import vllm_adapter

Adapter = Callable[[dict[str, Any]], dict[str, Any]]

_ADAPTERS: dict[str, Adapter] = {
    "vllm": vllm_adapter,
}


def apply_adapter(name, kwargs):
  """Apply the named adapter to ``kwargs``; identity when name is falsy or

  unknown (an unknown name degrades gracefully rather than failing the call).
  """
  if not name:
    return kwargs
  adapter = _ADAPTERS.get(name)
  return adapter(kwargs) if adapter is not None else kwargs


__all__ = ["Adapter", "apply_adapter", "vllm_adapter"]
