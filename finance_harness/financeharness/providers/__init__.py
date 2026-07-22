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

"""The model-agnostic backbone seam.

The loop talks to one normalized interface — :class:`Provider`, returning an
:class:`AssistantTurn` — and ``provider_for(profile)`` builds the right one for
a
:class:`ModelProfile`. Each provider uses its backbone's most native path:
``chat`` (OpenAI-compatible Chat Completions — works with vLLM / any compatible
endpoint + the page-reader) and ``gemini`` (the native google-genai SDK —
structured thoughts + thought_signature). Both surface reasoning/content deltas
through the same seam.

The lower-level ``complete()`` / ``stream()`` Chat Completions functions remain
for
the lightweight extraction paths (page reader, clarify, summarize).
Provider-specific Chat Completions quirks live in pluggable adapters.
"""

from typing import Any

from financeharness.providers.base import AssistantTurn, Provider, ToolCall
from financeharness.providers.client import build_create_kwargs, client_for, complete, stream
from financeharness.providers.profiles import (
    Generation,
    ModelProfile,
    available_backbones,
    get_profile,
    load_profiles,
)


def provider_for(profile, client = None):
  """Build the :class:`Provider` for a profile, dispatching on ``profile.provider``:

  ``gemini`` → native google-genai, else the OpenAI-compatible Chat Completions
  provider (works with vLLM / any compatible endpoint + the page reader).
  """
  if profile.provider == "gemini":
    from financeharness.providers.gemini import GeminiProvider

    return GeminiProvider(profile, client)
  from financeharness.providers.chat import ChatCompletionsProvider

  return ChatCompletionsProvider(profile, client)


__all__ = [
    "AssistantTurn",
    "Generation",
    "ModelProfile",
    "Provider",
    "ToolCall",
    "available_backbones",
    "build_create_kwargs",
    "client_for",
    "complete",
    "get_profile",
    "load_profiles",
    "provider_for",
    "stream",
]
