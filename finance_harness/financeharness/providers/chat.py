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

"""The OpenAI-compatible Chat Completions provider.

The native interface for the local vLLM backbone, and the path any
OpenAI-compatible
endpoint speaks. Wraps the :mod:`client` seam — which builds the request,
applies the
quirk adapter, and assembles streamed deltas into a ``ChatCompletion`` — and
normalizes the result into an :class:`AssistantTurn`.
"""

from __future__ import annotations

from typing import Any

from financeharness.providers.base import AssistantTurn, DeltaCallback, Provider, ToolCall
from financeharness.providers.client import complete, stream
from openai.types.chat import ChatCompletion


def turn_from_completion(cc):
  """Normalize a ``ChatCompletion`` (streamed or not) into an AssistantTurn."""
  if not cc.choices:
    return AssistantTurn(finish_reason="error")
  choice = cc.choices[0]
  msg = choice.message
  calls: list[ToolCall] = []
  for tc in getattr(msg, "tool_calls", None) or []:
    # Gemini rides a thought_signature on the tool call's extra_content; it must
    # round-trip or multi-turn tool calling 400s. Absent for other providers.
    extra = getattr(tc, "extra_content", None) or (
        getattr(tc, "model_extra", None) or {}
    ).get("extra_content")
    calls.append(
        ToolCall(
            id=tc.id,
            name=tc.function.name,
            arguments=tc.function.arguments,
            extra=extra or None,
        )
    )
  return AssistantTurn(
      content=msg.content or "",
      tool_calls=calls,
      finish_reason=choice.finish_reason or "stop",
  )


class ChatCompletionsProvider(Provider):
  """vLLM / OpenAI-compatible Chat Completions backbone."""

  async def generate(
      self,
      messages,
      *,
      tools = None,
      max_tokens = None,
      on_delta = None,
  ):
    kwargs: dict[str, Any] = {
        "tools": tools,
        "max_tokens": max_tokens,
        "client": self.client,
    }
    if on_delta is None:
      cc = await complete(self.profile, messages, **kwargs)
    else:
      cc = await stream(self.profile, messages, on_delta=on_delta, **kwargs)
    return turn_from_completion(cc)
