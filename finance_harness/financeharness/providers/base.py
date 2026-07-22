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

"""Provider abstraction — a uniform turn interface over heterogeneous backbones.

Each backbone speaks a different wire protocol (OpenAI-compatible Chat
Completions,
Gemini's native SDK). A :class:`Provider` hides that: it
maps the normalized chat history to its native request, issues the call
(streaming
reasoning/content deltas via ``on_delta`` when one is given), and returns a
single
:class:`AssistantTurn`. The loop, recovery, and history threading are written
once
against these normalized types and never touch a provider's native shape.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from financeharness.providers.profiles import ModelProfile

# on_delta(kind, text) with kind in {"reasoning", "content"} — the streaming seam.
DeltaCallback = Callable[[str, str], None]


@dataclass(frozen=True)
class ToolCall:
  """A normalized tool call.

  ``arguments`` is the raw JSON string the model emitted (parsed at dispatch).
  ``extra`` carries opaque provider state that must round-trip — e.g. Gemini's
  ``thought_signature`` — echoed back verbatim on the next turn or multi-turn
  tool calling fails.
  """

  id: str
  name: str
  arguments: str
  extra: dict[str, Any] | None = None


@dataclass
class AssistantTurn:
  """One assistant turn, normalized across providers: the visible content, any tool

  calls, and why generation stopped. (Reasoning is surfaced live via the
  on_delta
  streaming seam, not carried on the turn.)
  """

  content: str = ""
  tool_calls: list[ToolCall] = field(default_factory=list)
  finish_reason: str = "stop"
  # Opaque, provider-native turn items threaded back verbatim on the next request
  # — e.g. the OpenAI Responses reasoning items (encrypted) that must round-trip
  # statelessly. Carried in the history under "_native"; only the provider that
  # produced it reads it (the wire layer strips "_"-prefixed keys for everyone else).
  native: list[dict[str, Any]] | None = None

  def to_message(self):
    """Render this turn as a chat-history message for the next request."""
    msg: dict[str, Any] = {"role": "assistant", "content": self.content or ""}
    if self.tool_calls:
      msg["tool_calls"] = [
          {
              "id": tc.id,
              "type": "function",
              "function": {"name": tc.name, "arguments": tc.arguments},
              **({"extra_content": tc.extra} if tc.extra else {}),
          }
          for tc in self.tool_calls
      ]
    if self.native is not None:
      msg["_native"] = self.native
    return msg


class Provider(ABC):
  """A model backbone behind the normalized turn interface."""

  def __init__(self, profile, client = None):
    self.profile = profile
    self.client = client

  @abstractmethod
  async def generate(
      self,
      messages,
      *,
      tools = None,
      max_tokens = None,
      on_delta = None,
  ):
    """Issue one turn.

    Streams deltas via ``on_delta`` when given (else a single non-streaming
    call), and returns the assembled :class:`AssistantTurn`. Tool choice is
    always the provider default (``auto``).
    """
    Ellipsis
