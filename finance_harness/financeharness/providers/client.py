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

"""The backbone client seam — build a client and issue chat completions.

One place constructs the OpenAI-compatible ``AsyncOpenAI`` client and assembles
the create-kwargs (model-agnostic base + profile ``extra_body`` + quirk
adapter).
Both a non-streaming :func:`complete` and a streaming :func:`stream` are
provided;
the streaming path returns an assembled ``ChatCompletion`` of the *same shape*
as
the non-streaming one, so every downstream consumer is identical either way.

The ``client`` argument is injectable so the seam is unit-testable without a
network (pass a fake exposing ``chat.completions.create`` / ``.stream``).
"""

from __future__ import annotations

from typing import Any

from financeharness.providers.adapters import apply_adapter
from financeharness.providers.base import DeltaCallback
from financeharness.providers.profiles import ModelProfile
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)


def client_for(profile):
  """Construct the async OpenAI-compatible client for a profile.

  A keyless vLLM endpoint gets the conventional ``"EMPTY"`` placeholder.
  """
  return AsyncOpenAI(
      api_key=profile.resolve_api_key() or "EMPTY",
      base_url=profile.base_url,
      timeout=profile.timeout_s,
  )


def build_create_kwargs(
    profile,
    messages,
    *,
    tools = None,
    tool_choice = "auto",
    max_tokens = None,
):
  """Assemble the chat-completion kwargs for a profile.

  Vanilla OpenAI params first (so any compatible endpoint works), then the
  profile's ``extra_body``, then the quirk adapter. ``tools`` are omitted
  entirely when empty (endpoints treat that as "no tool calling").
  """
  g = profile.generation
  # Strip "_"-prefixed history keys (opaque provider state, e.g. another provider's
  # "_native" threading items) — they're not valid wire fields for Chat Completions.
  messages = [
      {k: v for k, v in m.items() if not k.startswith("_")} for m in messages
  ]
  kwargs: dict[str, Any] = {
      "model": profile.model,
      "messages": messages,
      "temperature": g.temperature,
      "top_p": g.top_p,
      "max_tokens": max_tokens if max_tokens is not None else g.max_tokens,
      "presence_penalty": g.presence_penalty,
  }
  if tools:
    kwargs["tools"] = tools
    kwargs["tool_choice"] = tool_choice
  if profile.extra_body:
    kwargs["extra_body"] = dict(profile.extra_body)
  return apply_adapter(profile.adapter, kwargs)


async def complete(
    profile,
    messages,
    *,
    tools = None,
    tool_choice = "auto",
    max_tokens = None,
    client = None,
):
  """Issue one non-streaming chat completion and return it."""
  cl = client or client_for(profile)
  kwargs = build_create_kwargs(
      profile,
      messages,
      tools=tools,
      tool_choice=tool_choice,
      max_tokens=max_tokens,
  )
  return await cl.chat.completions.create(**kwargs)


async def stream(
    profile,
    messages,
    *,
    tools = None,
    tool_choice = "auto",
    max_tokens = None,
    on_delta = None,
    client = None,
):
  """Stream one completion, surfacing reasoning/content deltas via ``on_delta``,

  and return the assembled ``ChatCompletion`` (same shape as :func:`complete`).

  Deliberately uses the raw ``create(stream=True)`` iterator rather than the
  SDK's beta ``.stream()`` auto-parse helper: the latter rejects non-``strict``
  function tools, and our tools aren't strict. We accumulate content / reasoning
  / tool-call deltas ourselves and rebuild a ``ChatCompletion`` so every
  downstream consumer (and the recovery layer's ``length`` check) is identical
  to the non-streaming path.
  """
  cl = client or client_for(profile)
  kwargs = build_create_kwargs(
      profile,
      messages,
      tools=tools,
      tool_choice=tool_choice,
      max_tokens=max_tokens,
  )
  kwargs["stream"] = True

  content_parts: list[str] = []
  tool_calls: dict[int, dict[str, str]] = {}
  finish_reason: str | None = None
  resp_id = "stream"
  model_name = profile.model
  created = 0

  async for chunk in await cl.chat.completions.create(**kwargs):
    resp_id = chunk.id or resp_id
    model_name = chunk.model or model_name
    created = chunk.created or created
    if not chunk.choices:
      continue
    choice = chunk.choices[0]
    if choice.finish_reason:
      finish_reason = choice.finish_reason
    delta = choice.delta
    reasoning = getattr(delta, "reasoning", None) or getattr(
        delta, "reasoning_content", None
    )
    if reasoning and on_delta is not None:
      on_delta("reasoning", reasoning)
    if delta.content:
      content_parts.append(delta.content)
      if on_delta is not None:
        on_delta("content", delta.content)
    for tc in delta.tool_calls or []:
      acc = tool_calls.setdefault(
          tc.index,
          {"id": "", "name": "", "arguments": "", "extra_content": None},
      )
      if tc.id:
        acc["id"] = tc.id
      if tc.function and tc.function.name:
        acc["name"] = tc.function.name
      if tc.function and tc.function.arguments:
        acc["arguments"] += tc.function.arguments
      # Gemini emits a `thought_signature` in the tool-call's extra_content (on
      # the first delta); it MUST be echoed back on the next turn or multi-turn
      # tool calling 400s. Non-streaming carries it natively — capture it here too.
      extra = getattr(tc, "extra_content", None) or (
          getattr(tc, "model_extra", None) or {}
      ).get("extra_content")
      if extra:
        acc["extra_content"] = extra

  return _assemble_completion(
      resp_id=resp_id,
      model=model_name,
      created=created,
      content="".join(content_parts),
      tool_calls=tool_calls,
      finish_reason=finish_reason,
  )


def _assemble_completion(
    *,
    resp_id,
    model,
    created,
    content,
    tool_calls,
    finish_reason,
):
  """Rebuild a ``ChatCompletion`` from accumulated streaming deltas — the same

  shape ``complete()`` returns, so the loop, recovery, and logging are agnostic
  to whether the call streamed.
  """
  calls = [
      ChatCompletionMessageToolCall(
          id=acc["id"] or f"call_{idx}",
          type="function",
          function=Function(name=acc["name"], arguments=acc["arguments"]),
          # preserve the provider's tool-call extra_content (Gemini thought_signature);
          # the SDK model allows extras, so it lands in model_extra for echo-back.
          **(
              {"extra_content": acc["extra_content"]}
              if acc.get("extra_content")
              else {}
          ),
      )
      for idx, acc in sorted(tool_calls.items())
  ]
  message = ChatCompletionMessage(
      role="assistant",
      content=content or None,
      tool_calls=calls or None,
  )
  choice = Choice(
      index=0,
      message=message,
      # default to "stop" only if the stream ended without a reason
      finish_reason=finish_reason or "stop",  # type: ignore[arg-type]
  )
  return ChatCompletion(
      id=resp_id,
      choices=[choice],
      created=created,
      model=model,
      object="chat.completion",
  )
