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

"""vLLM open-weight quirk adapter.

Some open-weight models served via vLLM enable "thinking mode" per-request
through the chat template and recommend a ``top_k`` alongside the
temperature/top_p defaults. vLLM accepts both through ``extra_body``. Kept off
the vanilla path so a hosted OpenAI-compatible call never sends these.
"""

from __future__ import annotations

import copy
from typing import Any

_DEFAULT_TOP_K = 20


def vllm_adapter(kwargs):
  """Inject ``top_k`` and ``enable_thinking`` into ``extra_body`` without

  clobbering any caller-provided extra_body keys. Pure (returns a new dict).
  """
  out = copy.deepcopy(kwargs)
  extra_body = dict(out.get("extra_body") or {})
  extra_body.setdefault("top_k", _DEFAULT_TOP_K)
  chat_template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
  chat_template_kwargs.setdefault("enable_thinking", True)
  extra_body["chat_template_kwargs"] = chat_template_kwargs
  out["extra_body"] = extra_body
  return out
