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

"""Shared LLM call wrapper for the curation pipeline.

Every curation filter follows the same shape: one Gemini call per question,
in JSON mode, with rate-limit-aware backoff. Pulling that into a single
helper keeps each filter focused on its prompt and the score it produces.
"""

from __future__ import annotations

import json
import time

from google.genai import types

MAX_RETRIES = 4
RETRY_BASE_DELAY = 2.0
TRANSIENT_MARKERS = (
    "429",
    "503",
    "RESOURCE_EXHAUSTED",
    "UNAVAILABLE",
    "overloaded",
)


def llm_json(
    client,
    model,
    prompt,
    *,
    max_retries = MAX_RETRIES,
    sleep=time.sleep,
):
  """Call Gemini in JSON mode and return the parsed dict (or ``None``)."""
  config = types.GenerateContentConfig(
      response_mime_type="application/json",
      automatic_function_calling=types.AutomaticFunctionCallingConfig(
          disable=True
      ),
  )
  for attempt in range(max_retries + 1):
    try:
      resp = client.models.generate_content(
          model=model, contents=prompt, config=config
      )
      if not resp.text:
        return None
      result = json.loads(resp.text)
      if isinstance(result, list):
        result = result[0] if result else {}
      return result if isinstance(result, dict) else None
    except Exception as e:  # noqa: BLE001
      err = str(e)
      if any(m in err for m in TRANSIENT_MARKERS) and attempt < max_retries:
        sleep(RETRY_BASE_DELAY * (2**attempt))
        continue
      return None
  return None
