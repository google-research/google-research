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

"""Gemini structured-output client used at every LLM call site in FinanceGym.

The public surface is small on purpose:

* :func:`get_client` returns a configured ``google.genai.Client``.
* :func:`generate` returns plain text.
* :func:`generate_structured` returns a parsed pydantic model.
* :func:`generate_with_retry` is the lower-level escape hatch for callers that
  need to pass a fully-built ``GenerateContentConfig``.

All paths share one exponential-backoff retry loop that re-tries the
transient classes Gemini surfaces (429, 503, RESOURCE_EXHAUSTED,
UNAVAILABLE, DEADLINE_EXCEEDED, network errors). Retry budget can be
tuned per-process via env vars:

* ``GEMINI_MAX_ATTEMPTS`` (default 6)
* ``GEMINI_BASE_DELAY``   (default 2.0 s)
* ``GEMINI_MAX_DELAY``    (default 64.0 s)

Callers must propagate exceptions — never catch and return the error as
the model's "answer". An outer retry layer needs the real exception to
decide what to do.

The model is configurable via the ``FINANCEGYM_MODEL`` environment variable
(or a ``model=...`` argument to any call site); it is not pinned to a specific
version. For reproducibility, set ``FINANCEGYM_MODEL`` to a fixed model id.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from google import genai
from google.genai import types

log = logging.getLogger(__name__)

# Configurable; defaults to a current Gemini alias rather than a pinned version.
DEFAULT_MODEL = os.environ.get("FINANCEGYM_MODEL", "gemini-flash-latest")

_TRANSIENT_MARKERS = (
    "429",
    "503",
    "RESOURCE_EXHAUSTED",
    "UNAVAILABLE",
    "DEADLINE_EXCEEDED",
    "timeout",
    "ConnectionError",
)


def get_client():
  """Construct a ``google.genai.Client``. Reads ``GOOGLE_API_KEY`` from env."""
  return genai.Client()


def generate_with_retry(
    client,
    *,
    model,
    contents,
    config = None,
    max_attempts = None,
    base_delay = None,
    max_delay = None,
):
  """Call ``client.models.generate_content`` with exponential-backoff retry.

  Re-tries only on transient markers; raises immediately on anything else,
  and re-raises the last exception after the budget is exhausted so the
  caller can decide how to handle it.
  """
  if max_attempts is None:
    max_attempts = int(os.environ.get("GEMINI_MAX_ATTEMPTS", "6"))
  if base_delay is None:
    base_delay = float(os.environ.get("GEMINI_BASE_DELAY", "2.0"))
  if max_delay is None:
    max_delay = float(os.environ.get("GEMINI_MAX_DELAY", "64.0"))

  last_exc: BaseException | None = None
  for attempt in range(max_attempts):
    try:
      return client.models.generate_content(
          model=model, contents=contents, config=config
      )
    except Exception as e:  # noqa: BLE001 — re-raised below
      err = str(e)
      transient = any(t in err for t in _TRANSIENT_MARKERS)
      last_exc = e
      if not transient or attempt + 1 >= max_attempts:
        break
      delay = min(base_delay * (2**attempt), max_delay)
      log.warning(
          "generate_content transient failure (attempt %d/%d): %s. Retrying in"
          " %.0fs",
          attempt + 1,
          max_attempts,
          err[:120],
          delay,
      )
      time.sleep(delay)
  assert last_exc is not None
  raise last_exc


def generate(
    prompt,
    system_instruction = None,
    model = DEFAULT_MODEL,
):
  """One-shot plain-text generation. Returns ``response.text``."""
  client = get_client()
  config = (
      types.GenerateContentConfig(system_instruction=system_instruction)
      if system_instruction
      else None
  )
  response = generate_with_retry(
      client, model=model, contents=prompt, config=config
  )
  return response.text


def generate_structured(
    prompt,
    response_schema,
    system_instruction = None,
    model = DEFAULT_MODEL,
):
  """Structured JSON generation; returns ``response.parsed`` (a pydantic model)."""
  client = get_client()
  config = types.GenerateContentConfig(
      response_mime_type="application/json",
      response_schema=response_schema,
      system_instruction=system_instruction,
  )
  response = generate_with_retry(
      client, model=model, contents=prompt, config=config
  )
  return response.parsed
