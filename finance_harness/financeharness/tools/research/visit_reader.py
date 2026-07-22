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

"""Reader-model extraction helpers for the `visit` tool."""

from __future__ import annotations

import re
from typing import Any

from financeharness.providers import ModelProfile, complete
from financeharness.runtime.config import RuntimeConfig
from financeharness.runtime.jsonutil import extract_json_obj
from openai import AsyncOpenAI

_READER_CHAR_BUDGET = 24000

EXTRACTOR_PROMPT = """\
You extract grounded content from a fetched webpage for a research agent. Return \
ONLY a JSON object with three fields:
  "accessible": true if the text is real article/report content; false if it is a \
paywall, login/bot wall, cookie/consent page, error page, or essentially empty.
  "summary": a faithful, information-dense summary of the page's substantive \
content (figures, facts, claims), using the page's own numbers. Empty string if \
not accessible.
  "evidence": one or two short verbatim quotes that support the key facts. Empty \
string if not accessible.
Report only what the page actually states. If it carries no real content (paywall, \
bot-wall, error, or essentially empty), set accessible=false with empty summary/evidence."""


def _salvage_reader_fields(raw):
  """Pull reader fields from JSON-shaped text the parser could not load."""

  def field(name):
    m = re.search(rf'"{name}"\s*:\s*"((?:[^"\\]|\\.)*)"', raw, re.DOTALL)
    if not m:
      return ""
    return (
        m.group(1)
        .replace('\\"', '"')
        .replace("\\n", " ")
        .replace("\\t", " ")
        .strip()
    )

  summary, evidence = field("summary"), field("evidence")
  acc = re.search(r'"accessible"\s*:\s*(true|false)', raw)
  accessible = acc.group(1) == "true" if acc else bool(summary)
  return {"accessible": accessible, "summary": summary, "evidence": evidence}


async def read_page(
    profile,
    text,
    *,
    goal,
    client,
    config,
):
  """Run the reader-model extraction over page text."""
  snippet = text[:_READER_CHAR_BUDGET]
  user = (
      f"Research goal: {goal}\n\n" if goal else ""
  ) + f"Webpage content:\n\n{snippet}"
  try:
    resp = await complete(
        profile,
        [
            {"role": "system", "content": EXTRACTOR_PROMPT},
            {"role": "user", "content": user},
        ],
        max_tokens=config.reader.max_tokens,
        client=client,
    )
    raw = resp.choices[0].message.content or ""
  except Exception as exc:  # noqa: BLE001 — reader/transport failure, not page inaccessibility
    return {
        "accessible": False,
        "summary": "",
        "evidence": "",
        "reader_error": str(exc),
    }

  obj = extract_json_obj(raw)
  if obj is None:
    if '"summary"' in raw or raw.lstrip().startswith("{"):
      salvaged = _salvage_reader_fields(raw)
      if salvaged["summary"]:
        return salvaged
      return {"accessible": False, "summary": "", "evidence": ""}
    return {
        "accessible": bool(raw.strip()),
        "summary": raw.strip(),
        "evidence": "",
    }
  return {
      "accessible": bool(obj.get("accessible", True)),
      "summary": str(obj.get("summary", "")).strip(),
      "evidence": str(obj.get("evidence", "")).strip(),
  }
