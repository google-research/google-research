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

"""Bottom-up taxonomy: discover natural categories, then relabel every question.

Two phases:

1. ``discover_taxonomy`` reads sampled batches and asks Gemini for the
   natural ``topic`` and ``reasoning_method`` groupings present in the
   data. Aggregating across batches gives a label set the actual
   questions support.
2. ``classify_question`` re-labels each question against that aggregated
   taxonomy.

The aggregation step here is intentionally simpler than the embedding-
based clustering in the source monorepo — we count category names across
the discovery batches and keep the most frequent ones. The benchmark
contract is the *final* labels written onto questions, not the
clustering method.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import logging
import random
import time

from financegym.common.llm import DEFAULT_MODEL, get_client
from financegym.curation._common import llm_json

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase 1: discover natural categories from sampled batches
# ---------------------------------------------------------------------------


DISCOVER_PROMPT_TEMPLATE = """You are a financial research taxonomy expert. Analyze these
{n} research questions and identify the NATURAL categories they fall into.

QUESTIONS:
{q_text}

Your task: identify natural groupings across TWO independent axes.

AXIS 1 — TOPIC: what is the question about?
AXIS 2 — REASONING METHOD: how should it be analyzed?

For each axis, provide 6-12 mutually exclusive categories at the same level of
granularity. Output JSON:

{{
  "topic_categories": [
    {{"name": "short_name", "description": "1-sentence definition", "count": N}}
  ],
  "reasoning_categories": [
    {{"name": "short_name", "description": "1-sentence definition", "count": N}}
  ]
}}"""


def _format_questions(batch):
  lines: list[str] = []
  for i, q in enumerate(batch):
    ents = ", ".join(q.get("entities", []))
    lines.append(f"[Q{i + 1}] {q.get('question', '')}\n  Entities: {ents}")
  return "\n\n".join(lines)


def discover_taxonomy(
    questions,
    *,
    client=None,
    model = DEFAULT_MODEL,
    n_batches = 30,
    batch_size = 50,
    seed = 42,
    sleep=time.sleep,
):
  """Aggregate discovered category lists across ``n_batches`` LLM calls."""
  if not questions:
    return {"topic_categories": [], "reasoning_categories": []}

  rng = random.Random(seed)
  cli = client or get_client()

  topic_count: Counter = Counter()
  reason_count: Counter = Counter()
  topic_desc: dict[str, str] = {}
  reason_desc: dict[str, str] = {}

  for _i in range(n_batches):
    if len(questions) > batch_size:
      batch = rng.sample(questions, batch_size)
    else:
      batch = list(questions)
    prompt = DISCOVER_PROMPT_TEMPLATE.format(
        n=len(batch), q_text=_format_questions(batch)
    )
    result = llm_json(cli, model, prompt, sleep=sleep)
    if not result:
      continue
    for cat in result.get("topic_categories") or []:
      name = (cat.get("name") or "").strip().lower()
      if not name:
        continue
      topic_count[name] += 1
      topic_desc.setdefault(name, cat.get("description", ""))
    for cat in result.get("reasoning_categories") or []:
      name = (cat.get("name") or "").strip().lower()
      if not name:
        continue
      reason_count[name] += 1
      reason_desc.setdefault(name, cat.get("description", ""))

  def _aggregate(counts, descs):
    return [
        {
            "name": name,
            "description": descs.get(name, ""),
            "batches_seen": count,
        }
        for name, count in counts.most_common()
    ]

  return {
      "topic_categories": _aggregate(topic_count, topic_desc),
      "reasoning_categories": _aggregate(reason_count, reason_desc),
  }


# ---------------------------------------------------------------------------
# Phase 2: classify each question against the discovered taxonomy
# ---------------------------------------------------------------------------


CLASSIFY_PROMPT_TEMPLATE = """Pick exactly one topic and one reasoning method for this question.

QUESTION: {question}

TOPIC OPTIONS:
{topic_list}

REASONING OPTIONS:
{reasoning_list}

Return JSON:
{{
  "topic": "<one topic name from the list>",
  "reasoning_type": "<one reasoning name from the list>",
  "reason": "1 sentence"
}}"""


def _format_options(cats):
  return "\n".join(f"- {c['name']}: {c.get('description','')}" for c in cats)


def classify_question(
    question,
    taxonomy,
    *,
    client=None,
    model = DEFAULT_MODEL,
    sleep=time.sleep,
):
  """Return ``{topic, reasoning_type}`` for one question, validated against the taxonomy."""
  topic_options = taxonomy.get("topic_categories") or []
  reason_options = taxonomy.get("reasoning_categories") or []
  if not topic_options or not reason_options:
    return None
  prompt = CLASSIFY_PROMPT_TEMPLATE.format(
      question=question.get("question", ""),
      topic_list=_format_options(topic_options),
      reasoning_list=_format_options(reason_options),
  )
  result = llm_json(client or get_client(), model, prompt, sleep=sleep)
  if not result:
    return None

  valid_topics = {c["name"] for c in topic_options}
  valid_reasons = {c["name"] for c in reason_options}
  topic = (result.get("topic") or "").strip().lower()
  reasoning = (result.get("reasoning_type") or "").strip().lower()
  if topic not in valid_topics or reasoning not in valid_reasons:
    return None
  return {
      "topic": topic,
      "reasoning_type": reasoning,
      "reason": result.get("reason", ""),
  }


def apply_taxonomy(
    questions,
    taxonomy,
    *,
    client=None,
    model = DEFAULT_MODEL,
    sleep=time.sleep,
):
  """Annotate each question with ``topic`` / ``reasoning_type`` in place."""
  cli = client or get_client()
  counts: dict[str, int] = defaultdict(int)
  for q in questions:
    cls = classify_question(q, taxonomy, client=cli, model=model, sleep=sleep)
    if cls:
      q["topic"] = cls["topic"]
      q["reasoning_type"] = cls["reasoning_type"]
      counts["classified"] += 1
    else:
      counts["unclassified"] += 1
  return questions, dict(counts)
