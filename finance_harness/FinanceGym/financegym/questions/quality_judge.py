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

"""Five-dimension quality judge for generated questions.

The judge scores a question on naturalness, analytical depth, relevance,
temporal design, and rubric quality (each 1–5), then accepts if the
average is at or above the threshold, no dimension is 1, and naturalness
is at least 3. The prompt is pinned in :data:`JUDGE_PROMPT_TEMPLATE` and
is part of the benchmark contract.
"""

from __future__ import annotations

import logging
import time

from financegym.common.llm import DEFAULT_MODEL, get_client
from financegym.questions.generate import _llm_json_call

log = logging.getLogger(__name__)


DEFAULT_MIN_AVERAGE = 3.8
MIN_NATURALNESS = 3
NO_ONES = 1


JUDGE_PROMPT_TEMPLATE = """You are reviewing a financial research question for a benchmark.
Score on these dimensions (1-5 each):

QUESTION: {question}
ENTITIES: {entities}
CUTOFF: {cutoff}

RUBRIC:
{rubric}

1. NATURALNESS — Does this sound like a real analyst would type it?
   5: Copy-paste from analyst workflow. Focused, clear, one thread.
   3: Reasonable but slightly academic.
   1: Sounds like a test question. Forced entities, excessive jargon.

2. ANALYTICAL DEPTH — Requires genuine multi-source synthesis?
   5: Needs 3+ data points from different sources.
   3: Needs 2 sources, straightforward.
   1: Single-fact lookup.

3. RELEVANCE — Would a financial professional care?
   5: Directly affects investment decisions. Timely, specific.
   3: Interesting background, not urgent.
   1: Trivial or too obscure.

4. TEMPORAL DESIGN — Antecedents pre-cutoff, consequents post-cutoff?
   5: Clean separation, consequents are real post-cutoff developments.
   3: Mostly correct, some static facts in consequents.
   1: No temporal structure.

5. RUBRIC QUALITY — Objectively verifiable criteria?
   5: Every item names specific entities, metrics, dates.
   3: Mix of specific and vague.
   1: Generic or unmeasurable.

OUTPUT JSON:
{{"scores": {{"naturalness": N, "depth": N, "relevance": N, "temporal": N, "rubric": N}},
 "average": N.N,
 "verdict": "accept/reject",
 "reasoning": "1 sentence"}}

accept if: average >= {threshold}, no dim == 1, naturalness >= 3
reject otherwise"""


def _rubric_to_text(rubric):
  out_lines: list[str] = []
  for item in rubric or []:
    if isinstance(item, dict):
      cat = item.get("category", "?")
      crit = item.get("criterion", "?")
      out_lines.append(f"  [{cat}] {crit}")
  return "\n".join(out_lines)


def judge_question(
    question,
    *,
    client=None,
    model = DEFAULT_MODEL,
    min_average = DEFAULT_MIN_AVERAGE,
    sleep=time.sleep,
):
  """Return the judge verdict dict (``scores``, ``average``, ``verdict``, ``reasoning``)."""
  prompt = JUDGE_PROMPT_TEMPLATE.format(
      question=question.get("question", ""),
      entities=", ".join(question.get("entities", [])),
      cutoff=question.get("cutoff", "?"),
      rubric=_rubric_to_text(question.get("rubric", [])),
      threshold=min_average,
  )
  result = _llm_json_call(client or get_client(), model, prompt, sleep=sleep)
  if not result:
    return {
        "scores": {},
        "average": 0,
        "verdict": "error",
        "reasoning": "LLM call failed",
    }

  verdict = str(result.get("verdict", "")).strip().lower()
  result["verdict"] = "accept" if verdict == "accept" else "reject"
  return result


def passes_quality_gate(
    judgment,
    *,
    min_average = DEFAULT_MIN_AVERAGE,
    min_naturalness = MIN_NATURALNESS,
):
  """Apply the documented gate: avg >= threshold, no 1s, naturalness >= 3."""
  if judgment.get("verdict") != "accept":
    return False
  if (judgment.get("average") or 0) < min_average:
    return False
  scores = judgment.get("scores", {}) or {}
  if any(v == NO_ONES for v in scores.values()):
    return False
  return scores.get("naturalness", 0) >= min_naturalness
