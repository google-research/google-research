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

"""Single-thread coherence filter — third stage of the curation funnel.

Some questions stitch two or three unrelated sub-questions with
", and how" connectors. They pass relevance individually but read as
"pieced-together" rather than as a single analytical thread. This filter
scores how well a question reads as one focused inquiry vs multiple
stitched threads.

Documented gate (research log 7.8): ``coherence >= 4``.

The script is reconstructed faithfully from the documented prompt and
scoring scheme — there was no standalone implementation in the source
monorepo, only a shared prompt with the relevance filter. We keep them
separate here so the funnel maps 1:1 onto its documented stages.
"""

from __future__ import annotations

import time

from financegym.common.llm import DEFAULT_MODEL, get_client
from financegym.curation._common import llm_json

COHERENCE_PROMPT_TEMPLATE = """You are reviewing a research question for SINGLE-THREAD COHERENCE.

QUESTION: {question}

Score (1-5) how well this reads as one focused analytical inquiry vs
multiple stitched sub-questions.

5: One clear analytical thread. Exactly one question. Natural.
4: Primarily one thread, with a minor related secondary aspect.
3: Two related threads, starting to feel stretched.
2: Two loosely connected sub-questions. Feels artificial.
1: Multiple unrelated sub-questions packed together.

Return JSON:
{{"coherence": N, "reason": "1 sentence"}}"""


def score_coherence(
    question,
    *,
    client=None,
    model = DEFAULT_MODEL,
    sleep=time.sleep,
):
  """One coherence call per question."""
  prompt = COHERENCE_PROMPT_TEMPLATE.format(
      question=question.get("question", "")
  )
  return llm_json(client or get_client(), model, prompt, sleep=sleep)


def passes_coherence(score, *, min_score = 4):
  """Documented gate: coherence >= 4."""
  if not score:
    return False
  return int(score.get("coherence", 0) or 0) >= min_score
