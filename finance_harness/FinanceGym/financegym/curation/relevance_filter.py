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

"""Wall-Street institutional relevance + naturalness filter.

Second stage of the curation funnel. The prompt frames the score from the
perspective of a multi-strategy firm with named desks, so the model is
calibrated to "would a desk assign an analyst to this" rather than the
much looser "is this a valid question".

Documented gate (research log 7.8): ``institutional_relevance >= 5 and
naturalness >= 4``.
"""

from __future__ import annotations

import time

from financegym.common.llm import DEFAULT_MODEL, get_client
from financegym.curation._common import llm_json

RELEVANCE_PROMPT_TEMPLATE = """You are a senior research director at a multi-strategy investment firm.
Your firm has the following desks, each with dedicated analysts:

- Equity Long/Short: large- and mid-cap stocks across sectors (Russell 1000).
- Credit & Fixed Income: IG/HY corporate bonds, sovereign debt, rates strategy.
- Global Macro: currencies, rates, commodities, geopolitics.
- Quantitative & Systematic: factor models, risk systems.
- Event-Driven / Special Situations: M&A, spin-offs, activist campaigns.
- Sector Specialists: Healthcare/Biotech, Technology/Semiconductors, Energy, Financials.

Now evaluate this research question:

QUESTION: {question}
ENTITIES: {entities}
CUTOFF DATE: {cutoff}

Score on TWO dimensions (1-5 each):

1. INSTITUTIONAL RELEVANCE — Would ANY desk at this firm assign an analyst
   to investigate this question?
   5: Multiple desks would want it answered.
   4: One desk would definitely investigate.
   3: A specialist might look into this if they cover the sector.
   2: Too narrow or obscure for institutional research.
   1: No desk would investigate.

2. NATURALNESS — Does this read like something an analyst would actually type?
   5: Exactly how a PM or analyst would phrase the query.
   4: Good question, minor adjustments would make it perfect.
   3: Slightly academic or over-engineered.
   2: Awkward framing, asks about things no practitioner would.
   1: Clearly artificial, reads like a test question.

Return JSON:
{{
  "institutional_relevance": N,
  "naturalness": N,
  "relevant_desks": ["e.g. equity_long_short, global_macro"],
  "reason": "1 sentence"
}}"""


def score_relevance(
    question,
    *,
    client=None,
    model = DEFAULT_MODEL,
    sleep=time.sleep,
):
  """Run one relevance + naturalness scoring call."""
  prompt = RELEVANCE_PROMPT_TEMPLATE.format(
      question=question.get("question", ""),
      entities=", ".join(question.get("entities", [])),
      cutoff=question.get("cutoff", ""),
  )
  return llm_json(client or get_client(), model, prompt, sleep=sleep)


def passes_relevance(
    score,
    *,
    min_relevance = 5,
    min_naturalness = 4,
):
  """Documented gate: institutional_relevance >= 5 and naturalness >= 4."""
  if not score:
    return False
  return (
      int(score.get("institutional_relevance", 0) or 0) >= min_relevance
      and int(score.get("naturalness", 0) or 0) >= min_naturalness
  )
