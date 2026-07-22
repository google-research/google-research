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

"""Independent feasibility audit — first filter in the curation funnel.

Each question is audited by one independent LLM call (no batching, no
neighbour context) on four dimensions and one flag. The default gate
matches the documented funnel: all dimensions ≥ 5, flag == ``"none"``.
"""

from __future__ import annotations

import logging
import time

from financegym.common.llm import DEFAULT_MODEL, get_client
from financegym.curation._common import llm_json

log = logging.getLogger(__name__)


AUDIT_PROMPT_TEMPLATE = """You are a senior financial research quality auditor. Independently
assess this single research question for feasibility as a benchmark task.

IMPORTANT CONTEXT: The corpus of articles covers January 2025 through December 2025.
Events and data from this period ARE verifiable in the corpus. The cutoff date splits
evidence into pre-cutoff (available to the research agent) and post-cutoff (for
verification).

QUESTION: {question}
THESIS: {thesis}
ENTITIES: {entities}
CUTOFF DATE: {cutoff}

ANTECEDENT RUBRIC (facts the agent should find pre-cutoff):
{antecedent_text}

CONSEQUENT RUBRIC (developments for verification post-cutoff):
{consequent_text}

Score this question on 4 dimensions (1-5 each):

1. PRACTITIONER RELEVANCE — Would a real portfolio manager, research analyst, or
   risk manager actually investigate this?
   5: Directly informs investment/risk decisions.
   3: Interesting background research but not urgent.
   1: Trivial or no financial professional would care.

2. ANTECEDENT VERIFIABILITY — Can the antecedent facts be found in public
   financial news articles from Jan-Dec 2025?
   5: Standard public data (earnings, filings, news).
   3: Partially verifiable.
   1: Requires proprietary/internal data only.

3. CONSEQUENT LOGICAL LINK — Do the consequent items logically follow from the
   dynamics described in the question and antecedents?
   5: Clear causal/thematic connection.
   3: Same entity/topic but the connection is weak.
   1: No meaningful connection.

4. ANSWERABILITY — Can an agentic research AI system, with access to ~122M financial
   news articles (Jan-Dec 2025), produce a meaningful 2-3 page research memo?
   5: Well-scoped, clear analytical path.
   3: Partially answerable.
   1: Essentially unanswerable from news alone.

Also identify any specific issues:
- impossible_prediction, hallucinated_fact, proprietary_data_needed,
  too_vague, entity_error, or none.

Return JSON:
{{
  "practitioner_relevance": N,
  "antecedent_verifiability": N,
  "consequent_link": N,
  "answerability": N,
  "flag": "none|impossible_prediction|hallucinated_fact|proprietary_data_needed|too_vague|entity_error",
  "issue_detail": "Brief explanation if flagged, empty string if none"
}}"""


AUDIT_DIMENSIONS = (
    "practitioner_relevance",
    "antecedent_verifiability",
    "consequent_link",
    "answerability",
)


def _rubric_text(rubric, category):
  items = [
      r["criterion"] for r in rubric or [] if r.get("category") == category
  ]
  if not items:
    return "  (none)"
  return "\n".join(f"  - {a}" for a in items)


def audit_one(
    question,
    *,
    client=None,
    model = DEFAULT_MODEL,
    sleep=time.sleep,
):
  """Run one feasibility audit. Returns the parsed JSON dict or ``None``."""
  rubric = question.get("rubric", [])
  prompt = AUDIT_PROMPT_TEMPLATE.format(
      question=question.get("question", ""),
      thesis=question.get("thesis", ""),
      entities=", ".join(question.get("entities", [])),
      cutoff=question.get("cutoff", "unknown"),
      antecedent_text=_rubric_text(rubric, "antecedent"),
      consequent_text=_rubric_text(rubric, "consequent"),
  )
  return llm_json(client or get_client(), model, prompt, sleep=sleep)


def passes_audit(
    audit,
    *,
    min_score = 5,
    allowed_flags = ("none",),
):
  """Apply the documented audit gate: all dims at ``min_score`` and flag allowed."""
  if not audit:
    return False
  if audit.get("flag", "none") not in allowed_flags:
    return False
  return all(int(audit.get(d, 0) or 0) >= min_score for d in AUDIT_DIMENSIONS)
