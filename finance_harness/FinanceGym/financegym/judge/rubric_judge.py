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

"""Five-tier (0-4) rubric judge — the FinanceGym scoring contract.

For each (question, agent-report) pair the judge scores every rubric item
on the same 0-4 scale and returns a JSON dict with one entry per item.
The judge has access to the question, thesis, cutoff, the full rubric
with antecedent/consequent labels, the pre- and post-cutoff edge
evidence, the source URLs, and the agent's report text.

The prompt is pinned in :data:`JUDGE_SYSTEM` and :func:`build_prompt`.
Both are part of the benchmark contract — any change must be tracked in
``docs/reproducibility.md``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutTimeoutError
import logging
import os
import time

from financegym.common.llm import DEFAULT_MODEL, get_client
from google.genai import types
from pydantic import BaseModel

log = logging.getLogger(__name__)

DEFAULT_REPORT_CHAR_LIMIT = 12_000
DEFAULT_EDGE_LIMIT = 30
DEFAULT_URL_LIMIT = 25
DEFAULT_CALL_TIMEOUT_S = int(os.environ.get("JUDGE_CALL_TIMEOUT_S", "300"))


# ---------------------------------------------------------------------------
# Pinned schema + prompt
# ---------------------------------------------------------------------------


class ItemScore(BaseModel):
  item_idx: int
  score: int  # 0..4
  reasoning: str


class JudgeOutput(BaseModel):
  scores: list[ItemScore]


JUDGE_SYSTEM = """You are a senior financial research reviewer scoring a research \
report against a rubric of specific factual claims.

Use this 0-4 scale for each rubric item:
  0 = NOT ADDRESSED: rubric item absent from report
  1 = MENTIONED: touched in passing; no substance, no evidence
  2 = PARTIAL: directionally correct but missing a key specific (number, date, entity, mechanism)
  3 = SUBSTANTIVE: correct with specifics; minor gap or weak attribution
  4 = FULLY GROUNDED: correct, specific, and attributed to a plausible source

Grounding rules:
- Items labeled [antecedent] should be verified against pre-cutoff evidence \
(data available at time of research)
- Items labeled [consequent] should be verified against post-cutoff evidence \
(what actually happened after the cutoff)
- The report's specific claims should be consistent with the evidence triples provided.
- A citation to a URL is only "plausible" if the URL appears in the provided source \
list or is from a reputable financial publication.

Be strict. A report can discuss a topic thoroughly and still score 1 if it does not \
address the specific rubric criterion. A report can reach 4 only with specific, \
verifiable facts matching the criterion."""


def _format_edges(edges, limit = DEFAULT_EDGE_LIMIT):
  if not edges:
    return "(none)"
  lines: list[str] = []
  for e in edges[:limit]:
    head = e.get("head", "?")
    rel = e.get("relation", "?")
    tail = e.get("tail", "?")
    ctx = (e.get("context") or "")[:180]
    if ctx:
      lines.append(f"  - {head} —{rel}→ {tail}  [{ctx}]")
    else:
      lines.append(f"  - {head} —{rel}→ {tail}")
  if len(edges) > limit:
    lines.append(f"  ... and {len(edges) - limit} more")
  return "\n".join(lines)


def _format_urls(urls, limit = DEFAULT_URL_LIMIT):
  if not urls:
    return "(none)"
  shown = urls[:limit]
  suffix = f"\n  ... and {len(urls) - limit} more" if len(urls) > limit else ""
  return "\n".join(f"  - {u}" for u in shown) + suffix


def build_prompt(
    question,
    report,
    *,
    report_char_limit = DEFAULT_REPORT_CHAR_LIMIT,
):
  """Assemble the judge prompt for one (question, report) pair."""
  rubric = question.get("rubric", [])
  md = question.get("metadata", {})
  rubric_text = "\n".join(
      f"  [{i}] [{r.get('category', '?')}] {r['criterion']}"
      for i, r in enumerate(rubric)
  )
  return (
      f"<question>\n{question.get('question', '')}\n</question>\n\n<thesis>\n{question.get('thesis', '')}\n</thesis>\n\n<cutoff>{question.get('cutoff', '')}</cutoff>\n\n<pre_cutoff_evidence>\nEvidence"
      " available at or before the cutoff"
      f" date:\n{_format_edges(md.get('pre_edge_evidence', []))}\n\nPre-cutoff"
      " source"
      f" URLs:\n{_format_urls(md.get('source_urls_pre', []))}\n</pre_cutoff_evidence>\n\n<post_cutoff_evidence>\nEvidence"
      " from after the cutoff (ground truth for consequent items"
      f" only):\n{_format_edges(md.get('post_edge_evidence', []))}\n\nPost-cutoff"
      " source"
      f" URLs:\n{_format_urls(md.get('source_urls_post', []))}\n</post_cutoff_evidence>\n\n<rubric>\nScore"
      " each item on the 0-4 scale. Items marked [antecedent] verify against"
      " pre-cutoff evidence; items marked [consequent] verify against"
      " post-cutoff"
      f" evidence.\n{rubric_text}\n</rubric>\n\n<report>\n{report[:report_char_limit]}\n</report>\n\nReturn"
      " JSON with one entry per rubric item (in the same order, item_idx"
      ' matching the bracket number):\n{"scores": [{"item_idx": 0,'
      ' "score": N, "reasoning": "brief"}, ...]}'
  )


# ---------------------------------------------------------------------------
# Judge call
# ---------------------------------------------------------------------------


def _align_scores(scores, rubric):
  """Re-order LLM-emitted scores to the rubric order, stamping category + criterion."""
  by_idx = {s["item_idx"]: s for s in scores}
  aligned: list[dict] = []
  for i, r in enumerate(rubric):
    s = by_idx.get(i)
    if s is None:
      aligned.append({
          "item_idx": i,
          "score": 0,
          "reasoning": "missing from judge output",
          "category": r.get("category", "unknown"),
          "criterion": r["criterion"],
      })
    else:
      s["category"] = r.get("category", "unknown")
      s["criterion"] = r["criterion"]
      aligned.append(s)
  return aligned


def _placeholder_scores(rubric, err):
  return [
      {
          "item_idx": i,
          "score": 0,
          "reasoning": f"judge failed: {err[:100]}",
          "category": r.get("category", "unknown"),
          "criterion": r["criterion"],
      }
      for i, r in enumerate(rubric)
  ]


def judge_one(
    question,
    report,
    *,
    client=None,
    model = DEFAULT_MODEL,
    max_retries = 4,
    call_timeout_s = DEFAULT_CALL_TIMEOUT_S,
    sleep=time.sleep,
):
  """Score one (question, report) pair. Returns the rubric-aligned score list.

  On persistent failure returns a list of score-0 placeholders so downstream
  aggregation can detect the failure via the ``judge failed:`` prefix.
  Returns ``None`` only when the question has no rubric to score against.
  """
  rubric = question.get("rubric", [])
  if not rubric:
    return None

  cli = client or get_client()
  config = types.GenerateContentConfig(
      response_mime_type="application/json",
      response_schema=JudgeOutput,
      system_instruction=JUDGE_SYSTEM,
      temperature=0.0,
      automatic_function_calling=types.AutomaticFunctionCallingConfig(
          disable=True
      ),
  )
  prompt = build_prompt(question, report)

  err = ""
  for attempt in range(max_retries + 1):
    try:
      with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(
            cli.models.generate_content,
            model=model,
            config=config,
            contents=prompt,
        )
        resp = fut.result(timeout=call_timeout_s)
      parsed = resp.parsed
      scores = [s.model_dump() for s in parsed.scores]
      return _align_scores(scores, rubric)
    except Exception as e:  # noqa: BLE001
      err = str(e)
      transient = isinstance(e, FutTimeoutError) or any(
          t in err for t in ("429", "503", "RESOURCE_EXHAUSTED", "timeout")
      )
      if attempt < max_retries and transient:
        sleep(2**attempt)
        continue
      log.warning("judge failed (attempt %d): %s", attempt + 1, err[:150])
      break
  return _placeholder_scores(rubric, err)


# ---------------------------------------------------------------------------
# Per-question normalized summary (used by aggregate.py)
# ---------------------------------------------------------------------------


def summarize(scores):
  """Per-question score sums + normalized ratios + distribution."""
  ant_items = [s for s in scores if s.get("category") == "antecedent"]
  con_items = [s for s in scores if s.get("category") == "consequent"]
  ant = sum(s["score"] for s in ant_items)
  ant_max = 4 * len(ant_items)
  con = sum(s["score"] for s in con_items)
  con_max = 4 * len(con_items)
  total = ant + con
  total_max = ant_max + con_max
  dist = {str(k): sum(1 for s in scores if s["score"] == k) for k in range(5)}
  return {
      "antecedent_sum": ant,
      "antecedent_max": ant_max,
      "antecedent_norm": round(ant / ant_max, 4) if ant_max else None,
      "consequent_sum": con,
      "consequent_max": con_max,
      "consequent_norm": round(con / con_max, 4) if con_max else None,
      "total_sum": total,
      "total_max": total_max,
      "total_norm": round(total / total_max, 4) if total_max else None,
      "score_dist": dist,
  }


def is_judge_failure(record):
  """A record is a judge failure if every score has the placeholder reasoning."""
  scs = record.get("scores", []) or []
  return bool(scs) and all(
      str(s.get("reasoning", "")).startswith("judge failed") for s in scs
  )


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------


def judge_pair_to_record(
    agent,
    question,
    answer,
    *,
    client=None,
    model = DEFAULT_MODEL,
):
  """Score one (question, answer) pair and stamp the result with axis labels."""
  report = answer.get("report", "") or ""
  scores = judge_one(question, report, client=client, model=model)
  if scores is None:
    return None
  return {
      "agent": agent,
      "question": question.get("question"),
      "cutoff": question.get("cutoff"),
      "topic": question.get("topic"),
      "sector": question.get("sector"),
      "reasoning_type": question.get("reasoning_type"),
      "situation_type": question.get("situation_type"),
      "report_chars": len(report),
      "report_words": len(report.split()),
      "scores": scores,
      **summarize(scores),
  }
