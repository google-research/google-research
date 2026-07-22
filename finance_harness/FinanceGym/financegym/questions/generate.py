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

"""Generate one research question from a mined situation + a chosen cutoff.

The prompt is pinned in :data:`QUESTION_PROMPT_TEMPLATE` and is part of the
benchmark contract — any change should be tracked in
``docs/reproducibility.md``. The LLM is given a deliberately type-agnostic
brief: it sees the pre-cutoff edge evidence and post-cutoff verification
edges but no situation-type label, so the generated question and rubric
reflect what naturally arises from the evidence.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
import json
import logging
import re
import time

from financegym.common.llm import DEFAULT_MODEL, get_client
from financegym.questions.entity_filter import (
    ENTITY_BLOCKLIST,
    categorize_relation,
    is_garbage_entity,
    normalize_sector,
)
from google.genai import types

log = logging.getLogger(__name__)

MIN_PRE_EDGES = 10
MIN_POST_EDGES = 5
MAX_RETRIES = 4
RETRY_BASE_DELAY = 2.0


# ---------------------------------------------------------------------------
# Evidence packing helpers (pure)
# ---------------------------------------------------------------------------


def format_edge_for_prompt(e, *, context_limit = 200):
  """One-line edge representation for the LLM prompt body."""
  return (
      f"  [{e['pub_date']}] {e['head']} → {e['relation']} → "
      f"{e['tail']}: {e.get('context', '')[:context_limit]}"
  )


def edge_record(e):
  """Full-fidelity edge dict used in the question's metadata block."""
  return {
      "head": e["head"],
      "relation": e["relation"],
      "tail": e["tail"],
      "pub_date": e.get("pub_date", ""),
      "domain": e.get("domain", ""),
      "url": e.get("url", ""),
      "context": e.get("context", ""),
  }


def split_pre_post(
    edges, cutoff
):
  """Split edges into pre-cutoff (<=) and post-cutoff (>)."""
  pre, post = [], []
  for e in edges:
    if e.get("pub_date", "") <= cutoff:
      pre.append(e)
    else:
      post.append(e)
  return pre, post


def _clean_entities(entities):
  if not isinstance(entities, list):
    return []
  out: list[str] = []
  for e in entities:
    if not isinstance(e, str) or len(e) < 2:
      continue
    if is_garbage_entity(e) or e in ENTITY_BLOCKLIST:
      continue
    if re.fullmatch(r"[\d.,]+", e):
      continue
    if " to " in e:
      continue
    if e.upper() in {"PUT", "CALL", "BUY", "SELL", "HOLD"}:
      continue
    out.append(e)
  return out


def _flatten_rubric(rubric):
  """Convert the LLM's nested ``{antecedent, consequent}`` rubric into the flat list."""
  if not isinstance(rubric, dict):
    return []
  flat: list[dict] = []
  for cat in ("antecedent", "consequent"):
    for item in rubric.get(cat, []) or []:
      if isinstance(item, str):
        flat.append({"criterion": item, "category": cat})
      elif isinstance(item, dict):
        item = dict(item)
        item["category"] = cat
        flat.append(item)
  return flat


# ---------------------------------------------------------------------------
# Evidence text builders per situation type
# ---------------------------------------------------------------------------


def build_multihop_evidence(
    situation, pre
):
  """Return ``(prompt_text, hop1_sample, hop2_sample, other_sample)`` or empty results."""
  cats = situation.get("category_sequence", [])
  ents = situation["focus_entities"]
  bridge = ents[1] if len(ents) >= 3 else ents[0]
  end = ents[2] if len(ents) >= 3 else "?"

  hop1 = [
      e
      for e in pre
      if bridge in (e["head"], e["tail"]) and ents[0] in (e["head"], e["tail"])
  ]
  hop2 = [
      e
      for e in pre
      if bridge in (e["head"], e["tail"]) and end in (e["head"], e["tail"])
  ]
  if not hop1 or not hop2:
    return "", [], [], []

  hop_ids = {id(e) for e in hop1 + hop2}
  other = [e for e in pre if id(e) not in hop_ids]
  hop1_sample = sorted(hop1, key=lambda x: x["pub_date"], reverse=True)[:15]
  hop2_sample = sorted(hop2, key=lambda x: x["pub_date"], reverse=True)[:15]
  other_sample = sorted(other, key=lambda x: x["pub_date"], reverse=True)[:10]

  cat1 = cats[0] if cats else "?"
  cat2 = cats[1] if len(cats) > 1 else "?"

  text = f"CONNECTION 1 ({ents[0]} ↔ {bridge}, {cat1} relations):\n"
  text += "\n".join(format_edge_for_prompt(e) for e in hop1_sample)
  text += f"\n\nCONNECTION 2 ({bridge} ↔ {end}, {cat2} relations):\n"
  text += "\n".join(format_edge_for_prompt(e) for e in hop2_sample)
  if other_sample:
    text += "\n\nADDITIONAL CONTEXT:\n" + "\n".join(
        format_edge_for_prompt(e) for e in other_sample
    )
  return text, hop1_sample, hop2_sample, other_sample


def build_narrative_evidence(
    situation,
    pre,
    cutoff,
    categories,
):
  """Walk the category arc and pull representative edges per phase."""
  arc = situation.get("category_arc", [])
  text = ""
  sample: list[dict] = []
  edges_used = 0
  for month, cat in arc:
    if month > cutoff[:7]:
      break
    month_edges = [
        e
        for e in pre
        if e["pub_date"][:7] == month
        and categorize_relation(e["relation"], categories) == cat
    ]
    if month_edges and edges_used < 50:
      chunk = sorted(month_edges, key=lambda x: x["pub_date"], reverse=True)[:8]
      text += f"\nPHASE ({month}, {cat}):\n" + "\n".join(
          format_edge_for_prompt(e) for e in chunk
      )
      sample.extend(chunk)
      edges_used += len(chunk)
  return text, sample


def build_tension_evidence(pre):
  """Diverse-relation sampling for tension-type situations."""
  by_rel: dict[str, list[dict]] = defaultdict(list)
  for e in pre:
    by_rel[e["relation"]].append(e)
  sample: list[dict] = []
  for _rel, rel_edges in sorted(by_rel.items(), key=lambda x: -len(x[1])):
    sample.extend(
        sorted(rel_edges, key=lambda x: x["pub_date"], reverse=True)[:3]
    )
    if len(sample) >= 50:
      break
  text = "\n".join(
      format_edge_for_prompt(e)
      for e in sorted(sample, key=lambda x: x["pub_date"])
  )
  return text, sample


# ---------------------------------------------------------------------------
# Pinned prompt
# ---------------------------------------------------------------------------


QUESTION_PROMPT_TEMPLATE = """You are a financial analyst using an agentic research system. You have
evidence from a knowledge graph extracted from financial news articles.

ENTITIES IN FOCUS: {focus}
EVIDENCE SCOPE: {n_entities} entities, {n_cats} relation categories, {n_months} months
CUTOFF DATE: {cutoff} (analyst can only see data up to this date)

PRE-CUTOFF EVIDENCE (available to the analyst):
{pre_text}

POST-CUTOFF DEVELOPMENTS (for designing verification only — the analyst cannot see these):
{post_text}

Based on this evidence, generate one research question that a financial
practitioner would naturally want to investigate. The question should be
grounded in the evidence above and require synthesising multiple data points
to answer — the kind of question that needs a 2-3 page research memo drawing
on earnings data, filings, market data, news, and regulatory documents.

OUTPUT JSON:
{{
  "question": "A natural research question, 30-50 words.",
  "thesis": "The core analytical question in one sentence.",
  "entities": ["1-4 real entities central to answering the question"],
  "rubric": {{
    "antecedent": ["3-5 specific, verifiable pre-cutoff facts."],
    "consequent": ["2-3 specific post-cutoff developments."]
  }},
  "skip": false,
  "skip_reason": null
}}

If the evidence doesn't support a natural question, set skip to true."""


# ---------------------------------------------------------------------------
# LLM call with retry
# ---------------------------------------------------------------------------


def _llm_json_call(
    client,
    model,
    prompt,
    *,
    max_retries = MAX_RETRIES,
    sleep=time.sleep,
):
  """Run a JSON-mode Gemini call, retrying transient errors."""
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
      transient = "429" in err or "RESOURCE_EXHAUSTED" in err
      if transient and attempt < max_retries:
        sleep(RETRY_BASE_DELAY * (2**attempt))
        continue
      return None
  return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_question(
    situation,
    cutoff,
    *,
    categories,
    client=None,
    model = DEFAULT_MODEL,
    sleep=time.sleep,
):
  """Produce one structured question dict, or ``None`` to skip."""
  sit_edges = situation["edges"]
  pre, post = split_pre_post(sit_edges, cutoff)
  if len(pre) < MIN_PRE_EDGES or len(post) < MIN_POST_EDGES:
    return None

  sit_type = situation["situation_type"]
  hop1_sample: list[dict] = []
  hop2_sample: list[dict] = []
  pre_sample: list[dict] = []

  if sit_type == "multihop_path":
    pre_text, hop1_sample, hop2_sample, other_sample = build_multihop_evidence(
        situation, pre
    )
    if not pre_text:
      return None
    pre_sample = hop1_sample + hop2_sample + other_sample
  elif sit_type == "temporal_narrative":
    pre_text, pre_sample = build_narrative_evidence(
        situation, pre, cutoff, categories
    )
    if not pre_text.strip():
      return None
  else:
    pre_text, pre_sample = build_tension_evidence(pre)

  post_sample = sorted(post, key=lambda e: e["pub_date"])[:20]
  post_text = "\n".join(format_edge_for_prompt(e) for e in post_sample)

  n_entities = len({e["head"] for e in pre} | {e["tail"] for e in pre})
  n_cats = len(
      {categorize_relation(e["relation"], categories) for e in pre}
      - {"other", "uncategorized"}
  )
  n_months = len({e["pub_date"][:7] for e in pre})

  prompt = QUESTION_PROMPT_TEMPLATE.format(
      focus=", ".join(situation["focus_entities"]),
      n_entities=n_entities,
      n_cats=n_cats,
      n_months=n_months,
      cutoff=cutoff,
      pre_text=pre_text,
      post_text=post_text,
  )

  result = _llm_json_call(client or get_client(), model, prompt, sleep=sleep)
  if not result or result.get("skip"):
    return None

  if "sector" in result:
    result["sector"] = normalize_sector(result["sector"])
  if "entities" in result:
    result["entities"] = _clean_entities(result["entities"])
  result["rubric"] = _flatten_rubric(result.get("rubric", {}))

  all_pre_evidence = (
      hop1_sample + hop2_sample if sit_type == "multihop_path" else pre_sample
  )

  result.update({
      "situation_type": sit_type,
      "focus_entities": situation["focus_entities"],
      "cutoff": cutoff,
      "pre_edges": len(pre),
      "post_edges": len(post),
  })
  if "category_sequence" in situation:
    result["category_sequence"] = situation["category_sequence"]
  if "category_arc" in situation:
    result["category_arc"] = situation["category_arc"]

  result["metadata"] = {
      "source_urls_pre": sorted(
          {e["url"] for e in all_pre_evidence if e.get("url")}
      ),
      "source_urls_post": sorted(
          {e["url"] for e in post_sample if e.get("url")}
      ),
      "source_domains": sorted({
          e["domain"] for e in all_pre_evidence + post_sample if e.get("domain")
      }),
      "pre_edge_evidence": [edge_record(e) for e in all_pre_evidence],
      "post_edge_evidence": [edge_record(e) for e in post_sample],
      "situation_signal": situation.get("signal", ""),
      "n_months": n_months,
      "n_entities_in_evidence": n_entities,
      "n_categories_in_evidence": n_cats,
      "total_pre_edges": len(pre),
      "total_post_edges": len(post),
  }
  return result


# ---------------------------------------------------------------------------
# Near-duplicate elimination (post-generation)
# ---------------------------------------------------------------------------


_STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "and",
    "or",
    "but",
    "with",
    "by",
    "from",
    "its",
    "their",
    "this",
    "that",
    "has",
    "have",
    "had",
    "been",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "can",
    "while",
    "despite",
    "given",
    "between",
    "against",
    "how",
    "what",
    "why",
    "which",
    "when",
    "where",
}
_WORD_RE = re.compile(r"\b[a-zA-Z]{3,}\b")


def _words(text):
  return set(_WORD_RE.findall(text.lower())) - _STOPWORDS


def _overlap(a, b):
  if not a or not b:
    return 0.0
  return len(a & b) / min(len(a), len(b))


def deduplicate_questions(
    questions,
    *,
    entity_overlap_threshold = 0.5,
    word_overlap_threshold = 0.4,
):
  """Drop near-duplicates by joint entity + word-set overlap."""
  items = sorted(questions, key=lambda q: -len(q.get("rubric", [])))
  kept: list[dict] = []
  for q in items:
    q_ents = set(q.get("entities", []))
    q_words = _words(q.get("question", ""))
    if any(
        _overlap(q_ents, set(existing.get("entities", [])))
        >= entity_overlap_threshold
        and _overlap(q_words, _words(existing.get("question", "")))
        >= word_overlap_threshold
        for existing in kept
    ):
      continue
    kept.append(q)
  return kept
