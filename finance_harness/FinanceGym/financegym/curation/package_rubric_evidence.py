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

"""Attach source-edge indices + article text to each rubric item.

The original generation pipeline preserved which edges seeded the
question as a whole, but not which edges seeded each individual rubric
criterion. This module reconstructs the link via a per-criterion Gemini
call against the pre-cutoff (for antecedents) or post-cutoff (for
consequents) edge evidence already stored under
``question["metadata"]["{pre,post}_edge_evidence"]``.

It also exports the article texts the rubric will be evaluated against
into a separate JSONL, so the entire benchmark is auditable offline (no
live-web fetches at evaluation time).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sqlite3
import time

from financegym.common.llm import DEFAULT_MODEL, generate_with_retry, get_client
from google.genai import types
from pydantic import BaseModel

log = logging.getLogger(__name__)


MAP_SYSTEM_INSTRUCTION = (
    "You are a financial research analyst mapping a rubric criterion to its"
    " supporting evidence.\n\nReturn the 0-based indices of the edges that"
    " *clearly and specifically* support the criterion. If no edge specifically"
    ' supports it, return an empty list.\n\nOutput JSON: {"indices": [int,'
    " ...]}."
)
MAX_CRITERION_EDGES = 100
MAX_TEXT_CHARS = 20_000


class _IndexList(BaseModel):
  indices: list[int]


# ---------------------------------------------------------------------------
# Phase A: map rubric items to source edges
# ---------------------------------------------------------------------------


def build_mapping_prompt(
    criterion, edges, max_edges = MAX_CRITERION_EDGES
):
  """Build the criterion-mapping prompt (testable pure function)."""
  items = []
  for i, e in enumerate(edges[:max_edges]):
    head = e.get("head", "?")
    rel = e.get("relation", "?")
    tail = e.get("tail", "?")
    ctx = (e.get("context") or "").replace("\n", " ")
    if len(ctx) > 400:
      ctx = ctx[:400] + "…"
    items.append(f"[{i}] {head} —{rel}→ {tail}  ({ctx})")
  return (
      f"<criterion>{criterion}</criterion>\n\n<candidate_edges>\n"
      + "\n".join(items)
      + '\n</candidate_edges>\n\nReturn JSON: {"indices": [...]}'
  )


def map_one_criterion(
    criterion,
    edges,
    *,
    client=None,
    model = DEFAULT_MODEL,
    max_attempts = 4,
):
  """Run the LLM mapping for one criterion. Returns the validated index list."""
  if not edges:
    return []
  cli = client or get_client()
  config = types.GenerateContentConfig(
      response_mime_type="application/json",
      response_schema=_IndexList,
      system_instruction=MAP_SYSTEM_INSTRUCTION,
      temperature=0.0,
      automatic_function_calling=types.AutomaticFunctionCallingConfig(
          disable=True
      ),
  )
  try:
    resp = generate_with_retry(
        cli,
        model=model,
        contents=build_mapping_prompt(criterion, edges),
        config=config,
        max_attempts=max_attempts,
    )
  except Exception as e:  # noqa: BLE001
    log.warning("map_one_criterion failed: %s", str(e)[:120])
    return []
  parsed = resp.parsed
  if parsed is None:
    return []
  return sorted(
      {i for i in parsed.indices if isinstance(i, int) and 0 <= i < len(edges)}
  )


def annotate_question(
    record,
    *,
    client=None,
    model = DEFAULT_MODEL,
):
  """Attach ``source_edge_indices`` to every rubric item in ``record``."""
  md = record.get("metadata") or {}
  pre_edges = md.get("pre_edge_evidence") or []
  post_edges = md.get("post_edge_evidence") or []
  rubric = record.get("rubric") or []
  cli = client or get_client()
  for item in rubric:
    cat = item.get("category", "antecedent")
    edges = pre_edges if cat == "antecedent" else post_edges
    item["source_edge_indices"] = map_one_criterion(
        item.get("criterion", ""), edges, client=cli, model=model
    )
  return record


# ---------------------------------------------------------------------------
# Phase B: export article texts referenced by the rubric evidence
# ---------------------------------------------------------------------------


def collect_referenced_urls(records):
  """Return the de-duplicated set of URLs referenced by any record's edges."""
  seen: set[str] = set()
  for r in records:
    md = r.get("metadata") or {}
    for ev in md.get("pre_edge_evidence") or []:
      url = ev.get("url")
      if url:
        seen.add(url)
    for ev in md.get("post_edge_evidence") or []:
      url = ev.get("url")
      if url:
        seen.add(url)
  return sorted(seen)


def export_article_texts(
    records,
    metadata_path,
    corpus_db_path,
    output_path,
    *,
    max_text_chars = MAX_TEXT_CHARS,
):
  """Resolve every referenced URL to its article text via ``corpus.db``.

  Step 1 — Stream ``metadata.jsonl`` once to map url → doc_id (the SQLite
  store is indexed on doc_id, not url).
  Step 2 — Look up each doc_id in ``corpus.db``, truncate, and write one
  JSONL line per URL.
  Returns a small ``{written, missing}`` stats dict.
  """
  urls_wanted = set(collect_referenced_urls(records))
  if not urls_wanted:
    Path(output_path).write_text("")
    return {"written": 0, "missing": 0}

  url_to_doc: dict[str, str] = {}
  with open(metadata_path) as f:
    for line in f:
      if not line.strip():
        continue
      try:
        rec = json.loads(line)
      except json.JSONDecodeError:
        continue
      url = rec.get("url")
      if url and url in urls_wanted and url not in url_to_doc:
        url_to_doc[url] = rec.get("doc_id", "")
        if len(url_to_doc) == len(urls_wanted):
          break

  conn = sqlite3.connect(str(corpus_db_path))
  written = 0
  missing = 0
  try:
    with open(output_path, "w") as out:
      for url in sorted(urls_wanted):
        doc_id = url_to_doc.get(url)
        text = ""
        if doc_id:
          row = conn.execute(
              "SELECT text FROM docs WHERE doc_id = ?", (doc_id,)
          ).fetchone()
          if row:
            text = row[0] or ""
        if not text:
          missing += 1
          out.write(
              json.dumps({
                  "url": url,
                  "doc_id": doc_id or "",
                  "text": "",
                  "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "error": "not_found",
              })
              + "\n"
          )
          continue
        truncated = len(text) > max_text_chars
        out.write(
            json.dumps({
                "url": url,
                "doc_id": doc_id,
                "text": text[:max_text_chars],
                "text_len_orig": len(text),
                "truncated": truncated,
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            })
            + "\n"
        )
        written += 1
  finally:
    conn.close()

  return {"written": written, "missing": missing}
