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

"""Classify each finance-domain article and extract entity-relation triples.

Reads the ``metadata.jsonl`` + ``texts.jsonl`` written by
:mod:`financegym.corpus.extract_embed`, filters by the finance domain
whitelist, and calls Gemini with a pinned structured-output schema.
Outputs an append-only edge CSV plus a per-document JSONL with raw LLM
results for debugging.

The Gemini prompt itself is part of the benchmark contract — it defines
what counts as a "finance document" and the relation taxonomy. It is
pinned in :data:`SYSTEM_INSTRUCTION` and any change should be tracked in
``docs/reproducibility.md``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import logging
from pathlib import Path
import threading
import time

from financegym.common.llm import DEFAULT_MODEL, get_client
from google.genai import types
from pydantic import BaseModel
from tqdm import tqdm

log = logging.getLogger(__name__)


EDGE_COLUMNS = [
    "head",
    "relation",
    "tail",
    "context",
    "url",
    "domain",
    "pub_date",
    "crawl_date",
]

TEXT_LIMIT = 2_000
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0


# ---------------------------------------------------------------------------
# Pinned schema + prompt (benchmark contract)
# ---------------------------------------------------------------------------


class Relation(BaseModel):
  head: str
  relation: str
  tail: str
  context: str


class DocResult(BaseModel):
  is_finance: bool
  relations: list[Relation] | None = None


SYSTEM_INSTRUCTION = """\
You are a financial document classifier and knowledge graph extractor.

Step 1 — Classification:
Set is_finance to true if a financial analyst, portfolio manager, or equity \
researcher would cite this document in a research report. The key test: does \
the document contain specific financial facts or material corporate events \
tied to named companies, institutions, or markets? Examples:
- "Stellantis reported a 27% decline in Q3 revenue"
- "Goldman Sachs initiated coverage on TDOC with a $14 price target"
- "The Federal Reserve raised rates by 25bp in March"
- "BP formed a $300M joint venture with Verenium"
- "The SEC opened an investigation into Wirecard's accounting practices"

Set is_finance to false for everything else, including general business news \
without financial data or corporate events, product reviews, sports, \
politics, and promotional content.

Step 2 — Extraction (only if is_finance is true):
Extract entity-relation triples. Use concise snake_case relation verbs \
covering the kinds of relationships a financial analyst would track, e.g.:
- Corporate actions: acquired, partnered_with, invested_in, launched, spun_off
- Financial data: reported_revenue, reported_eps, set_price_target
- People & governance: ceo_of, appointed, resigned, analyst_at
- Competition & structure: competes_with, subsidiary_of, owns, operates_in
- Analyst & regulatory: upgraded, downgraded, filed_lawsuit, regulated_by

Entity normalization:
- Publicly traded US companies → stock ticker (e.g. "AAPL", "MSFT")
- People → full name (e.g. "Tim Cook", "Jerome Powell")
- Organizations → official name (e.g. "Goldman Sachs", "Federal Reserve")

For each triple, provide a one-sentence context grounded in the document.

If is_finance is false, set relations to null."""


def _make_config():
  return types.GenerateContentConfig(
      response_mime_type="application/json",
      response_schema=DocResult,
      system_instruction=SYSTEM_INSTRUCTION,
      automatic_function_calling=types.AutomaticFunctionCallingConfig(
          disable=True
      ),
  )


# ---------------------------------------------------------------------------
# Pure helpers (domain matching, candidate loading)
# ---------------------------------------------------------------------------


def load_whitelist(path):
  """Load the curated domain whitelist (``finance_domains_clean.json``)."""
  data = json.loads(Path(path).read_text())
  return set(data["finance_domains"])


def domain_matches(domain, whitelist):
  """``True`` if the domain is in the whitelist or is a subdomain of one."""
  if domain in whitelist:
    return True
  return any(domain.endswith("." + w) for w in whitelist)


def load_candidates(
    metadata_path,
    texts_path,
    whitelist,
):
  """Read metadata + texts in lock-step, keeping only whitelisted domains."""
  metas: list[dict] = []
  with open(metadata_path) as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      try:
        metas.append(json.loads(line))
      except json.JSONDecodeError:
        break

  keep_idx = {
      i
      for i, m in enumerate(metas)
      if domain_matches(m.get("domain", ""), whitelist)
  }
  if not keep_idx:
    return []

  out: list[dict] = []
  with open(texts_path) as f:
    for i, line in enumerate(f):
      if i >= len(metas):
        break
      if i not in keep_idx:
        continue
      line = line.strip()
      if not line:
        continue
      try:
        td = json.loads(line)
      except json.JSONDecodeError:
        break
      m = metas[i]
      out.append({
          "doc_id": m["doc_id"],
          "url": m.get("url", ""),
          "domain": m.get("domain", ""),
          "pub_date": m.get("pub_date", ""),
          "crawl_date": m.get("crawl_date", ""),
          "text": td.get("text", ""),
      })
  return out


def relations_to_edge_rows(doc):
  """Project one processed doc's relations into edge CSV rows."""
  if not doc.get("is_finance"):
    return []
  rows: list[dict] = []
  for r in doc.get("relations", []) or []:
    rows.append({
        "head": r["head"],
        "relation": r["relation"],
        "tail": r["tail"],
        "context": r["context"],
        "url": doc.get("url", ""),
        "domain": doc.get("domain", ""),
        "pub_date": doc.get("pub_date", ""),
        "crawl_date": doc.get("crawl_date", ""),
    })
  return rows


# ---------------------------------------------------------------------------
# Per-doc LLM call with retry + thread-safe stats
# ---------------------------------------------------------------------------


_stats_lock = threading.Lock()
_stats = {"success": 0, "retried": 0, "failed": 0}


def get_stats():
  with _stats_lock:
    return dict(_stats)


def reset_stats():
  with _stats_lock:
    for k in _stats:
      _stats[k] = 0


def process_doc(
    doc,
    *,
    model = DEFAULT_MODEL,
    client=None,
    config = None,
    max_retries = MAX_RETRIES,
    sleep=time.sleep,
):
  """Classify + extract for one document with rate-limit backoff."""
  cli = client or get_client()
  cfg = config or _make_config()
  text = (doc.get("text") or "")[:TEXT_LIMIT]

  for attempt in range(max_retries + 1):
    try:
      resp = cli.models.generate_content(model=model, config=cfg, contents=text)
      parsed = resp.parsed
      doc["is_finance"] = parsed.is_finance
      doc["relations"] = (
          [r.model_dump() for r in parsed.relations or []]
          if parsed.relations
          else []
      )
      with _stats_lock:
        _stats["success"] += 1
        if attempt > 0:
          _stats["retried"] += 1
      return doc
    except Exception as e:  # noqa: BLE001
      err = str(e)
      rate_limited = "429" in err or "RESOURCE_EXHAUSTED" in err
      if rate_limited and attempt < max_retries:
        sleep(RETRY_BASE_DELAY * (2**attempt))
        continue
      with _stats_lock:
        _stats["failed"] += 1
      doc["_llm_failed"] = True
      return doc
  return doc


# ---------------------------------------------------------------------------
# Orchestrator + simple stats reporter
# ---------------------------------------------------------------------------


def run_extraction(
    candidates,
    *,
    model = DEFAULT_MODEL,
    workers = 32,
    checkpoint_path = None,
    jsonl_path = None,
    edges_path = None,
):
  """Run the LLM pass over ``candidates`` and append-write outputs.

  Returns ``(total_processed, finance_count, relation_count)``. Honours
  a checkpoint file: documents whose ``doc_id`` is already in
  ``checkpoint_path`` are skipped.
  """
  done: set[str] = set()
  if checkpoint_path and checkpoint_path.exists():
    done = set(checkpoint_path.read_text().splitlines()) - {""}
  candidates = [c for c in candidates if c["doc_id"] not in done]
  if not candidates:
    return 0, 0, 0

  cp_file = open(checkpoint_path, "a") if checkpoint_path else None  # noqa: SIM115
  jsonl_file = open(jsonl_path, "a") if jsonl_path else None  # noqa: SIM115
  edges_file = None
  edges_writer = None
  if edges_path:
    edges_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not edges_path.exists() or edges_path.stat().st_size == 0
    edges_file = open(edges_path, "a", newline="")  # noqa: SIM115
    edges_writer = csv.DictWriter(edges_file, fieldnames=EDGE_COLUMNS)
    if write_header:
      edges_writer.writeheader()

  finance_count = 0
  relation_count = 0
  total = 0
  reset_stats()
  client = get_client()
  config = _make_config()
  t0 = time.time()

  with ThreadPoolExecutor(max_workers=workers) as pool:
    futs = {
        pool.submit(
            process_doc, c, model=model, client=client, config=config
        ): c
        for c in candidates
    }
    for fut in tqdm(as_completed(futs), total=len(futs), desc="extract"):
      res = fut.result()
      total += 1
      if res.get("_llm_failed"):
        continue
      if cp_file:
        cp_file.write(res["doc_id"] + "\n")
      if jsonl_file:
        jsonl_file.write(
            json.dumps({
                "doc_id": res.get("doc_id", ""),
                "url": res.get("url", ""),
                "domain": res.get("domain", ""),
                "pub_date": res.get("pub_date", ""),
                "is_finance": res.get("is_finance", False),
                "relations": res.get("relations", []),
            })
            + "\n"
        )
      if res.get("is_finance"):
        finance_count += 1
        relation_count += len(res.get("relations", []))
        if edges_writer:
          edges_writer.writerows(relations_to_edge_rows(res))
      if total % 200 == 0:
        for f in (cp_file, jsonl_file, edges_file):
          if f:
            f.flush()

  for f in (cp_file, jsonl_file, edges_file):
    if f:
      f.flush()
      f.close()

  log.info(
      "extracted %d edges from %d finance docs (of %d) in %.0fs",
      relation_count,
      finance_count,
      total,
      time.time() - t0,
  )
  return total, finance_count, relation_count


def summarize_edges(edges_path):
  """Return basic stats for an existing edge CSV (entities, relations, months)."""
  heads: Counter = Counter()
  tails: Counter = Counter()
  rels: Counter = Counter()
  months: Counter = Counter()
  total = 0
  with open(edges_path) as f:
    for row in csv.DictReader(f):
      heads[row["head"]] += 1
      tails[row["tail"]] += 1
      rels[row["relation"]] += 1
      pub = row.get("pub_date", "")
      if pub and len(pub) >= 7:
        months[pub[:7]] += 1
      total += 1
  return {
      "total_edges": total,
      "unique_heads": len(heads),
      "unique_tails": len(tails),
      "unique_entities": len(set(heads) | set(tails)),
      "unique_relations": len(rels),
      "top_entities": dict((heads + tails).most_common(15)),
      "top_relations": dict(rels.most_common(15)),
      "edges_by_month": dict(sorted(months.items())),
  }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
  parser = argparse.ArgumentParser(
      description="Build the finance entity graph."
  )
  parser.add_argument(
      "--texts", required=True, help="texts.jsonl from extract_embed"
  )
  parser.add_argument(
      "--metadata", required=True, help="metadata.jsonl from extract_embed"
  )
  parser.add_argument(
      "--whitelist", required=True, help="finance_domains JSON file"
  )
  parser.add_argument(
      "--output", default="data/graphs/edges.csv", help="Edge CSV output path"
  )
  parser.add_argument(
      "--jsonl-output",
      default="output/corpus/graph_results.jsonl",
      help="Per-doc LLM results",
  )
  parser.add_argument("--model", default=DEFAULT_MODEL)
  parser.add_argument("--workers", type=int, default=32)
  parser.add_argument("--dry-run", action="store_true")
  args = parser.parse_args()
  logging.basicConfig(
      level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s"
  )

  wl = load_whitelist(args.whitelist)
  candidates = load_candidates(args.metadata, args.texts, wl)
  log.info("loaded %d whitelist-matching candidates", len(candidates))
  if args.dry_run or not candidates:
    return

  jsonl = Path(args.jsonl_output)
  edges = Path(args.output)
  jsonl.parent.mkdir(parents=True, exist_ok=True)
  edges.parent.mkdir(parents=True, exist_ok=True)
  checkpoint = jsonl.with_suffix(".checkpoint")

  run_extraction(
      candidates,
      model=args.model,
      workers=args.workers,
      checkpoint_path=checkpoint,
      jsonl_path=jsonl,
      edges_path=edges,
  )

  if edges.exists():
    for k, v in summarize_edges(edges).items():
      log.info("  %s: %s", k, v)


if __name__ == "__main__":
  main()
