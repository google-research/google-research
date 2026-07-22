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

"""Build a curated finance domain whitelist from corpus metadata.

Three subcommands, all driven by the same on-disk inputs/outputs so they
can be replayed independently:

* ``count``   — read one or more ``metadata.jsonl`` snapshots (the output
  of :mod:`financegym.corpus.extract_embed`) and count documents per
  domain. Pure Python, no LLM.
* ``curate``  — send the top domains to Gemini and keep those that look
  like credible English-language information sources.
* ``cleanup`` — a stricter second Gemini pass that keeps only domains a
  US-focused financial practitioner would actually read.

The two LLM passes are deliberately split: the first errs on the side of
breadth (saves cost on the strict pass), the second errs on the side of
relevance.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
from pathlib import Path
import time
from urllib.parse import urlparse

from financegym.common.llm import DEFAULT_MODEL, get_client
from google.genai import types
from pydantic import BaseModel
from tqdm import tqdm

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def extract_domain(url):
  """Normalize a URL to its bare hostname (``www.`` stripped, lowercased)."""
  try:
    netloc = urlparse(url).netloc.lower()
    return netloc[4:] if netloc.startswith("www.") else netloc
  except Exception:
    return ""


# Document-count buckets used by the count CLI summary. Order matters for
# the printed report.
DOMAIN_BUCKETS: list[tuple[str, int]] = [
    ("100+", 100),
    ("50-99", 50),
    ("20-49", 20),
    ("10-19", 10),
    ("5-9", 5),
    ("2-4", 2),
    ("1", 1),
]


def bucket_for(count):
  """Return the bucket label a per-domain count falls into."""
  for label, floor in DOMAIN_BUCKETS:
    if count >= floor:
      return label
  return DOMAIN_BUCKETS[-1][0]


def select_by_coverage(
    domain_counts,
    total_docs,
    *,
    coverage = 0.95,
):
  """Return the head-set of domains that collectively cover ``coverage``."""
  ordered = sorted(domain_counts, key=lambda d: -domain_counts[d])
  out: list[str] = []
  running = 0
  threshold = coverage * total_docs
  for d in ordered:
    out.append(d)
    running += domain_counts[d]
    if running >= threshold:
      break
  return out


# ---------------------------------------------------------------------------
# Step 1 — count domains
# ---------------------------------------------------------------------------


def count_metadata_jsonl(paths):
  """Count per-domain occurrences across one or more metadata JSONL files.

  Returns ``(Counter, total_docs)``. Lines that don't parse are skipped.
  """
  counts: Counter = Counter()
  total = 0
  for p in paths:
    with open(p) as f:
      for line in f:
        line = line.strip()
        if not line:
          continue
        try:
          rec = json.loads(line)
        except json.JSONDecodeError:
          continue
        total += 1
        dom = rec.get("domain") or extract_domain(rec.get("url", ""))
        if dom:
          counts[dom] += 1
  return counts, total


def run_count(input_paths, output_path):
  """Count domain frequencies and write ``domain_counts.json``."""
  files: list[Path] = []
  for p in input_paths:
    pp = Path(p)
    if pp.is_dir():
      files.extend(sorted(pp.rglob("metadata.jsonl")))
    elif pp.is_file():
      files.append(pp)
  if not files:
    raise FileNotFoundError(
        f"no metadata.jsonl files found under {input_paths}"
    )

  counts, total = count_metadata_jsonl(files)
  log.info(
      "scanned %d metadata files, %d docs, %d unique domains",
      len(files),
      total,
      len(counts),
  )

  out = Path(output_path)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(
      json.dumps({
          "total_docs": total,
          "unique_domains": len(counts),
          "domain_counts": dict(counts.most_common()),
      })
  )
  return out


# ---------------------------------------------------------------------------
# Step 2 — broad-pass LLM curation
# ---------------------------------------------------------------------------


class _CurationResult(BaseModel):
  """Schema for the broad-pass LLM call.

  The ``finance_domains`` key is kept for backwards compatibility with the saved
  JSON shape.
  """

  finance_domains: list[str]


CURATE_SYSTEM = """\
You are reviewing website domains for SOURCE QUALITY. Given a list of domains, \
identify which ones are credible, established, ENGLISH-LANGUAGE information \
sources with editorial standards.

KEEP domains that are:
- English-language news organizations (national, regional, or industry-specific)
- English-language data/research platforms
- Government agencies, central banks, regulatory bodies (English sites)
- Major consulting, audit, or professional services firms
- Academic institutions and research organizations (English)
- Established wire services with editorial oversight
- Well-known English-language publications

REMOVE domains that are:
- Non-English language sites
- Low-quality press release aggregators with no editorial oversight
- Personal blogs or small unknown sites
- SEO-driven content farms
- Promotional or affiliate marketing sites
- Cryptocurrency exchange platforms
- Job listing or salary comparison sites
- E-commerce stores, coupon sites, product listing pages
- Sites with no clear editorial reputation or institutional backing

Return ONLY English-language credible domains. When in doubt, REMOVE."""


def _classify_batch(
    client, model, system, schema, prompt
):
  """One Gemini call returning the list of kept domains."""
  config = types.GenerateContentConfig(
      response_mime_type="application/json",
      response_schema=schema,
      system_instruction=system,
      automatic_function_calling=types.AutomaticFunctionCallingConfig(
          disable=True
      ),
  )
  resp = client.models.generate_content(
      model=model, config=config, contents=prompt
  )
  parsed = resp.parsed
  # The two schemas have different field names; read whichever is present.
  keys = (
      getattr(parsed, "finance_domains", None)
      or getattr(parsed, "keep_domains", None)
      or []
  )
  return [k.lower().strip() for k in keys]


def run_curate(
    counts_path,
    output_path,
    *,
    top_n = None,
    coverage = 0.95,
    model = DEFAULT_MODEL,
    batch_size = 200,
    workers = 16,
):
  """Broad-pass LLM curation. Writes the curated JSON, returns its path."""
  counts = json.loads(Path(counts_path).read_text())
  domain_counts: dict[str, int] = counts["domain_counts"]
  total_docs = counts["total_docs"]

  if top_n:
    candidates = list(domain_counts)[:top_n]
  else:
    candidates = select_by_coverage(
        domain_counts, total_docs, coverage=coverage
    )

  batches = [
      candidates[i : i + batch_size]
      for i in range(0, len(candidates), batch_size)
  ]
  log.info(
      "curating %d candidates in %d batches (model=%s)",
      len(candidates),
      len(batches),
      model,
  )

  client = get_client()
  kept: set[str] = set()
  candidate_set = {d.lower() for d in candidates}

  t0 = time.time()
  with ThreadPoolExecutor(max_workers=workers) as pool:
    futures = [
        pool.submit(
            _classify_batch,
            client,
            model,
            CURATE_SYSTEM,
            _CurationResult,
            "Classify these domains:\n" + "\n".join(b),
        )
        for b in batches
    ]
    for fut in tqdm(as_completed(futures), total=len(futures), desc="curate"):
      for d in fut.result():
        if d in candidate_set:
          kept.add(d)

  log.info(
      "curate done in %.0fs: kept %d / %d",
      time.time() - t0,
      len(kept),
      len(candidates),
  )
  out = Path(output_path)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(
      json.dumps(
          {
              "finance_domains": sorted(kept),
              "domain_doc_counts": {
                  d: domain_counts.get(d, 0) for d in sorted(kept)
              },
              "metadata": {
                  "source": str(counts_path),
                  "domains_classified": len(candidates),
                  "coverage_threshold": coverage,
                  "model": model,
                  "total_finance_domains": len(kept),
                  "total_docs_in_corpus": total_docs,
              },
          },
          indent=2,
      )
  )
  return out


# ---------------------------------------------------------------------------
# Step 3 — strict-pass LLM cleanup
# ---------------------------------------------------------------------------


class _CleanupResult(BaseModel):
  keep_domains: list[str]


CLEANUP_SYSTEM = """\
You are identifying which website domains a US-focused financial practitioner \
would read regularly and cite in their work.

Think about what a Wall Street equity researcher, US buy-side analyst, \
portfolio manager at a US fund, or macro strategist covering US markets would \
have in their daily reading list. Keep domains that cover:
- US stock market news, analysis, and commentary
- US-focused investment research, analyst ratings, price targets
- Corporate finance for US-listed companies: earnings, M&A, IPOs, executive changes
- US macro and economics: Federal Reserve policy, US indicators, Treasury yields
- Global events with direct US market impact
- Financial data platforms and market data providers used by US practitioners
- Major business publications that substantially cover US financial markets

Non-US regional sources should be kept ONLY if they are widely read by US \
practitioners for their US market coverage. Currency exchange rate pages and \
non-market financial services sites should be removed.

Return ONLY the domains relevant to US-focused financial practitioners."""


def run_cleanup(
    input_path,
    output_path,
    *,
    model = DEFAULT_MODEL,
    batch_size = 100,
    workers = 4,
):
  """Strict-pass LLM cleanup. Writes the final JSON, returns its path."""
  data = json.loads(Path(input_path).read_text())
  domains: list[str] = data["finance_domains"]
  counts: dict[str, int] = data.get("domain_doc_counts", {})

  batches = [
      domains[i : i + batch_size] for i in range(0, len(domains), batch_size)
  ]
  log.info("cleanup over %d domains in %d batches", len(domains), len(batches))

  client = get_client()
  candidate_set = {d.lower() for d in domains}
  kept: set[str] = set()

  t0 = time.time()
  with ThreadPoolExecutor(max_workers=workers) as pool:
    futures = [
        pool.submit(
            _classify_batch,
            client,
            model,
            CLEANUP_SYSTEM,
            _CleanupResult,
            "Review these domains. Return only the ones to KEEP:\n"
            + "\n".join(b),
        )
        for b in batches
    ]
    for fut in tqdm(as_completed(futures), total=len(futures), desc="cleanup"):
      for d in fut.result():
        if d in candidate_set:
          kept.add(d)

  removed = candidate_set - kept
  log.info(
      "cleanup done in %.0fs: kept %d / %d (removed %d)",
      time.time() - t0,
      len(kept),
      len(domains),
      len(removed),
  )
  out = Path(output_path)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(
      json.dumps(
          {
              "finance_domains": sorted(kept),
              "domain_doc_counts": {d: counts.get(d, 0) for d in sorted(kept)},
              "removed_domains": sorted(removed),
              "metadata": {
                  **data.get("metadata", {}),
                  "cleanup_model": model,
                  "pre_cleanup_count": len(domains),
                  "post_cleanup_count": len(kept),
                  "removed_count": len(removed),
              },
          },
          indent=2,
      )
  )
  return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
  parser = argparse.ArgumentParser(
      description="Build the finance domain whitelist."
  )
  sub = parser.add_subparsers(dest="step", required=True)

  p = sub.add_parser("count", help="Count domains in metadata.jsonl snapshots.")
  p.add_argument(
      "input", nargs="+", help="metadata.jsonl files or dirs to scan"
  )
  p.add_argument("--output", default="output/corpus/domain_counts.json")

  p = sub.add_parser("curate", help="Broad-pass LLM curation.")
  p.add_argument("input", help="domain_counts.json from `count`")
  p.add_argument(
      "--output", default="output/corpus/finance_domains_curated.json"
  )
  p.add_argument("--top", type=int, default=None)
  p.add_argument("--coverage", type=float, default=0.95)
  p.add_argument("--model", default=DEFAULT_MODEL)
  p.add_argument("--batch-size", type=int, default=200)
  p.add_argument("--workers", type=int, default=16)

  p = sub.add_parser("cleanup", help="Strict-pass LLM cleanup.")
  p.add_argument("input", help="finance_domains JSON from `curate`")
  p.add_argument("--output", default="output/corpus/finance_domains_clean.json")
  p.add_argument("--model", default=DEFAULT_MODEL)
  p.add_argument("--batch-size", type=int, default=100)
  p.add_argument("--workers", type=int, default=4)

  args = parser.parse_args()
  logging.basicConfig(
      level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s"
  )

  if args.step == "count":
    run_count(args.input, args.output)
  elif args.step == "curate":
    run_curate(
        args.input,
        args.output,
        top_n=args.top,
        coverage=args.coverage,
        model=args.model,
        batch_size=args.batch_size,
        workers=args.workers,
    )
  elif args.step == "cleanup":
    run_cleanup(
        args.input,
        args.output,
        model=args.model,
        batch_size=args.batch_size,
        workers=args.workers,
    )


if __name__ == "__main__":
  main()
