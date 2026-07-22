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

"""Group raw relation labels into a small LLM-curated taxonomy.

The question-generation pipeline operates on *categories* of relations
("corporate_action", "financial_report", "people_governance", ...) rather
than on the raw verbs the extractor produced. This module:

1. Counts raw relation frequencies from the edge CSV (pure Python).
2. Asks Gemini to group the frequent relations into a small set of
   thematic categories used downstream.

Splitting it from question generation lets the categorization be
recomputed cheaply whenever the edge CSV changes.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import logging
from pathlib import Path

from financegym.common.llm import DEFAULT_MODEL, get_client
from google.genai import types
from pydantic import BaseModel

log = logging.getLogger(__name__)


COUNT_BUCKETS: list[tuple[str, int]] = [
    ("1000+", 1000),
    ("500-999", 500),
    ("100-499", 100),
    ("50-99", 50),
    ("10-49", 10),
    ("5-9", 5),
    ("1-4", 1),
]


def count_relations(edges_path):
  """Tally ``relation`` values in an edge CSV. Returns ``(counter, total)``."""
  counts: Counter = Counter()
  total = 0
  with open(edges_path) as f:
    for row in csv.DictReader(f):
      counts[row["relation"]] += 1
      total += 1
  return counts, total


def bucket_for(n):
  for label, floor in COUNT_BUCKETS:
    if n >= floor:
      return label
  return COUNT_BUCKETS[-1][0]


def run_count(edges_path, output_path):
  """Write ``relation_counts.json`` and return its path."""
  counts, total = count_relations(edges_path)
  out = Path(output_path)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(
      json.dumps(
          {
              "total_edges": total,
              "unique_relations": len(counts),
              "relation_counts": dict(counts.most_common()),
          },
          indent=2,
      )
  )
  return out


# ---------------------------------------------------------------------------
# LLM categorization
# ---------------------------------------------------------------------------


class _CategoryAssignments(BaseModel):
  """One category per element."""

  relation: str
  category: str


class _Categorization(BaseModel):
  """LLM output: per-relation category labels."""

  assignments: list[_CategoryAssignments]


CATEGORIZE_SYSTEM = """\
You group raw verb-style relation labels into a small set of THEMATIC \
CATEGORIES for downstream financial-research question generation.

Pick a category for each input relation from this list (or invent a new \
snake_case category only when none fit):
  corporate_action       (acquired, partnered_with, invested_in, launched, ...)
  financial_report       (reported_revenue, reported_eps, beat_estimate, ...)
  people_governance      (ceo_of, appointed, resigned, board_member_of, ...)
  market_competition     (competes_with, subsidiary_of, supplier_to, ...)
  analyst_action         (upgraded, downgraded, set_price_target, ...)
  regulatory             (sued, fined, settled_with, regulated_by, ...)
  macro_policy           (raised_rate, set_target, signed_bill, ...)
  research_market_data   (covered_in, listed_on, indexed_by, ...)

Return one (relation, category) pair for every input verb."""


def run_categorize(
    counts_path,
    output_path,
    *,
    min_count = 10,
    model = DEFAULT_MODEL,
):
  """Send frequent relations to Gemini for grouping; write categories JSON."""
  data = json.loads(Path(counts_path).read_text())
  all_counts: dict[str, int] = data["relation_counts"]
  frequent = sorted(
      (r for r, c in all_counts.items() if c >= min_count),
      key=lambda r: -all_counts[r],
  )

  log.info(
      "categorizing %d frequent relations (min_count=%d)",
      len(frequent),
      min_count,
  )
  client = get_client()
  config = types.GenerateContentConfig(
      response_mime_type="application/json",
      response_schema=_Categorization,
      system_instruction=CATEGORIZE_SYSTEM,
      automatic_function_calling=types.AutomaticFunctionCallingConfig(
          disable=True
      ),
  )
  resp = client.models.generate_content(
      model=model,
      config=config,
      contents="Categorize:\n" + "\n".join(frequent),
  )
  parsed: _Categorization = resp.parsed

  by_category: dict[str, list[str]] = {}
  for a in parsed.assignments:
    by_category.setdefault(a.category, []).append(a.relation)

  out = Path(output_path)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(
      json.dumps(
          {
              "categories": {k: sorted(v) for k, v in by_category.items()},
              "metadata": {
                  "source": str(counts_path),
                  "frequent_relations": len(frequent),
                  "min_count": min_count,
                  "model": model,
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
  parser = argparse.ArgumentParser(description="Build relation categories.")
  sub = parser.add_subparsers(dest="step", required=True)

  p = sub.add_parser("count", help="Count relation frequencies in an edge CSV.")
  p.add_argument("--edges", required=True)
  p.add_argument("--output", default="output/corpus/relation_counts.json")

  p = sub.add_parser("categorize", help="LLM grouping of frequent relations.")
  p.add_argument("--counts", required=True)
  p.add_argument("--output", default="output/corpus/relation_categories.json")
  p.add_argument("--min-count", type=int, default=10)
  p.add_argument("--model", default=DEFAULT_MODEL)

  args = parser.parse_args()
  logging.basicConfig(
      level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s"
  )

  if args.step == "count":
    run_count(args.edges, args.output)
  else:
    run_categorize(
        args.counts, args.output, min_count=args.min_count, model=args.model
    )


if __name__ == "__main__":
  main()
