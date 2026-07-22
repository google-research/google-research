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

"""Generate a single research question from a tiny synthetic situation.

This example shows the v3 question pipeline in miniature: a hand-built
``situation`` dict (the kind that :mod:`financegym.questions.situation_mining`
produces), a chosen cutoff, and the categorical taxonomy the generator
needs for relation grouping. It calls Gemini for the actual generation,
so ``GOOGLE_API_KEY`` must be set.
"""

from __future__ import annotations

import json
from pprint import pp

from financegym.questions.generate import generate_question

CATEGORIES = {
    "corporate_action": ["acquired", "partnered_with", "launched"],
    "financial_report": [
        "reported_revenue",
        "reported_eps",
        "beat_estimate",
        "missed_estimate",
    ],
    "people_governance": ["ceo_of", "appointed", "resigned"],
    "analyst_action": ["upgraded", "downgraded", "set_price_target"],
}


def _edge(
    head, rel, tail, day, url = "", domain = ""
):
  return {
      "head": head,
      "relation": rel,
      "tail": tail,
      "pub_date": day,
      "url": url or f"https://example.com/{head}-{day}",
      "domain": domain or "example.com",
      "context": f"{head} {rel} {tail} on {day}",
  }


def main():
  # A tension situation: AAPL has both beat and miss signals over several months.
  pre_edges = [
      _edge("AAPL", "beat_estimate", f"Q{i}", f"2025-03-{(i % 28) + 1:02d}")
      for i in range(12)
  ]
  post_edges = [
      _edge("AAPL", "missed_estimate", f"Q{i}", f"2025-04-{(i % 28) + 1:02d}")
      for i in range(8)
  ]
  situation = {
      "situation_type": "tension_earnings_surprise",
      "focus_entities": ["AAPL"],
      "signal": "AAPL has both beat_estimate and missed_estimate signals",
      "edges": pre_edges + post_edges,
  }

  out = generate_question(
      situation,
      cutoff="2025-03-31",
      categories=CATEGORIES,
  )

  if out is None:
    print("Generator returned None — try a richer situation.")
    return

  print(json.dumps({k: v for k, v in out.items() if k != "metadata"}, indent=2))
  print("\nmetadata.source_domains:")
  pp(out["metadata"]["source_domains"])


if __name__ == "__main__":
  main()
