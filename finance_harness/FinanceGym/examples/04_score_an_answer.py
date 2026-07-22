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

"""Score one agent answer against its question's rubric.

Demonstrates the FinanceGym scoring contract end-to-end:

1. A question (with rubric + pre/post evidence metadata).
2. An agent's report.
3. The five-tier judge.
4. The macro summary + aggregate over a list of records.

Requires ``GOOGLE_API_KEY`` for the Gemini judge call.
"""

from __future__ import annotations

from pprint import pp

from financegym.judge.aggregate import axis_breakdown, macro
from financegym.judge.rubric_judge import judge_pair_to_record

QUESTION = {
    "question": "Why did AAPL revenue decelerate in Q1 2025?",
    "thesis": "Q1 2025 revenue deceleration",
    "cutoff": "2025-03-31",
    "topic": "earnings",
    "sector": "technology",
    "reasoning_type": "causal",
    "rubric": [
        {
            "category": "antecedent",
            "criterion": (
                "Identify the Q1 2025 revenue figure for AAPL and the YoY"
                " change."
            ),
        },
        {
            "category": "antecedent",
            "criterion": (
                "Mention macroeconomic headwinds (USD strength, China demand)"
                " cited by management."
            ),
        },
        {
            "category": "consequent",
            "criterion": (
                "Note the Q2 guidance issued by AAPL after Q1 results."
            ),
        },
    ],
    "metadata": {
        "pre_edge_evidence": [{
            "head": "AAPL",
            "relation": "reported_revenue",
            "tail": "$94.8B",
            "pub_date": "2025-02-01",
            "url": "https://example.com/aapl-q1-2025",
            "domain": "example.com",
            "context": "Apple reported Q1 2025 revenue of $94.8B, down 4% YoY.",
        }],
        "post_edge_evidence": [{
            "head": "AAPL",
            "relation": "guidance_q2",
            "tail": "low-single-digit growth",
            "pub_date": "2025-04-01",
            "url": "https://example.com/aapl-guidance",
            "domain": "example.com",
            "context": (
                "Apple guided Q2 2025 to low-single-digit YoY revenue growth."
            ),
        }],
        "source_urls_pre": ["https://example.com/aapl-q1-2025"],
        "source_urls_post": ["https://example.com/aapl-guidance"],
    },
}

REPORT = """\
Apple reported Q1 2025 revenue of $94.8B, a 4% YoY decline driven primarily
by USD strength and softer iPhone demand in China that management cited on
the earnings call. Management subsequently guided Q2 2025 to low-single-digit
revenue growth, signalling expected recovery in the next quarter.
"""


def main():
  record = judge_pair_to_record("example-agent", QUESTION, {"report": REPORT})
  if record is None:
    print("Judge returned None — check that the question has a rubric.")
    return
  print("per-question record:")
  pp({
      k: record[k]
      for k in ("agent", "total_sum", "total_max", "total_norm", "score_dist")
  })
  print("\nrubric breakdown:")
  for s in record["scores"]:
    print(f"  [{s['category']:10s}] {s['score']}/4  {s['criterion'][:80]}")

  # An aggregate over a single record is trivial but exercises the API.
  print("\nmacro across records:", macro([record]))
  print("topic breakdown:", axis_breakdown([record], "topic"))


if __name__ == "__main__":
  main()
