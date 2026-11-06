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

#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ResearchRubrics evaluation pipeline.

1. Converts our JSONL output to markdown files
2. Runs the ResearchRubrics rubric evaluation (Gemini)
3. Calculates and reports compliance scores

Usage:
    python eval_researchrubrics.py --input results.jsonl --output-dir
    /path/to/output
    python eval_researchrubrics.py --input results.jsonl --output-dir
    /path/to/output --max-concurrent 10
"""

import argparse
import asyncio
import collections
import json
import os
import pathlib
import re
import sys
from typing import Any

RESEARCHRUBRICS_REPO = pathlib.Path(
    os.environ.get("RESEARCHRUBRICS_REPO", "/path/to/researchrubrics")
)
RESEARCHRUBRICS_DATA = (
    RESEARCHRUBRICS_REPO / "data" / "researchrubrics" / "processed_data.jsonl"
)

# Add researchrubrics src to path
sys.path.insert(0, str(RESEARCHRUBRICS_REPO / "src" / "evaluate_rubrics"))


def extract_answer(final_response):
  """Extract content from <answer>...</answer> tags, or return full response."""
  match = re.search(r"<answer>(.*?)</answer>", final_response, re.DOTALL)
  if match:
    return match.group(1).strip()
  return final_response.strip()


def strip_citations(text):
  """Remove <cite id="...">...</cite> tags, keeping inner text."""
  return re.sub(
      r'<cite\s+id="[^"]*">(.*?)</cite>', r"\1", text, flags=re.DOTALL
  )


def build_snippet_source_map(full_traces):
  """Build snippet_id -> {url, title} map from full_traces.tool_calls.documents."""
  source_map = {}
  if not full_traces:
    return source_map
  for tc in full_traces.get("tool_calls", []):
    docs = tc.get("documents", [])
    if not docs:
      continue
    for doc in docs:
      if isinstance(doc, dict):
        doc_id = doc.get("id", "")
        url = doc.get("url", "").strip()
        title = doc.get("title", "").strip()
        if doc_id and (url or title):
          source_map[doc_id] = {"url": url, "title": title}
  return source_map


def convert_citations_to_numbered(text, source_map):
  """Convert <cite id="S_xxx">claim</cite> to claim [N] with references at bottom.

  Uses source_map to resolve snippet IDs to actual URLs/titles.

  Args:
    text: Input text containing <cite> tags.
    source_map: Mapping from snippet ID to source metadata.

  Returns:
    Text with numbered citations and appended References section.
  """
  ref_map = {}  # snippet_id -> ref_number
  counter = [0]

  def _replace(match):
    ids_str = match.group(1)
    claim_text = match.group(2)
    ids = [s.strip() for s in ids_str.split(",")]
    ref_nums = []
    for sid in ids:
      if sid not in ref_map:
        counter[0] += 1
        ref_map[sid] = counter[0]
      ref_nums.append(str(ref_map[sid]))
    return f"{claim_text} [{', '.join(ref_nums)}]"

  converted = re.sub(
      r'<cite\s+id="([^"]*)">(.*?)</cite>', _replace, text, flags=re.DOTALL
  )

  if ref_map:
    converted += "\n\n---\n\n## References\n\n"
    for sid, num in sorted(ref_map.items(), key=lambda x: x[1]):
      src = source_map.get(sid, {})
      url = src.get("url", "")
      title = src.get("title", "")
      if url:
        converted += (
            f"[{num}] {title} — {url}\n\n" if title else f"[{num}] {url}\n\n"
        )
      elif title:
        converted += f"[{num}] {title}\n\n"
      else:
        converted += f"[{num}] (source not available)\n\n"

  return converted


def convert_jsonl_to_markdown(
    input_jsonl, output_dir, no_citation = False
):
  """Convert our JSONL generation output to markdown files.

  Args:
    input_jsonl: Path to input JSONL file.
    output_dir: Directory to write markdown responses.
    no_citation: If True, strip citation tags instead of converting.

  Returns:
    List of (sample_id, markdown_path) tuples.
  """
  output_path = pathlib.Path(output_dir) / "agent_responses"
  output_path.mkdir(parents=True, exist_ok=True)

  results = []
  with open(input_jsonl, "r") as f:
    for line in f:
      if not line.strip():
        continue
      data = json.loads(line)
      sample_id = data.get("example_id", "")
      final_response = data.get("final_response", "")

      # Extract answer content
      answer_text = extract_answer(final_response)

      # Convert citations to numbered format with real URLs
      if no_citation:
        answer_text = strip_citations(answer_text)
      else:
        source_map = build_snippet_source_map(data.get("full_traces", {}))
        answer_text = convert_citations_to_numbered(answer_text, source_map)

      # Write as markdown
      md_path = output_path / f"{sample_id}.md"
      md_path.write_text(answer_text, encoding="utf-8")
      results.append((sample_id, md_path))

  print(f"Converted {len(results)} responses to markdown in {output_path}")
  return results


def load_rubrics_map():
  """Load rubrics from processed_data.jsonl, keyed by sample_id."""
  rubrics_map = {}
  with open(RESEARCHRUBRICS_DATA, "r") as f:
    for line in f:
      if not line.strip():
        continue
      data = json.loads(line)
      rubrics_map[data["sample_id"]] = data["rubrics"]
  return rubrics_map


async def run_evaluation(
    input_jsonl,
    output_dir,
    max_concurrent = 20,
    model = "gemini-2.5-pro",
    no_citation = False,
):
  """Run full ResearchRubrics evaluation pipeline."""
  del max_concurrent

  # Step 1: Convert JSONL to markdown
  print("=" * 60)
  print("Step 1: Converting JSONL to markdown files")
  print("=" * 60)
  converted = convert_jsonl_to_markdown(
      input_jsonl, output_dir, no_citation=no_citation
  )

  if not converted:
    print("No responses to evaluate!")
    return

  # Step 2: Load rubrics
  print("\n" + "=" * 60)
  print("Step 2: Loading rubrics")
  print("=" * 60)
  rubrics_map = load_rubrics_map()
  print(f"Loaded rubrics for {len(rubrics_map)} tasks")

  # Step 3: Run evaluation
  print("\n" + "=" * 60)
  print(f"Step 3: Running rubric evaluation ({len(converted)} reports)")
  print("=" * 60)

  from evaluate_single_report import evaluate_task_rubrics  # pylint: disable=g-import-not-at-top,g-importing-member

  all_results = []
  compliance_scores = {}

  for i, (sample_id, md_path) in enumerate(converted, 1):
    if sample_id not in rubrics_map:
      print(f"  [{i}/{len(converted)}] Skipping {sample_id}: no rubrics found")
      continue

    print(f"  [{i}/{len(converted)}] Evaluating: {sample_id}")
    try:
      results_df, compliance_score = await evaluate_task_rubrics(
          str(md_path),
          rubrics_jsonl_path=str(RESEARCHRUBRICS_DATA),
          save_results=False,
          model=model,
      )

      compliance_scores[sample_id] = compliance_score

      # Get axis info from rubrics
      sample_rubrics = rubrics_map[sample_id]
      for idx, row in results_df.iterrows():
        axis = (
            sample_rubrics[idx]["axis"]
            if idx < len(sample_rubrics)
            else "Unknown"
        )
        all_results.append({
            "sample_id": sample_id,
            "rubric_title": row["rubric_title"],
            "verdict": row["verdict"],
            "score": row["score"],
            "confidence": row["confidence"],
            "reasoning": row.get("reasoning", "")[:500],
            "tokens_used": row.get("tokens_used", 0),
            "success": row.get("success", True),
            "weight": row.get("weight", 1.0),
            "axis": axis,
        })

    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f"  Failed to evaluate {sample_id}: {e}")
      continue

  # Step 4: Save results
  print("\n" + "=" * 60)
  print("Step 4: Saving results")
  print("=" * 60)

  results_dir = pathlib.Path(output_dir)
  results_dir.mkdir(parents=True, exist_ok=True)

  # Save per-rubric results
  results_jsonl = results_dir / "researchrubrics_rubric_results.jsonl"
  with open(results_jsonl, "w") as f:
    for result in all_results:
      f.write(json.dumps(result) + "\n")
  print(f"  Rubric results: {results_jsonl}")

  # Save summary
  avg_compliance = (
      sum(compliance_scores.values()) / len(compliance_scores)
      if compliance_scores
      else 0.0
  )

  verdict_counts = collections.defaultdict(int)
  for r in all_results:
    verdict_counts[r["verdict"]] = verdict_counts.get(r["verdict"], 0) + 1

  # Per-axis breakdown
  axis_scores = collections.defaultdict(list)
  for r in all_results:
    axis_scores[r.get("axis", "Unknown")].append(r["score"])
  per_axis = {
      axis: {
          "mean": round(sum(scores) / len(scores), 4),
          "count": len(scores),
          "satisfied": sum(1 for s in scores if s > 0),
      }
      for axis, scores in sorted(axis_scores.items())
  }

  summary = {
      "task": "researchrubrics",
      "total_samples": len(compliance_scores),
      "average_compliance_score": round(avg_compliance, 4),
      "score": round(avg_compliance, 4),
      "per_axis_scores": per_axis,
      "per_sample_scores": {
          k: round(v, 4) for k, v in sorted(compliance_scores.items())
      },
      "verdict_distribution": dict(verdict_counts),
  }

  summary_path = results_dir / "researchrubrics_eval_results.json"
  with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
  print(f"  Summary: {summary_path}")

  # Print summary
  print("\n" + "=" * 60)
  print("RESEARCHRUBRICS EVALUATION SUMMARY")
  print("=" * 60)
  print(f"  Total Samples Evaluated: {len(compliance_scores)}")
  print(f"  Average Compliance Score: {avg_compliance:.4f}")
  print("\n  Per-Axis Scores:")
  for axis, info in per_axis.items():
    print(
        f"    {axis}: {info['mean']:.4f} ({info['satisfied']}/{info['count']}"
        " satisfied)"
    )
  print("\n  Verdict Distribution:")
  for verdict, count in sorted(verdict_counts.items()):
    pct = count / len(all_results) * 100 if all_results else 0
    print(f"    {verdict}: {count} ({pct:.1f}%)")


def main():
  parser = argparse.ArgumentParser(
      description="ResearchRubrics evaluation pipeline"
  )
  parser.add_argument(
      "--input", required=True, help="Input JSONL file from generation"
  )
  parser.add_argument(
      "--output-dir", required=True, help="Output directory for results"
  )
  parser.add_argument(
      "--max-concurrent", type=int, default=20, help="Max concurrent API calls"
  )
  parser.add_argument("--model", default="gemini-2.5-pro", help="Judge model")
  parser.add_argument(
      "--no-citation",
      action="store_true",
      help="Strip citation tags before evaluation",
  )
  args = parser.parse_args()

  asyncio.run(
      run_evaluation(
          input_jsonl=args.input,
          output_dir=args.output_dir,
          max_concurrent=args.max_concurrent,
          model=args.model,
          no_citation=args.no_citation,
      )
  )


if __name__ == "__main__":
  main()
