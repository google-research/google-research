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

"""Convert <cite id="...">claim</cite> to academic citation format.

For snippet_search (Semantic Scholar): [N, "Paper Title"]
For google_search: [N, website_title — URL]

Appends a References section at the bottom.

Usage:
    python convert_citations_academic.py input.jsonl output.jsonl
"""

import json
import re
import sys
from typing import Any


def extract_snippet_titles(generated_text):
  """Extract snippet_id -> title from tool output snippets in generated_text."""
  title_map = {}
  pattern = r'<snippet id="?([^">\s]+)"?>\s*\nTitle:\s*(.*?)(?:\n|$)'
  for match in re.finditer(pattern, generated_text):
    sid = match.group(1).strip()
    title = match.group(2).strip()
    if title:
      title_map[sid] = {"title": title, "type": "paper"}
  return title_map


def build_snippet_to_doc_map(
    full_traces, generated_text
):
  """Map snippet IDs (from generated_text) to document metadata (from tool_calls.documents).

  Snippet IDs like '84d1e2c4-0' appear in the tool output text.
  Document IDs like '9d8767dd-...' are in the documents list.
  They correspond positionally within each tool call (excluding the summary
  doc).

  Args:
    full_traces: Trace dictionary containing tool_calls and documents.
    generated_text: Raw generated text containing snippet tags.

  Returns:
    Mapping from snippet/document ID to metadata dictionary.
  """
  snippet_to_meta = {}

  for tc in full_traces.get("tool_calls", []):
    tool_name = tc.get("tool_name", "")
    docs = tc.get("documents", [])
    if not docs:
      continue

    # For snippet_search, first doc is summary, rest are papers
    if "snippet" in tool_name:
      paper_docs = [d for d in docs if not d.get("id", "").endswith("-summary")]
    else:
      paper_docs = docs

    # Match docs to snippet IDs by title
    for doc in paper_docs:
      doc_title = doc.get("title", "").strip()
      authors = doc.get("authors", [])
      year = doc.get("year")
      url = doc.get("url", "").strip()
      doc_id = doc.get("id", "")

      if not doc_title:
        continue

      author_short = _format_author_short(authors) if authors else None
      meta = {
          "title": doc_title,
          "url": url,
          "authors": author_short,
          "year": year,
          "type": "paper" if "snippet" in tool_name else "web",
      }

      # Store by doc_id (for direct matching)
      snippet_to_meta[doc_id] = meta

      # Also try to find matching snippet IDs by title in generated_text
      # Snippet IDs in tool_output look like: <snippet id=HASH-INDEX>
      # We match by title since the same paper title appears in both
      for sid, sinfo in extract_snippet_titles(generated_text).items():
        if sinfo.get("title", "").strip() == doc_title:
          snippet_to_meta[sid] = meta

    # For google_search docs, doc_id IS the snippet ID (S_xxx format)
    if "google" in tool_name:
      for doc in paper_docs:
        doc_id = doc.get("id", "")
        title = doc.get("title", "").strip()
        url = doc.get("url", "").strip()
        if doc_id:
          snippet_to_meta[doc_id] = {
              "title": title,
              "url": url,
              "authors": None,
              "year": None,
              "type": "web",
          }

  return snippet_to_meta


def _format_author_short(authors):
  """Format authors as 'First et al.' or 'First and Second'."""
  if not authors:
    return None
  # Get last names (assume "First Last" format)
  last_names = []
  for a in authors:
    parts = a.strip().split()
    if parts:
      last_names.append(parts[-1])
  if len(last_names) == 1:
    return last_names[0]
  elif len(last_names) == 2:
    return f"{last_names[0]} and {last_names[1]}"
  else:
    return f"{last_names[0]} et al."


def extract_doc_metadata(full_traces):
  """Extract document metadata from full_traces.tool_calls.documents."""
  meta_map = {}
  for tc in full_traces.get("tool_calls", []):
    tool_name = tc.get("tool_name", "")
    for doc in tc.get("documents", []):
      doc_id = doc.get("id", "")
      title = doc.get("title", "").strip()
      url = doc.get("url", "").strip()
      authors = doc.get("authors", [])
      year = doc.get("year")
      if not doc_id or not title:
        continue
      author_short = _format_author_short(authors) if authors else None
      if "snippet" in tool_name:
        meta_map[doc_id] = {
            "title": title,
            "url": url,
            "type": "paper",
            "authors": author_short,
            "year": year,
        }
      elif "google" in tool_name:
        meta_map[doc_id] = {
            "title": title,
            "url": url,
            "type": "web",
            "authors": None,
            "year": None,
        }
  return meta_map


def convert_citations(text, source_map):
  """Convert <cite id="...">claim</cite> to claim [N, source] format."""
  ref_map = {}
  counter = [0]

  def _replace(match):
    ids_str = match.group(1)
    claim_text = match.group(2)
    ids = [s.strip() for s in ids_str.split(",")]
    ref_labels = []
    for sid in ids:
      if sid not in ref_map:
        src = source_map.get(sid)
        if src:
          counter[0] += 1
          ref_map[sid] = {
              "num": counter[0],
              "title": src["title"],
              "url": src.get("url", ""),
              "type": src.get("type", "paper"),
              "authors": src.get("authors"),
              "year": src.get("year"),
          }
      if sid in ref_map:
        info = ref_map[sid]
        # Build inline label: [N, Author et al., Year] or [N]
        if info["authors"] and info["year"]:
          ref_labels.append(f"{info['num']}, {info['authors']}, {info['year']}")
        elif info["authors"]:
          ref_labels.append(f"{info['num']}, {info['authors']}")
        else:
          ref_labels.append(str(info["num"]))

    if ref_labels:
      return f"{claim_text} [{'; '.join(ref_labels)}]"
    return claim_text

  converted = re.sub(
      r'<cite\s+id="([^"]*)">(.*?)</cite>', _replace, text, flags=re.DOTALL
  )

  if ref_map:
    converted += "\n\nReferences:\n"
    for _, info in sorted(ref_map.items(), key=lambda x: x[1]["num"]):
      num = info["num"]
      title = info["title"]
      url = info["url"]
      authors = info["authors"]
      year = info["year"]
      if info["type"] == "paper":
        parts = [f"[{num}]"]
        if authors:
          parts.append(authors)
        if year:
          parts.append(f"({year}).")
        parts.append(f'"{title}."')
        converted += " ".join(parts) + "\n"
      elif url:
        converted += f"[{num}] {title} — {url}\n"
      else:
        converted += f"[{num}] {title}\n"

  return converted


def process_example(data):
  """Process a single JSONL example."""
  if "final_response" not in data or "<cite" not in data["final_response"]:
    return data

  gen_text = data.get("full_traces", {}).get("generated_text", "")
  full_traces = data.get("full_traces", {})

  # Build unified source map: snippet_id -> {title, authors, year, url, type}
  # Priority: build_snippet_to_doc_map (has authors/year) > snippet_titles
  source_map = extract_snippet_titles(gen_text)
  doc_map = extract_doc_metadata(full_traces)
  snippet_doc_map = build_snippet_to_doc_map(full_traces, gen_text)

  source_map.update(doc_map)
  source_map.update(snippet_doc_map)  # highest priority — has authors/year

  if source_map:
    data["final_response"] = convert_citations(
        data["final_response"], source_map
    )

  return data


def main():
  if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} <input.jsonl> <output.jsonl>")
    sys.exit(1)

  input_path, output_path = sys.argv[1], sys.argv[2]
  count = 0
  converted = 0

  with open(input_path, "r") as fin, open(output_path, "w") as fout:
    for line in fin:
      if not line.strip():
        continue
      data = json.loads(line)
      count += 1
      had_cite = "<cite" in data.get("final_response", "")
      data = process_example(data)
      if had_cite and "<cite" not in data.get("final_response", ""):
        converted += 1
      fout.write(json.dumps(data, ensure_ascii=False) + "\n")

  print(f"Processed {count} examples, converted citations in {converted}")


if __name__ == "__main__":
  main()
