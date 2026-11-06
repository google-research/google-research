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

r"""Download Google Search batch job results from the Gemini Batch API.

Polls the batch job until completion, then downloads and saves results.

Usage:
    python tool_call_stage/tool_batch_download.py \
        --run_id my_run --round 1 --part 1
"""

import argparse
import json
import os
import time

from google import genai


def _format_search_tool_output(entry):
  """Formats a Gemini grounded search response into a <tool_output> block."""
  if entry.get("error"):
    return (
        "<tool_output>\nError executing Google Search:"
        f" {entry['error']}\n</tool_output>"
    )
  response = entry.get("response", entry)
  if not isinstance(response, dict):
    return (
        "<tool_output>\nSearch results were blocked or failed.\n</tool_output>"
    )
  candidates = response.get("candidates", [])
  if not candidates or not isinstance(candidates, list):
    return (
        "<tool_output>\nSearch results were blocked or failed.\n</tool_output>"
    )
  cand = candidates[0] if isinstance(candidates[0], dict) else {}
  content = cand.get("content", {}) if isinstance(cand, dict) else {}
  parts = content.get("parts", []) if isinstance(content, dict) else []
  text_parts = [
      p.get("text", "")
      for p in parts
      if isinstance(p, dict) and not p.get("thought") and p.get("text")
  ]
  summary_text = "".join(text_parts).strip()
  if not summary_text:
    return "<tool_output>\n\nNo grounding data available.\n</tool_output>"
  return f"<tool_output>\n{summary_text}\n</tool_output>"


def main():
  parser = argparse.ArgumentParser(
      description="Download Google Search batch results."
  )
  parser.add_argument("--run_id", type=str, required=True)
  parser.add_argument("--round", type=int, required=True)
  parser.add_argument("--part", type=int, default=1)
  parser.add_argument("--input_dir", type=str, default="files")
  parser.add_argument("--poll_interval", type=int, default=30)
  args = parser.parse_args()

  api_key = os.environ.get("GEMINI_API_KEY")
  if not api_key:
    print("Error: GEMINI_API_KEY not set")
    exit(1)

  client = genai.Client(api_key=api_key)

  google_dir = os.path.join(
      args.input_dir,
      "rounds_processing",
      f"round_{args.round}",
      "google_search",
  )
  rounds_meta_file = os.path.join(
      google_dir, f"{args.run_id}_batch_meta_part_{args.part}.json"
  )
  legacy_meta_file = os.path.join(
      args.input_dir,
      args.run_id,
      f"round_{args.round}",
      f"part_{args.part}",
      "google_search_batch_meta.json",
  )
  meta_file = (
      rounds_meta_file
      if os.path.exists(rounds_meta_file)
      else legacy_meta_file
  )

  if not os.path.exists(meta_file):
    print(f"Job metadata not found: {meta_file}")
    exit(0)

  with open(meta_file, "r", encoding="utf-8") as f:
    meta = json.load(f)

  job_name = meta["job_name"]
  print(f"Polling batch job: {job_name}")

  while True:
    job = client.batches.get(name=job_name)
    state = job.state.name if hasattr(job.state, "name") else str(job.state)
    print(f"  Status: {state}")

    if state in ("SUCCEEDED", "JOB_STATE_SUCCEEDED"):
      print("Batch job completed!")
      break
    elif state in (
        "FAILED",
        "JOB_STATE_FAILED",
        "CANCELLED",
        "JOB_STATE_CANCELLED",
    ):
      print(f"Batch job ended with state: {state}")
      exit(1)

    time.sleep(args.poll_interval)

  # Download raw batch responses
  raw_entries = []
  dest_file_name = (
      getattr(job.dest, "file_name", None)
      if hasattr(job, "dest") and job.dest
      else None
  )
  if not dest_file_name and hasattr(job, "dest") and isinstance(job.dest, str):
    dest_file_name = job.dest

  if dest_file_name:
    content = client.files.download(file=dest_file_name)
    text_content = (
        content.decode("utf-8") if isinstance(content, bytes) else str(content)
    )
    for line in text_content.splitlines():
      if line.strip():
        raw_entries.append(json.loads(line))
  elif hasattr(job, "dest") and job.dest and getattr(
      job.dest, "inlined_responses", None
  ):
    for resp in job.dest.inlined_responses:
      dumped = (
          resp.model_dump(mode="json") if hasattr(resp, "model_dump") else resp
      )
      raw_entries.append(
          dumped if isinstance(dumped, dict) else {"response": dumped}
      )
  elif hasattr(job, "response") and job.response:
    inline_list = (
        getattr(job.response, "inlined_responses", None)
        or getattr(job.response, "inline_responses", None)
        or []
    )
    for resp in inline_list:
      dumped = (
          resp.model_dump(mode="json") if hasattr(resp, "model_dump") else resp
      )
      raw_entries.append(
          dumped if isinstance(dumped, dict) else {"response": dumped}
      )
  else:
    print(f"Warning: Could not find results. Job: {job}")
    exit(1)

  # Join with mapping_part_{part}.jsonl and write batched_output.jsonl
  mapping_file = meta.get(
      "mapping_file",
      os.path.join(google_dir, f"{args.run_id}_mapping_part_{args.part}.jsonl"),
  )
  mapping_by_key = {}
  ordered_items = []
  if os.path.exists(mapping_file):
    with open(mapping_file, "r", encoding="utf-8") as f:
      for line in f:
        if line.strip():
          rec = json.loads(line)
          key = str(rec.get("key", ""))
          orig = rec.get("original_data", {})
          mapping_by_key[key] = orig
          ordered_items.append(orig)

  os.makedirs(google_dir, exist_ok=True)
  output_file = os.path.join(
      google_dir, f"{args.run_id}_part_{args.part}_batched_output.jsonl"
  )
  with open(output_file, "w", encoding="utf-8") as f:
    for idx, entry in enumerate(raw_entries):
      key = str(entry.get("key", f"req-{idx}"))
      if key in mapping_by_key:
        record = dict(mapping_by_key[key])
      elif idx < len(ordered_items):
        record = dict(ordered_items[idx])
      else:
        record = {"id": key, "round": args.round}
      record["tool_output"] = _format_search_tool_output(entry)
      f.write(json.dumps(record, ensure_ascii=False) + "\n")

  print(f"Results saved to: {output_file}")


if __name__ == "__main__":
  main()
