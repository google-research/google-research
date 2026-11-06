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

r"""Download and process model response batch job results from the Gemini Batch API.

Polls the batch job until completion, then downloads and processes the results.

Usage:
    python model_call_stage/model_response_batch_download.py \
        --run_id my_run --round 1 --part 1
"""

import argparse
import json
import os
import time

from google import genai


def _extract_response_text(resp_obj):
  """Extracts visible text and optional thinking text from a Gemini response."""
  response = resp_obj.get("response", resp_obj)
  if not isinstance(response, dict):
    return "", ""
  candidates = response.get("candidates", [])
  if not candidates or not isinstance(candidates, list):
    return "", ""
  content = (
      candidates[0].get("content", {})
      if isinstance(candidates[0], dict)
      else {}
  )
  parts = content.get("parts", []) if isinstance(content, dict) else []
  text_parts = []
  thinking_parts = []
  for part in parts:
    if not isinstance(part, dict):
      continue
    part_text = part.get("text", "")
    if not part_text:
      continue
    if part.get("thought"):
      thinking_parts.append(part_text)
    else:
      text_parts.append(part_text)
  return "".join(text_parts), "".join(thinking_parts)


def main():
  parser = argparse.ArgumentParser(
      description="Download model response batch results."
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

  # Read job metadata
  work_dir = os.path.join(
      args.input_dir,
      args.run_id,
      f"round_{args.round}",
      f"part_{args.part}",
  )
  meta_file = os.path.join(work_dir, "batch_job_meta.json")

  if not os.path.exists(meta_file):
    print(f"Job metadata not found: {meta_file}")
    print("Run model_response_batch_submit.py first.")
    exit(1)

  with open(meta_file, "r", encoding="utf-8") as f:
    meta = json.load(f)

  job_name = meta["job_name"]
  print(f"Polling batch job: {job_name}")

  # Poll until completion
  while True:
    job = client.batches.get(name=job_name)
    state = job.state.name if hasattr(job.state, "name") else str(job.state)
    print(f"  Status: {state}")

    if state in ("SUCCEEDED", "JOB_STATE_SUCCEEDED"):
      print("Batch job completed successfully!")
      break
    elif state in ("FAILED", "JOB_STATE_FAILED"):
      print(f"Batch job failed: {job}")
      exit(1)
    elif state in ("CANCELLED", "JOB_STATE_CANCELLED"):
      print("Batch job was cancelled.")
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
    print("Warning: Could not find batch results in expected location.")
    print(f"Job details: {job}")
    exit(1)

  # Load original items mapping to merge with model_response
  mapping_file = meta.get(
      "mapping_file", os.path.join(work_dir, "batch_mapping.jsonl")
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

  output_dir = os.path.join(
      args.input_dir,
      "rounds_processing",
      f"round_{args.round}",
      "model_output",
  )
  os.makedirs(output_dir, exist_ok=True)
  output_file = os.path.join(output_dir, f"{args.run_id}_{args.part}.jsonl")

  with open(output_file, "w", encoding="utf-8") as f:
    for idx, entry in enumerate(raw_entries):
      key = str(entry.get("key", f"req-{idx}"))
      if key in mapping_by_key:
        record = dict(mapping_by_key[key])
      elif idx < len(ordered_items):
        record = dict(ordered_items[idx])
      else:
        record = {"id": key, "round": args.round}

      model_text, thinking_text = _extract_response_text(entry)
      record["model_response"] = model_text
      if thinking_text:
        record["internal_thinking"] = thinking_text
      f.write(json.dumps(record, ensure_ascii=False) + "\n")

  print(f"Results saved to: {output_file}")
  print(f"Total responses: {len(raw_entries)}")


if __name__ == "__main__":
  main()
