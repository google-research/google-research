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

r"""Submit Google Search tool call batch job using the Gemini Batch API.

Converts parsed tool call inputs into Gemini batch requests with Google Search
grounding enabled, uploads via File API, and submits a batch job.

Usage:
    python tool_call_stage/tool_batch_submit.py \
        --run_id my_run --round 1 --part 1
"""

import argparse
import json
import os

from google import genai


def main():
  parser = argparse.ArgumentParser(
      description="Submit Google Search batch job via Gemini Batch API."
  )
  parser.add_argument("--run_id", type=str, required=True)
  parser.add_argument("--round", type=int, required=True)
  parser.add_argument("--part", type=int, default=1)
  parser.add_argument(
      "--model_name", type=str, default="gemini-3-flash-preview"
  )
  parser.add_argument("--input_dir", type=str, default="files")
  args = parser.parse_args()

  api_key = os.environ.get("GEMINI_API_KEY")
  if not api_key:
    print("Error: GEMINI_API_KEY not set")
    exit(1)

  client = genai.Client(api_key=api_key)

  # Read pre-built batch requests from processing_model_output_to_tool_input.py
  google_dir = os.path.join(
      args.input_dir,
      "rounds_processing",
      f"round_{args.round}",
      "google_search",
  )
  pre_built_batch_file = os.path.join(
      google_dir, f"{args.run_id}_batch_req_part_{args.part}.jsonl"
  )
  mapping_file = os.path.join(
      google_dir, f"{args.run_id}_mapping_part_{args.part}.jsonl"
  )
  legacy_input_file = os.path.join(
      args.input_dir,
      args.run_id,
      f"round_{args.round}",
      f"part_{args.part}",
      "google_search_input.jsonl",
  )

  if os.path.exists(pre_built_batch_file):
    batch_file = pre_built_batch_file
    with open(batch_file, "r", encoding="utf-8") as fin:
      count = sum(1 for line in fin if line.strip())
    meta_file = os.path.join(
        google_dir, f"{args.run_id}_batch_meta_part_{args.part}.json"
    )
  elif os.path.exists(legacy_input_file):
    batch_file = legacy_input_file.replace(
        "google_search_input.jsonl", "google_search_batch.jsonl"
    )
    count = 0
    with open(legacy_input_file, "r", encoding="utf-8") as fin, open(
        batch_file, "w", encoding="utf-8"
    ) as fout:
      for idx, line in enumerate(fin):
        if not line.strip():
          continue
        item = json.loads(line)
        if "request" in item:
          fout.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
          query = item.get("query", "")
          request = {
              "key": item.get("key", f"tool-{idx}"),
              "request": {
                  "contents": [{"parts": [{"text": query}]}],
                  "tools": [{"google_search": {}}],
                  "generation_config": {"temperature": 0.0},
              },
          }
          fout.write(json.dumps(request, ensure_ascii=False) + "\n")
        count += 1
    meta_file = legacy_input_file.replace(
        "google_search_input.jsonl", "google_search_batch_meta.json"
    )
  else:
    print(f"No Google Search inputs found: {pre_built_batch_file}")
    exit(0)

  if count == 0:
    print("No search queries to process.")
    exit(0)

  print(f"Prepared {count} search requests -> {batch_file}")

  # Upload and submit
  uploaded_file = client.files.upload(file=batch_file)
  print(f"Uploaded: {uploaded_file.name}")

  batch_job = client.batches.create(
      model=args.model_name,
      src=uploaded_file.name,
      config={
          "display_name": (
              f"{args.run_id}_round_{args.round}_part_{args.part}_google_search"
          ),
      },
  )
  print(f"Batch job created: {batch_job.name}")

  # Save metadata
  with open(meta_file, "w", encoding="utf-8") as f:
    json.dump(
        {
            "job_name": batch_job.name,
            "run_id": args.run_id,
            "round": args.round,
            "part": args.part,
            "num_requests": count,
            "mapping_file": mapping_file,
        },
        f,
        indent=2,
    )
  print(f"Job metadata saved to: {meta_file}")


if __name__ == "__main__":
  main()
