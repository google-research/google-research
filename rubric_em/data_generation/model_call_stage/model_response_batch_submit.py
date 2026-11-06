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

r"""Submit model response batch job using the Gemini Batch API (File API).

Reads a local JSONL input file, uploads it via the File API,
and submits a batch prediction job.

Usage:
    python model_call_stage/model_response_batch_submit.py \
        --run_id my_run --round 1 --part 1 \
        --model_name gemini-3-flash-preview
"""

import argparse
import json
import os

from google import genai
from model_call_stage import prompt


def build_batch_requests(
    input_file,
    model_name,
    prompt_version,
    default_round = 1,
):
  """Converts input data to Gemini batch request JSONL and item mappings."""
  del model_name

  requests = []
  mappings = []
  with open(input_file, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
      if not line.strip():
        continue
      item = json.loads(line)
      question = item.get(
          "problem", item.get("question", item.get("prompt", ""))
      )
      history = item.get("conversation", item.get("history", {}))
      round_num = item.get("round", default_round)
      answer_type = item.get("answer_type", "long_form")
      req_key = str(item.get("id", f"req-{idx}"))

      system_prompt = prompt.get_system_prompt(prompt_version, round_num)
      user_prompt = prompt.get_user_prompt(
          prompt_version, question, history, round_num, answer_type
      )

      request = {
          "key": req_key,
          "request": {
              "contents": [{"parts": [{"text": user_prompt}]}],
              "system_instruction": {"parts": [{"text": system_prompt}]},
              "generation_config": {
                  "temperature": 0.7,
                  "max_output_tokens": 8192,
              },
          },
      }
      requests.append(request)
      mappings.append({"key": req_key, "original_data": item})

  return requests, mappings


def main():
  parser = argparse.ArgumentParser(
      description="Submit model response batch job via Gemini Batch API."
  )
  parser.add_argument("--run_id", type=str, required=True)
  parser.add_argument("--round", type=int, required=True)
  parser.add_argument("--part", type=int, default=1)
  parser.add_argument(
      "--model_name", type=str, default="gemini-3-flash-preview"
  )
  parser.add_argument("--prompt", type=str, default="scratchpad")
  parser.add_argument("--input_dir", type=str, default="files")
  parser.add_argument("--thinking_level", type=str, default="NONE")
  args = parser.parse_args()

  api_key = os.environ.get("GEMINI_API_KEY")
  if not api_key:
    print("Error: GEMINI_API_KEY not set")
    exit(1)

  client = genai.Client(api_key=api_key)

  # Determine input file (prefer rounds_processing path used by pipeline)
  rounds_input_file = os.path.join(
      args.input_dir,
      "rounds_processing",
      f"round_{args.round}",
      "input",
      f"{args.run_id}.jsonl",
  )
  legacy_input_file = os.path.join(
      args.input_dir,
      args.run_id,
      f"round_{args.round}",
      f"part_{args.part}",
      "model_input.jsonl",
  )
  input_file = (
      rounds_input_file
      if os.path.exists(rounds_input_file)
      else legacy_input_file
  )

  if not os.path.exists(input_file):
    print(f"Input file not found: {input_file}")
    print("No data to process.")
    exit(222)  # Early-stop signal

  work_dir = os.path.join(
      args.input_dir,
      args.run_id,
      f"round_{args.round}",
      f"part_{args.part}",
  )
  os.makedirs(work_dir, exist_ok=True)

  # Build batch request JSONL and mapping JSONL
  batch_file = os.path.join(work_dir, "batch_requests.jsonl")
  mapping_file = os.path.join(work_dir, "batch_mapping.jsonl")
  requests, mappings = build_batch_requests(
      input_file, args.model_name, args.prompt, default_round=args.round
  )

  if not requests:
    print("No requests to submit.")
    exit(222)

  with open(batch_file, "w", encoding="utf-8") as f:
    for req in requests:
      f.write(json.dumps(req, ensure_ascii=False) + "\n")

  with open(mapping_file, "w", encoding="utf-8") as f:
    for mapping in mappings:
      f.write(json.dumps(mapping, ensure_ascii=False) + "\n")

  print(f"Prepared {len(requests)} batch requests -> {batch_file}")

  # Upload via File API
  print("Uploading batch file...")
  uploaded_file = client.files.upload(file=batch_file)
  print(f"Uploaded: {uploaded_file.name}")

  # Submit batch job
  print(f"Submitting batch job with model: {args.model_name}")
  batch_job = client.batches.create(
      model=args.model_name,
      src=uploaded_file.name,
      config={
          "display_name": (
              f"{args.run_id}_round_{args.round}_part_{args.part}_model"
          ),
      },
  )
  print(f"Batch job created: {batch_job.name}")

  # Save job metadata for download step
  meta_file = os.path.join(work_dir, "batch_job_meta.json")
  with open(meta_file, "w", encoding="utf-8") as f:
    json.dump(
        {
            "job_name": batch_job.name,
            "run_id": args.run_id,
            "round": args.round,
            "part": args.part,
            "model": args.model_name,
            "num_requests": len(requests),
            "input_file": input_file,
            "mapping_file": mapping_file,
        },
        f,
        indent=2,
    )
  print(f"Job metadata saved to: {meta_file}")


if __name__ == "__main__":
  main()
