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

"""Merges tool outputs and filters errors for the next round input."""

import argparse
import json
import pathlib
import sys
from typing import List


def process_and_merge_jsonl(
    input_file_paths,
    next_round_input_path,
    error_file_path,
    include_error_tool_call,
):
  """Merges tool outputs from input files and filters error tool calls."""
  error_patterns = [
      "<tool_output>\nSearch results were blocked or failed.",
      "<tool_output>\n\nNo grounding data available.\n</tool_output>",
      "<tool_output>\nError executing Google Search:",
      (
          "<tool_output>\nSearch results were blocked by Gemini's safety"
          " filters.\n</tool_output>"
      ),
      "<tool_output>\nError executing Snippet Search",
      "<tool_output>\nInvalid tool routed to S2 handler:",
      "<tool_output>\nNo valid tool call requested by model",
      "<tool_output>\nNo valid tool call found in response.",
      "<tool_output>\nNo tool call requested by model.",
  ]

  valid_count = 0
  error_count = 0
  written_count = 0

  #
  with open(next_round_input_path, "w", encoding="utf-8") as out_f, open(
      error_file_path, "w", encoding="utf-8"
  ) as err_f:

    for file_path in input_file_paths:
      print(f"Processing: {file_path}")

      with open(file_path, "r", encoding="utf-8") as in_f:
        for line in in_f:
          line = line.strip()
          if not line:
            continue

          try:
            data = json.loads(line)

            #
            tool_output = data.get("tool_output", "")
            is_error = any(p in tool_output for p in error_patterns)

            #
            current_round = data.get("round", 1)

            if "conversation" not in data:
              data["conversation"] = {}

            #
            if any(
                k in data
                for k in ["internal_thinking", "model_response", "tool_output"]
            ):
              internal_thinking = data.pop("internal_thinking", None)
              model_response = data.pop("model_response", None)
              data.pop("tool_output", None)

              #
              data["conversation"][str(current_round)] = {
                  "internal_thinking": internal_thinking,
                  "model_response": model_response,
                  "tool_output": tool_output,
              }

            #
            data["round"] = current_round + 1

            #

            if not include_error_tool_call:
              if is_error:
                data["error_reason"] = "Matched error pattern in tool_output"
                err_f.write(json.dumps(data, ensure_ascii=False) + "\n")
                error_count += 1
              else:
                out_f.write(json.dumps(data, ensure_ascii=False) + "\n")
                valid_count += 1
                written_count += 1
            else:
              out_f.write(json.dumps(data, ensure_ascii=False) + "\n")
              written_count += 1
              if is_error:
                data["error_reason"] = "Matched error pattern in tool_output"
                err_f.write(json.dumps(data, ensure_ascii=False) + "\n")
                error_count += 1
              else:
                valid_count += 1

          except json.JSONDecodeError:
            print("Warning: skipping a line that is not valid JSON.")

  if not include_error_tool_call:
    print("-" * 50)
    print("Processing complete.")
    print(
        f"✅ Valid items ready for next round: {valid_count} ->"
        f" {next_round_input_path}"
    )
    print(
        f"⚠️ Error items filtered out:       {error_count} -> {error_file_path}"
    )
  else:
    print("-" * 50)
    print("Processing complete.")
    print(
        f"✅ Valid items ready for next round: {valid_count} ->"
        f" {next_round_input_path}"
    )
    print(
        f"⚠️ Error items not filtered out:    {error_count} ->"
        f" {error_file_path}"
    )
  return written_count


#
if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Merge tool outputs and filter errors for next round"
  )
  parser.add_argument(
      "--run_id",
      type=str,
      default="first_run",
      help="Meta_Run_id_for_current_runs",
  )
  parser.add_argument(
      "--round", type=int, required=True, help="which round just finished?"
  )
  parser.add_argument(
      "--include_error_tool_call",
      default=False,
      action="store_true",
      help="whether exclude error tool calls",
  )
  args = parser.parse_args()

  prefix = pathlib.Path(f"./files/rounds_processing/round_{args.round}")

  #
  input_files = []

  #
  gs_files = list(
      prefix.glob(f"google_search/{args.run_id}_part_*_batched_output.jsonl")
  )
  input_files.extend([str(p) for p in gs_files])

  #
  s2_files = list(
      prefix.glob(f"semantic_scholar/{args.run_id}_part_*_async_output.jsonl")
  )
  input_files.extend([str(p) for p in s2_files])

  #
  no_tool_files = list(prefix.glob(f"no_tool/{args.run_id}.jsonl"))
  input_files.extend([str(p) for p in no_tool_files])

  #
  next_round = args.round + 1

  #
  out_dir = pathlib.Path(f"./files/rounds_processing/round_{next_round}/input")
  out_dir.mkdir(parents=True, exist_ok=True)
  next_round_input = out_dir / f"{args.run_id}.jsonl"

  #
  error_dir = prefix / "error_tool_calls"
  error_dir.mkdir(parents=True, exist_ok=True)
  error_file = error_dir / f"{args.run_id}_errors.jsonl"

  #
  remaining = process_and_merge_jsonl(
      input_files,
      str(next_round_input),
      str(error_file),
      args.include_error_tool_call,
  )
  if remaining == 0:
    sys.exit(222)
