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

"""Converts and merges round JSONL files into benchmark output format."""

import argparse
import json
import os
import pathlib
import re
from typing import Any
import uuid


def process_output(original_data):
  """Processes conversation rounds into full_traces and tool_calls.

  Args:
    original_data: Raw record dictionary containing conversation rounds.

  Returns:
    Formatted benchmark record dictionary.
  """
  example_id = original_data.get("id")
  problem = original_data.get("problem")

  generated_text_parts = []
  tool_calls = []
  searched_links = set()

  final_response = ""

  #
  rounds_data = {}

  #
  conversation = original_data.get("conversation", {})
  for k, v in conversation.items():
    if str(k).isdigit():
      rounds_data[int(k)] = v

  #
  #
  if "round" in original_data:
    curr_round = int(original_data["round"])
    rounds_data[curr_round] = {
        "model_response": original_data.get("model_response", ""),
        "tool_output": original_data.get("tool_output", ""),
    }
  #
  elif "model_response" in original_data:
    #
    rounds_data[0] = {
        "model_response": original_data.get("model_response", ""),
        "tool_output": original_data.get("tool_output", ""),
    }

  #
  sorted_round_keys = sorted(rounds_data.keys())

  for round_key in sorted_round_keys:
    round_data = rounds_data[round_key]
    model_response = round_data.get("model_response", "")
    tool_output_str = round_data.get("tool_output", "")

    if model_response:
      generated_text_parts.append(model_response)

      #
      ans_match = re.search(
          r"<answer>\n?(.*?)\n?</answer>", model_response, re.DOTALL
      )
      if ans_match:
        final_response = ans_match.group(1).strip()

      #
      tool_call_match = re.search(
          r'<call_tool name="([^"]+)"[^>]*>(.*?)</call_tool>',
          model_response,
          re.DOTALL,
      )

      if tool_call_match:
        tool_name = tool_call_match.group(1)
        tool_query = tool_call_match.group(2)
        call_id = str(uuid.uuid4())[:8]

        if tool_output_str:
          generated_text_parts.append(tool_output_str)

          clean_output = tool_output_str
          if clean_output.startswith("<tool_output>"):
            clean_output = clean_output[len("<tool_output>") :].strip()
          if clean_output.endswith("</tool_output>"):
            clean_output = clean_output[: -len("</tool_output>")].strip()

          urls = re.findall(r'(https?://[^\s<"]+)', clean_output)
          for url in urls:
            searched_links.add(url)

          tool_calls.append({
              "tool_name": tool_name,
              "output": clean_output,
              "called": True,
              "timeout": False,
              "runtime": 0.0,
              "error": "",
              "call_id": call_id,
              "raw_output": None,
              "input_params": {"query": tool_query},
          })

  #
  if not final_response and sorted_round_keys:
    final_response = (
        rounds_data[sorted_round_keys[-1]].get("model_response", "").strip()
    )

  generated_text = "\n".join(generated_text_parts)
  full_traces = {
      "generated_text": generated_text,
      "total_tokens": len(generated_text) // 4,
      "tool_call_count": len(tool_calls),
      "stopped_reason": "natural" if final_response else "unknown",
      "tool_calls": tool_calls,
  }

  additional_output_data = {
      "browsed_links": [],
      "searched_links": list(searched_links),
      "total_tool_calls": len(tool_calls),
      "total_failed_tool_calls": 0,
      "failed_tool_call_errors": [],
  }

  return {
      "example_id": example_id,
      "problem": problem,
      "final_response": final_response,
      "full_traces": full_traces,
      "additional_output_data": additional_output_data,
      "original_data": original_data,
  }


def main():
  parser = argparse.ArgumentParser(
      description=(
          "Process and merge round JSONL files into a single benchmark output."
      )
  )
  parser.add_argument(
      "--run_id",
      type=str,
      required=True,
      help="The run_id to filter and process files for.",
  )
  args = parser.parse_args()

  run_id = args.run_id
  input_dir = pathlib.Path("files/complete")
  output_dir = pathlib.Path("files/benchmark_output")

  output_dir.mkdir(parents=True, exist_ok=True)

  if not input_dir.exists():
    print(f"Error: Input directory '{input_dir}' does not exist.")
    return

  #
  # 1. {run_id}_round_{x}.jsonl
  # 2. {run_id}_remaining_round_{x}.jsonl
  # 3. {run_id}_remaining{y}_round_{x}.jsonl
  #
  pattern_str = rf"^{re.escape(run_id)}(?:_remaining(\d*))?_round_(\d+)\.jsonl$"
  pattern = re.compile(pattern_str)

  matched_files = []

  for filename in os.listdir(input_dir):
    match = pattern.match(filename)
    if match:
      y_str = match.group(1)
      x_str = match.group(2)  #

      #
      #
      #
      #
      if y_str is None:
        y_val = -1
      elif not y_str:
        y_val = 0
      else:
        y_val = int(y_str)

      x_val = int(x_str)

      matched_files.append(((y_val, x_val), input_dir / filename))

  if not matched_files:
    print(
        f"No files found matching the pattern for run_id '{run_id}' in"
        f" '{input_dir}'."
    )
    return

  #
  matched_files.sort(key=lambda item: item[0])

  #
  #
  all_raw_data_dict = {}

  print(
      f"Found {len(matched_files)} files for run_id '{run_id}'. Reading &"
      " Merging..."
  )
  for _, file_path in matched_files:
    print(f"  - Reading {file_path.name}")
    with open(file_path, "r", encoding="utf-8") as f:
      for line in f:
        line = line.strip()
        if line:
          data = json.loads(line)
          example_id = data.get("id")
          if example_id is not None:
            all_raw_data_dict[example_id] = data

  print(f"\nTotal unique records merged: {len(all_raw_data_dict)}")

  #
  processed_data = []
  #
  for example_id in sorted(all_raw_data_dict.keys()):
    processed_data.append(process_output(all_raw_data_dict[example_id]))

  output_file = output_dir / f"{run_id}.jsonl"
  print(f"Writing processed data to '{output_file}'...")

  with open(output_file, "w", encoding="utf-8") as f:
    for item in processed_data:
      f.write(json.dumps(item, ensure_ascii=False) + "\n")

  print("Done!")


if __name__ == "__main__":
  main()
