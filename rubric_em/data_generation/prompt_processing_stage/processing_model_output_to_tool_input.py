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

"""Splits model outputs into tool call batches and finished tasks."""

import argparse
import json
import math
import pathlib
import re
from typing import Any


def parse_tool_call(model_response):
  """Parses a <call_tool> block and normalizes the tool name."""
  pattern = r"<call_tool\s+([^>]+)>(.*?)</call_tool>"
  match = re.search(pattern, model_response, re.DOTALL)
  if not match:
    return None, None, None, None

  attr_str = match.group(1).replace(
      '\\"', '"'
  )  # Unescape quoted attribute values.
  query = match.group(2).strip()

  name_match = re.search(r'name=["\'](.*?)["\']', attr_str)
  if name_match:
    raw_name = name_match.group(1).lower()
    if "google" in raw_name:
      norm_name = "google_search"
    elif "snippet" in raw_name or "scholar" in raw_name:
      norm_name = "snippet_search"
    else:
      norm_name = raw_name.replace(" ", "_")

    #
    clean_attrs = re.sub(r'name=["\'].*?["\']', f'name="{norm_name}"', attr_str)
    return norm_name, query, match.group(0), clean_attrs
  return None, query, match.group(0), attr_str


def prepare_google_batch_request(query):
  return {
      "contents": [{
          "role": "user",
          "parts": [
              {"text": f"Please search the web to answer this query: {query}"}
          ],
      }],
      "systemInstruction": {
          "parts": [{
              "text": (
                  "You MUST use the google_search tool to answer the user's"
                  " query. DO NOT answer solely from your internal knowledge,"
                  " even if you are confident. Every substantial claim must be"
                  " grounded in the search results. Provide specific details"
                  " and context from the external sources to ensure accuracy."
              )
          }]
      },
      "tools": [{
          "googleSearch": {
              #
              "excludeDomains": ["vertex-ai-empty-struct-bug-workaround.com"]
          }
      }],
      #
  }


def main():
  parser = argparse.ArgumentParser(
      description="Split dataset into S2, Google Batch, and Finished chunks."
  )
  parser.add_argument(
      "--run_id",
      type=str,
      default="first_run",
      help="Meta_Run_id_for_current_runs",
  )
  parser.add_argument(
      "--round", type=int, required=True, help="which round is current stage?"
  )
  parser.add_argument(
      "--num_s2_splits",
      type=int,
      default=1,
      help="How many chunks for Semantic Scholar.",
  )
  parser.add_argument(
      "--num_google_splits",
      type=int,
      default=1,
      help="How many chunks for Google Search.",
  )
  args = parser.parse_args()

  #
  finished_dir = pathlib.Path("files/complete")
  prefix = "files/rounds_processing"
  google_dir = pathlib.Path(f"{prefix}/round_{args.round}/google_search")
  s2_dir = pathlib.Path(f"{prefix}/round_{args.round}/semantic_scholar")
  no_tool_dir = pathlib.Path(f"{prefix}/round_{args.round}/no_tool")
  input_dir = pathlib.Path(f"{prefix}/round_{args.round}/model_output")

  #
  error_no_tool_dir = pathlib.Path(f"files/error_no_tool_files/{args.run_id}")
  # ====================================================================

  #
  for d in [finished_dir, google_dir, s2_dir, no_tool_dir, error_no_tool_dir]:
    d.mkdir(parents=True, exist_ok=True)

  google_tasks = []
  s2_tasks = []
  no_tool_tasks = []
  finished_tasks = []

  #
  #

  prefix_glob = f"{args.run_id}_"

  input_files = [
      p
      for p in input_dir.glob(f"{args.run_id}_*.jsonl")
      if p.stem.removeprefix(prefix_glob).isdigit()
  ]
  # input_files = list(input_dir.glob(f"{args.run_id}_*.jsonl"))
  print("Processing.... ", input_files)
  # exit()

  if not input_files:
    print(
        f"Error: No files matching '{args.run_id}*.jsonl' found in {input_dir}"
    )
    return

  print(f"Found {len(input_files)} file(s) to process in {input_dir}...")

  for input_file in input_files:
    print(f"  -> Reading data from {input_file.name}...")
    try:
      with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
          if not line.strip():
            continue
          data = json.loads(line)
          model_response = data.get("model_response", "")

          #
          if (
              isinstance(model_response, str)
              and "</answer>" in model_response[-100:]
          ):
            data["finished"] = True
            finished_tasks.append(data)
            continue

          #
          tool_name, query, original_tag, final_attrs = parse_tool_call(
              model_response
          )

          if tool_name == "google_search":
            standard_tag = (
                f'<call_tool name="google_search">{query}</call_tool>'
            )
            data["model_response"] = model_response.replace(
                original_tag, standard_tag
            )
            data["parsed_query"] = query
            google_tasks.append(data)

          elif tool_name == "snippet_search":
            standard_tag = f"<call_tool {final_attrs}>{query}</call_tool>"
            data["model_response"] = model_response.replace(
                original_tag, standard_tag
            )
            s2_tasks.append(data)
          else:
            no_tool_tasks.append(data)

    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f"Error processing {input_file}: {e}")
  # ====================================================================

  print("-" * 50)
  print("Total aggregated stats:")
  print(f"Found {len(finished_tasks)} finished items.")
  print(f"Found {len(google_tasks)} Google Search calls.")
  print(f"Found {len(s2_tasks)} Semantic Scholar calls.")
  print(f"Found {len(no_tool_tasks)} items needing no_tool handling.")
  print("-" * 50)

  #
  if finished_tasks:
    finished_file = finished_dir / f"{args.run_id}_round_{args.round}.jsonl"
    with open(finished_file, "w", encoding="utf-8") as out:
      for item in finished_tasks:
        out.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved Finished: {finished_file}")

  # 2. Split Semantic Scholar
  if s2_tasks:
    s2_chunk_size = math.ceil(len(s2_tasks) / args.num_s2_splits)
    for i in range(args.num_s2_splits):
      chunk = s2_tasks[i * s2_chunk_size : (i + 1) * s2_chunk_size]
      if not chunk:
        continue
      s2_file = s2_dir / f"{args.run_id}_{i+1}.jsonl"
      with open(s2_file, "w", encoding="utf-8") as out:
        for item in chunk:
          out.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved Semantic Scholar parts to {s2_dir}")

  # 3. Save No Tool Items
  if no_tool_tasks:
    no_tool_file = no_tool_dir / f"{args.run_id}.jsonl"
    #
    error_no_tool_file = error_no_tool_dir / f"round_{args.round}.jsonl"
    # =================================================================

    #
    with open(no_tool_file, "w", encoding="utf-8") as out, open(
        error_no_tool_file, "w", encoding="utf-8"
    ) as err_out:
      for item in no_tool_tasks:
        item["tool_output"] = (
            "<tool_output>\nNo valid tool call requested by"
            " model.\n</tool_output>"
        )
        out.write(json.dumps(item, ensure_ascii=False) + "\n")
        err_out.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved No Tool items to {no_tool_file}")
    print(f"Also saved No Tool copies to {error_no_tool_file}")

  # 4. Split Google Search (Mapping & Request)
  if google_tasks:
    google_chunk_size = math.ceil(len(google_tasks) / args.num_google_splits)
    for i in range(args.num_google_splits):
      chunk = google_tasks[i * google_chunk_size : (i + 1) * google_chunk_size]
      if not chunk:
        continue

      batch_req_file = google_dir / f"{args.run_id}_batch_req_part_{i+1}.jsonl"
      mapping_file = google_dir / f"{args.run_id}_mapping_part_{i+1}.jsonl"

      with open(batch_req_file, "w", encoding="utf-8") as req_out, open(
          mapping_file, "w", encoding="utf-8"
      ) as map_out:
        for idx, item in enumerate(chunk):
          item_id = item.get("id", f"req-{idx}")
          query = item.pop("parsed_query")
          req_dict = prepare_google_batch_request(query)
          req_out.write(
              json.dumps(
                  {"key": str(item_id), "request": req_dict}, ensure_ascii=False
              )
              + "\n"
          )
          map_out.write(
              json.dumps(
                  {"key": str(item_id), "original_data": item},
                  ensure_ascii=False,
              )
              + "\n"
          )
    print(f"Saved Google Search parts to {google_dir}")

  print("-" * 50)
  print("Splitting complete!")


if __name__ == "__main__":
  main()
