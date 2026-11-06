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

"""Asynchronously executes Semantic Scholar snippet search tool calls."""

import argparse
import asyncio
import hashlib
import json
import os
import pathlib
import random
import re
from typing import Any, Dict
import aiohttp
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 1. ROBUST TOOL PARSER
# ---------------------------------------------------------------------------


def parse_and_clean_tool_call(model_response):
  """Robustly parses the tool call and normalizes the tool name.

  Args:
    model_response: Raw model response text containing a <call_tool> tag.

  Returns:
    Tuple of (normalized_name, query, original_tag_match,
    clean_attributes_string).
  """
  pattern = r"<call_tool\s+([^>]+)>(.*?)</call_tool>"
  match = re.search(pattern, model_response, re.DOTALL)
  if not match:
    return None, None, None, None

  attr_str = match.group(1).replace('\\"', '"')  # Clean escaped quotes
  query = match.group(2).strip()

  name_match = re.search(r'name=["\'](.*?)["\']', attr_str, re.IGNORECASE)
  if name_match:
    raw_name = name_match.group(1).lower()
    # Normalization logic
    if "snippet" in raw_name or "scholar" in raw_name:
      norm_name = "snippet_search"
    else:
      norm_name = raw_name.replace(" ", "_")

    # Reconstruct clean attributes with the normalized name
    clean_attrs = re.sub(
        r'name=["\'].*?["\']',
        f'name="{norm_name}"',
        attr_str,
        flags=re.IGNORECASE,
    )
    return norm_name, query, match.group(0), clean_attrs

  return None, query, match.group(0), attr_str


# ---------------------------------------------------------------------------
# 2. ASYNC TOOL IMPLEMENTATIONS (Semantic Scholar Only)
# ---------------------------------------------------------------------------


async def execute_snippet_search_async(
    query, attrs, session
):
  """Asynchronous Semantic Scholar Snippet Search with full jitter exponential backoff."""
  url = "https://api.semanticscholar.org/graph/v1/snippet/search"

  params = {"query": query}
  if "year" in attrs:
    params["year"] = attrs["year"]
  if "venue" in attrs:
    params["venue"] = attrs["venue"]
  if "fieldsOfStudy" in attrs:
    params["fieldsOfStudy"] = attrs["fieldsOfStudy"]

  try:
    limit = int(attrs.get("limit", 10))
  except ValueError:
    limit = 10
  params["limit"] = limit

  headers = {}
  s2_api_key = os.environ.get("S2_API_KEY")
  if s2_api_key:
    headers["x-api-key"] = s2_api_key

  max_retries = 6  #
  base_delay = 2.0
  data = []

  for attempt in range(max_retries):
    try:
      async with session.get(
          url,
          params=params,
          headers=headers,
          timeout=aiohttp.ClientTimeout(total=35),
      ) as res:
        if res.status in [429, 500, 502, 503, 504]:
          res.raise_for_status()  #

        res.raise_for_status()
        json_res = await res.json()
        data = json_res.get("data", [])
        break

    except Exception as e:  # pylint: disable=broad-exception-caught
      if attempt < max_retries - 1:
        # Exponential Backoff with Full Jitter
        #
        temp_sleep = min(60, base_delay * (2**attempt))
        sleep_time = random.uniform(base_delay, temp_sleep)

        #
        err_msg = str(e)[:50] if str(e) else type(e).__name__
        tqdm.write(
            f"    [!] S2 Error (Attempt {attempt+1}/{max_retries}): {err_msg}."
            f" Retrying in {sleep_time:.1f}s..."
        )
        await asyncio.sleep(sleep_time)
      else:
        return (
            f"<tool_output>\nError executing Snippet Search after {max_retries}"
            f" retries: {e}\n</tool_output>"
        )

  hash_input = f"snippet_search:{query}:{str(attrs)}"
  base_hash = hashlib.md5(hash_input.encode("utf-8")).hexdigest()[:8]

  output_parts = []
  for index, item in enumerate(data):
    paper_info = item.get("paper", {})
    title = paper_info.get("title", "No Title").strip()

    snippet_info = item.get("snippet", {})
    if snippet_info.get("snippetKind") == "title":
      snippet_text = ""
    else:
      snippet_text = snippet_info.get("text", "").strip()

    if title or snippet_text:
      snippet_id = f"{base_hash}-{index}"
      snippet_block = (
          f"<snippet id={snippet_id}>\n"
          f"Title: {title}\n"
          f"Snippet: {snippet_text}\n"
          "</snippet>"
      )
      output_parts.append(snippet_block)

  final_text = "\n".join(output_parts) if output_parts else "No snippets found."
  return f"<tool_output>\n{final_text}\n</tool_output>"


# ---------------------------------------------------------------------------
# 3. BATCH PROCESSING & CONCURRENCY
# ---------------------------------------------------------------------------


async def process_single_item(
    item,
    session,
    semaphore,
):
  """Processes a single JSONL item asynchronously, enforcing concurrency limits."""
  async with semaphore:
    model_response = item.get("model_response", "")

    if "<call_tool" in model_response:
      tool_name, query, original_tag, final_attrs = parse_and_clean_tool_call(
          model_response
      )

      if tool_name == "snippet_search":
        #
        standard_tag = f"<call_tool {final_attrs}>{query}</call_tool>"
        item["model_response"] = model_response.replace(
            original_tag, standard_tag
        )

        #
        attrs_dict = {}
        for attr_match in re.finditer(r'(\w+)="([^"]*)"', final_attrs):
          attrs_dict[attr_match.group(1)] = attr_match.group(2)

        tool_output = await execute_snippet_search_async(
            query, attrs_dict, session
        )
      elif tool_name:
        tool_output = (
            "<tool_output>\nInvalid tool routed to S2 handler:"
            f" {tool_name}\n</tool_output>"
        )
      else:
        tool_output = (
            "<tool_output>\nNo valid tool call found in"
            " response.\n</tool_output>"
        )
    else:
      tool_output = (
          "<tool_output>\nNo tool call requested by model.\n</tool_output>"
      )

    item["tool_output"] = tool_output
    return item


async def process_batch(
    batch,
    session,
    semaphore,
    output_file,
):
  """Processes a batch concurrently and appends results to disk."""
  tasks = [process_single_item(item, session, semaphore) for item in batch]
  results = await asyncio.gather(*tasks)

  with open(output_file, "a", encoding="utf-8") as outfile:
    for res in results:
      outfile.write(json.dumps(res, ensure_ascii=False) + "\n")


async def main_async():
  parser = argparse.ArgumentParser(
      description="Asynchronously execute Semantic Scholar tool calls."
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
      "--part", type=int, default=1, help="which part of the split is this?"
  )
  parser.add_argument(
      "--batch_size",
      type=int,
      default=20,
      help="Number of items to process before writing to disk",
  )
  parser.add_argument(
      "--max_concurrency",
      type=int,
      default=2,
      help="Maximum number of simultaneous API calls",
  )
  args = parser.parse_args()

  if not os.environ.get("S2_API_KEY"):
    print(
        "WARNING: S2_API_KEY is not set. Semantic Scholar limits will be"
        " severe."
    )

  # ---------------------------------------------------------
  # PATH CONFIGURATION
  # ---------------------------------------------------------
  base_dir = pathlib.Path(
      f"files/rounds_processing/round_{args.round}/semantic_scholar"
  )
  input_file = base_dir / f"{args.run_id}_{args.part}.jsonl"
  output_file = base_dir / f"{args.run_id}_part_{args.part}_async_output.jsonl"

  print(f"Reading from: {input_file}")
  print(f"Writing to:   {output_file}")

  if not input_file.exists():
    print(f"Error: The input file {input_file} was not found.")
    return

  # ---------------------------------------------------------
  # RESUME LOGIC & PROGRESS CALCULATION
  # ---------------------------------------------------------
  processed_ids = set()
  error_target_string = "<tool_output>\nError executing Snippet Search"
  valid_output_lines = []

  if output_file.exists():
    with open(output_file, "r", encoding="utf-8") as outfile:
      for line in outfile:
        if not line.strip():
          continue
        try:
          data = json.loads(line)
          if "id" in data:
            tool_output = data.get("tool_output", "")

            if error_target_string in tool_output:
              continue

            processed_ids.add(data["id"])
            valid_output_lines.append(line)
        except json.JSONDecodeError:
          pass

    with open(output_file, "w", encoding="utf-8") as outfile:
      for valid_line in valid_output_lines:
        outfile.write(valid_line)

  print("Scanning input file to calculate progress...")
  total_items = sum(
      1 for _ in open(input_file, "r", encoding="utf-8") if _.strip()
  )
  remaining_items = total_items - len(processed_ids)

  print("-" * 50)
  print(f"Total items in input: {total_items}")
  print(f"Already processed:    {len(processed_ids)}")
  print(f"Items to process:     {remaining_items}")
  print("-" * 50)

  if remaining_items <= 0:
    print("All items have already been processed cleanly. Exiting.")
    return

  semaphore = asyncio.Semaphore(args.max_concurrency)
  current_batch = []

  async with aiohttp.ClientSession() as aio_session:
    with tqdm(
        total=remaining_items, desc=f"Executing S2 Part {args.part}", unit="req"
    ) as pbar:
      with open(input_file, "r", encoding="utf-8") as infile:
        for line in infile:
          if not line.strip():
            continue

          data = json.loads(line)
          item_id = data.get("id")

          if item_id and item_id in processed_ids:
            continue

          current_batch.append(data)

          if len(current_batch) >= args.batch_size:
            await process_batch(
                current_batch, aio_session, semaphore, output_file
            )
            pbar.update(len(current_batch))
            current_batch = []

        if current_batch:
          await process_batch(
              current_batch, aio_session, semaphore, output_file
          )
          pbar.update(len(current_batch))

  print("-" * 50)
  print(
      f"All Semantic Scholar tasks for Part {args.part} completed successfully!"
  )


if __name__ == "__main__":
  asyncio.run(main_async())
