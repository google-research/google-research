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

"""Chronicle Weaver: Orchestrator Script

Automated Multi-keyword Dataset Builder
"""

import argparse
import asyncio
import copy
import logging
import os
import signal
import time
from typing import List, Optional

from dataset_agent.event_pipeline import run_event_centric_pipeline
from dataset_agent.modules.module_00_setup import load_and_validate_config
from dataset_agent.modules.module_05_initial_events import generate_time_chunks
from dataset_agent.modules.task_runtime import (
    build_heartbeat_payload,
    build_task_outcome_payload,
    get_terminal_state,
    set_terminal_state,
    write_heartbeat,
    write_task_outcome,
)
from dotenv import load_dotenv


def normalize_output_root(output_root):
  if not output_root:
    return None
  expanded = os.path.expanduser(output_root)
  return os.path.abspath(expanded)


def build_project_output_root(
    project_name, output_root
):
  normalized_root = normalize_output_root(output_root)
  if normalized_root:
    return os.path.join(normalized_root, project_name)
  return project_name


def sanitize_keyword(keyword):
  """Sanitize keyword to create a safe directory name

  Args:
      keyword: The original keyword

  Returns:
      Sanitized string safe for use as directory name
  """
  return (
      keyword.replace(" ", "_")
      .replace("/", "_")
      .replace("\\", "_")
      .replace(":", "_")
      .lower()
  )


class ControlledTermination(Exception):
  """Raised when the orchestrator receives a termination signal."""


async def heartbeat_pump(
    config, runtime_view, interval_seconds = 5
):
  while True:
    write_heartbeat(
        config,
        build_heartbeat_payload(
            task_id=runtime_view["task_id"],
            keyword=runtime_view["keyword"],
            project_name=runtime_view["project_name"],
            stage=runtime_view["stage"],
            status="running",
            time_range=runtime_view["time_range"],
            config_path=runtime_view["config_path"],
        ),
    )
    await asyncio.sleep(interval_seconds)


async def main():
  """Main orchestrator function"""
  # Setup basic logging for orchestrator
  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
      handlers=[logging.StreamHandler()],
  )
  logger = logging.getLogger(__name__)
  active_runtime_config = None
  heartbeat_task = None
  runtime_view = None

  def handle_termination(signum, _frame):
    signal_name = signal.Signals(signum).name.lower()
    raise ControlledTermination(f"received_{signal_name}")

  signal.signal(signal.SIGTERM, handle_termination)
  signal.signal(signal.SIGINT, handle_termination)

  start_time = time.time()
  print(f"Orchestrator started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
  print("-" * 80)

  try:
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description="Chronicle Weaver Orchestrator"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration YAML file.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help=(
            "Optional root directory for generated datasets and runtime"
            " outputs. Overrides paths.output_root in the config."
        ),
    )
    parser.add_argument(
        "--validate-config-only",
        action="store_true",
        help=(
            "Load and validate the YAML config without running networked"
            " pipeline stages."
        ),
    )
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Load base configuration from the provided path
    base_config = load_and_validate_config(args.config)
    if args.output_root:
      base_config.setdefault("paths", {})["output_root"] = args.output_root
    output_root = normalize_output_root(
        base_config.get("paths", {}).get("output_root")
    )
    output_project_root = build_project_output_root(
        base_config["project_name"], output_root
    )

    if args.validate_config_only:
      keywords = base_config.get("keywords_to_process", [])
      date_range = base_config.get("date_range", {})
      time_range = base_config.get("time_range", {})
      start_date = (
          base_config.get("start_date")
          or date_range.get("start_date")
          or date_range.get("start")
          or time_range.get("start")
      )
      end_date = (
          base_config.get("end_date")
          or date_range.get("end_date")
          or date_range.get("end")
          or time_range.get("end")
      )
      print("CONFIG VALID")
      print(f"Project: {base_config.get('project_name', 'unknown')}")
      print(f"Keywords: {len(keywords)}")
      print(f"Date range: {start_date} to {end_date}")
      return

    # Read keywords from config
    keywords = base_config.get("keywords_to_process", [])
    if not keywords:
      print(
          "No keywords found in config.yaml. Please add keywords_to_process"
          " list."
      )
      return

    # Read processing mode configuration
    processing_mode = base_config.get(
        "processing_mode",
        {"mode": "full", "status_file": "processed_keywords.log"},
    )
    processed_keywords = set()

    # Load processed keywords if in incremental mode
    if processing_mode.get("mode") == "incremental":
      status_file = processing_mode.get(
          "status_file", "data/processed_keywords.log"
      )
      if os.path.exists(status_file):
        with open(status_file, "r", encoding="utf-8") as f:
          processed_keywords = set(line.strip() for line in f if line.strip())
        print(
            f"Loaded {len(processed_keywords)} previously processed keywords"
            f" from {status_file}"
        )
      else:
        print(f"Status file {status_file} not found. Starting fresh.")

    print("=" * 80)
    print("CHRONICLE WEAVER ORCHESTRATOR - Multi-keyword Dataset Builder")
    print("=" * 80)
    print(f"Base project: {base_config['project_name']}")
    print(
        "Output root:"
        f" {output_root if output_root else 'current working directory'}"
    )
    print(f"Domain: {base_config.get('domain', 'Not specified')}")
    print(f"Keywords to process: {len(keywords)}")
    print(f"Processing mode: {processing_mode.get('mode', 'full')}")
    print(
        f"Time range: {base_config['time_range']['start']} to"
        f" {base_config['time_range']['end']}"
    )
    print("=" * 80)

    results_summary = []

    # Process each keyword
    for i, keyword in enumerate(keywords, 1):
      print(f"\n[{i}/{len(keywords)}] Processing keyword: '{keyword}'")
      print("-" * 60)

      # Check if any time blocks for this keyword need processing
      if processing_mode.get("mode") == "incremental":
        status_file = processing_mode.get(
            "status_file", "data/processed_keywords.log"
        )
        processed_items = load_processed_time_blocks(status_file)

        pending_time_blocks = get_pending_time_blocks(
            keyword, base_config["time_range"], processed_items
        )

        if not pending_time_blocks:
          print(
              f"Keyword '{keyword}' skipped because all time blocks are already"
              " complete."
          )
          results_summary.append({
              "keyword": keyword,
              "status": "SKIPPED_ALL_BLOCKS_COMPLETED",
              "output_dir": f"data/output/{sanitize_keyword(keyword)}",
              "rows": 0,
              "columns": 0,
          })
          continue

        print(
            f"Found {len(pending_time_blocks)} pending time blocks to"
            " process..."
        )

      try:

        if processing_mode.get("mode") == "incremental":

          time_blocks_to_process = pending_time_blocks
        else:

          time_blocks_to_process = generate_time_chunks(
              base_config["time_range"]
          )

        successful_blocks = 0

        for block_idx, time_block in enumerate(time_blocks_to_process, 1):
          print(
              f"\n  [{block_idx}/{len(time_blocks_to_process)}] Time block:"
              f" {time_block['start']} to {time_block['end']}"
          )

          modified_config = copy.deepcopy(base_config)
          time_block_id = create_time_block_id(keyword, time_block)

          try:
            modified_config["target_variable"] = keyword
            modified_config["time_range"] = time_block

            sanitized_keyword = sanitize_keyword(keyword)
            time_block_suffix = f"{time_block['start']}_to_{time_block['end']}"
            output_dir = os.path.join(
                output_project_root,
                base_config["domain"],
                "data",
                "output",
                sanitized_keyword,
                time_block_suffix,
            )

            modified_config["paths"]["raw_trends"] = os.path.join(
                output_dir, "raw_trends.csv"
            )
            modified_config["paths"]["raw_weather"] = os.path.join(
                output_dir, "raw_weather.csv"
            )
            modified_config["paths"]["raw_nasdaq"] = os.path.join(
                output_dir, "raw_nasdaq.csv"
            )
            modified_config["paths"]["final_dataset"] = os.path.join(
                output_dir, "final_dataset.csv"
            )
            modified_config["paths"]["log_file"] = os.path.join(
                output_dir, "run.log"
            )
            modified_config["paths"]["research_output_dir"] = os.path.join(
                output_dir, "research_trees"
            )
            modified_config["project_name"] = (
                f"{base_config['project_name']}_{sanitized_keyword}_{time_block_suffix}"
            )
            modified_config["_cw_task_id"] = time_block_id
            modified_config["_cw_config_path"] = os.path.abspath(args.config)

            print(f"  Output directory: {output_dir}")

            active_runtime_config = modified_config
            runtime_view = {
                "task_id": time_block_id,
                "keyword": keyword,
                "project_name": modified_config["project_name"],
                "stage": "orchestrator_preflight",
                "time_range": time_block,
                "config_path": os.path.abspath(args.config),
            }
            write_heartbeat(
                modified_config,
                build_heartbeat_payload(
                    task_id=time_block_id,
                    keyword=keyword,
                    project_name=modified_config["project_name"],
                    stage=runtime_view["stage"],
                    status="running",
                    time_range=time_block,
                    config_path=os.path.abspath(args.config),
                ),
            )
            heartbeat_task = asyncio.create_task(
                heartbeat_pump(modified_config, runtime_view)
            )
            runtime_view["stage"] = "event_pipeline"

            result = await run_event_centric_pipeline(modified_config)

            runtime_view["stage"] = "orchestrator_finalize"
            terminal_state = get_terminal_state(modified_config)
            if terminal_state is None:
              if result is not None:
                terminal_state = set_terminal_state(
                    modified_config,
                    "accepted",
                    "pipeline_completed_without_explicit_terminal_state",
                    runtime_view["stage"],
                    total_events=len(result),
                    verified_events=len(result),
                    safe_to_accept=True,
                    workflow_outcome="accepted",
                )
              else:
                terminal_state = set_terminal_state(
                    modified_config,
                    "failed",
                    "pipeline_returned_none_without_terminal_state",
                    runtime_view["stage"],
                    total_events=0,
                    verified_events=0,
                    safe_to_accept=False,
                    workflow_outcome="failed",
                )

            write_task_outcome(
                modified_config,
                build_task_outcome_payload(modified_config, terminal_state),
            )

            terminal_status = terminal_state["terminal_status"]
            if terminal_status != "failed":
              successful_blocks += 1
              print(f"  COMPLETED: terminal_status={terminal_status}")

              if processing_mode.get("mode") == "incremental":
                status_file = processing_mode.get(
                    "status_file", "data/processed_keywords.log"
                )
                os.makedirs(os.path.dirname(status_file), exist_ok=True)

                with open(status_file, "a", encoding="utf-8") as f:
                  f.write(f"{time_block_id}\n")

                print(f"  RECORDED incremental status: {time_block_id}")
            else:
              print(
                  "  TERMINAL FAILURE:"
                  f" {terminal_state.get('reason', 'unknown_failure')}"
              )

          except ControlledTermination as e:
            runtime_stage = (
                runtime_view["stage"] if runtime_view else "orchestrator_signal"
            )
            terminal_state = set_terminal_state(
                modified_config,
                "failed",
                str(e),
                runtime_stage,
                total_events=0,
                verified_events=0,
                safe_to_accept=False,
                workflow_outcome="failed",
            )
            write_task_outcome(
                modified_config,
                build_task_outcome_payload(modified_config, terminal_state),
            )
            print(f"  TERMINATED: {str(e)}")
            raise
          except Exception as e:
            runtime_stage = (
                runtime_view["stage"]
                if runtime_view
                else "orchestrator_exception"
            )
            terminal_state = set_terminal_state(
                modified_config,
                "failed",
                f"orchestrator_exception: {str(e)}",
                runtime_stage,
                total_events=0,
                verified_events=0,
                safe_to_accept=False,
                workflow_outcome="failed",
            )
            write_task_outcome(
                modified_config,
                build_task_outcome_payload(modified_config, terminal_state),
            )
            print(f"  ERROR: {str(e)}")
            logger.error(
                f"Error processing time block for keyword {keyword}: {str(e)}"
            )
          finally:
            if heartbeat_task is not None:
              heartbeat_task.cancel()
              try:
                await heartbeat_task
              except asyncio.CancelledError:
                pass
              heartbeat_task = None
            active_runtime_config = None
            runtime_view = None

        if successful_blocks > 0:
          results_summary.append({
              "keyword": keyword,
              "status": "SUCCESS",
              "output_dir": f"data/output/{sanitize_keyword(keyword)}",
              "successful_blocks": successful_blocks,
              "total_blocks": len(time_blocks_to_process),
          })
          print(
              f"SUCCESS: Keyword '{keyword}' completed"
              f" {successful_blocks}/{len(time_blocks_to_process)} blocks"
          )
        else:
          results_summary.append({
              "keyword": keyword,
              "status": "FAILED",
              "output_dir": f"data/output/{sanitize_keyword(keyword)}",
              "successful_blocks": 0,
              "total_blocks": len(time_blocks_to_process),
          })
          print(f"FAILED: Keyword '{keyword}' produced no successful blocks")

      except ControlledTermination as e:
        results_summary.append({
            "keyword": keyword,
            "status": "TERMINATED",
            "output_dir": f"data/output/{sanitize_keyword(keyword)}",
            "error": str(e),
        })
        print(f"TERMINATED: Processing keyword '{keyword}': {str(e)}")
        raise
      except Exception as e:

        results_summary.append({
            "keyword": keyword,
            "status": "ERROR",
            "output_dir": f"data/output/{sanitize_keyword(keyword)}",
            "error": str(e),
        })
        print(f"ERROR: Processing keyword '{keyword}': {str(e)}")
        logger.error(f"Error processing keyword {keyword}: {str(e)}")

    # Print final summary
    print("\n" + "=" * 80)
    print("RUN SUMMARY")
    print("=" * 80)

    total_keywords = len(results_summary)
    successful_keywords = len(
        [r for r in results_summary if r["status"] == "SUCCESS"]
    )
    skipped_keywords = len([
        r
        for r in results_summary
        if r["status"] == "SKIPPED_ALL_BLOCKS_COMPLETED"
    ])

    for result in results_summary:
      if result["status"] == "SUCCESS":
        blocks_info = f"{result['successful_blocks']}/{result['total_blocks']}"
        print(f"SUCCESS {result['keyword']}: {blocks_info} blocks")
      elif result["status"] == "SKIPPED_ALL_BLOCKS_COMPLETED":
        print(
            f"SKIPPED {result['keyword']}: all time blocks were already"
            " completed"
        )
      elif result["status"] == "FAILED":
        blocks_info = f"{result['successful_blocks']}/{result['total_blocks']}"
        print(f"FAILED {result['keyword']}: {blocks_info} blocks")
      else:
        print(f"ERROR {result['keyword']}: {result['status']}")

    print(f"\nCompleted keywords: {successful_keywords}/{total_keywords}")
    if skipped_keywords > 0:
      print(f"Skipped keywords: {skipped_keywords} (already completed)")

    print("=" * 80)
    print("ORCHESTRATOR FINISHED!")
    print("=" * 80)

    return results_summary

  except ControlledTermination as e:
    print(f"ORCHESTRATOR TERMINATED: {str(e)}")
    return results_summary
  except Exception as e:
    print(f"ORCHESTRATOR FAILED: {str(e)}")
    raise
  finally:
    end_time = time.time()
    duration = end_time - start_time

    # Format duration into hh:mm:ss
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)

    print("-" * 80)
    print(f"Orchestrator finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total execution time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print("-" * 80)


def create_time_block_id(keyword, time_block):
  """Create a stable identifier in the form keyword|start_end."""
  return f"{keyword}|{time_block['start']}_{time_block['end']}"


def load_processed_time_blocks(status_file):
  """Return only the time blocks that have not yet been marked as processed."""
  processed_items = set()
  if os.path.exists(status_file):
    with open(status_file, "r", encoding="utf-8") as f:
      for line in f:
        line = line.strip()
        if line and "|" in line:
          processed_items.add(line)
  return processed_items


def get_pending_time_blocks(
    keyword, time_range, processed_items
):
  """"""

  all_time_blocks = generate_time_chunks(time_range)
  pending_blocks = []

  for time_block in all_time_blocks:
    time_block_id = create_time_block_id(keyword, time_block)
    if time_block_id not in processed_items:
      pending_blocks.append(time_block)

  return pending_blocks


if __name__ == "__main__":
  asyncio.run(main())
