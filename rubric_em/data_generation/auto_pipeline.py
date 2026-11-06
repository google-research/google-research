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

"""Automated step-wise SFT data generation pipeline orchestrator."""

import argparse
import subprocess
import time


def run_step_live(step_name, command):
  """Executes a pipeline step command and checks its exit code."""
  print(f"\n{'='*60}")
  print(f"▶️ [{step_name}] Executing:\n{command}")
  print(f"{'='*60}\n")

  result = subprocess.run(command, shell=True, check=False)

  # ======== [Added] ========
  if result.returncode == 222:
    return 222  # Return early-stop signal
  # ============================================

  if result.returncode != 0:
    print(f"\n❌ {step_name} FAILED! Pipeline stopped.")
    exit(1)
  print(f"\n✅ {step_name} SUCCESS.")
  return 0


def main():
  parser = argparse.ArgumentParser(
      description="Fully Automated Step-Wise SFT Pipeline Orchestrator"
  )
  parser.add_argument(
      "--start_round", type=int, required=True, help="Round to start from"
  )
  parser.add_argument(
      "--end_round", type=int, required=True, help="Round to end at"
  )
  parser.add_argument("--run_id", type=str, default="first_run")
  parser.add_argument("--part", type=str, default="1")
  parser.add_argument("--thinking_level", type=str, default=None)

  # ======== [Added] ========
  parser.add_argument(
      "--model_name",
      type=str,
      default="gemini-3.1-pro-preview",
      help="The Gemini model to use.",
  )
  parser.add_argument(
      "--prompt",
      type=str,
      default="scratchpad",
      help="Which prompt version to use.",
  )
  # =================================================

  # ======== [Existing] ========
  parser.add_argument(
      "--include_error_tool_call",
      action="store_true",
      help="whether to include error tool calls",
  )
  # =================================================

  args = parser.parse_args()

  base_cmd = (
      "bash scripts/run_pipeline.sh {mode} --round {round} --run_id"
      f" {args.run_id}"
  )

  for current_round in range(args.start_round, args.end_round + 1):
    print("\n" + "🚀" * 20)
    print(f"   STARTING ROUND {current_round}")
    print("🚀" * 20)

    # Step 1: Model Response
    cmd = (
        base_cmd.format(mode="model_response", round=current_round)
        + f" --part {args.part}"
    )
    if args.thinking_level:
      cmd += f" --thinking_level {args.thinking_level}"

    # ======== [Added] ========
    cmd += f" --model_name {args.model_name} --prompt {args.prompt}"
    # =======================================================================

    status = run_step_live(f"Round {current_round} - Model Response", cmd)

    # ======== [Existing] ========
    if status == 222:
      print(
          "\n🎉 ALL CLEAR! No more data to process. Pipeline early-stopped at"
          f" Round {current_round}."
      )
      break
    # ==========================================================

    # Step 2: Parse Model to Tool
    cmd = base_cmd.format(mode="parsing_model_to_tool", round=current_round)
    run_step_live(f"Round {current_round} - Parse Model Output", cmd)

    # Step 3: Tool Calls
    print(
        f"\n[Round {current_round} - Tool Calls] Starting Semantic Scholar and"
        " Google Search in PARALLEL..."
    )

    s2_cmd = (
        base_cmd.format(mode="semantic_scholar", round=current_round)
        + f" --part {args.part} --batch_size 20 --max_concurrency 2"
    )
    print(f"  -> Launching Semantic Scholar (Background): {s2_cmd}")
    s2_process = subprocess.Popen(s2_cmd, shell=True)

    gs_cmd = (
        base_cmd.format(mode="google_search", round=current_round)
        + f" --part {args.part}"
    )
    run_step_live(f"Round {current_round} - Google Search", gs_cmd)

    print(
        "\n⏳ Google Search branch finished (or skipped). Waiting for Semantic"
        " Scholar..."
    )
    s2_process.wait()
    # Note: handle S2 script exit gracefully
    if s2_process.returncode != 0:
      print("❌ Semantic Scholar FAILED!")
      exit(1)
    print("✅ Semantic Scholar SUCCESS.")

    # Step 4: Parse Tool to Input
    cmd = base_cmd.format(mode="parsing_tool_to_input", round=current_round)

    # ======== [Existing] ========
    if args.include_error_tool_call:
      cmd += " --include_error_tool_call"
    # ================================================================

    #
    status = run_step_live(f"Round {current_round} - Parse Tool Outputs", cmd)

    #
    if status == 222:
      print(
          "\n🎉 ALL CLEAR! No more data to process. Pipeline early-stopped at"
          f" Round {current_round}."
      )
      break

    print(f"\n🎉 ROUND {current_round} COMPLETED SUCCESSFULLY!")
    time.sleep(3)

  print("\n" + "=" * 60)
  print("🏆 PIPELINE EXECUTION FULLY COMPLETED!")
  print("=" * 60)


if __name__ == "__main__":
  main()
