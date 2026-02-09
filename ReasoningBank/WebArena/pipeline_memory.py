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

import os
import json
import argparse
from subprocess import Popen
import random

def main():
    # collect examples
    config_files = [
        os.path.join("config_files", f) for f in os.listdir("config_files")
        if f.endswith(".json") and f.split(".")[0].isdigit()
    ]
    config_files = sorted(config_files, key=lambda x: int(x.split("/")[-1].split(".")[0]))
    config_list = [json.load(open(f)) for f in config_files]
    if args.website == "multi":
        config_flags = []
        for config in config_list:
            if len(config["sites"]) != 1 and "map" not in config['sites']:
                config_flags.append(True)
            else:
                config_flags.append(False)
    else:
        config_flags = [config["sites"][0] == args.website for config in config_list]
    task_ids = [config["task_id"] for config, flag in zip(config_list, config_flags) if flag]
    # random.shuffle(task_ids)

    if args.end_index == None:
        args.end_index = len(task_ids)

    # num = args.output_dir.split("_")[-1]

    for tid in task_ids[args.start_index: args.end_index]:

        if int(tid) <= args.prev_id:
            continue

        # step 1: run inference
        process = Popen([
            "python", "run.py",
            "--task_name", f"webarena.{tid}",
            "--memory_path", f"memories_{args.memory_mode}/{args.website}.txt",
            "--model_name", args.model,
            "--results_path", f"{args.output_dir}",
        ])
        process.wait()

        # step 2: run evaluation
        process = Popen([
            "python", "-m", "autoeval.evaluate_trajectory",
            "--result_dir", f"{args.output_dir}/webarena.{tid}",
            "--model", args.model,
            "--log_dir", f"autoeval/logs_{args.memory_mode}_{args.website}",
        ])
        process.wait()

        # step 3: extract new memory items
        process = Popen([
            "python", "induce_reasoningbank.py",
            "--result_dir", args.output_dir,
            "--task", f"webarena.{tid}",
            "--criteria", args.judge,
            "--memory_mode", args.memory_mode,
            "--model", args.model,
            "--output_path", f"memories_{args.memory_mode}/{args.website}.jsonl"
        ])
        process.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--website", type=str, required=True,
                        choices=["shopping", "shopping_admin", "gitlab", "reddit", "multi"])
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--model", type=str, default="gemini-2.5-flash",
                        choices=["gemini-2.5-flash", "claude-3-7-sonnet@20250219", "gemini-2.5-pro", "google/gemma-3-12b-it"])
    parser.add_argument("--prev_id", type=int, default=-1)
    parser.add_argument("--memory_mode", type=str, default="reasoningbank")
    parser.add_argument("--judge", type=str, default="auto_eval")
    args = parser.parse_args()

    main()
