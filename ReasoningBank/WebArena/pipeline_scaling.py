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
    config_flags = [config["sites"][0] == args.website for config in config_list]
    task_ids = [config["task_id"] for config, flag in zip(config_list, config_flags) if flag]
    # random.shuffle(task_ids)

    if args.end_index == None:
        args.end_index = len(task_ids)

    for tid in task_ids[args.start_index: args.end_index]:

        if int(tid) <= args.prev_id:
            continue

        procs = []
        for i in range(args.num_trials):
            # step 1: run async inference
            env = os.environ.copy()
            if args.website == "shopping":
                env["WA_SHOPPING"] = f"http://127.0.0.1:{args.start_port + i}"  # 每个进程不同端口
            elif args.website == "shopping_admin":
                env["WA_SHOPPING_ADMIN"] = f"http://127.0.0.1:{args.start_port + i}/admin"
            elif args.website == "gitlab":
                env["WA_GITLAB"] = f"http://127.0.0.1:{args.start_port + i}"
            elif args.website == "reddit":
                env["WA_REDDIT"] = f"http://127.0.0.1:{args.start_port + i}"

            p = Popen([
                "python", "run.py",
                "--task_name", f"webarena.{tid}",
                "--memory_path", f"memories_scaling/{args.website}.txt",
                "--model_name", args.model,
                "--results_path", f"{args.output_dir}/results_{i}",
            ], env=env)
            # process.wait()
            procs.append(p)

        exit_codes = [p.wait() for p in procs]
        # check if there are any non-zero exit codes
        if any(code != 0 for code in exit_codes):
            print(f"[WARN] webarena.{tid} with exit code: {exit_codes}")

        # step 2: update memory
        process = Popen([
            "python", "induce_scaling.py",
            "--result_dir", f"{args.output_dir}/results_{i}",
            "--output_path", f"memories_scaling/{args.website}.jsonl",
            "--task", f"webarena.{tid}",
            "--num_samples", str(args.num_trials),
        ])
        process.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--website", type=str, required=True,
                        choices=["shopping", "shopping_admin", "gitlab", "reddit", "map"])
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--num_trials", type=int, default=1,
                        help="Number of trials to run for each task.")
    parser.add_argument("--prev_id", type=int, default=-1)
    parser.add_argument("--model", type=str, default="gemini-2.5-flash",
                        choices=["gpt-3.5", "gpt-4", "gpt-4o", "gemini-2.5-flash", "gemini-2.5-pro"])
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save the results.")
    parser.add_argument("--start_port", type=int, default=8010,
                        help="Starting port number for the web server instances.")
    args = parser.parse_args()

    main()
