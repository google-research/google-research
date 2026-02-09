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
import random
import argparse
import re
import gzip
import pickle

from google import genai
from functools import partial
from google.genai.types import HttpOptions, GenerateContentConfig
import time
from memory_management import select_memory
# import openai
# openai.api_key = os.environ["OPENAI_API_KEY"]
# from openai import OpenAI
client = genai.Client(http_options=HttpOptions(api_version="v1"))

from prompt.instruction import PARALLEL_SI, PARALLEL_AWM_SI

# %% load examples
def load_blocks(path):
    """Load blank-line separated blocks from the log file."""
    blocks, block = [], []
    for line in open(path, 'r'):
        if line.strip() == "":
            blocks.append(block)
            block = []
        else:
            if line.strip():
                block.append(line.strip())
    assert len(blocks) % 2 == 0
    return blocks

def remove_invalid_steps(actions):
    """Remove invalid steps from the action sequence."""
    valid_actions = []
    for a in actions:
        if "click(" in a:
            arg = a[a.index("(")+1: a.index(")")]
            try:
                if type(eval(arg)) == str and type(eval(arg[1:-1])) == int:
                    valid_actions.append(a)
            except:
                continue
        elif "fill(" in a:
            arg = a[a.index("(")+1: a.index(",")].strip()
            if type(eval(arg)) == str:
                valid_actions.append(a)
        elif "scroll(" in a or "noop(" in a:
            continue
        else:
            valid_actions.append(a)
    return valid_actions

def load_step_pkl(path):
    with gzip.open(path, 'rb') as f:
        data = pickle.load(f)
    msgs = data.agent_info.get("chat_messages", [])[:3]
    return [str(m) for m in msgs]


def load_trajectory(folder):
    step_files = sorted(
        [f for f in os.listdir(folder) if re.match(r"step_\d+\.pkl\.gz", f)],
        key=lambda f: int(re.findall(r"\d+", f)[0])
    )
    traj = []
    for f in step_files:
        try:
            msgs = load_step_pkl(os.path.join(folder, f))
            traj.append(msgs[2])
        except Exception as e:
            continue
    return traj

def extract_think_and_action(path):
    """Extract the task trajectory from the log file."""
    log_text = open(path, 'r').read()
    lines = log_text.splitlines()
    think_list = []
    action_list = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("action:"):
            # Parse the full action block (can span multiple lines)
            action_lines = []
            if line.strip() != "action:":
                action_lines.append(line[len("action:"):].strip())
            i += 1
            while i < len(lines) and lines[i].strip() != "":
                action_lines.append(lines[i].strip())
                i += 1
            action_text = "".join(action_lines).strip()

            # Now look backward for the most recent loop-INFO thinking block
            thinking_lines = []
            for j in range(i - 1, -1, -1):
                if "browsergym.experiments.loop - INFO -" in lines[j]:
                    thinking = lines[j].split("browsergym.experiments.loop - INFO -", 1)[-1].strip()
                    thinking_lines.insert(0, thinking)
                    break
            thinking_text = "\n".join(thinking_lines).strip()
            think_list.append(thinking_text)
            action_list.append(action_text)
        else:
            i += 1

    assert len(think_list) == len(action_list)
    return think_list, action_list

def format_trajectory(think_list, action_list):
    trajectory = []
    for t, a in zip(think_list, action_list):
        # acts = '\n'.join(a)
        acts = a
        trajectory.append(f"<think>\n{t}\n</think>\n<action>\n{acts}\n</action>")
    return '\n\n'.join(trajectory)


def format_examples(examples, flag=False):
    """Format examples to the prompt."""
    formatted_examples = []
    for ex in examples:
        trajectory = format_trajectory(ex["think_list"], ex["action_list"])
        formatted_examples.append(f"Query: {ex['query']}\nTrajectory:\n{trajectory}")
    # return '\n\n'.join(["## Concrete Examples"] + formatted_examples + ["## Summary Workflow"])
    if flag:
        return '\n\n'.join(["## Query and Trajectory Generated Using Previous Memory"] + formatted_examples + ["## Correctness Signal"]+ ["The result is CORRECT."] + ["## Updated Memory"])
    else:
        return '\n\n'.join(["## Query and Trajectory Generated Using Previous Memory"] + formatted_examples + ["## Correctness Signal"]+ ["The result is INCORRECT."] + ["## Updated Memory"])


def llm_generate(prompt, args, verbose = False, si = None):
    """Call gpt model to generate workflows."""
    if verbose:
        print("Prompt:\n", prompt, '\n\n')
    response = client.models.generate_content(
        model=args.model,
        contents=prompt,
        config=GenerateContentConfig(
            temperature=1.0,
            max_output_tokens=65536,
            system_instruction=si.strip() if si else None,
        )
    )
    response = response.text
    if verbose: print(response)
    return response.split("\n\n")


def get_info(f, status = None):

    # get query -> task objective
    task_id = f.split('/')[-1].split("_")[0].split(".")[1]
    config_path = os.path.join("config_files", f"{task_id}.json")
    config = json.load(open(config_path))
    query = config["intent"]

    template_id = config["intent_template_id"]  # for deduplication
    traj = load_trajectory(f)

    # add to template dict
    if status == 'success':
        wdict = {"query": query, "template_id": template_id, "steps": traj, "status": "success"}
    elif status == 'fail':
        wdict = {"query": query, "template_id": template_id, "steps": traj, "status": "fail"}

    return wdict

def main():

    trajectories = ""

    for i in range(int(args.num_samples)):

        res_dir = args.result_dir
        cur_task = os.path.join(res_dir, args.task)

        # load feedback from auto-evaluation
        if args.criteria == "gt":
            reward = json.load(open(os.path.join(cur_task, "summary_info.json")))["cum_reward"]
        elif args.criteria == "autoeval":
            reward = json.load(open(os.path.join(cur_task, f"{args.model}_autoeval.json")))[0]["rm"]
        else:
            raise ValueError(f"Invalid criteria: {args.criteria}.")

        if reward == 0:
            status = "success"
        else:
            status = "fail"

        ex = get_info(cur_task, status)

        if i == 0:
            trajectories += f"**Query:** {ex['query']}\n\n"
        trajectories += f"**Trajectory {i+1} :**\n"
        trajectories += "\n\n".join(ex["steps"]) + "\n\n"

    generated_memory_item = llm_generate(trajectories, args, True, si=PARALLEL_SI)

    # write to jsonl file
    with open(args.output_path, 'a') as f:
        f.write(json.dumps({
            "task_id": args.task.split(".")[-1],
            "query": ex["query"],
            "memory_items": generated_memory_item,
            "template_id": ex["template_id"]
        }) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, default="results_base_new",
                        help="Path to the result directory. Support multiple directories separated by space.")
    parser.add_argument("--output_path", type=str, default=None, required=True,
                        help="Path to the output file.")
    parser.add_argument("--criteria", type=str, default="gt",
                        choices=["gt", "autoeval"],
                        help="'gt': only use examples with gold reward, 'autoeval': use examples with autoeval reward.")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash",
                        choices=["gpt-3.5", "gpt-4", "gpt-4o", "gemini-2.5-flash"])
    parser.add_argument("--task", type=str, default="webarena.21")
    parser.add_argument("--num_samples", type=int, default=1, help="samples per task")
    args = parser.parse_args()

    main()