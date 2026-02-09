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
import traceback
from autoeval.evaluator import Evaluator
from autoeval.clients import CLIENT_DICT


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
    print(blocks)
    assert len(blocks) % 2 == 0
    return blocks

def remove_invalid_steps(actions):
    """Remove invalid steps from the action sequence."""
    valid_actions = []
    for a in actions:
        if "click(" in a:
            arg = a[a.index("(")+1: a.index(")")]
            if type(eval(arg)) == str:
                valid_actions.append(a)
        elif "fill(" in a:
            arg = a[a.index("(")+1: a.index(",")].strip()
            if type(eval(arg)) == str:
                valid_actions.append(a)
        else:
            valid_actions.append(a)
    return valid_actions

# def extract_think_and_action(path: str) -> tuple[list[str], list[str]]:
#     """Extract the task trajectory from the log file."""
#     blocks = load_blocks(path)
#     think_list, action_list = [], []
#     for i in range(1, len(blocks), 2):
#         # action
#         b = blocks[i]
#         actions = remove_invalid_steps(b[1:])
#         if len(actions) == 0: continue
#         action_list.append(actions)
#         # think
#         b = blocks[i-1]
#         idx = b[-1].index("browsergym.experiments.loop - INFO -")
#         think_list.append(b[-1][idx+36: ].strip())

#     assert len(think_list) == len(action_list)

#     # TODO: merge same actions
#     return think_list, action_list

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

def extract_response(action):
    s, e = action.index("(")+1, action.index(")")
    return action[s: e]


def process_sample(
    idx, traj_info, log_save_path,
    model, eval_version,
):
    clients = {model: CLIENT_DICT[model](model_name=model)}
    evaluator = Evaluator(clients, log_save_path=log_save_path + "/trajs")
    try:
        out, _ = evaluator(traj_info, model, eval_version)
        eval_result = None
        if out["status"].lower() == "success": eval_result = True
        else: eval_result = False
        return [{
                "idx": idx,
                "gt": traj_info["eval"],
                "rm": eval_result,
                "thoughts": out["thoughts"],
                "uid": traj_info["traj_name"],
        }]
    except Exception as e:
        print(f"Error on {idx}, {e}")
        print(traceback.format_exc())
        return {
            "idx": idx,
            "gt": traj_info["eval"],
            "rm": None,
            "thoughts": None,
            "uid": traj_info["traj_name"],
        }


def main():
    # load task config
    task_id = args.result_dir.split('/')[-1].split(".")[1]
    config_path = os.path.join("config_files", f"{task_id}.json")
    config = json.load(open(config_path))

    # load trajectory log
    log_path = os.path.join(args.result_dir, "experiment.log")
    think_list, action_list = extract_think_and_action(log_path)
    # actions = [act for acts in action_list for act in acts]
    actions = [act for act in action_list]
    if "send_msg_to_user" in action_list[-1][0]:
        response = extract_response(action_list[-1][0])
    else:
        response = ""

    # load summary info
    summary_path = os.path.join(args.result_dir, "summary_info.json")
    summary = json.load(open(summary_path, 'r'))

    # collect traj info
    image_paths = [
        os.path.join(args.result_dir, f) for f in os.listdir(args.result_dir)
        if f.startswith("screenshot_step_") and f.endswith(".jpg")
    ]
    image_paths = sorted(image_paths, key=lambda x: int(x.split('/')[-1].split("_")[-1].split(".")[0]))
    traj_info = {
        "intent": config["intent"],
        "response": response,
        "captions": think_list,
        "actions": actions,
        "traj_name": config["task_id"],
        "image_paths": image_paths,
        "images": image_paths,
        "eval": summary["cum_reward"]
    }

    # evaluate trajectory
    log_save_path = os.path.join(args.log_dir, args.result_dir.split('/')[-1])
    print("Log Save Path:", log_save_path)
    if not os.path.exists(log_save_path):
        os.makedirs(log_save_path)
        os.makedirs(log_save_path + "/trajs")
    eval_info = process_sample(
        idx=config["task_id"], traj_info=traj_info,
        log_save_path=log_save_path,
        model=args.model, eval_version=args.prompt,
    )
    output_eval_path = os.path.join(args.result_dir, f"{args.model}_autoeval.json")
    json.dump(eval_info, open(output_eval_path, 'w'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True,
                        help="Path to the result directory, e.g., 'webarena.0'.")
    # autoeval
    parser.add_argument("--model", type=str, default="gemini-2.5-flash",
                        choices=['gemini-2.5-flash', 'claude-3-7-sonnet@20250219', 'gemini-2.5-pro', "google/gemma-3-12b-it"])
    parser.add_argument("--prompt", type=str, default="text",
                        choices=["text", "vision"])
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Path to the output directory.")

    args = parser.parse_args()

    if args.model == "gpt-4o" and args.prompt != "vision":
        print(f"Waring: use vision prompt by default for {args.model}.")
        args.prompt = "vision"

    main()
