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
from autoeval.prompts import *


class Evaluator:
    def __init__(self, lm_clients, log_save_path=None):
        self.lm_clients = lm_clients
        self.log_save_path = log_save_path

    def __call__(self, info, client="gpt-3.5", version="naive"):
        assert (client in self.lm_clients), \
            f"Client {client} not found in {self.lm_clients.keys()}"
        if version == "text":
            eval_info, eval_str, prompt = self.eval_text(info, client)
        elif version == "vision":
            eval_info, eval_str, prompt = self.eval_vision(info, client)
        else:
            raise NotImplementedError(f"Version {version} not implemented")

        if self.log_save_path:
            with open(self.log_save_path + "/outputs.jsons", "a") as f:
                f.write(
                    json.dumps(
                        {
                            "id": info["traj_name"],
                            "eval_info": eval_info,
                        }
                    )
                    + "\n"
                )
            with open(f"{self.log_save_path}/{info['traj_name']}.md", "w") as md_file:
                md_file.write(f"## Intent\n\n{info['intent']}\n\n")
                md_file.write(f"## RM\n\n{eval_str}\n\n")
                md_file.write(f"## Final Response {info['response']}\n\n")
                if "captions" in info and info['captions'] is not None:
                    md_file.write("## Captions\n\n")
                    for idx, cap in enumerate(info["captions"]):
                        md_file.write(f"===============")
                        md_file.write(f"{cap}\n")
                md_file.write("\n## Images\n\n")
                for idx, img in enumerate(info["image_paths"]):
                    rel_img_path = os.path.relpath(img, self.log_save_path)
                    md_file.write(f"![Image {idx+1}]({rel_img_path})\n")

                if "config" in info:
                    md_file.write("## Config\n\n")
                    cofig_str = json.dumps(info["config"], indent=4)
                    md_file.write(f"```json\n{cofig_str}\n```\n")
                if prompt:
                    md_file.write("## Main Prompt\n\n")
                    md_file.write(f"```md\n{prompt}\n```\n")
        return eval_info, prompt

    def eval_text(self, info, client):
        response = info["response"] if info["response"] else "None"
        lm_client = self.lm_clients[client]
        action_history = ""
        for idx, act in enumerate(info["actions"]):
            action_history += f"{idx+1}: {act}\n"
        prompt, sys_msg = build_text_eval_prompt(
            info["captions"][-1], info["intent"], response, action_history
        )
        msg_str, _ = lm_client.one_step_chat(prompt, system_msg=sys_msg)
        msg_dict = {
            "thoughts": extract_content(msg_str, "Thoughts:"),
            "status": extract_content(msg_str, "Status:").replace('"', ""),
        }
        return msg_dict, msg_str, prompt

    def eval_vision(self, info, client):
        assert client == "gpt-4v" or client == "gpt-4o"
        action_history = ""
        for idx, act in enumerate(info["actions"]):
            action_history += f"{idx+1}: {act}\n"
        prompt, sys_msg = build_vision_eval_prompt(
            info["intent"], info["response"], action_history
        )
        img = info["images"][-1]

        lm_client = self.lm_clients[client]
        msg_str, _ = lm_client.one_step_chat(
            text=prompt, image=img, system_msg=sys_msg
        )
        del info["images"]
        msg_dict = {
            "thoughts": extract_content(msg_str, "Thoughts:"),
            "status": extract_content(msg_str, "Status:").replace('"', ""),
        }
        return msg_dict, msg_str, prompt
