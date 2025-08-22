# coding=utf-8
# Copyright 2025 The Google Research Authors.
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
import shutil
from typing import List
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft.utils.save_and_load import set_peft_model_state_dict, get_peft_model_state_dict

def lora_merge(weights, lora_name_list, output_name, gpu_id, directly_load_safetensors = 0):
    """merge lora models with weights, fast or slow based on directly_load_safetensors."""

    # the slow merge: load models as AutoModelForCausalLM, merge them, save them again
    if not directly_load_safetensors:
        lora_state_dict_list = []
        for lora_name in lora_name_list:
            model = AutoModelForCausalLM.from_pretrained(lora_name).to(f"cuda:{gpu_id}")
            lora_state_dict_list.append(get_peft_model_state_dict(model))
            if lora_name != lora_name_list[-1]:
                del model
            # torch.cuda.empty_cache()

        final_state_dict = {}
        for i in range(len(lora_state_dict_list)):
            if i == 0:
                for key in lora_state_dict_list[i].keys():
                    final_state_dict[key] = weights[i] * lora_state_dict_list[i][key]
            else:
                for key in lora_state_dict_list[i].keys():
                    assert key in final_state_dict.keys()
                    final_state_dict[key] += weights[i] * lora_state_dict_list[i][key]

        model = AutoModelForCausalLM.from_pretrained(lora_name_list[0]).to(f"cuda:{gpu_id}")
        set_peft_model_state_dict(model, final_state_dict)
        if os.path.exists(output_name):
            shutil.rmtree(output_name)
        model.save_pretrained(output_name)
    else:
        # the fast merge: load only state_dicts, merge them, save only state_dicts
        # apply to the setting that models share the same architecture, sharding, and adapter format
        lora_state_dict_list = []
        for lora_name in lora_name_list:
            state_dict_this = load_file(os.path.join(lora_name, "adapter_model.safetensors"), device="cpu")
            lora_state_dict_list.append(state_dict_this)

        final_state_dict = {}
        for i in range(len(lora_state_dict_list)):
            if i == 0:
                for key in lora_state_dict_list[i].keys():
                    final_state_dict[key] = weights[i] * lora_state_dict_list[i][key]
            else:
                for key in lora_state_dict_list[i].keys():
                    assert key in final_state_dict.keys()
                    final_state_dict[key] += weights[i] * lora_state_dict_list[i][key]

        if not os.path.exists(output_name):
            os.mkdir(output_name)
        save_file(final_state_dict, os.path.join(output_name, "adapter_model.safetensors"))

        return final_state_dict

    # del model
    # del lora_state_dict_list
    # del final_state_dict
    # torch.cuda.empty_cache()


# sanity check example
# lora_merge([0.3, 0.6, 0.8], ["./initial_experts/lima", "./initial_experts/cot", "./initial_experts/science"], "./new", 0, directly_load_safetensors=1)