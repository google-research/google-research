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

"""downloading the SFTed experts from Huggingface and save to initial_experts."""

import os
from transformers import AutoModelForCausalLM

if __name__ == "__main__":
    for model_name in ["code_alpaca", "cot", "flan_v2", "gemini_alpaca", "lima", "oasst1", "open_orca", "science", "sharegpt", "wizardlm"]:
        if os.path.exists(model_name):
            continue
        model = AutoModelForCausalLM.from_pretrained("bunsenfeng/"+model_name)
        model.save_pretrained(model_name)
