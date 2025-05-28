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

r"""Huggingface Llama2 inference."""

import torch
import transformers
from transformers import AutoTokenizer


model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
    return_full_text=False,
)

if pipeline.tokenizer.pad_token_id is None:
    pipeline.tokenizer.pad_token_id = tokenizer.pad_token_id

if pipeline.model.config.pad_token_id is None:
    pipeline.model.config.pad_token_id = tokenizer.pad_token_id


while True:
    sequences = pipeline(
        'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        min_new_tokens=10,
        max_new_tokens=20,
    )
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")
