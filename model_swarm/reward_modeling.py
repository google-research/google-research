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

import json
import math
import time
import torch
import datetime
from tqdm import tqdm
from datasets import load_dataset
from tenacity import retry, wait_random_exponential, stop_after_attempt
from sklearn.metrics import accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

# pre-calculated reward model scores and response length distributions based on the dev set of rm.json
RM_HIGH = 2.65625
RM_LOW = -3.060546875
LEN_HIGH = 201
LEN_LOW = 10

def get_reward_scores(pairs, gpu_id, mode ="default"):
    """
    Args:
        pairs: a list of pairs of the form:
            [
                {"role": "user", "content": "Hello! What's your name?"},
                {"role": "assistant", "content": "My name is InternLM2! A helpful AI assistant. What can I do for you?"}
            ]
        gpu_id: the gpu id to use
        mode: default, concise, verbose, reverse
    Returns:
        a list of scores of the same length as pairs
    """

    model = AutoModel.from_pretrained(
    "internlm/internlm2-7b-reward",
    device_map="cuda:"+str(gpu_id),
    torch_dtype=torch.float16,
    trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2-7b-reward", trust_remote_code=True)

    # reduce if OOM
    BATCH_SIZE = 10
    scores = []

    for i in range(0, len(pairs), BATCH_SIZE):
        end_loc = min(len(pairs), i + BATCH_SIZE)
        examples = pairs[i:end_loc]
        scores += model.get_scores(tokenizer, examples)

    # normalize reward model scores
    scores = [(x - RM_LOW) / (RM_HIGH - RM_LOW) for x in scores]
    # cutoff
    scores = [min(1, x) for x in scores]
    scores = [max(0, x) for x in scores]

    if not mode == "default":
        lengths = []
        for i in range(len(pairs)):
            response = pairs[i][1]["content"]
            input_ids = tokenizer(response, return_tensors="pt")
            lengths.append(len(input_ids[0]))

        # print(scores)

        if mode == "verbose":
            lengths = [(x - LEN_LOW) / (LEN_HIGH - LEN_LOW) for x in lengths]
        elif mode == "concise":
            lengths = [1 - ((x - LEN_LOW) / (LEN_HIGH - LEN_LOW)) for x in lengths]
        lengths = [min(1, x) for x in lengths]
        lengths = [max(0, x) for x in lengths]

        # print(scores)
        # print(lengths)

        final_scores = [(scores[i] + lengths[i]) / 2 for i in range(len(scores))]
        scores = final_scores

    if mode == "reverse":
        scores = [-x for x in scores]

    del model
    del tokenizer
    torch.cuda.empty_cache()

    return scores