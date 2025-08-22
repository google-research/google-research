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

import argparse
import csv

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


toxicity = evaluate.load("ybelkada/toxicity", "DaNLP/da-electra-hatespeech-detection", module_type="measurement")
ds = load_dataset("OxAISH-AL-LLM/wiki_toxic", split="test")

parser = argparse.ArgumentParser(description="Evaluate de-toxified models")
parser.add_argument("--model_type", default="all", type=str, help="Relative path to the source model folder")
parser.add_argument("--output_file", default="toxicity.csv", type=str, help="Relative path to the source model folder")
parser.add_argument("--batch_size", default=64, type=int, help="Batch size")
parser.add_argument("--num_samples", default=400, type=int, help="Number of samples")
parser.add_argument("--context_length", default=2000, type=int, help="Number of samples")
parser.add_argument("--max_new_tokens", default=30, type=int, help="Max new tokens for generation")
args = parser.parse_args()


if args.model_type == "all":
    MODELS_TO_TEST = [
        "ybelkada/gpt-neo-125m-detox",
        "EleutherAI/gpt-neo-125M",
        "EleutherAI/gpt-neo-2.7B",
        "ybelkada/gpt-neo-2.7B-detox",
        "ybelkada/gpt-j-6b-sharded-bf16",
        "ybelkada/gpt-j-6b-detoxs",
    ]
elif args.model_type == "gpt-neo":
    MODELS_TO_TEST = [
        "ybelkada/gpt-neo-125m-detox",
        "EleutherAI/gpt-neo-125M",
        "EleutherAI/gpt-neo-2.7B",
        "ybelkada/gpt-neo-2.7B-detox",
    ]
elif args.model_type == "gpt-j":
    MODELS_TO_TEST = [
        "ybelkada/gpt-j-6b-sharded-bf16",
        "ybelkada/gpt-j-6b-detox",
    ]
else:
    MODELS_TO_TEST = [args.model_type]
NUM_SAMPLES = args.num_samples
BATCH_SIZE = args.batch_size
output_file = args.output_file
max_new_tokens = args.max_new_tokens
context_length = args.context_length
device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

# consider only toxic prompts
ds = ds.filter(lambda x: x["label"] == 1)

toxicities = {}

# open a csv file
file = open(f"{output_file}", "w", newline="")
writer = csv.writer(file)
# add first rows
writer.writerow(["model_id", "mean_toxicity", "std_toxicity"])


for model_id in tqdm(MODELS_TO_TEST):
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": device}, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    input_texts = []

    for i, example in enumerate(ds):
        # set seed
        torch.manual_seed(42)

        input_text = example["comment_text"]
        input_texts.append(input_text[:2000])

        if i > NUM_SAMPLES:
            break

        if (i + 1) % BATCH_SIZE == 0:
            inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to(device)
            inputs.input_ids = inputs.input_ids[:context_length]
            inputs.attention_mask = inputs.attention_mask[:context_length]
            outputs = model.generate(**inputs, do_sample=True, max_new_tokens=max_new_tokens, use_cache=True)
            generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            generated_texts = [
                generated_text.replace(input_texts[i], "") for i, generated_text in enumerate(generated_texts)
            ]
            toxicity_score = toxicity.compute(predictions=generated_texts)
            input_texts = []

            if model_id not in toxicities:
                toxicities[model_id] = []
            toxicities[model_id].extend(toxicity_score["toxicity"])

    # last batch
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to(device)
    outputs = model.generate(**inputs, do_sample=True, max_new_tokens=30)
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    generated_texts = [generated_text.replace(input_texts[i], "") for i, generated_text in enumerate(generated_texts)]
    toxicity_score = toxicity.compute(predictions=generated_texts)
    toxicities[model_id].extend(toxicity_score["toxicity"])

    # compute mean & std using np
    mean = np.mean(toxicities[model_id])
    std = np.std(toxicities[model_id])

    # save to file
    writer.writerow([model_id, mean, std])

    # print
    print(f"Model: {model_id} - Mean: {mean} - Std: {std}")

    model = None
    torch.cuda.empty_cache()

# close file
file.close()
