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

#!/bin/bash

import os
import json
import torch
import random
import argparse

from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

HOME, SCRATCH = os.environ['HOME'], os.environ['SCRATCH']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_lm(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    model.to(device)
    model.eval()

    model.resize_token_embeddings(len(tokenizer))

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    tokenizer.padding_side = "left"

    return model, tokenizer, None


def process_prompt(text):
    text = text.replace('’', "'")
    text = text.replace('‘', "'")
    text = text.replace('“', '"')
    text = text.replace('”', '"')
    return text


def batch_generate(
    model,
    tokenizer,
    batch,
    num_returns=1,
    do_sample=True,
    top_p=0.9,
    top_k=30,
    num_beams=1,
    min_new_tokens=20,
    max_new_tokens=10,
):
    encoded_prompt = tokenizer(batch, padding='longest', add_special_tokens=False, return_tensors="pt")
    encoded_prompt = encoded_prompt.to(device)

    input_ids, attention_mask = encoded_prompt['input_ids'], encoded_prompt['attention_mask']

    prompt_len = input_ids.shape[1]
    output_sequences = model.generate(
        input_ids,
        attention_mask=attention_mask,
        min_new_tokens=min_new_tokens,
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        repetition_penalty=1.0,
        do_sample=do_sample,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        num_return_sequences=num_returns,
    )

    batch_gens = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
    batch_cons = tokenizer.batch_decode(output_sequences[:, prompt_len:], skip_special_tokens=True)

    return batch_gens, batch_cons


def main(args):
    # Load LM
    model, tokenizer, config = load_lm(os.path.join(SCRATCH, args.lm_dir))
    print("- LM loaded:", os.path.join(SCRATCH, args.lm_dir))

    # load RealToxicityPrompt dataset
    with open(args.prompts_file, 'r') as f:
        prompts = [json.loads(line) for line in f]

    if args.sample_size is not None:
        prompts_sampled = random.sample(prompts, min(args.sample_size, len(prompts)))
    else:
        prompts_sampled = prompts
    print(f"- evaluate {len(prompts_sampled)}/{len(prompts)} samples from {args.prompts_file}")

    # Generate continuations
    continuations = []
    batch = []
    for s in tqdm(prompts_sampled):
        batch.append(process_prompt(s['prompt']['text']))

        if len(batch) == args.batch_size:
            _, batch_cons = batch_generate(
                model,
                tokenizer,
                batch,
                num_returns=args.num_returns,
                do_sample=args.do_sample,
                top_p=args.top_p,
                top_k=args.top_k,
                num_beams=args.num_beams,
                max_new_tokens=args.max_length,
                min_new_tokens=args.min_length,
            )
            assert len(batch_cons) == args.batch_size * args.num_returns

            for i in range(args.batch_size):
                sample_cons = [batch_cons[i * args.num_returns + j] for j in range(args.num_returns)]
                continuations.append(sample_cons)

            batch = []

    if len(batch) > 0:
        _, batch_cons = batch_generate(
            model,
            tokenizer,
            batch,
            num_returns=args.num_returns,
            do_sample=args.do_sample,
            top_p=args.top_p,
            top_k=args.top_k,
            num_beams=args.num_beams,
            max_new_tokens=args.max_length,
            min_new_tokens=args.min_length,
        )
        assert len(batch_cons) == len(batch) * args.num_returns

        for i in range(len(batch)):
            sample_cons = [batch_cons[i * args.num_returns + j] for j in range(args.num_returns)]
            continuations.append(sample_cons)

    # Save generations
    assert len(continuations) == len(prompts_sampled)
    for gens, pmt in zip(continuations, prompts_sampled):
        pmt['generations'] = []
        for g in gens:
            pmt['generations'].append({'text': g})

    with open(os.path.join(SCRATCH, args.output_file), 'w') as wf:
        for item in prompts_sampled:
            wf.write(json.dumps(item) + "\n")

    print("Generation saved at: ", os.path.join(SCRATCH, args.output_file))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_file",
        type=str,
        default='realtoxicityprompts-data/generations_rescored/prompted/test.jsonl',
    )
    parser.add_argument(
        "--lm_dir",
        type=str,
        default='huggingface/gpt2-xl',
        help="Language model directory."
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default='realtoxicityprompts-data/prompts_rescored.jsonl',
        help="A jsonl file."
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of samples to evalute.",
    )
    parser.add_argument(
        "--num_returns",
        type=int,
        default=25,
    )
    parser.add_argument(
        '--do_sample',
        action='store_true',
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )

    args = parser.parse_args()
    main(args)