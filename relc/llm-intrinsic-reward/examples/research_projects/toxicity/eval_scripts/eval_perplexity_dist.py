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

"""
evaluate generated output for diversity (dist-n) and fluency (perplexity according to GPT2-XL)
"""

import os
import math
import torch
import click
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


def conditional_perplexity(generations_df, model, tokenizer, device='cuda'):
    perplexities = []
    ct = 0
    # for every prompt
    for i, row in tqdm(generations_df.iterrows(), total=len(generations_df.index), desc='Evaluating fluency'):
        prompt = row.prompt['text']
        prompt_input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        prompt_loss = model(prompt_input_ids, labels=prompt_input_ids)[0] * (prompt_input_ids.shape[1]-1)
        # for every generation conditioned on the prompt
        generations = [g['text'] for g in row['generations']]
        for gen in generations:
            full_input_ids = tokenizer.encode(prompt+gen, return_tensors='pt').to(device)
            full_loss = model(full_input_ids, labels=full_input_ids)[0] * (full_input_ids.shape[1]-1)
            loss = (full_loss - prompt_loss) / (full_input_ids.shape[1] - prompt_input_ids.shape[1])
            ppl = math.exp(loss.item())
            if ppl < 1e4:   # for sanity
                perplexities.append(ppl)
    return np.nanmean(perplexities)


def distinctness(generations_df):
    dist1, dist2, dist3 = [], [], []
    # calculate dist1, dist2, dist3 across generations for every prompt
    for i, row in tqdm(generations_df.iterrows(), total=len(generations_df.index), desc='Evaluating diversity'):
        generations = [g['text'] for g in row['generations']]
        unigrams, bigrams, trigrams = set(), set(), set()
        total_words = 0
        for gen in generations:
            o = gen.split(' ')
            total_words += len(o)
            unigrams.update(o)
            for i in range(len(o) - 1):
                bigrams.add(o[i] + '_' + o[i+1])
            for i in range(len(o) - 2):
                trigrams.add(o[i] + '_' + o[i+1] + '_' + o[i+2])
        dist1.append(len(unigrams) / total_words)
        dist2.append(len(bigrams) / total_words)
        dist3.append(len(trigrams) / total_words)

    # take the mean across prompts
    return np.nanmean(dist1), np.nanmean(dist2), np.nanmean(dist3)


def main(args):
    assert os.path.exists(args.generations_file), f"{args.generations_file} doesn't exit!"
    generations_df = pd.read_json(args.generations_file, lines=True)

    # calculate diversity
    dist1, dist2, dist3 = distinctness(generations_df)

    # write output results
    with open(args.output_file, 'w') as fo:
        for i, dist_n in enumerate([dist1, dist2, dist3]):
            fo.write(f'dist-{i+1} = {dist_n}\n')
            print(f'dist-{i+1} = {dist_n}\n')

    # calculate fluency
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_model = AutoModelForCausalLM.from_pretrained('gpt2-xl').to(device)
    eval_tokenizer = AutoTokenizer.from_pretrained('gpt2-xl')
    torch.cuda.empty_cache()
    with torch.no_grad():
        ppl = conditional_perplexity(generations_df, eval_model, eval_tokenizer, device=device)

    # write output results
    with open(args.output_file, 'a') as fo:
        fo.write(f'perplexity = {ppl}')
        print(f'perplexity = {ppl}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generations_file",
        type=str,
        help="A jsonl file with two columns: 'text' and 'generations'."
    )
    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()
    main(args)