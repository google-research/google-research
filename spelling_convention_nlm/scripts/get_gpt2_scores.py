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

"""Generate scores for Prompt/Target pairs using GPT2."""

# Necessary imports, and seed set for reproducibility.

import math
import pickle

import pandas as pd
import torch
from torch import nn
import tqdm
from transformers import GPT2LMHeadModel
from transformers import GPT2TokenizerFast
from transformers import set_seed


set_seed(42)

# Set the following 3 paramters to match your system.
# Location of prompts.
PROMPT_PATH = 'gen_examples.common_only.16k_sub_multiplied_full.tsv'
# Are we scoring adjacent or non-adjacent prompts?
ADJACENCY = 'adj'  # 'adj' or 'nonadj'
# Directory where scores should be written to disk.
OUTPUT_PATH = ''

raw_data = pd.read_csv(PROMPT_PATH, sep='\t')

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token = '<|endoftext|>'
model = GPT2LMHeadModel.from_pretrained('gpt2')
model_gpu = model.to('cuda')
# model_gpu = model # If GPU not available.

# What conditions are we in?
cond = ADJACENCY
batch_size = 16
max_len = 32

# Setup individual samples
joint_prompt = 'joint_' + cond + '_prompt'
joint_target_us_us = 'joint_target_us_us'
joint_target_uk_uk = 'joint_target_uk_uk'
eos_token = '<|endoftext|>'


samples = []

for jp, jtus, jtuk in zip(
    raw_data[joint_prompt],
    raw_data[joint_target_us_us],
    raw_data[joint_target_uk_uk],
):
  usp = jtus.split('<extra_id_0>')[1].split('<extra_id_1>')[0]
  ust = jtus.split('<extra_id_1>')[1].split('<extra_id_2>')[0]
  ukp = jtuk.split('<extra_id_0>')[1].split('<extra_id_1>')[0]
  ukt = jtuk.split('<extra_id_1>')[1].split('<extra_id_2>')[0]

  us_head = eos_token + jp.split('<extra_id_1>')[0].replace('<extra_id_0>', usp)
  us_tail = jp.split('<extra_id_1>')[1] + eos_token

  uk_head = eos_token + jp.split('<extra_id_1>')[0].replace('<extra_id_0>', ukp)
  uk_tail = jp.split('<extra_id_1>')[1] + eos_token

  # Basic tokenization without padding so that we can get IDX limits.
  us_head_tokens = tokenizer(us_head, return_tensors='pt')['input_ids']
  us_target_tokens = tokenizer(ust, return_tensors='pt')['input_ids']
  us_tail_tokens = tokenizer(us_tail, return_tensors='pt')['input_ids']

  uk_head_tokens = tokenizer(uk_head, return_tensors='pt')['input_ids']
  uk_target_tokens = tokenizer(ukt, return_tensors='pt')['input_ids']
  uk_tail_tokens = tokenizer(uk_tail, return_tensors='pt')['input_ids']

  # IDX limits for adding up the appropriate scores
  us_us_head_limit = us_head_tokens.shape[1]
  us_us_target_limit = us_us_head_limit + us_target_tokens.shape[1]
  us_us_full_limit = us_us_target_limit + us_tail_tokens.shape[1]

  us_uk_head_limit = us_head_tokens.shape[1]
  us_uk_target_limit = us_uk_head_limit + uk_target_tokens.shape[1]
  us_uk_full_limit = us_uk_target_limit + uk_tail_tokens.shape[1]

  uk_us_head_limit = uk_head_tokens.shape[1]
  uk_us_target_limit = uk_us_head_limit + us_target_tokens.shape[1]
  uk_us_full_limit = uk_us_target_limit + uk_tail_tokens.shape[1]

  uk_uk_head_limit = uk_head_tokens.shape[1]
  uk_uk_target_limit = uk_uk_head_limit + uk_target_tokens.shape[1]
  uk_uk_full_limit = uk_uk_target_limit + uk_tail_tokens.shape[1]

  # These are the pre-padded actual inputs that we will run through the model.
  j_us_us = torch.cat((us_head_tokens, us_target_tokens, us_tail_tokens), 1)
  assert j_us_us.shape[1] < max_len
  j_us_us = nn.functional.pad(
      j_us_us, (0, max_len - j_us_us.shape[1]), 'constant', 50256
  )

  j_us_uk = torch.cat((us_head_tokens, uk_target_tokens, uk_tail_tokens), 1)
  assert j_us_uk.shape[1] < max_len
  j_us_uk = nn.functional.pad(
      j_us_uk, (0, max_len - j_us_uk.shape[1]), 'constant', 50256
  )

  j_uk_us = torch.cat((uk_head_tokens, us_target_tokens, us_tail_tokens), 1)
  assert j_uk_us.shape[1] < max_len
  j_uk_us = nn.functional.pad(
      j_uk_us, (0, max_len - j_uk_us.shape[1]), 'constant', 50256
  )

  j_uk_uk = torch.cat((uk_head_tokens, uk_target_tokens, uk_tail_tokens), 1)
  assert j_uk_uk.shape[1] < max_len
  j_uk_uk = nn.functional.pad(
      j_uk_uk, (0, max_len - j_uk_uk.shape[1]), 'constant', 50256
  )

  samples.append((
      j_us_us,  # 0
      j_us_uk,  # 1
      j_uk_us,  # 2
      j_uk_uk,  # 3
      (us_us_head_limit, us_us_target_limit, us_us_full_limit),
      (us_uk_head_limit, us_uk_target_limit, us_uk_full_limit),
      (uk_us_head_limit, uk_us_target_limit, uk_us_full_limit),
      (uk_uk_head_limit, uk_uk_target_limit, uk_uk_full_limit),
  ))


total_items = len(samples)
total_batches = math.ceil(total_items / batch_size)

out_dict = {'us_us': [], 'us_uk': [], 'uk_us': [], 'uk_uk': []}

for b in tqdm.tqdm(range(total_batches)):
  subsample = samples[b * batch_size : (b + 1) * batch_size]
  for m, lset in enumerate(['us_us', 'us_uk', 'uk_us', 'uk_uk']):
    # Concatenate the inputs.
    model_batch = torch.cat([x[m] for x in subsample], 0)
    model_batch_gpu = model_batch.to('cuda')
    # model_batch_gpu = model_batch  # If GPU not available.

    # Run the model.
    logits = model_gpu(model_batch_gpu)['logits']

    # Get conditional to target, conditional to end, and joint scores.
    for i, ss in enumerate(subsample):
      # Get all the limits.
      head_limit, target_limit, full_limit = subsample[i][m + 4]

      # Get all the prob types
      tgt_range = range(head_limit - 1, target_limit - 1)
      tgts = model_batch[i, head_limit:target_limit]
      totarg = float(sum([logits[i, j, k] for j, k in zip(tgt_range, tgts)]))

      tgt_range = range(head_limit - 1, full_limit - 1)
      tgts = model_batch[i, head_limit:full_limit]
      toend = float(sum([logits[i, j, k] for j, k in zip(tgt_range, tgts)]))

      tgt_range = range(0, full_limit - 1)
      tgts = model_batch[i, 1:full_limit]
      full = float(sum([logits[i, j, k] for j, k in zip(tgt_range, tgts)]))

      out_dict[lset].append((totarg, toend, full))


with open(OUTPUT_PATH + cond + '_scores.pickle', 'wb') as f:
  pickle.dump(out_dict, f)
