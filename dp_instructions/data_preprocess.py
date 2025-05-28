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

"""This code implements the pre-processing steps.

See Appendix E of
[Privacy-Preserving Instructions for Aligning Large Language Models]
(https://arxiv.org/abs/2402.13659).
"""

import datasets
from datasets import load_dataset
from nltk.tokenize import word_tokenize
import numpy as np


# Load the LMSYS-1M dataset.
print('Loading dataset, this may take few minutes.')
lmsys_1m = load_dataset('lmsys/lmsys-chat-1m', split='train')


# First step, apply the following rules:
# 1. Only keep English conversations.
# 2. Remove redacted conversations.
# 3. Remove conversations flagged by OpenAI moderation tool.
# 4. Only keep first round instructions.
language = lmsys_1m['language']
redacted = lmsys_1m['redacted']
moderation = lmsys_1m['openai_moderation']

remaining_instructions = []
for i, sample in enumerate(lmsys_1m):
  if (
      language[i] == 'English'
      and not redacted[i]
      and not moderation[i][-1]['flagged']
  ):
    first_round = sample['conversation'][0]
    assert first_round['role'] == 'user'
    instruction = first_round['content']
    remaining_instructions.append(instruction)

num_samples = len(remaining_instructions)
print('Number of samples after first step:', num_samples)

# Second step, sequence-level de-duplication.
helper_dict = {}
seq_dedup_remaining_instructions = []
for inst in remaining_instructions:
  if inst not in helper_dict:
    helper_dict[inst] = 1
    seq_dedup_remaining_instructions.append(inst)

num_samples = len(seq_dedup_remaining_instructions)
print('Number of samples after sequence-level de-duplication:', num_samples)


# Third step, n-gram de-duplication.
print('Running n-gram de-duplication, this may take few minutes.')


def get_string_n_grams(s, ng_size):
  tokens = word_tokenize(s)
  num_n_grams = len(tokens) - ng_size + 1
  tmp_grams = []
  for j in range(num_n_grams):
    tmp_gram = ' '.join(tokens[j : j + ng_size])
    tmp_grams.append(tmp_gram)
  return tmp_grams


ngram_size = 10
helper_dict = {}
ngram_dedup_remaining_instructions = []
# de-duplication based on a n-gram dictionary
for inst in seq_dedup_remaining_instructions:
  n_grams = get_string_n_grams(inst, ngram_size)
  keep = True
  for n_gram in n_grams:
    if n_gram not in helper_dict:
      helper_dict[n_gram] = 1
    else:
      keep = False
      break
  if keep:
    ngram_dedup_remaining_instructions.append(inst)

num_samples = len(ngram_dedup_remaining_instructions)
print('Number of samples after n-gram de-duplication:', num_samples)


# Final step, filter out unusual repetitions.
final_instructions = []
for inst in ngram_dedup_remaining_instructions:
  inst = inst.lower()
  if 'write an instruction of' in inst and 'words in chemical industry' in inst:
    continue
  elif (
      'write an article about' in inst and 'words in chemical industry' in inst
  ):
    continue
  elif 'give me an introduction over' in inst and 'a chemical company' in inst:
    continue
  elif 'write an introduction of' in inst and 'in chemical industry' in inst:
    continue
  else:
    final_instructions.append(inst)

num_samples = len(final_instructions)
print(
    'Number of samples after filtering out unusual repetitions:', num_samples
)  # 188819


np.random.shuffle(final_instructions)

print(
    f'Splitting the dataset into train ({180000}), validation ({5000}), and'
    f' test ({num_samples-185000}).'
)
train_dataset = datasets.Dataset.from_dict(
    {'instruction': final_instructions[0:180000]}
)
val_dataset = datasets.Dataset.from_dict(
    {'instruction': final_instructions[180000:185000]}
)
test_dataset = datasets.Dataset.from_dict(
    {'instruction': final_instructions[185000:]}
)

dataset_dict = datasets.DatasetDict(
    {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
)
dataset_dict.save_to_disk(
    'dp_finetuning/data/chatbot_arena_instructions_train180k'
)
