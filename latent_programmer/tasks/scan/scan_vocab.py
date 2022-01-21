# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Vocabulary and vocab-related utilities for SCAN dataset.

For a task with N parts:
* The maximum number of input tokens is 5N - 1, e.g., each part could be
  "walk around left thrice", with parts separated by conjunctions.
* The maximum number of output tokens is 25N - 1, e.g., if each part is
  "walk around left thrice" then there are 24 output tokens per part, with a
  separator between parts.

These counts don't include any BOS or EOS tokens at the beginning or end.
"""

from typing import Dict, List, Tuple

BOS = 'BOS'
EOS = 'EOS'
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2

# A token to separate partial programs.
SEP = '|'

INPUT_VOCAB = ['walk', 'look', 'run', 'jump',
               'turn', 'left', 'right',
               'opposite', 'around',
               'twice', 'thrice',
               'and', 'after']

OUTPUT_VOCAB = ['I_WALK', 'I_LOOK', 'I_RUN', 'I_JUMP',
                'I_TURN_LEFT', 'I_TURN_RIGHT', SEP]

SCAN_VOCAB = INPUT_VOCAB + OUTPUT_VOCAB


def build_token_tables():
  """Get mappings from ids to tokens and vice versa."""
  # Reserve 0, 1, 2 for padding, bos, eos.
  id_token_table = {id + 3: token for id, token in enumerate(SCAN_VOCAB)}
  id_token_table[BOS_ID] = BOS
  id_token_table[EOS_ID] = EOS
  token_id_table = {token: id for id, token in id_token_table.items()}
  return id_token_table, token_id_table


def encode(tokens, token_id_table):
  return [token_id_table[token] for token in tokens]


def encode_str(tokens_str, token_id_table):
  return encode(tokens_str.strip().split(), token_id_table)


def decode(ids, id_token_table):
  return [id_token_table[id] for id in ids if id != PAD_ID]
