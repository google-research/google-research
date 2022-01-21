# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Create dictionaries for mapping input and program tokens to ids."""

BOS = 'BOS'
EOS = 'EOS'

def build_input_token_tables():
  tokens = [
    'walk',
    'look',
    'jump',
    'turn left',
    'turn right',
    'left',
    'right',
    'turn opposite left',
    'turn opposite right',
    'opposite left',
    'opposite right',
    'turn around left',
    'turn around right',
    'around left',
    'around right',
    'twice',
    'thrice',
    'and',
    'after',
  ]
  # Reserve 0 for padding.
  id_token_table = {id+1: token for id, token in enumerate(tokens)}
  token_id_table = {token: id for id, token in id_token_table.items()}
  return (id_token_table, token_id_table)


def build_program_token_tables():
  """Get mapping from ids to program tokens."""
  tokens = [
    'WALK',
    'LOOK',
    'RUN',
    'JUMP',
    'LTURN',
    'RTURN'
  ]

  # Build dictiontaries. Reserve 0, 1, 2 for padding, bos, eos.
  id_token_table = {id+3: token for id, token in enumerate(tokens)}
  id_token_table[1] = BOS
  id_token_table[2] = EOS
  token_id_table = {token: id for id, token in id_token_table.items()}
  return (id_token_table, token_id_table)
