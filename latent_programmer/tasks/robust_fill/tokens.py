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

"""Create dictionaries for mapping program tokens to ids."""

from latent_programmer.tasks.robust_fill import dsl


def build_token_tables():
  """Get mapping from ids to program tokens."""
  tokens = [
      # Expressions
      dsl.Compose,
      dsl.ConstStr,
      dsl.SubStr,
      dsl.GetSpan,
      # Nestings
      dsl.GetToken,
      dsl.ToCase,
      dsl.Replace,
      dsl.Trim,
      dsl.GetUpto,
      dsl.GetFrom,
      dsl.GetFirst,
      dsl.GetAll,
      # New functions
      dsl.Substitute,
      dsl.SubstituteAll,
      dsl.Remove,
      dsl.RemoveAll,
  ]

  # Primitive types
  for character in list(dsl.CHARACTER):   # Includes delimiter.
    tokens.append(character)

  for t in dsl.Type:
    tokens.append(t)

  for case in dsl.Case:
    tokens.append(case)

  for boundary in dsl.Boundary:
    tokens.append(boundary)

  for k in range(dsl.POSITION[0], dsl.POSITION[1] + 1):  # Includes index.
    tokens.append(k)

  # Build dictiontaries. Reserve 0, 1, 2 for padding, bos, eos.
  id_token_table = {id+3: token for id, token in enumerate(tokens)}
  id_token_table[1] = dsl.BOS
  id_token_table[2] = dsl.EOS
  token_id_table = {token: id for id, token in id_token_table.items()}
  return (id_token_table, token_id_table)
