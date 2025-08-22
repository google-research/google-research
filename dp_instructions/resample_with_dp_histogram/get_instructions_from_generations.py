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

"""Get instructions from generations."""

import argparse
import os
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--generation_folder', type=str, required=True)
parser.add_argument('--output_path', type=str, required=True)

args = parser.parse_args()


def get_instructions_from_generations(fname):  # pylint: disable=redefined-outer-name
  """Get instructions from generations."""
  with open(fname, 'r') as f:
    lines = f.readlines()

  current_instruction = ''

  instructions = []  # pylint: disable=redefined-outer-name
  for i, line in enumerate(lines):
    if 'Generated:' not in line:
      current_instruction += line
    elif 'Generated:' in line and i != 0:
      current_instruction = current_instruction.strip()
      instructions.append(current_instruction)
      current_instruction = ''

  return instructions


def check_duplication(instructions):  # pylint: disable=redefined-outer-name
  res_dict = {}
  for i, ins in enumerate(instructions):
    if ins not in res_dict:
      res_dict[ins] = [i]
    else:
      res_dict[ins].append(i)
      print('duplicate:', res_dict[ins])


files = os.listdir(args.generation_folder)
files = [os.path.join(f, 'generations_gpu0.txt') for f in files]

if not files:
  raise ValueError('No files found in the generation folder.')

all_instructions = []


for fname in files:  # pylint: disable=redefined-outer-name
  instructions = get_instructions_from_generations(
      args.generation_folder + '/' + fname
  )

  all_instructions.extend(instructions)


def deduplicate(instructions):  # pylint: disable=redefined-outer-name
  res_dict = {}
  new_instructions = []
  for _, ins in enumerate(instructions):
    if ins not in res_dict:
      res_dict[ins] = 1
      new_instructions.append(ins)
    else:
      pass

  return new_instructions


all_instructions = deduplicate(all_instructions)

print(len(all_instructions))

assert args.output_path.endswith('.pkl')
pickle.dump(all_instructions, open(args.output_path, 'wb'))
