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

"""Batch label commands for annotation."""

import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('--instruction_file', type=str, required=True)
parser.add_argument('--output_file', type=str, required=True)
parser.add_argument('--samples_to_label', type=int, default=180000)

args = parser.parse_args()


SAMPLES_PER_COMMAND = 5000


CMD_TEMPLATE = (
    'python query_openai_labelling.py --samples_to_label {} --instruction_file'
    ' {} --output_file {}'
)


num_commands = args.samples_to_label // SAMPLES_PER_COMMAND
if args.samples_to_label % SAMPLES_PER_COMMAND != 0:
  num_commands += 1
# in case some instructions are failed
num_commands += 1
for j in range(num_commands):
  print(
      f'currently labelling {args.instruction_file},'
      f' {j+1}/{num_commands} command'
  )
  cmd = CMD_TEMPLATE.format(
      SAMPLES_PER_COMMAND, args.instruction_file, args.output_file
  )
  os.system(cmd)
