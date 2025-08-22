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

"""Used to convert the PACIFIC dataset into a format for Vertex AI SFT.

Used to convert the dataset at https://github.com/dengyang17/PACIFIC/ to Gemini
1.0 Pro format for running SFT and ACT on Vertex AI.

Internally, the training code will convert this dataset into a format that
HuggingFace based `ACTTrainer` can use.
"""

import argparse
from collections import defaultdict
import json
from typing import Dict, List

from act.data.constants import (
    _MODEL_ROLE,
    _SYSTEM_ROLE,
    _USER_ROLE,
)
import numpy as np


def process_table_text(table):
  table_text = []
  for row in table:
    table_text.append("{} : ".format(row[0]) + " | ".join(row[1:]))
  return table_text


def process_paragraph_text(paras):
  paragraph_text = []
  for para in paras:
    text = para["text"].replace("\n", " ")
    paragraph_text.append(text)
  return paragraph_text


def assess_clarification(question):
  if question['req_clari']:
    return True
  return False


def process_conversation(conversation, user_role=_USER_ROLE,
                         model_role=_MODEL_ROLE):
  conver = []
  for question in conversation:
    user = {}
    user['content'] = question['question']
    user['role'] = user_role

    system = {}
    answer = str(question['answer'])
    if question['answer_type'] == 'arithmetic':
      derivation = question['derivation']
      if derivation:
        answer = derivation + " = " + answer
    system['content'] = answer
    system['role'] = model_role

    if assess_clarification(question):
      user['requires_clarification'] = True
    else:
      user['requires_clarification'] = False

    conver.append(user)
    conver.append(system)
  return conver


def get_initial_row(paragraph_text, table_text, system_role=_SYSTEM_ROLE):
  INITIAL_MESSAGE = "[context]\n{}\n[data]\n{}\n".format(
      '\n'.join(paragraph_text), '\n'.join(table_text)
  )
  initial_row = {'content': INITIAL_MESSAGE, 'role': system_role}
  return initial_row


def get_initial_system(paragraph_text, table_text, model_role=_MODEL_ROLE):
  return None


def process_example(example, args):
    para_text = process_paragraph_text(example['paragraphs'])
    table_text = process_table_text(example['table']['table'])

    initial_row = get_initial_row(para_text, table_text,
                                  system_role=args.system_role)

    sample = defaultdict(list)
    sample['messages'] = []

    initial_system = get_initial_system(para_text, table_text,
                                        model_role=args.model_role)
    sample['messages'].append(initial_row)
    if initial_system:
      sample['messages'].append(initial_system)
    row = process_conversation(example['questions'],
                               user_role=args.user_role,
                               model_role=args.model_role)
    sample['messages'].extend(row)
    return sample


def process_dataset(data, args):
  processed_dataset = []
  for dat in data:
    processed_dataset.append(process_example(dat, args))
  return processed_dataset


def parse_arguments():
  """Parse the arguments."""
  parser = argparse.ArgumentParser(description="Convert PACIFIC Dataset.")
  parser.add_argument(
      "--path",
      type=str,
      default=None,
      help="Path to the dataset.",
  )
  parser.add_argument(
      "--results_path",
      type=str,
      help="Where to write the results.",
  )
  parser.add_argument(
      "--user_role",
      type=str,
      default=_USER_ROLE,
      help="Default user role.",
  )
  parser.add_argument(
      "--system_role",
      type=str,
      default=_SYSTEM_ROLE,
      help="Default system role.",
  )
  parser.add_argument(
      "--model_role",
      type=str,
      default=_MODEL_ROLE,
      help="Default model role.",
  )
  args = parser.parse_args()
  return args


def main():
  args = parse_arguments()
  with open(args.path, "r") as f:
    data = json.load(f)
  processed_dataset = process_dataset(data, args)
  with open(args.results_path, "w") as f:
    for row in processed_dataset:
      json.dump(row, f)
      f.write("\n")


if __name__ == "__main__":
  main()
