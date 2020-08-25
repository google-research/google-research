# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

import numpy
import json
import argparse
import os, sys
import copy

from airdialogue.prepro.tokenize_lib import tokenize_kb
from airdialogue.evaluator.selfplay_utils import compute_reward
from airdialogue.evaluator.evaluator_main import action_obj_to_str


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--ref_file', required=True, type=str, help='path to reference file')
  parser.add_argument(
      '--ref_kb', required=True, type=str, help='path to reference kb file')
  parser.add_argument(
      '--output_dir', required=True, type=str, help='path to output dir')
  args = parser.parse_args()
  print(args)

  ref_data = []
  ref_score = 0
  with open(args.ref_file, 'r') as ref_file:
    with open(args.ref_kb, 'r') as kb_file:
      for l, kb_line in zip(ref_file, kb_file):
        _json = json.loads(l)
        kb = tokenize_kb(json.loads(kb_line))
        action = action_obj_to_str(_json['action'])
        expected_action = action_obj_to_str(_json['expected_action'])
        if 'reward' not in _json:
          score = compute_reward(action, expected_action, kb)
          _json['reward'] = score[0]
        ref_score += _json['reward']
        ref_data.append(_json)
  print('# of reference dialogue: ', len(ref_data))
  print('avg ref reward: ', ref_score / len(ref_data))

  result_data = []
  for s in ref_data:
    dia = s['dialogue']
    new_s = copy.deepcopy(s)
    customer_lines = len([l for l in dia if l.startswith('customer: ')])
    agent_lines = len([l for l in dia if l.startswith('agent: ')])
    gen_lines = len([l for l in dia if l.startswith('agent_tgt: ')])
    assert agent_lines + customer_lines + gen_lines == len(dia)
    if not dia[0].startswith('customer: '):
      new_s['dialogue'] = ['customer: '] + new_s['dialogue']
    if dia[-1].startswith('customer: '):
      new_s['dialogue'] = new_s['dialogue'][:-1]
    assert len(new_s['dialogue']) % 3 == 0
    new_s['ref_customer_response'] = [
        l.replace('customer:', '').lstrip().rstrip()
        for l in new_s['dialogue'][::3]
    ]
    new_s['ref_agent_response'] = [
        l.replace('agent:', '').lstrip().rstrip()
        for l in new_s['dialogue'][1::3]
    ]
    new_s['gen_agent_response'] = [
        l.replace('agent_tgt:', '').lstrip().rstrip()
        for l in new_s['dialogue'][2::3]
    ]
    # print(len(new_s['ref_agent_response']))

    result_data.append(new_s)
  os.makedirs(args.output_dir, exist_ok=True)
  print('saving to :', args.output_dir)
  with open(os.path.join(args.output_dir, 'data.json'), 'w') as save_file:
    for i, r in enumerate(result_data):
      save_file.write(json.dumps(r))
      save_file.write('\n')


if __name__ == '__main__':
  main()
