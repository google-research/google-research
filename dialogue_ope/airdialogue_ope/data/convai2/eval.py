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

import json
import os
from copy import deepcopy
from tqdm import tqdm

from parlai.core.agents import create_agent
from parlai.scripts.script import ParlaiScript
from parlai.core.params import ParlaiParser
from parlai.utils.strings import normalize_reply
import parlai.utils.logging as logging

logging.disable()

SILENCE = '__SILENCE__'
PERSONA_PREFIX = 'your persona: '

import model_configs

from flashtool import Logger
import sys


def setup_args():
  parser = ParlaiParser(True, True)
  parser.add_argument(
      '--eval-dir', type=str, default='evaluation_logs_reproducible')
  parser.add_argument('--save-dir', type=str, default='cleaned_logs')
  return parser


def _run_conversation(conversation_id, conversation, tgt_agent, ref_agent):
  tgt_agent.reset()
  ref_agent.reset()
  model_persona = conversation['model_persona']
  model_persona = PERSONA_PREFIX + model_persona.replace(
      '\n', '\n' + PERSONA_PREFIX)

  new_dialog = []
  for turn_id, turn in enumerate(conversation['dialog']):
    speaker = turn['speaker']
    reference_text = turn['text']

    if turn_id == 0 and speaker == 'model':
      silenced_text = model_persona + '\n' + SILENCE
      observed = ref_agent.observe({
          'id': 'SPEAKER_2',
          'text': silenced_text,
          'episode_done': False
      })
      observed = tgt_agent.observe({
          'id': 'SPEAKER_2',
          'text': silenced_text,
          'episode_done': False
      })
      new_dialog.append({'speaker': 'human_evaluator', 'text': SILENCE})
    elif turn_id == 0 and speaker == 'human_evaluator':
      reference_text = model_persona + '\n' + reference_text

    if speaker == 'human_evaluator':
      observed = ref_agent.observe({
          'id': 'SPEAKER_2',
          'text': reference_text,
          'episode_done': False
      })
      observed = tgt_agent.observe({
          'id': 'SPEAKER_2',
          'text': reference_text,
          'episode_done': False
      })
      new_dialog.append({'speaker': 'human_evaluator', 'text': turn['text']})
    if speaker == 'model':
      ref_response = ref_agent.batch_act([ref_agent.observation])[0]
      ref_agent.self_observe(ref_response)
      tgt_response = tgt_agent.batch_act([tgt_agent.observation])[0]
      tgt_agent.self_observe(deepcopy(ref_response))
      assert tgt_response['id'] == ref_response['id']

      response_normalized = normalize_reply(ref_response['text'])
      if response_normalized != reference_text:
        logging.error(
            f'{conversation_id}:{turn_id}: ref {repr(reference_text)} '
            f'!= resp {repr(response_normalized)}. Context:\n{repr(observed)}')
        return False
      response_normalized = normalize_reply(tgt_response['text'])
      new_dialog.append({'speaker': 'model', 'text': turn['text']})
      new_dialog.append({'speaker': 'tgt_model', 'text': response_normalized})

      #else:
      #    logging.info(f'{conversation_id}:{turn_id} OK')
  conversation['dialog'] = new_dialog
  return True


def main():
  parser = setup_args()
  opt = parser.parse_args()
  for tgt_agent in model_configs.ALL_SETTINGS:
    save_dir = os.path.join(opt['save_dir'], tgt_agent)
    os.makedirs(save_dir, exist_ok=True)
    for ref_agent in model_configs.ALL_SETTINGS:
      #if ref_agent == tgt_agent:
      #    continue
      print('Evaluating {} <-> {}'.format(tgt_agent, ref_agent))
      eval_single(opt, tgt_agent, ref_agent, save_dir)


def eval_single(opt, tgt_agent, ref_agent, save_dir):
  eval_file_path = os.path.join(opt['eval_dir'], ref_agent + '.jsonl')
  save_file_path = os.path.join(save_dir, ref_agent + '.jsonl')

  model_config = deepcopy(opt)
  for k, v in getattr(model_configs, tgt_agent).items():
    model_config[k] = v
    model_config['override'][k] = v
  tgt_agent = create_agent(model_config, True)
  model_config = deepcopy(opt)
  for k, v in getattr(model_configs, ref_agent).items():
    model_config[k] = v
    model_config['override'][k] = v
  ref_agent = create_agent(model_config, True)

  with open(eval_file_path) as eval_file, open(save_file_path,
                                               'w') as save_file:
    num_match = 0
    errorids = []
    for i, line in tqdm(enumerate(eval_file)):
      if not line.strip():
        continue
      conversation = json.loads(line)
      if _run_conversation(i, conversation, tgt_agent, ref_agent):
        num_match += 1

        if conversation['dialog'][-1]['speaker'] != 'tgt_model':
          conversation['dialog'].pop()
        assert len(conversation['dialog']) % 3 == 0
        conversation['reward'] = conversation.pop('evaluation_results')
        # max score is 4
        for k in conversation['reward'].keys():
          conversation['reward'][k] /= 4.0
        conversation['reward']['reward'] = sum([
            r for k, r in conversation['reward'].items()
        ]) / len(conversation['reward'])
        save_file.write(json.dumps(conversation) + '\n')
      else:
        errorids.append(i)
    print('Matched: {}/{}'.format(num_match, (num_match + len(errorids))))
    print('Error IDs: ', errorids)


if __name__ == '__main__':
  main()
