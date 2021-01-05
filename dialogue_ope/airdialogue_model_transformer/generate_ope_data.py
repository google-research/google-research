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

# Adapted from https://github.com/facebookresearch/ParlAI/issues/2855

import json
import os
from copy import deepcopy
from tqdm import tqdm

from parlai.core.agents import create_agent, create_agent_from_model_file
from parlai.core.params import ParlaiParser
import parlai.utils.logging as logging

logging.disable()

SILENCE = '__SILENCE__'
PERSONA_PREFIX = 'your persona: '

ALL_SETTINGS = [
    '5K',
    '10K',
    '20K',
    '30K',
    '40K',
    '50K',
    '75K',
    '100K',
    '150K',
    '200K',
    '250K',
    'full',
    '5K_w',
    '10K_w',
    '20K_w',
    '30K_w',
    '40K_w',
    '50K_w',
    '75K_w',
    '100K_w',
    '150K_w',
    '200K_w',
    '250K_w',
    'full_w',
]

from flashtool import Logger
import sys


def setup_args():
  parser = ParlaiParser(True, True)
  parser.add_argument('--eval-dir', type=str, default='outputs/selfchat_eval/')
  parser.add_argument('--log-file', type=str, default='log.jsonl')
  parser.add_argument(
      '--save-dir', type=str, default='outputs/selfchat_ope_data/')
  parser.add_argument('--tgt-agent', type=str, default=None)
  return parser


def get_context(kb):
  # prepend kb
  reservation = kb['reservation']
  tickets = kb['kb']
  keys = [
      'price', 'num_connections', 'class', 'airline', 'departure_airport',
      'departure_month', 'departure_day', 'departure_time_num',
      'return_airport', 'return_month', 'return_day', 'return_time_num'
  ]
  ticket_text = []
  ticket_text.append('flight ' + 'None' + ' , ' + ' , '.join(
      [k.replace('num_', '').replace('_num', '') + ' ' + 'None' for k in keys]))
  for t in tickets:
    assert set(t.keys()) == set(keys + ['flight_number']), f'{t}'
    ticket_text.append('flight ' + str(t['flight_number']) + ' , ' +
                       ' , '.join([
                           k.replace('num_', '').replace('_num', '') + ' ' +
                           str(t[k]).strip() for k in keys
                       ]))
    # replace for consistency
  if reservation != 0:
    t = tickets[reservation - 1000]
    ticket_text.append('reservation ' + str(t['flight_number']) + ' , ' +
                       ' , '.join([
                           k.replace('num_', '').replace('_num', '') + ' ' +
                           str(t[k]).strip() for k in keys
                       ]))
  else:
    ticket_text.append('reservation ' + 'None' + ' , ' + ' , '.join([
        k.replace('num_', '').replace('_num', '') + ' ' + 'None' for k in keys
    ]))
  return ticket_text


def _run_conversation(conversation_id, conversation, tgt_agent, ref_agent):
  tgt_agent.reset()
  ref_agent.reset()

  # process context
  kb = conversation['kb']
  ticket_text = get_context(kb)

  tgt_agent_encoder_state = None

  new_dialog = []
  for turn_id, turn in enumerate(conversation['conversation']):
    speaker = turn['speaker']
    reference_text = turn['text']

    if speaker == 'customer':
      assert turn_id % 2 == 0
      act = {'id': 'customer', 'text': reference_text, 'episode_done': False}
      act['tickets'] = ticket_text[:-1]
      act['reservation'] = ticket_text[-1]
      act['return_encoder_state'] = True
      # the following is just padding
      act['action_name'] = 'none'
      act['action_flight'] = []
      act['action_intent'] = 'book'
      act['action_status'] = 'book'
      observed = ref_agent.observe(act)
      observed = tgt_agent.observe(act)
      new_dialog.append({'speaker': 'human_evaluator', 'text': turn['text']})
    if speaker == 'agent':
      assert turn_id % 2 == 1
      ref_response = ref_agent.batch_act([ref_agent.observation])[0]
      ref_agent.self_observe(ref_response)
      tgt_response = tgt_agent.batch_act([tgt_agent.observation])[0]
      tgt_agent.self_observe(deepcopy(ref_response))
      assert tgt_response['id'] == ref_response['id']

      if ref_response['text'] != reference_text:
        logging.error(
            f'{conversation_id}:{turn_id}: ref {repr(reference_text)} '
            f'!= resp {repr(ref_response["text"])}. Context:\n{repr(observed)}')
        import ipdb
        ipdb.set_trace()
        return False
      new_dialog.append({'speaker': 'model', 'text': turn['text']})
      new_dialog.append({'speaker': 'tgt_model', 'text': tgt_response['text']})
      tgt_agent_encoder_state = tgt_response['encoder_states']

      #else:
      #    logging.info(f'{conversation_id}:{turn_id} OK')
  conversation['dialog'] = new_dialog
  reward = tgt_agent.get_air_score(
      tgt_agent_encoder_state,
      conversation['expected_action'],
      kb,
  )
  conversation['reward'] = {
      'reward': reward['reward'],
      'name_score': reward['name_score'],
      'flight_score': reward['flight_score'],
      'status_score': reward['status_score']
  }
  conversation.pop('conversation')
  return True


def main():
  parser = setup_args()
  opt = parser.parse_args()
  if opt['tgt_agent'] is None:
    tgt_agent_list = ALL_SETTINGS
  else:
    tgt_agent_list = [opt['tgt_agent']]
  for tgt_agent in tgt_agent_list:
    save_dir = os.path.join(opt['save_dir'], tgt_agent)
    os.makedirs(save_dir, exist_ok=True)
    for ref_agent in ALL_SETTINGS:
      #if ref_agent == tgt_agent:
      #    continue
      print('Evaluating {} <-> {}'.format(tgt_agent, ref_agent))
      eval_single(opt, tgt_agent, ref_agent, save_dir)


def eval_single(opt, tgt_agent, ref_agent, save_dir):
  eval_file_path = opt['eval_dir'] + ref_agent + '/' + opt['log_file']
  save_file_path = os.path.join(save_dir, ref_agent + '.jsonl')

  model_mf = 'outputs/agent_' + tgt_agent + '/model'
  model_optf = 'outputs/agent_' + tgt_agent + '/model.opt'
  with open(model_optf) as f:
    model_opt = json.load(f)
  model_opt['interactive_mode'] = True
  tgt_agent = create_agent_from_model_file(model_mf, model_opt)

  model_mf = 'outputs/agent_' + ref_agent + '/model'
  model_optf = 'outputs/agent_' + ref_agent + '/model.opt'
  with open(model_optf) as f:
    model_opt = json.load(f)
  model_opt['interactive_mode'] = True
  ref_agent = create_agent_from_model_file(model_mf, model_opt)

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

        assert conversation['dialog'][-1]['speaker'] == 'tgt_model'
        assert len(conversation['dialog']) % 3 == 0
        conversation['reward_ref'] = conversation.pop('report')
        save_file.write(json.dumps(conversation) + '\n')
      else:
        errorids.append(i)
    print('Matched: {}/{}'.format(num_match, (num_match + len(errorids))))
    print('Error IDs: ', errorids)


if __name__ == '__main__':
  main()
