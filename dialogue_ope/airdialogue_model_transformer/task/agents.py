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

from parlai.core.teachers import FixedDialogTeacher
from parlai.core.message import Message
from .build import build
import os
import json
import tqdm


def _path(opt):
  # build the data if it does not exist
  build(opt)

  # set up path to data (specific to each dataset)
  jsons_path = os.path.join(opt['datapath'], 'airdialogue_data', 'airdialogue')
  return jsons_path


def normalize_action(example):
  example['action']['intent'] = example['intent']['goal']
  if example['action']['intent'] == 'change' \
          and example['action']['status'] == 'change':
    example['action']['status'] = 'book'
  return example


def is_sample_valid(example, kb):
  return example['correct_sample']


def load_from_path(agent,
                   data_path,
                   kb_path,
                   size,
                   load_intent=False,
                   load_kb=False,
                   load_expected_action=False):
  """
    intent and kb have been already loaded in dialogue
    if you want to additionaly load intent or kb set load_intent/load_kb to true
    """
  num_tobeload = size
  with open(data_path) as f_data, open(kb_path) as f_kb:
    for line_data, line_kb in tqdm.tqdm(zip(f_data, f_kb)):
      # import ipdb; ipdb.set_trace()
      example = json.loads(line_data)
      kb = json.loads(line_kb)
      if not is_sample_valid(example, kb):
        continue
      example = normalize_action(example)
      if load_kb:
        agent.kbs.append(kb)
      if load_intent:
        agent.intents.append(' | '.join(
            [f"{k} : {v}" for k, v in example['intent'].items()]))
        agent.intent_objs.append(example['intent'])
      if load_expected_action:
        agent.expected_actions.append(example['expected_action'])

      if example['dialogue'][0].startswith(agent.agenttype):
        example['dialogue'] = ['__SILENCE__'] + example['dialogue']

      num_turns = len(example['dialogue']) // 2
      example['dialogue'] = example['dialogue'][:2 * num_turns]
      example['dialogue'] = [s.split(': ')[-1] for s in example['dialogue']]
      agent.num_ex += num_turns

      if agent.agenttype in ['customer', 'both']:
        # prepend intent
        keys = [
            'goal', 'name', 'max_price', 'max_connections', 'class',
            'airline_preference', 'departure_airport', 'departure_month',
            'departure_day', 'departure_time', 'return_airport', 'return_month',
            'return_day', 'return_time'
        ]
        intent = example['intent']
        for k in keys:
          if k not in intent:
            intent[k] = 'None'
        assert set(intent.keys()) == set(keys), f'{intent}'
        example['dialogue'][0] = ' , '.join([k + ' ' + str(intent[k]).strip() for k in keys]) \
            + '\n' + example['dialogue'][0]
      if agent.agenttype in ['agent', 'both']:
        # prepend kb
        reservation = kb['reservation']
        tickets = kb['kb']
        keys = [
            'price', 'num_connections', 'class', 'airline', 'departure_airport',
            'departure_month', 'departure_day', 'departure_time_num',
            'return_airport', 'return_month', 'return_day', 'return_time_num'
        ]
        ticket_text = []
        ticket_text.append('flight ' + 'None' + ' , ' + ' , '.join([
            k.replace('num_', '').replace('_num', '') + ' ' + 'None'
            for k in keys
        ]))
        for t in tickets:
          assert set(t.keys()) == set(keys + ['flight_number']), f'{t}'
          ticket_text.append('flight ' + str(t['flight_number']) + ' , ' +
                             ' , '.join([
                                 k.replace('num_', '').replace('_num', '') +
                                 ' ' + str(t[k]).strip() for k in keys
                             ]))
          # replace for consistency
        if reservation != 0:
          t = tickets[reservation - 1000]
          ticket_text.append('reservation ' + str(t['flight_number']) + ' , ' +
                             ' , '.join([
                                 k.replace('num_', '').replace('_num', '') +
                                 ' ' + str(t[k]).strip() for k in keys
                             ]))
        else:
          ticket_text.append('reservation ' + 'None' + ' , ' + ' , '.join([
              k.replace('num_', '').replace('_num', '') + ' ' + 'None'
              for k in keys
          ]))
        example['dialogue'][0] = '\n'.join(list(ticket_text) +
                                           [example['dialogue'][0]])

      agent.messages.append(example['dialogue'])
      agent.actions.append(example['action'])

      num_tobeload -= 1
      if num_tobeload == 0:
        break


class AirDialogueTeacher(FixedDialogTeacher):
  """
    AirDialogue Teacher.

    This also contains other files (dev_kb.json, train_kb.json) with flight data
    about
    return flights, price, connections, flight airlines, departure times, and
    other
    flight information. More information and related paper can be found at
    <https://github.com/google/airdialogue>.
    """

  @classmethod
  def add_cmdline_args(cls, argparser):
    #super(AirDialogueTeacher, cls).add_cmdline_args(argparser)
    group = argparser.add_argument_group('Air Dialogue Teacher')
    group.add_argument(
        '--valid-size',
        type=int,
        default=-1,
        help='size of validation set (default: -1 , use all valid data)')

  def __init__(self, opt, shared=None):
    super().__init__(opt, shared)
    jsons_path = _path(opt)
    self.datatype = opt['datatype'].split(':')[0]
    defaultagent = 'agent'
    defaultsize = -1
    task = opt.get('task', f'airdialogue:{defaultagent}:{defaultsize}')
    self.agenttype = task.split(':')[1] if len(
        task.split(':')) > 1 else defaultagent
    self.datasize = int(
        task.split(':')[2]) if len(task.split(':')) > 2 else defaultsize
    self.validsize = opt.get('valid_size')
    assert self.agenttype in ['agent', 'customer'
                             ], 'datatype: {}'.format(opt['datatype'])

    if shared is not None:
      self.messages = shared['messages']
      self.actions = shared['actions']
      self.num_ex = shared['num_ex']
    else:
      self.messages = []
      self.actions = []
      self.num_ex = 0
      self._setup_data(jsons_path)
    # self._setup_data(jsons_path)
    self.id = 'airdialogue'
    self.reset()

  def _setup_data(self, jsons_path):
    if self.datatype.startswith('valid'):
      data_path = os.path.join(jsons_path, 'dev_data.json')
      kb_path = os.path.join(jsons_path, 'dev_kb.json')
      size = self.validsize
    else:
      data_path = os.path.join(jsons_path, 'train_data.json')
      kb_path = os.path.join(jsons_path, 'train_kb.json')
      size = self.datasize
    load_from_path(self, data_path, kb_path, size)

  def share(self):
    shared = super().share()
    shared['messages'] = self.messages
    shared['actions'] = self.actions
    shared['num_ex'] = self.num_ex
    return shared

  def num_examples(self):
    return self.num_ex

  def num_episodes(self):
    return len(self.messages)

  def get(self, episode_idx, entry_idx=0):
    log_idx = entry_idx * 2
    entry = self.messages[episode_idx][log_idx]
    last_backnforth_idx = len(self.messages[episode_idx]) - 2
    episode_done = log_idx >= last_backnforth_idx
    label_text = self.messages[episode_idx][log_idx + 1]
    labels = [label_text]
    action = {
        'id': self.id,
        'text': entry,
        'episode_done': episode_done,
        'labels': labels,
    }
    if entry_idx == 0:
      entrys = action['text'].split('\n')
    else:
      entrys = self.messages[episode_idx][0].split('\n')
      entrys[-1] = entry

    if self.agenttype == 'agent':
      action['tickets'] = entrys[:-2]
      action['reservation'] = entrys[-2]
      action['text'] = entrys[-1]
      action['action_name'] = self.actions[episode_idx]['name']
      action['action_flight'] = self.actions[episode_idx]['flight']
      action['action_status'] = self.actions[episode_idx]['status']
      action['action_intent'] = self.actions[episode_idx]['intent']
    elif self.agenttype == 'customer':
      action['intent'] = entrys[0]
      action['text'] = entrys[1]
      assert len(entrys) == 2
    return Message(action)


class DefaultTeacher(AirDialogueTeacher):
  pass


class AgentTeacher(AirDialogueTeacher):
  pass


class CustomerTeacher(AirDialogueTeacher):
  pass


class BothTeacher(AirDialogueTeacher):
  pass
