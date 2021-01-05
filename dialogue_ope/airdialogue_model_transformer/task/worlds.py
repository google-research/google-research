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

from copy import deepcopy
import json
import random
import os
import string
import numpy as np

from parlai.core.agents import create_agent
from parlai.core.message import Message
from parlai.core.worlds import DialogPartnerWorld, validate
from parlai.tasks.self_chat.worlds import SelfChatWorld as SelfChatBaseWorld
from parlai.utils.misc import warn_once
from .agents import _path, load_from_path


class InteractiveWorld(DialogPartnerWorld):
  """
    Interactive world for Airdialogue.

    Used for models trained on the task `-t wizard_of_wikipedia`. Automatically
    retrieves knowledge from Wikipedia based on the conversation history using a
    TF-IDF
    retriever. Then uses a Transformer-based model to select a checked sentence
    from
    these retrieved passages.
    """

  @staticmethod
  def add_cmdline_args(argparser):
    group = argparser.add_argument_group('Air Interactive World Args')
    group.add_argument(
        '--intent-choice',
        type=int,
        default=3,
        help='number of intent choices in dialogue (default: 3)')

  def __init__(self, opt, agents, shared=None):
    super().__init__(opt, agents, shared)
    print('[ loading airdialogue.. ]')
    self.opt = opt
    self.cnt = 0
    self.human_agent = self.agents[0]
    self.model_agent = self.agents[1]

    defaultagent = 'agent'
    defaultsize = -1
    task = opt.get('task', f'airdialogue:{defaultagent}:{defaultsize}')
    self.agenttype = task.split(':')[1] if len(
        task.split(':')) > 1 else defaultagent
    self.datasize = int(
        task.split(':')[2]) if len(task.split(':')) > 2 else defaultsize

    if shared is not None:
      self.messages = shared['messages']
      self.actions = shared['actions']
      self.expected_actions = shared['expected_actions']
      self.num_ex = shared['num_ex']
      self.intents = shared['intents']
      self.intent_objs = shared['intent_objs']
      self.kbs = shared['kbs']
    else:
      self.messages = []
      self.actions = []
      self.expected_actions = []
      self.intents = []
      self.intent_objs = []
      self.kbs = []
      self.num_ex = 0
      jsons_path = _path(opt)
      self._setup_data(jsons_path)

    self.num_intent_choice = opt.get('intent_choice', 3)

  def _setup_data(self, jsons_path):
    data_path = os.path.join(jsons_path, 'dev_data.json')
    kb_path = os.path.join(jsons_path, 'dev_kb.json')
    size = self.datasize
    load_from_path(
        self,
        data_path,
        kb_path,
        size,
        load_intent=True,
        load_kb=True,
        load_expected_action=True)

  def _get_new_intent(self):
    random.seed()
    intent_ids = random.sample(
        range(len(self.intents)), self.num_intent_choice - 1)
    intents = [self.intents[i] for i in intent_ids]
    intents.append('[OTHER INTENT]')
    letters = list(string.ascii_uppercase)[:self.num_intent_choice]
    intent_list = {x: y for x, y in zip(letters, intents)}
    intent_text = '\n'.join(
        ['{}: {}'.format(k, v) for k, v in intent_list.items()])
    intent_id_list = {x: y for x, y in zip(letters[:-1], intent_ids)}

    done = False
    while not done:
      self.human_agent.observe({
          'text':
              'Your role is {}\nPlease choose one of the following intents by typing '
              'A, B, C, ..., etc. : \n\n{}\n'.format(self.agenttype,
                                                     intent_text)
      })
      intent_act = self.human_agent.act()
      choice = intent_act['text'][0].upper()
      if choice in intent_list:
        if intent_list[choice] == '[OTHER INTENT]':
          intent_ids = random.sample(
              range(len(self.intents)), self.num_intent_choice - 1)
          intents = [self.intents[i] for i in intent_ids]
          intents.append('[OTHER INTENT]')
          letters = list(string.ascii_uppercase)[:self.num_intent_choice]
          intent_list = {x: y for x, y in zip(letters, intents)}
          intent_text = '\n'.join(
              ['{}: {}'.format(k, v) for k, v in intent_list.items()])
          intent_id_list = {x: y for x, y in zip(letters[:-1], intent_ids)}
        else:
          done = True
      else:
        self.human_agent.observe(
            {'text': 'Invalid response, please try again.'})

    self.human_agent.observe(
        {'text': f'[Your chosen intent is: {intent_list[choice]}]'})
    chosen_id = intent_id_list[choice]
    expected_action = self.expected_actions[chosen_id]
    self.human_agent.observe(
        {'text': f'[expected action is: {expected_action}]'})
    for flight in expected_action['flight']:
      expected_flight = flight - 1000
      # import ipdb; ipdb.set_trace()
      expected_flight = self.kbs[chosen_id]['kb'][expected_flight]
      self.human_agent.observe(
          {'text': f'[expected flight is: {expected_flight}]'})
    if len(expected_action['flight']) == 0:
      self.human_agent.observe({'text': f'[expected flight is: None]'})
    reservation = self.kbs[chosen_id]['reservation']
    self.human_agent.observe(
        {'text': f'[reservation flight is: {reservation}]'})

    return chosen_id

  def _add_context(self, action):
    entrys = self.messages[self.context_id][0].split('\n')
    entrys[-1] = action['text']

    if self.agenttype == 'agent':
      action['tickets'] = entrys[:-2]
      action['reservation'] = entrys[-2]
      # the following are actually not used in eval just for calculate loss
      # need to remove in the future
      action['action_name'] = self.actions[self.context_id]['name']
      action['action_flight'] = self.actions[self.context_id]['flight']
      action['action_status'] = self.actions[self.context_id]['status']
      action['action_intent'] = self.actions[self.context_id]['intent']
    elif self.agenttype == 'customer':
      action['intent'] = entrys[0]
      assert len(entrys) == 2
    action['return_encoder_state'] = True
    return action

  def reset(self):
    super().reset()
    self.cnt = 0
    self.context_id = None
    self.model_agent.reset()
    self.human_agent.reset()
    self.acts = [None, None]

  def get_air_score(self):
    score_obj = self.model_agent.get_air_score(
        self.acts[1]['encoder_states'], self.expected_actions[self.context_id],
        self.kbs[self.context_id])
    score_text = '\n'.join([f" - {k}: {v}" for k, v in score_obj.items()])
    for flight in score_obj['flight']:
      chosen_flight = self.kbs[self.context_id]['kb'][flight - 1000]
      score_text += f'\nChosen Flight: {chosen_flight}'
    self.human_agent.observe({
        'id': 'Final Agent Prediction',
        'text': '\n' + score_text
    })
    return score_obj

  def parley(self):
    """
        Loop between model and human.
        """

    if self.cnt == 0:
      self.context_id = self._get_new_intent()
      self.acts = [None, None]
      self.human_first = random.choice([0, 1])

    # possibly get human act first
    if self.cnt == 0 and not self.human_first:
      self.acts[0] = Message({'text': '__SILENCE__', 'episode_done': False})
    else:
      try:
        self.acts[0] = self.human_agent.act()
      except StopIteration:
        if self.agenttype != 'customer':
          self.get_air_score()
        print('[ CHAT DONE ]')
        print('\n[ Preparing new chat... ]\n')
        self.reset()
        return

    act = deepcopy(self.acts[0])

    # add context to the model observation
    act = self._add_context(act)

    # model observes context and human (apprentice) act
    self.model_agent.observe(validate(act))

    # model agent act
    self.acts[1] = self.model_agent.act()

    # human (apprentice) agent observes model act
    # remove encoder_states to prevent output
    act = deepcopy(self.acts[1])
    if 'encoder_states' in act:
      del act['encoder_states']
    self.human_agent.observe(validate(act))

    self.update_counters()
    self.cnt += 1

    if self.episode_done():
      print('[ CHAT DONE ]')
      print('\n[ Preparing new chat... ]\n')
      self.cnt = 0
      self.model_agent.reset()


class InteractiveCustomerWorld(InteractiveWorld):
  pass


class InteractiveAgentWorld(InteractiveWorld):
  pass


class SelfChatBothWorld(InteractiveWorld):

  def __init__(self, opt, agents, shared=None):
    super().__init__(opt, agents, shared)
    assert self.agenttype == 'both', 'agenttype must be both for selfplay'

    if opt['model_file'].split(':')[0] == 'human':
      print('[Human Evaluation]')
      self.human_eval = True
    else:
      self.human_eval = False

    self.customer_agent = self.agents[0]
    self.agent_agent = self.agents[1]

    self.max_turn_cnt = self.opt.get('selfchat_max_turns', 10)
    self.episode_cnt = 0
    self.agent_encoder_states = None

    self.score = None
    self.gather_rewards = {
        'reward': [],
        'flight_score': [],
        'name_score': [],
        'status_score': [],
    }
    self.start_cid = self.opt.get('start_cid', 0)

  @staticmethod
  def add_cmdline_args(argparser):
    group = argparser.add_argument_group('Air SelfChat World Args')
    group.add_argument(
        '--start-cid', type=int, default=0, help='offset of contextid')

  def display(self):
    s = super().display()
    if self.cnt == 0:
      s += '\n==============================\n'
    return s

  def _add_context(self, action, agenttype):
    entrys = self.messages[self.context_id][0].split('\n')
    entrys[-1] = action['text']

    if agenttype == 'agent':
      action['tickets'] = entrys[:-3]
      action['reservation'] = entrys[-3]
      # the following are actually not used in eval just for calculate loss
      # need to remove in the future
      action['action_name'] = self.actions[self.context_id]['name']
      action['action_flight'] = self.actions[self.context_id]['flight']
      action['action_status'] = self.actions[self.context_id]['status']
      action['action_intent'] = self.actions[self.context_id]['intent']
    elif agenttype == 'customer':
      action['intent'] = entrys[-2]
    return action

  def episode_done(self):
    # add a heuristic for episode_done
    # this one will break the current parlai selfplay script
    if self.acts[0] is not None and self.acts[1] is not None:
      if 'thank you' in self.acts[0]['text'].lower(
      ) and 'thank you' in self.acts[1]['text'].lower():
        return True
      if 'have a nice day' in self.acts[0]['text'].lower(
      ) or 'have a nice day' in self.acts[1]['text'].lower():
        return True
      if 'thank you' in self.acts[0]['text'].lower(
      ) and 'welcome' in self.acts[1]['text'].lower():
        return True
      if 'welcome' in self.acts[0]['text'].lower(
      ) and 'thank you' in self.acts[1]['text'].lower():
        return True
      if self.human_done:
        return True
    return self.cnt >= self.max_turn_cnt

  def get_air_score(self):
    score_obj = self.model_agent.get_air_score(
        self.agent_encoder_states, self.expected_actions[self.context_id],
        self.kbs[self.context_id])
    score_text = '\n'.join([f" - {k}: {v}" for k, v in score_obj.items()])
    for flight in score_obj['flight']:
      chosen_flight = self.kbs[self.context_id]['kb'][flight - 1000]
      score_text += f'\nChosen Flight: {chosen_flight}'
    return score_obj, score_text

  def write(self, logger, reports, outdir):
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, 'log.jsonl')
    conversations = logger.get_logs()
    # dont really how it works
    # hack to remove empty logs
    conversations = [i for i in conversations if len(i) > 0]

    def format_conv(conv):
      new_conv = []
      for i in conv:
        new_conv.append({'speaker': 'customer', 'text': i[0]['text']})
        new_conv.append({'speaker': 'agent', 'text': i[1]['text']})
      return new_conv

    if len(conversations) != len(reports):
      print('WARNING! length difference')
      import ipdb
      ipdb.set_trace()
    with open(outfile, 'w') as fout:
      #import ipdb; ipdb.set_trace()
      for conv, re in zip(conversations, reports):
        r = {}
        r['conversation'] = format_conv(conv)
        r['report'] = re
        context_id = re['id']
        r['expected_action'] = self.expected_actions[context_id]
        r['intent'] = self.intent_objs[context_id]
        r['kb'] = self.kbs[context_id]
        fout.write(json.dumps(r) + '\n')

  def report(self):
    for k, v in self.gather_rewards.items():
      v.append(self.score[k])
      v = np.array(v).mean()
      print(f"Gather {k} : {v}")
    r = deepcopy(self.score)
    r['id'] = self.context_id
    return r

  def reset(self):
    #self.reset()
    self.customer_agent.reset()
    self.agent_agent.reset()
    self.episode_cnt += 1
    self.cnt = 0
    self.acts = [None, None]

  def customer_obs(self, act):
    _act = act
    self.predefine_acts = []
    if self.human_eval:
      _act = {}
      _act['text'] = act['text']
      _act['id'] = act['id']
      if self.cnt == 0:
        _act['intent'] = act['intent']

      # define some template reponses to ease human eval
      intent = self.intent_objs[self.context_id]
      if self.cnt == 0:
        print(intent)
        if intent['goal'] == 'book':
          self.predefine_acts.append('Hi, I want to book a ticket.')
        else:
          self.predefine_acts.append(
              f"Hi, I want to {intent['goal']} a reservation.")
      else:
        self.predefine_acts.append(f"My name is {intent['name']}")
        if intent['goal'] in ['book', 'change']:
          self.predefine_acts.append(
              f"My origin is {intent['departure_airport']} and destination is {intent['return_airport']}."
          )

          # Add dates
          MONTH_DICT = {
              'Jan': '01',
              'Feb': '02',
              'Mar': '03',
              'Apr': '04',
              'May': '05',
              'Jun': '06',
              'Jul': '07',
              'Aug': '08',
              'Sep': '09',
              'Oct': '10',
              'Nov': '11',
              'Dec': '12'
          }
          m1 = MONTH_DICT[intent['departure_month'][:3]]
          m2 = MONTH_DICT[intent['return_month'][:3]]
          d1 = m1 + '/' + intent['departure_day']
          if 'departure_time' in intent and intent['departure_time'] != 'None':
            d1 += ' ' + intent['departure_time']
          d2 = m2 + '/' + intent['return_day']
          if 'return_time' in intent and intent['return_time'] != 'None':
            d2 += ' ' + intent['return_time']
          self.predefine_acts.append(f"Start on {d1} and return on {d2}.")

          # Add specification
          spec = ''
          if 'max_connections' in intent:
            spec += f"The connection limit is {intent['max_connections']} . "
          if 'max_price' in intent:
            spec += f"The price limit is {intent['max_price']} . "
          pref = []
          if 'class' in intent and intent['class'] != 'None':
            pref.append(f"{intent['class']} class")
          if 'airline' in intent:
            pref.append(f"{intent['airline']} airline")
          if len(pref) == 1:
            spec += f"And I prefer {pref[0]} ."
          elif len(pref) == 1:
            spec += f"And I prefer {pref[0]} and {pref[1]} ."

          self.predefine_acts.append(spec)

        self.predefine_acts.extend(
            ['Yes.', 'Ok.', 'Thank you.', "That's fine, thank you."])
        if 'sorry' in _act['text'] or 'no reservation' in _act['text']:
          # say that's fine
          self.predefine_acts = [self.predefine_acts[-1]
                                ] + self.predefine_acts[:-1]
        elif 'airport' in _act['text'] or 'scource' in _act['text'] or 'destination' in _act['text'] \
            or 'details' in _act['text'] or 'codes' in _act['text']:
          # say airport
          self.predefine_acts = [
              self.predefine_acts[1]
          ] + self.predefine_acts[0:1] + self.predefine_acts[2:]
        elif 'dates' in _act['text']:
          # say dates
          self.predefine_acts = [
              self.predefine_acts[2]
          ] + self.predefine_acts[0:2] + self.predefine_acts[3:]
        elif 'proceed for booking' in _act['text'] or 'shall' in _act['text'] or 'are you ok with' in _act['text'] \
            or 'would you like' in _act['text'] or 'can i' in _act['text']:
          # say yes
          self.predefine_acts = [
              self.predefine_acts[-4]
          ] + self.predefine_acts[:-4] + self.predefine_acts[-3:]
        elif 'wait' in _act['text']:
          # say ok
          self.predefine_acts = [
              self.predefine_acts[-3]
          ] + self.predefine_acts[:-3] + self.predefine_acts[-2:]
        elif 'booked' in _act['text'] or 'has been' in _act['text'] or \
             'is done' in _act['text'] or 'is confirmed' in _act['text']:
          # say thank you
          self.predefine_acts = [
              self.predefine_acts[-2]
          ] + self.predefine_acts[:-2] + self.predefine_acts[-1:]
      try:
        if self.customer_agent.ref_data is not None:
          ref_text = self.customer_agent.ref_data[self.context_id][self.cnt * 2
                                                                   + 2]['text']
          self.predefine_acts = [ref_text] + self.predefine_acts
      except:
        pass

      for i, t in enumerate(self.predefine_acts):
        _act[f"Act -{i}"] = t

    self.customer_agent.observe(validate(_act))

  def customer_act(self):
    if not self.human_eval or len(self.predefine_acts) == 0:
      return self.customer_agent.act()
    else:
      act = self.customer_agent.act()
      text = act['text']
      if len(text) == 2 and text[0] == '-' and text[1:].isdigit():
        text = text[1:]
        if int(text) < len(self.predefine_acts):
          act.force_set('text', self.predefine_acts[int(text)])
          act.force_set('id', 'customer')
          print(act['text'])
      if 'thank you' in act['text'].lower():
        self.human_done = True
      return act

  def parley(self):
    """
        Loop between model and human.
        """
    self.human_done = False
    if self.cnt == 0:
      self.context_id = self.episode_cnt + self.start_cid
      self.acts = [None, None]
      self.agent_first = False

    # possibly get customer act first
    if self.cnt == 0 and not self.agent_first:
      self.acts[0] = Message({
          'id': 'customer',
          'text': '__SILENCE__',
          'episode_done': False
      })
    else:
      if self.cnt == 0:
        preact = Message({'text': '__SILENCE__', 'episode_done': False})
        preact = self._add_context(preact, 'customer')
        self.customer_obs(preact)
      act = self.customer_act()
      self.acts[0] = act

    # add context to the model observation
    act = deepcopy(self.acts[0])
    act = self._add_context(act, 'agent')
    act['return_encoder_state'] = True

    # agent observes context and human (apprentice) act
    self.agent_agent.observe(validate(act))

    # agent agent act
    act = self.agent_agent.act()
    self.agent_encoder_states = act.pop('encoder_states')
    self.acts[1] = act

    # customer agent observes model act
    act = deepcopy(self.acts[1])
    act = self._add_context(act, 'customer')
    self.customer_obs(act)

    self.update_counters()
    self.cnt += 1

    if self.episode_done():
      score_obj, score_text = self.get_air_score()
      self.score = score_obj
      #print(score_text)
      return True

    return False
