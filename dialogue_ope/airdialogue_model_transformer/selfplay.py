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

"""Allows a model to self-chat on a given task.
"""
from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent, create_agent_from_model_file
from parlai.agents.local_human.local_human import LocalHumanAgent
from parlai.core.worlds import create_task
from parlai.utils.world_logging import WorldLogger
from parlai.utils.misc import TimeLogger
from parlai.core.script import ParlaiScript, register_script
import parlai.utils.logging as logging

import math
import json
import random


def setup_args(parser=None):
  if parser is None:
    parser = ParlaiParser(True, True, 'Generate self-chats of a model')
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('-d', '--display-examples', type='bool', default=True)
  parser.add_argument(
      '--display-ignore-fields',
      type=str,
      default='label_candidates,text_candidates',
      help='Do not display these fields',
  )
  parser.add_argument(
      '-st',
      '--selfchat-task',
      type='bool',
      default=True,
      help='Create a self chat version of the task',
  )
  parser.add_argument(
      '--num-self-chats',
      type=int,
      default=1,
      help='Number of self chats to run')
  parser.add_argument(
      '--selfchat-max-turns',
      type=int,
      default=6,
      help='The number of dialogue turns before self chat ends',
  )
  parser.add_argument(
      '--seed-messages-from-task',
      action='store_true',
      help='Automatically seed conversation with messages from task dataset.',
  )
  parser.add_argument(
      '--outfile', type=str, default=None, help='File to save self chat logs')
  parser.add_argument(
      '--save-format',
      type=str,
      default='conversations',
      choices=['conversations', 'parlai'],
      help='Format to save logs in. conversations is a jsonl format, parlai is a text format.',
  )
  parser.add_argument(
      '-pmf',
      '--partner-model-file',
      default=None,
      help='Define a different partner for self chat',
  )
  parser.add_argument(
      '--partner-opt-file',
      default=None,
      help='Path to file containing opts to override for partner',
  )
  parser.set_defaults(interactive_mode=True, task='self_chat')
  WorldLogger.add_cmdline_args(parser)
  return parser


def _run_self_chat_episode(opt, world, world_logger):
  bsz = opt.get('batchsize', 1)
  num_turns = opt['selfchat_max_turns']

  num_parleys = math.ceil(num_turns / bsz)
  for _ in range(num_parleys):
    finished = world.parley()
    world_logger.log(world)

    if opt['display_examples']:
      print(world.display())

    if finished:
      break

  if opt['display_examples']:
    print('-- end of episode --')

  world.reset()
  world_logger.reset_world()  # flush this episode


class MyLocalHumanAgent(LocalHumanAgent):

  def __init__(self, opt):
    super().__init__(opt)
    if ':' in opt['model_file']:
      self.ref_data = []
      ref_file = opt['model_file'].split(':')[1]
      with open(ref_file) as f:
        self.ref_data = [json.loads(l)['conversation'] for l in f]
    else:
      self.ref_data = None


def self_chat(opt):
  random.seed(opt['seed'])
  partner = opt['partner_model_file']
  assert partner is not None
  partner_opt_file = opt.get('partner_opt_file')
  if partner_opt_file:
    assert partner_opt_file == partner + '.opt', ('Unless you think it is save,'
                                                  ' you can remove assert')
  else:
    partner_opt_file = partner + '.opt'

  # Create agents
  if opt['model_file'].split(':')[0] == 'human':
    agent1 = MyLocalHumanAgent(opt)
    assert partner is not None
  else:
    agent1 = create_agent(opt, requireModelExists=True)
  if partner is None:
    # Self chat with same model
    agent2 = agent1.clone()
  else:
    # Self chat with different models
    if partner_opt_file:
      print(f"WARNING: Loading override opts from: {partner_opt_file}")
      with open(partner_opt_file) as f:
        partner_opt = json.load(f)
    else:
      partner_opt = {}
    partner_opt['interactive_mode'] = opt.get('interactive_mode', True)
    print(
        f"WARNING: Setting partner interactive mode to: {partner_opt['interactive_mode']}"
    )
    agent2 = create_agent_from_model_file(partner, partner_opt)

  # Set IDs
  agent1.id = agent1.id + '_1'
  agent2.id = agent2.id + '_2'

  model_id = agent1.id + '_' + agent2.id

  world = create_task(opt, user_agents=[agent1, agent2])

  # Set up world logging
  logger = WorldLogger(opt)
  log_time = TimeLogger()

  # Run some self chats.
  all_report = []
  if opt['num_self_chats'] < 0:
    opt['num_self_chats'] = len(world.messages)

  for i in range(opt['num_self_chats']):
    _run_self_chat_episode(opt, world, logger)
    report = world.report()
    text, report = log_time.log(i + 1, opt['num_self_chats'], report)
    logging.info(text)
    all_report.append(report)

    world.write(logger, all_report, opt['outfile'])

  # Save chats
  if opt['outfile'] is None:
    outfile = '/tmp/{}_selfchat'.format(model_id)
  else:
    outfile = opt['outfile']

  if opt['save_format'] == 'conversations' and hasattr(world, 'write'):
    # use self chat specific world to write conversation
    # this might be useful for logging extra contextual
    # information (like personas)
    world.write(logger, all_report, outfile)
  else:
    # use default logger write function
    logger.write(outfile, world, opt['save_format'])

  return logger.get_logs()


@register_script('self_chat')
class SelfChat(ParlaiScript):

  @classmethod
  def setup_args(cls):
    return setup_args()

  def run(self):
    return self_chat(self.opt)


if __name__ == '__main__':
  SelfChat.main()
