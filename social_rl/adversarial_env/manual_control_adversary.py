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

# Lint as: python3
"""Script used for debugging the environment via command line.

The env is rendered as a string so it can be used over ssh.
"""
import argparse
from absl import logging
import gym
import matplotlib.pyplot as plt
import numpy as np

# Import needed to trigger env registration, so pylint: disable=unused-import
from social_rl import gym_multigrid


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--env_name', type=str, default='MultiGrid-Adversarial-v0',
      help='Name of multi-agent adversarial environment.')
  parser.add_argument('--test_domain_randomization', default=False,
                      action='store_true')
  return parser.parse_args()


def get_user_input_agent(env):
  """Validates user keyboard input to obtain valid actions for agent.

  Args:
    env: Instance of MultiGrid environment

  Returns:
    An array of integer actions.
  """
  max_action = max(env.Actions).value
  min_action = min(env.Actions).value

  # Print action commands for user convenience.
  print('Actions are:')
  for act in env.Actions:
    print('\t', str(act.value) + ':', act.name)

  prompt = 'Enter action for agent, or q to quit: '

  # Check user input
  while True:
    user_cmd = input(prompt)
    if user_cmd == 'q':
      return False

    actions = user_cmd.split(',')
    if len(actions) != env.n_agents:
      logging.info('Uh oh, you entered commands for %i agents but there is '
                   '%i. Try again?', len(actions), str(env.n_agents))
      continue

    valid = True
    for i, a in enumerate(actions):
      if not a.isdigit() or int(a) > max_action or int(a) < min_action:
        logging.info('Uh oh, action %i is invalid.', i)
        valid = False

    if valid:
      break
    else:
      logging.info('All actions must be an integer between %i and %i',
                   min_action, max_action)

  return [int(a) for a in actions if a]


def get_user_input_environment(env, reparam=False):
  """Validate action input for adversarial environment role.

  Args:
    env: Multigrid environment object.
    reparam: True if this is a reparameterized version of the environment.

  Returns:
    Integer action.
  """
  max_action = env.adversary_action_dim - 1
  min_action = 0

  # Check if using the reparameterized environment, in which case objects are
  # placed by the adversary using a different action space.
  if not reparam:
    obj_type = 'wall'
    if env.choose_goal_last:
      if env.adversary_step_count == env.adversary_max_steps - 1:
        obj_type = 'agent'
      elif env.adversary_step_count == env.adversary_max_steps - 2:
        obj_type = 'goal'
    else:
      # In the reparameterized environment.
      if env.adversary_step_count == 0:
        obj_type = 'goal'
      elif env.adversary_step_count == 1:
        obj_type = 'agent'
    prompt = 'Place ' + obj_type + ': enter an integer between ' + \
        str(max_action) + ' and ' + str(min_action) + ': '
  else:
    prompt = 'Enter an integer between ' + str(max_action) + \
             ' and ' + str(min_action) + ': '

  # Check user input
  while True:
    user_cmd = input(prompt)
    if user_cmd == 'q':
      return False

    if (not user_cmd.isdigit() or int(user_cmd) > max_action or
        int(user_cmd) < min_action):
      print('Invalid action. All actions must be an integer between',
            min_action, 'and', max_action)
    else:
      break

  return user_cmd


def main(args):
  env = gym.make(args.env_name)
  obs = env.reset()

  print('You are playing the role of the adversary to place blocks in '
        'the environment.')
  if 'Reparameterized' in args.env_name:
    print('You will move through the spots in the grid in order from '
          'left to right, top to bottom. At each step, place the goal '
          '(0), the agent (1), a wall (2), or skip (3)')
  print(env)

  reparam = 'Reparameterized' in args.env_name

  # Adversarial environment loop
  if not args.test_domain_randomization:
    while True:
      if reparam:
        print('Place the next tile at coordinates ({0}, {1})'.format(
            obs['x'][0], obs['y'][0]))

      action = get_user_input_environment(env, reparam)
      if not action:
        break

      obs, _, done, _ = env.step_adversary(int(action))
      plt.imshow(env.render('rgb_array'))
      print(env)

      if env.adversary_step_count > 1 and not env.choose_goal_last:
        env.compute_shortest_path()
        if not env.passable:
          print(
              'There is no possible path between the start position and goal.')
        else:
          print('The shortest path between the start position and goal '
                'is length', env.shortest_path_length)

      if done:
        break
  else:
    env.reset_random()

  if env.choose_goal_last:
    if not env.passable:
      print(
          'There is no possible path between the start position and goal.')
    else:
      print('The shortest path between the start position and goal '
            'is length', env.shortest_path_length)

  print('Finished. A total of', env.n_clutter_placed, 'blocks were placed.')
  print('Goal was placed at a distance of', env.distance_to_goal)

  # Agent-environment interaction loop
  for name in ['agent', 'adversary agent']:
    logging.info('Now control the %i', name)
    obs = env.reset_agent()
    reward_hist = []
    for i in range(env.max_steps+1):
      print(env)

      logging.info('Observation:')
      for k in obs.keys():
        if isinstance((obs[k]), list):
          logging.info(k, len(obs[k]))
        else:
          logging.info(k, obs[k].shape)

      actions = get_user_input_agent(env)
      if not actions:
        return

      obs, rewards, done, _ = env.step(actions)

      for k in obs.keys():
        logging.info(k, np.array(obs[k]).shape)

      reward_hist.append(rewards)
      plt.imshow(env.render('rgb_array'))
      print('Step:', i)
      print('Rewards:', rewards)
      print('Collective reward history:', reward_hist)
      print('Cumulative collective reward:', np.sum(reward_hist))

      if done:
        logging.info('Game over')
        break


if __name__ == '__main__':
  main(parse_args())
