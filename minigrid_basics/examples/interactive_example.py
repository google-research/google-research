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

r"""Example that uses Gym-Minigrid in an interactive way.

The user will be able to interact with the environment by using the keyboard.

Sample run:

  ```
  python -m minigrid_basics.examples.interactive_example \
    --gin_bindings="MonMiniGridEnv.stochasticity=0.1"
  ```

"""

import os

from absl import app
from absl import flags
import gin
import gym
import gym_minigrid  # pylint: disable=unused-import
from gym_minigrid.wrappers import RGBImgObsWrapper
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf

from minigrid_basics.custom_wrappers import tabular_wrapper
from minigrid_basics.envs import mon_minigrid


FLAGS = flags.FLAGS

flags.DEFINE_string('file_path', '/tmp/minigrid/interactive',
                    'Path in which we will save the observations.')
flags.DEFINE_string('env_name', 'classic_fourrooms', 'Name of the environment.')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override default parameter values '
    '(e.g. "MonMiniGridEnv.stochasticity=0.1").')


DIR_MAPPINGS = ['>', 'v', '<', '^']
# The action names must match those in mon_minigrid.py.
ACTION_MAPPINGS = {
    'w': 'up',
    's': 'down',
    'a': 'left',
    'd': 'right',
}


def draw_ascii_view(env):
  """Draw an ASCII version of the grid.

  Will use special characters for goal and lava, and will display the agent's
  direction.

  Args:
    env: MiniGrid environment.
  """
  agent_pos = env.agent_pos
  agent_dir = env.agent_dir
  print('-' * env.width)
  for y in range(env.height):
    line = ''
    for x in range(env.width):
      grid_pos = x + y * env.width
      if env.pos_to_state[grid_pos] < 0:
        line += '*'
      elif np.array_equal((x, y), agent_pos):
        line += DIR_MAPPINGS[agent_dir]
      else:
        cell = env.grid.get(x, y)
        if cell is not None:
          if cell.type == 'goal':
            line += 'g'
          elif cell.type == 'lava':
            line += '%'
          else:
            line += ' '
        else:
          line += ' '
    print(line)
  print('-' * env.width)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  gin.parse_config_files_and_bindings(
      [os.path.join(mon_minigrid.GIN_FILES_PREFIX,
                    '{}.gin'.format(FLAGS.env_name))],
      bindings=FLAGS.gin_bindings,
      skip_unknown=False)
  env_id = mon_minigrid.register_environment()
  env = gym.make(env_id)
  env = RGBImgObsWrapper(env)  # Get pixel observations
  # Get tabular observation and drop the 'mission' field:
  env = tabular_wrapper.TabularWrapper(env, get_rgb=True)
  env.reset()

  num_frames = 0
  max_num_frames = 500

  if not tf.io.gfile.exists(FLAGS.file_path):
    tf.io.gfile.makedirs(FLAGS.file_path)

  print('Available actions:')
  for a in ACTION_MAPPINGS:
    print('\t{}: "{}"'.format(ACTION_MAPPINGS[a], a))
  print()
  undisc_return = 0
  while num_frames < max_num_frames:
    draw_ascii_view(env)
    a = input('action: ')
    if a not in ACTION_MAPPINGS:
      print('Unrecognized action.')
      continue
    action = env.DirectionalActions[ACTION_MAPPINGS[a]].value
    obs, reward, done, _ = env.step(action)
    undisc_return += reward
    num_frames += 1

    print('t:', num_frames, '   s:', obs['state'])
    # Draw environment frame just for simple visualization
    plt.imshow(obs['image'])
    path = os.path.join(FLAGS.file_path, 'obs_{}.png'.format(num_frames))
    plt.savefig(path)
    plt.clf()

    if done:
      break

  print('Undiscounted return: %.2f' % undisc_return)
  env.close()


if __name__ == '__main__':
  app.run(main)
