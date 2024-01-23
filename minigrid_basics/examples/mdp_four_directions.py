# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

r"""Uses Gym-Minigrid, a custom environment, custom actions, and the MDP wrapper.

Gym-Minigrid has a larger action space that is not standard in reinforcement
learning. By default, the actions are {rotate left, rotate right, forward, pick
up object, drop object, toggle/activate object, done}. This example uses a class
overridden to have the standard 4 directional actions: {left, right, up, down}.

We pass this environment through the custom MDPWrapper to obtain access to the
transition dynamics `transition_probs` and reward dynamics `rewards`. We then
use these to compute `values` via value iteration.

We also make use of the `ColoringWrapper` to render the computed values on the
grid.

Sample run:

  ```
  python -m minigrid_basics.examples.mdp_four_directions \
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
from matplotlib import cm
from matplotlib import colors
import matplotlib.pylab as plt
import numpy as np

from minigrid_basics.custom_wrappers import coloring_wrapper
from minigrid_basics.custom_wrappers import mdp_wrapper
from minigrid_basics.envs import mon_minigrid


FLAGS = flags.FLAGS

flags.DEFINE_string('values_image_file', None,
                    'Path prefix to use for saving the observations.')
flags.DEFINE_string('env', 'classic_fourrooms', 'Environment to run.')
flags.DEFINE_float('tolerance', 0.001, 'Error tolerance for value iteration.')
flags.DEFINE_float('gamma', 0.9, 'Discount factor to use for value iteration.')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override default parameter values '
    '(e.g. "MonMiniGridEnv.stochasticity=0.1").')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  gin.parse_config_files_and_bindings(
      [os.path.join(mon_minigrid.GIN_FILES_PREFIX, '{}.gin'.format(FLAGS.env))],
      bindings=FLAGS.gin_bindings,
      skip_unknown=False)
  env_id = mon_minigrid.register_environment()
  env = gym.make(env_id)
  env = RGBImgObsWrapper(env)  # Get pixel observations
  # Get tabular observation and drop the 'mission' field:
  env = mdp_wrapper.MDPWrapper(env)
  env = coloring_wrapper.ColoringWrapper(env)
  values = np.zeros(env.num_states)
  error = FLAGS.tolerance * 2
  i = 0
  while error > FLAGS.tolerance:
    new_values = np.copy(values)
    for s in range(env.num_states):
      max_value = 0.
      for a in range(env.num_actions):
        curr_value = (env.rewards[s, a] +
                      FLAGS.gamma * np.matmul(env.transition_probs[s, a, :],
                                              values))
        if curr_value > max_value:
          max_value = curr_value
      new_values[s] = max_value
    error = np.max(abs(new_values - values))
    values = new_values
    i += 1
    if i % 1000 == 0:
      print('Error after {} iterations: {}'.format(i, error))
  print('Found V* in {} iterations'.format(i))
  print(values)
  if FLAGS.values_image_file is not None:
    cmap = cm.get_cmap('plasma', 256)
    norm = colors.Normalize(vmin=min(values), vmax=max(values))
    obs_image = env.render_custom_observation(env.reset(), values, cmap,
                                              boundary_values=[1.0, 4.5])
    m = cm.ScalarMappable(cmap=cmap, norm=norm)
    m.set_array(obs_image)
    plt.imshow(obs_image)
    plt.colorbar(m)
    plt.savefig(FLAGS.values_image_file)
    plt.clf()
  env.close()


if __name__ == '__main__':
  app.run(main)
