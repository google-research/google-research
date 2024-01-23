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

r"""Simple example that uses Gym-Minigrid and a custom environment.

This is arguably the simplest example one can have on Gym-Minigrid. Here we have
a random agent interacting with the environment. In this case, I'm using a
custom environment, which is likely what one will do in their research. We are
writing the agent observations to the disk just as a simple way to get some
feedback of what is going on.

Sample run:

  ```
  python -m minigrid_basics.examples.rw_standard \
    --gin_bindings="MonMiniGridEnv.stochasticity=0.1"
  ```

"""

import os

from absl import app
from absl import flags
import gin
import gym
import gym_minigrid  # pylint: disable=unused-import
from gym_minigrid.wrappers import ImgObsWrapper
from gym_minigrid.wrappers import RGBImgObsWrapper
import matplotlib.pylab as plt
import tensorflow as tf

from minigrid_basics.envs import mon_minigrid

FLAGS = flags.FLAGS

flags.DEFINE_string('file_path', '/tmp/rw_standard',
                    'Path in which we will save the observations.')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override default parameter values '
    '(e.g. "MonMiniGridEnv.stochasticity=0.1").')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  gin.parse_config_files_and_bindings(
      [os.path.join(mon_minigrid.GIN_FILES_PREFIX, 'classic_fourrooms.gin')],
      bindings=FLAGS.gin_bindings,
      skip_unknown=False)
  env_id = mon_minigrid.register_environment()
  env = gym.make(env_id)
  env = RGBImgObsWrapper(env)  # Get pixel observations
  env = ImgObsWrapper(env)     # Get rid of the 'mission' field
  env.reset()

  num_frames = 0
  max_num_frames = 500

  if not tf.io.gfile.exists(FLAGS.file_path):
    tf.io.gfile.makedirs(FLAGS.file_path)

  undisc_return = 0
  while num_frames < max_num_frames:
    # Act randomly
    obs, reward, done, _ = env.step(env.action_space.sample())
    undisc_return += reward
    num_frames += 1

    # Draw environment frame just for simple visualization
    plt.imshow(obs)
    path = os.path.join(FLAGS.file_path, 'obs_{}.png'.format(num_frames))

    plt.savefig(path)
    plt.clf()

    if done:
      break

  print('Undiscounted return: %.2f' % undisc_return)
  env.close()


if __name__ == '__main__':
  app.run(main)
