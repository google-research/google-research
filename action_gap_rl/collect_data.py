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

# Lint as: python3
"""Data collection phase."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
import gym
import numpy as np
import replay


FLAGS = flags.FLAGS
flags.DEFINE_string('outdir', None, 'Directory where data will be written.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  env = gym.make('Pendulum-v0')

  behavior_policy = lambda obs: env.action_space.sample()
  # null_policy = lambda obs: np.zeros(env.action_space.shape)

  num_episodes = 100
  episode_length = 200

  memory = replay.Memory()

  for _ in range(num_episodes):
    # collect a trajectory
    obs = env.reset()
    memory.log_init(obs)

    for _ in range(episode_length):
      act = behavior_policy(obs)
      next_obs, reward, term, _ = env.step(act)
      memory.log_experience(obs, act, reward, next_obs)
      if term:
        break
      obs = next_obs

  s = memory.serialize()

  # Save pickle file
  with open(os.path.join(FLAGS.outdir, 'pendulum.pickle'), 'wb') as f:
    f.write(s)

  # Sanity check serialization.
  m2 = replay.Memory()
  m2.unserialize(s)
  print(np.array_equal(m2.entered_states(), memory.entered_states()))
  print(np.array_equal(m2.exited_states(), memory.exited_states()))
  print(np.array_equal(m2.attempted_actions(), memory.attempted_actions()))
  print(np.array_equal(m2.observed_rewards(), memory.observed_rewards()))


if __name__ == '__main__':
  app.run(main)
