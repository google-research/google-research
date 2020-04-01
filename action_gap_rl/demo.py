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

"""Runs an experiment, consisting of data collection, training, and eval."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from absl import app
from absl import flags
from absl import logging
import gym

FLAGS = flags.FLAGS

TimeStep = collections.namedtuple('TimeStep', 't,a,s,r')


class UniformContinuousAgent(object):

  def __init__(self, action_space):
    self._action_space = action_space

  def __call__(self, _):
    return self._action_space.sample()  # Samples uniformly.


def sample_episode(env, policy, max_episode_length):
  """Sample one episode from the environment."""
  state = env.reset()
  done = False
  t = 0
  rollout = []
  while not done and t < max_episode_length:
    action = policy(state)
    next_state, reward, done, _ = env.step(action)
    reward = max(min(reward, 1), -1)
    t += 1
    rollout.append(TimeStep(t=t, a=action, s=state, r=reward))
    state = next_state
  return rollout


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  env = gym.make('Pendulum-v0')
  agent = UniformContinuousAgent(env.action_space)

  rollout = sample_episode(env, agent, max_episode_length=1000)
  logging.info('Epside length: %d', len(rollout))
  logging.info('rollout:\n%s', rollout)


if __name__ == '__main__':
  app.run(main)
