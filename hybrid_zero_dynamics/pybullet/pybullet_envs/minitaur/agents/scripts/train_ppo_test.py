# Copyright 2017 The TensorFlow Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the PPO algorithm usage example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import itertools

import tensorflow as tf

from google3.robotics.reinforcement_learning.agents import ppo
from google3.robotics.reinforcement_learning.agents import tools
from google3.robotics.reinforcement_learning.agents.scripts import configs
from google3.robotics.reinforcement_learning.agents.scripts import networks
from google3.robotics.reinforcement_learning.agents.scripts import train


FLAGS = tf.app.flags.FLAGS


class PPOTest(tf.test.TestCase):

  def test_no_crash_cheetah(self):
    nets = networks.ForwardGaussianPolicy, networks.RecurrentGaussianPolicy
    for network in nets:
      config = self._define_config()
      with config.unlocked:
        config.env = 'HalfCheetah-v1'
        config.max_length = 200
        config.steps = 1000
        config.network = network
      for score in train.train(config, env_processes=True):
        float(score)

  def test_no_crash_ant(self):
    nets = networks.ForwardGaussianPolicy, networks.RecurrentGaussianPolicy
    for network in nets:
      config = self._define_config()
      with config.unlocked:
        config.env = 'Ant-v1'
        config.max_length = 200
        config.steps = 1000
        config.network = network
      for score in train.train(config, env_processes=True):
        float(score)

  def test_no_crash_observation_shape(self):
    nets = networks.ForwardGaussianPolicy, networks.RecurrentGaussianPolicy
    observ_shapes = (1,), (2, 3), (2, 3, 4)
    for network, observ_shape in itertools.product(nets, observ_shapes):
      config = self._define_config()
      with config.unlocked:
        config.env = functools.partial(
            tools.MockEnvironment, observ_shape, action_shape=(3,),
            min_duration=15, max_duration=15)
        config.max_length = 20
        config.steps = 100
        config.network = network
      for score in train.train(config, env_processes=False):
        float(score)

  def test_no_crash_variable_duration(self):
    config = self._define_config()
    with config.unlocked:
      config.env = functools.partial(
          tools.MockEnvironment, observ_shape=(2, 3), action_shape=(3,),
          min_duration=5, max_duration=25)
      config.max_length = 25
      config.steps = 200
      config.network = networks.RecurrentGaussianPolicy
    for score in train.train(config, env_processes=False):
      float(score)

  def _define_config(self):
    # Start from the example configuration.
    locals().update(configs.default())
    # pylint: disable=unused-variable
    # General
    algorithm = ppo.PPOAlgorithm
    num_agents = 2
    update_every = 4
    use_gpu = False
    # Network
    policy_layers = 20, 10
    value_layers = 20, 10
    # Optimization
    update_epochs_policy = 2
    update_epochs_value = 2
    # pylint: enable=unused-variable
    return tools.AttrDict(locals())


if __name__ == '__main__':
  FLAGS.config = 'unused'
  tf.test.main()
