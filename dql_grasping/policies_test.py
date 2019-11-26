# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

# Lint as: python2, python3
"""Unit tests for checking policy correctness.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl.testing import parameterized
import numpy as np
from six.moves import range
import tensorflow as tf

from dql_grasping import grasping_env
from dql_grasping import policies
from dql_grasping import tf_critics

FLAGS = flags.FLAGS


class DummyGreedyPolicy(policies.RandomGraspingPolicyD4):
  """Dummy greedy policy."""

  def sample_action(self, obs, explore_prob):
    del explore_prob
    return [3, 3, 3, 3], None


class PoliciesTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(('i1', 0), ('i2', .5), ('i3', 1))
  def testPerStepSwitch(self, explore_prob):
    # Check that exploration is occurring as frequently as we expect.
    np.random.seed(0)
    policy = policies.PerStepSwitchPolicy(policies.RandomGraspingPolicyD4,
                                          DummyGreedyPolicy)
    actions = []
    for _ in range(100):
      action, _ = policy.sample_action(None, explore_prob)
      actions.append(action)
    actions = np.array(actions)
    empirical_explore_prob = np.sum(actions[:, 0] < 3)/100
    self.assertAllClose([empirical_explore_prob], [explore_prob], atol=.03)

  def testCEMACtor(self):
    np.random.seed(0)
    env = grasping_env.KukaGraspingProceduralEnv(
        downsample_width=48, downsample_height=48,
        continuous=True, remove_height_hack=True, render_mode='DIRECT')
    policy = policies.CEMActorPolicy(tf_critics.cnn_ia_v1,
                                     state_shape=(1, 48, 48, 3),
                                     action_size=4,
                                     use_gpu=False,
                                     batch_size=64,
                                     build_target=False,
                                     include_timestep=True)
    policy.reset()
    obs = env.reset()
    action, debug = policy.sample_action(obs, 0)
    self.assertLen(action, 4)
    self.assertIn('q', debug)
    self.assertIn('final_params', debug)

  def testDDPGPolicy(self):
    np.random.seed(0)
    env = grasping_env.KukaGraspingProceduralEnv(
        downsample_width=48, downsample_height=48,
        continuous=True, remove_height_hack=True, render_mode='DIRECT')
    policy = policies.DDPGPolicy(tf_critics.cnn_v0,
                                 tf_critics.cnn_ia_v1,
                                 state_shape=(1, 48, 48, 3),
                                 action_size=4,
                                 use_gpu=False,
                                 build_target=False,
                                 include_timestep=True)
    policy.reset()
    obs = env.reset()
    action, debug = policy.sample_action(obs, 0)
    self.assertLen(action, 4)
    self.assertIn('q', debug)


if __name__ == '__main__':
  tf.test.main()
