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

"""Tests for muzero.core."""

import tensorflow as tf
from muzero import core


class CoreTest(tf.test.TestCase):

  def test_make_target(self):
    num_unroll_steps = 3
    td_steps = -1
    rewards = [1., 2., 3., 4.]
    # Assume 4 different actions.
    policy_distributions = [
        [0.7, 0.1, 0.1, 0.1],
        [0.1, 0.7, 0.1, 0.1],
        [0.1, 0.1, 0.7, 0.1],
        [0.1, 0.1, 0.1, 0.7],
    ]
    discount = 0.9

    target = core.Episode.make_target(
        state_index=0,
        num_unroll_steps=num_unroll_steps,
        td_steps=td_steps,
        rewards=rewards,
        policy_distributions=policy_distributions,
        discount=discount)
    self.assertEqual(core.Target(
        value_mask=(1., 1., 1., 1.),
        reward_mask=(0., 1., 1., 1.),
        policy_mask=(1., 1., 1., 1.),
        value=(rewards[0] + rewards[1] * discount \
                + rewards[2] * discount**2 + rewards[3] * discount**3,
               rewards[1] + rewards[2] * discount + rewards[3] * discount**2,
               rewards[2] + rewards[3] * discount,
               rewards[3]),
        reward=(rewards[3], rewards[0], rewards[1], rewards[2]),
        visits=tuple(policy_distributions)), target)

    target = core.Episode.make_target(
        state_index=2,
        num_unroll_steps=num_unroll_steps,
        td_steps=td_steps,
        rewards=rewards,
        policy_distributions=policy_distributions,
        discount=discount)
    self.assertEqual(
        core.Target(
            value_mask=(1., 1., 1., 1.),
            reward_mask=(0., 1., 1., 0.),
            policy_mask=(1., 1., 0., 0.),
            value=(rewards[2] + rewards[3] * discount, rewards[3], 0., 0.),
            reward=(rewards[1], rewards[2], rewards[3], 0.),
            visits=tuple(policy_distributions[2:] +
                         [policy_distributions[0]] * 2)), target)


if __name__ == '__main__':
  tf.test.main()
