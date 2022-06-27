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

"""Tests for multi_mc_replay_buffer."""

from typing import Sequence

from absl.testing import absltest
import numpy as np

from aux_tasks.auxiliary_mc import multi_mc_replay_buffer


def _fill_replay_buffer(buffer,
                        num_steps = 20,
                        obs_shape = (84, 84)):
  obs = np.zeros(obs_shape, dtype=np.uint8)
  action = 0
  reward = 1.0
  terminal = False
  for _ in range(num_steps):
    buffer.add(obs, action, reward, terminal)


class MultiMcReplayBufferTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    self._obs_shape = (84, 84)
    self._stack_size = 4
    self._replay_capacity = 1_000
    self._batch_size = 4
    self._update_horizon = 10
    self._buffer = multi_mc_replay_buffer.MultiMCReplayBuffer(
        observation_shape=self._obs_shape,
        stack_size=self._stack_size,
        replay_capacity=self._replay_capacity,
        batch_size=self._batch_size,
        update_horizon=self._update_horizon,
        num_additional_discount_factors=4)

  def test_replay_buffer_returns_correct_returns_for_multiple_gamma(self):
    extra_discounts = (1.0, 0.9, 0.5, 0.0)
    # Since we want to test with a backup of 10 steps, fill up > 10 transitions.
    _fill_replay_buffer(self._buffer, num_steps=20)

    batch = self._buffer.sample_transition_batch(
        batch_size=1,
        # Avoid 0 transitions at the beginning of the buffer.
        indices=(self._stack_size,),
        extra_discounts=extra_discounts)
    extra_returns = batch[8]  # The extra returns are 9th in the batch.

    # We compute the expected extra returns naively for transparency.
    extra_discounts = np.asarray(extra_discounts)
    ones = np.ones((4,))
    expected_extra_returns = np.zeros((4,), dtype=np.float32)
    for _ in range(10):
      expected_extra_returns = ones + (extra_discounts * expected_extra_returns)
    expected_extra_returns = expected_extra_returns.reshape(1, 4)

    np.testing.assert_allclose(extra_returns, expected_extra_returns)

  def test_replay_buffer_raises_error_if_incorrect_num_discount_factors(self):
    extra_discounts = [0.9 for _ in range(5)]  # Expects 4 discounts.
    # Since we want to test with a backup of 10 steps, fill up > 10 transitions.
    _fill_replay_buffer(self._buffer, num_steps=20)

    with self.assertRaises(ValueError):
      self._buffer.sample_transition_batch(
          batch_size=1,
          # Avoid 0 transitions at the beginning of the buffer.
          indices=(self._stack_size,),
          extra_discounts=extra_discounts)


if __name__ == '__main__':
  absltest.main()
