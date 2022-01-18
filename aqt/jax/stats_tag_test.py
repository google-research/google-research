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

"""Tests for aqt.jax.stats_tag."""

import math

from absl.testing import absltest
from absl.testing import parameterized
from jax import random
import jax.numpy as jnp
import numpy as onp

from aqt.jax import test_utils
from aqt.jax.stats_tag import StatsTag

test_utils.configure_jax()


class StatsTagTest(parameterized.TestCase):

  def setUp(self):
    super(StatsTagTest, self).setUp()
    self.rng = random.PRNGKey(0)
    key1, key2 = random.split(self.rng)
    self.key = key2
    self.x = random.normal(key1, (4, 3, 2))
    self.x2 = jnp.ones((4, 3, 2))
    self.x3 = jnp.arange(-3, -3 + (4 * 3 * 2)).reshape((4, 3, 2))
    self.x4 = jnp.ones((11, 13, 15))

  def init_model(self,
                 init_x,
                 channel_axis,
                 update_stats,
                 num_indices_per_ax=10):
    tag = StatsTag(
        num_indices_per_ax=num_indices_per_ax,
        channel_axis=channel_axis,
        update_stats=update_stats)
    init_state = tag.init(self.key, init_x, mask=None)
    return tag, init_state

  def test_masking(self):
    x = jnp.array([[1, -2, 3], [4, 5, 6]]).astype(jnp.float32)
    mask = jnp.array([[False, True, True], [True, True, False]])
    model, init_state = self.init_model(
        init_x=x, channel_axis=1, update_stats=True)
    _, new_state = model.apply(init_state, x, mask=mask, mutable='stats_tag')
    stats = new_state['stats_tag']
    onp.testing.assert_allclose(stats['min_per_ch'], [4, -2, 3])
    onp.testing.assert_allclose(stats['max_per_ch'], [4, 5, 3])
    mean_channel2 = (-2 + 5) / 2
    onp.testing.assert_allclose(stats['mean_per_ch'], [4, mean_channel2, 3])
    stddev_channel2 = math.sqrt(
        ((-2 - mean_channel2)**2 + (5 - mean_channel2)**2) / 2)
    onp.testing.assert_allclose(stats['stddev_per_ch'], [0, stddev_channel2, 0])
    absdev_channel2 = (abs(-2 - mean_channel2) + abs(5 - mean_channel2)) / 2
    onp.testing.assert_allclose(stats['absdev_per_ch'], [0, absdev_channel2, 0])
    sttdev_uncentered_channel2 = math.sqrt(((-2)**2 + 5**2) / 2)
    onp.testing.assert_allclose(stats['stddev_per_ch_uncentered'],
                                [4, sttdev_uncentered_channel2, 3])
    absdev_uncentered_channel2 = (abs(-2) + abs(5)) / 2
    onp.testing.assert_allclose(stats['absdev_per_ch_uncentered'],
                                [4, absdev_uncentered_channel2, 3])

    # Test correctness in presence of broadcasting and with channel_axis=None.
    # The mask, which is 1x3, has to be broadcasted to match 'x', which is
    # 2x3. After masking and flattening, 'x' should be equivalent to
    # [1, 3, -4, 6], so we compare to statistics computed on that array.

    mask = jnp.array([True, False, True]).reshape(1, 3)
    x = jnp.array([[1, 2, 3], [-4, 5, 6]])
    model, init_state = self.init_model(
        init_x=x, channel_axis=None, update_stats=True)
    _, state = model.apply(init_state, x, mask=mask, mutable='stats_tag')
    stats = state['stats_tag']
    onp.testing.assert_allclose(stats['min_per_ch'], -4)
    onp.testing.assert_allclose(stats['max_per_ch'], 6)
    mean = (1 + 3 + -4 + 6) / 4
    onp.testing.assert_allclose(stats['mean_per_ch'], mean)
    stddev = jnp.sqrt(
        ((1 - mean)**2 + (3 - mean)**2 + (-4 - mean)**2 + (6 - mean)**2) / 4)
    onp.testing.assert_allclose(stats['stddev_per_ch'], stddev)
    stddev_uncentered = jnp.sqrt((1**2 + 3**2 + (-4)**2 + 6**2) / 4)
    onp.testing.assert_allclose(stats['stddev_per_ch_uncentered'],
                                stddev_uncentered)
    absdev = (onp.abs(1 - mean) + onp.abs(3 - mean) + onp.abs(-4 - mean) +
              onp.abs(6 - mean)) / 4
    onp.testing.assert_allclose(stats['absdev_per_ch'], absdev)
    absdev_uncentered = (onp.abs(1) + onp.abs(3) + onp.abs(-4) + onp.abs(6)) / 4
    onp.testing.assert_allclose(stats['absdev_per_ch_uncentered'],
                                absdev_uncentered)

    # Test with broadcasting and channel_axis=1
    model, init_state = self.init_model(
        init_x=x, channel_axis=(0,), update_stats=True)
    _, state = model.apply(init_state, x, mask=mask, mutable='stats_tag')
    stats = state['stats_tag']
    mean = [(1 + 3) / 2, (-4 + 6) / 2]
    onp.testing.assert_allclose(stats['mean_per_ch'], mean)
    max_per_ch = [3, 6]
    onp.testing.assert_allclose(stats['max_per_ch'], max_per_ch)

  @parameterized.named_parameters(
      dict(
          testcase_name='no_channel_dim',
          channel_axis=None,
          expected_shape=(1,)),
      dict(
          testcase_name='one_channel_dim', channel_axis=-1,
          expected_shape=(2,)),
      dict(
          testcase_name='one_channel_dim_tuple',
          channel_axis=(2,),
          expected_shape=(2,)),
      dict(
          testcase_name='two_channel_dims',
          channel_axis=(1, 2),
          expected_shape=(3, 2)),
  )
  def test_init_and_shape(self, channel_axis, expected_shape):
    _, init_state = self.init_model(
        init_x=self.x, channel_axis=channel_axis, update_stats=True)
    for key in [
        'min_per_ch', 'max_per_ch', 'mean_per_ch', 'stddev_per_ch',
        'absdev_per_ch', 'stddev_per_ch_uncentered', 'absdev_per_ch_uncentered'
    ]:
      onp.testing.assert_array_equal(init_state['stats_tag'][key],
                                     onp.zeros(expected_shape))

  @parameterized.named_parameters(
      dict(
          testcase_name='no_channel_dim',
          channel_axis=None,
          expected_shape=(1,)),
      dict(
          testcase_name='one_channel_dim',
          channel_axis=-1,
          expected_shape=(2,)),
      dict(
          testcase_name='two_channel_dims',
          channel_axis=(1, 2),
          expected_shape=(3, 2)),
  )
  def test_tag_stats_ones(self, channel_axis, expected_shape):
    model, init_state = self.init_model(
        init_x=self.x, channel_axis=channel_axis, update_stats=True)
    _, state_0 = model.apply(
        init_state, self.x2, mask=None, mutable='stats_tag')
    stats = state_0['stats_tag']
    for key in ['min_per_ch', 'max_per_ch', 'mean_per_ch',
                'stddev_per_ch_uncentered', 'absdev_per_ch_uncentered']:
      onp.testing.assert_array_equal(stats[key], onp.ones(expected_shape))
    for key in ['stddev_per_ch', 'absdev_per_ch']:
      onp.testing.assert_array_equal(stats[key], onp.zeros(expected_shape))

  @parameterized.named_parameters(
      dict(
          testcase_name='no_channel_dim',
          channel_axis=None,
          reduction_axis=(0, 1, 2)),
      dict(
          testcase_name='one_channel_dim',
          channel_axis=-1,
          reduction_axis=(0, 1)),
      dict(
          testcase_name='two_channel_dims',
          channel_axis=(1, 2),
          reduction_axis=(0,)),
  )
  def test_tag_stats_numpy_comparison(self, channel_axis, reduction_axis):
    model, init_state = self.init_model(
        init_x=self.x, channel_axis=channel_axis, update_stats=True)
    _, state_0 = model.apply(
        init_state, self.x2, mask=None, mutable='stats_tag')
    _, state_1 = model.apply(state_0, self.x3, mask=None, mutable='stats_tag')
    stats = state_1['stats_tag']
    onp.testing.assert_allclose(stats['min_per_ch'],
                                onp.min(self.x3, axis=reduction_axis))
    onp.testing.assert_allclose(stats['max_per_ch'],
                                onp.max(self.x3, axis=reduction_axis))
    onp.testing.assert_allclose(stats['mean_per_ch'],
                                onp.mean(self.x3, axis=reduction_axis))
    onp.testing.assert_allclose(stats['stddev_per_ch'],
                                onp.std(self.x3, axis=reduction_axis))
    onp.testing.assert_allclose(
        stats['absdev_per_ch'],
        onp.mean(
            onp.abs(self.x3 -
                    onp.mean(self.x3, axis=reduction_axis, keepdims=True)),
            axis=reduction_axis))
    onp.testing.assert_allclose(
        stats['stddev_per_ch_uncentered'],
        onp.sqrt(onp.mean(onp.square(self.x3), axis=reduction_axis)))
    onp.testing.assert_allclose(stats['absdev_per_ch_uncentered'],
                                onp.mean(onp.abs(self.x3), axis=reduction_axis))

  @parameterized.named_parameters(
      dict(
          testcase_name='no_channel_dim',
          channel_axis=None,
          num_indices_per_ax=5,
          expected_shape=(1,)),
      dict(
          testcase_name='one_channel_dim',
          channel_axis=-1,
          num_indices_per_ax=5,
          expected_shape=(5,)),
      dict(
          testcase_name='two_channel_dims',
          channel_axis=(1, 2),
          num_indices_per_ax=6,
          expected_shape=(6, 6)),
  )
  def test_channel_axis_sampling(self, channel_axis, num_indices_per_ax,
                                 expected_shape):
    model, init_state = self.init_model(
        init_x=self.x4,
        channel_axis=channel_axis,
        update_stats=True,
        num_indices_per_ax=num_indices_per_ax)
    _, state_0 = model.apply(
        init_state, self.x4, mask=None, mutable='stats_tag')
    onp.testing.assert_array_equal(state_0['stats_tag']['min_per_ch'],
                                   onp.ones(expected_shape))


if __name__ == '__main__':
  absltest.main()
