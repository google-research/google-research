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

"""Tests for aqt.jax.stats."""

from absl.testing import absltest
from absl.testing import parameterized
from jax import random
import jax.numpy as jnp
import numpy as onp

from aqt.jax import test_utils
from aqt.jax.stats import masked_mean
from aqt.jax.stats import masked_sum
from aqt.jax.stats import Stats

test_utils.configure_jax()


class StatsTest(parameterized.TestCase):

  def setUp(self):
    super(StatsTest, self).setUp()
    self.key = random.PRNGKey(0)

  def test_create_empty_stats_using_initializer(self):
    stats = Stats.stats_initializer(shape=())
    self.assertEqual(stats.n, 0)
    self.assertEqual(stats.mean, 0.)
    self.assertEqual(stats.mean_abs, 0.)
    self.assertEqual(stats.mean_sq, 0.)

  # TODO(wanglisa/shivaniagrawal): parametrize tests for values of alpha.
  @parameterized.named_parameters(
      dict(
          testcase_name='axisnone_1dim',
          axis=None,
          sample_shape=(2,),
          stats_shape=(1,),
      ),
      dict(
          testcase_name='axisnone_2dim',
          axis=None,
          sample_shape=(2, 3),
          stats_shape=(1, 1),
      ),
      dict(
          testcase_name='axisnone_3dim',
          axis=None,
          sample_shape=(2, 3, 4),
          stats_shape=(1, 1, 1),
      ),
      dict(
          testcase_name='axis0_2dim',
          axis=0,
          sample_shape=(2, 3),
          stats_shape=(1, 3),
      ),
      dict(
          testcase_name='axis0_3dim',
          axis=0,
          sample_shape=(2, 3, 4),
          stats_shape=(1, 3, 4),
      ),
      dict(
          testcase_name='axis0-1_3dim',
          axis=(0, 1),
          sample_shape=(2, 3, 4),
          stats_shape=(1, 1, 4),
      ),
  )
  def test_update_stats_with_different_axes(self, axis, sample_shape,
                                            stats_shape):
    stats = Stats.stats_initializer(shape=stats_shape)
    for i in range(-1, 4):
      x = i * jnp.ones(sample_shape)
      stats = Stats.create_updated_stats(stats, x, axis=axis)
    self.assertEqual(stats.n, 5)

    exp_mean = (-1 + 0 + 1 + 2 + 3) / 5. * jnp.ones(stats_shape)
    onp.testing.assert_allclose(stats.mean, exp_mean)

    exp_mean_abs = (1 + 0 + 1 + 2 + 3) / 5. * jnp.ones(stats_shape)
    onp.testing.assert_allclose(stats.mean_abs, exp_mean_abs)

    exp_mean_sq = (
        (-1)**2 + 0**2 + 1**2 + 2**2 + 3**2) / 5. * jnp.ones(stats_shape)
    onp.testing.assert_allclose(stats.mean_sq, exp_mean_sq)

  def test_per_channel_average(self):
    """Stats should be different per channel."""
    stats = Stats.stats_initializer(shape=())
    for i in range(-1, 4):
      x = i * jnp.array([[1., 2.], [2., 4.], [3., 6.]])
      stats = Stats.create_updated_stats(stats, x, axis=(0,))
    self.assertEqual(stats.n, 5)

    # For i in range(-1, 4), ith array would be
    # [[i    , 2 * i]
    #  [i * 2, 2 * (i * 2)]
    #. [i * 3, 2 * (i * 3)]]

    exp_mean_ch0 = (-1 + 0 + 1 + 2 + 3) * (1 + 2 + 3) / 15.
    exp_mean_ch1 = (-2 + 0 + 2 + 4 + 6) * (1 + 2 + 3) / 15.
    exp_mean = jnp.array([[exp_mean_ch0, exp_mean_ch1]])
    onp.testing.assert_allclose(stats.mean, exp_mean)

    exp_mean_abs_ch0 = (1 + 0 + 1 + 2 + 3) * (1 + 2 + 3) / 15.
    exp_mean_abs_ch1 = (2 + 0 + 2 + 4 + 6) * (1 + 2 + 3) / 15.
    exp_mean_abs = jnp.array([[exp_mean_abs_ch0, exp_mean_abs_ch1]])
    onp.testing.assert_allclose(stats.mean_abs, exp_mean_abs)

    exp_mean_sq_ch0 = (
        (-1)**2 + 0**2 + 1**2 + 2**2 + 3**2) * (1**2 + 2**2 + 3**2) / 15.
    exp_mean_sq_ch1 = (
        (-2)**2 + 0**2 + 2**2 + 4**2 + 6**2) * (1**2 + 2**2 + 3**2) / 15.
    exp_mean_sq = jnp.array([[exp_mean_sq_ch0, exp_mean_sq_ch1]])
    onp.testing.assert_allclose(stats.mean_sq, exp_mean_sq)

  def test_masking(self):
    # We will simulate a situation where we have two batches with two tokens
    # each, and the second token of the second batch is padding. The channel
    # dimension is three.

    stats = Stats.stats_initializer(shape=(1, 1, 2))
    # The shape of 'x' is [batch index, token index, channel index]
    x = jnp.reshape(jnp.arange(8), (2, 2, 2)).astype(jnp.float32)
    token_mask = jnp.array([[True, True], [True, False]])
    mask = token_mask[Ellipsis,
                      None]  # Broadcast the mask over the channel dimension
    stats = Stats.create_updated_stats(stats, x, axis=(0, 1), mask=mask)
    exp_mean_ch0 = (0 + 2 + 4) / 3
    exp_mean_ch1 = (1 + 3 + 5) / 3
    exp_mean = jnp.array([[[exp_mean_ch0, exp_mean_ch1]]])
    onp.testing.assert_allclose(stats.mean, exp_mean)
    onp.testing.assert_allclose(stats.mean_abs, exp_mean)
    exp_mean_sq_ch0 = (0**2 + 2**2 + 4**2) / 3
    exp_mean_sq_ch1 = (1**2 + 3**2 + 5**2) / 3
    exp_mean_sq = jnp.array([[[exp_mean_sq_ch0, exp_mean_sq_ch1]]])
    onp.testing.assert_allclose(stats.mean_sq, exp_mean_sq)
    exp_max = [[[4, 5]]]
    onp.testing.assert_allclose(stats.mean_batch_maximum, exp_max)
    exp_min = [[[0, 1]]]
    onp.testing.assert_allclose(stats.mean_batch_minimum, exp_min)

    # Now do the same, but with axis=None
    stats = Stats.stats_initializer(shape=())
    stats = Stats.create_updated_stats(stats, x, axis=None, mask=mask)
    exp_mean = (0 + 1 + 2 + 3 + 4 + 5) / 6
    onp.testing.assert_allclose(stats.mean, [[[exp_mean]]])
    exp_mean_sq = (0**2 + 1**2 + 2**2 + 3**2 + 4**2 + 5**2) / 6
    onp.testing.assert_allclose(stats.mean_sq, [[[exp_mean_sq]]])
    onp.testing.assert_allclose(stats.mean_batch_maximum, [[[5]]])
    onp.testing.assert_allclose(stats.mean_batch_minimum, [[[0]]])

    # Also try with reduction axis equal to the broadcasting axis.
    # In this case, we expect a 0 when taking the mean over the
    # array slice that consists solely of masked elements, since only masked
    # elements will not update the initial value of 0.
    stats = Stats.stats_initializer(shape=(2, 2, 1))
    stats = Stats.create_updated_stats(stats, x, axis=(2,), mask=mask)
    exp_mean = [[[(0 + 1) / 2], [(2 + 3) / 2]], [[(4 + 5) / 2], [0]]]
    onp.testing.assert_allclose(stats.mean, exp_mean)
    exp_mean_sq = [[[(0**2 + 1**2) / 2], [(2**2 + 3**2) / 2]],
                   [[(4**2 + 5**2) / 2], [0]]]
    onp.testing.assert_allclose(stats.mean_sq, exp_mean_sq)
    exp_max = [[[1], [3]], [[5], [0]]]
    onp.testing.assert_allclose(stats.mean_batch_maximum, exp_max)
    exp_min = [[[0], [2]], [[4], [0]]]
    onp.testing.assert_allclose(stats.mean_batch_minimum, exp_min)

  @parameterized.named_parameters(
      dict(
          testcase_name='axis_none',
          axis=None,
          exp_masked_sum=(0 + 1 + 2 + 3 + 4 + 5),
          exp_masked_mean=(0 + 1 + 2 + 3 + 4 + 5) / 6,
      ),
      dict(
          testcase_name='axis_01',
          axis=(0, 1),
          exp_masked_sum=onp.array([0 + 2 + 4, 1 + 3 + 5]),
          exp_masked_mean=onp.array([(0 + 2 + 4) / 3, (1 + 3 + 5) / 3]),
      ),
      dict(
          testcase_name='axis_0',
          axis=(0,),
          exp_masked_sum=onp.array([[0 + 4, 1 + 5], [2 + 0, 3 + 0]]),
          exp_masked_mean=onp.array([[(0 + 4) / 2, (1 + 5) / 2],
                                     [(2 + 0) / 1, (3 + 0) / 1]]),
      ),
  )
  def test_masked_sum_and_masked_mean(self, axis, exp_masked_sum,
                                      exp_masked_mean):
    x = jnp.reshape(jnp.arange(8), (2, 2, 2)).astype(jnp.float32)
    mask = jnp.array([[[True, True], [True, True]],
                      [[True, True], [False, False]]])
    masked_sum_res = masked_sum(
        x, mask=mask, axis=axis, paxis_name=None, keepdims=False)
    onp.testing.assert_allclose(masked_sum_res, exp_masked_sum)

    masked_mean_res = masked_mean(
        x, mask=mask, axis=axis, paxis_name=None, keepdims=False)
    onp.testing.assert_allclose(masked_mean_res, exp_masked_mean)

  @parameterized.named_parameters(
      dict(
          testcase_name='axis_none_no_mask',
          x=onp.reshape(onp.arange(4), (2, 2)).astype(onp.float32),
          mask=None,
          axis=None,
          exp_mean=[[(1 + 2 + 3) / 3]],
      ),
      dict(
          testcase_name='axis_none_with_mask',
          x=onp.reshape(onp.arange(8), (2, 2, 2)).astype(onp.float32),
          mask=onp.array([[[True, True], [True, True]],
                          [[True, True], [False, False]]]),
          axis=None,
          exp_mean=[[[(1 + 2 + 3 + 4 + 5) / 5]]],
      ),
      # Because mean of x will have div by 0, it's an invalid mean, so
      # it should not update the initial value of 0.
      dict(
          testcase_name='div_by_0_axis_none_no_mask',
          x=onp.array([0, 0]).astype(onp.float32),
          mask=None,
          axis=None,
          exp_mean=[0],
      ),
      dict(
          testcase_name='div_by_0_axis_none_with_mask',
          x=onp.array([0, 1]).astype(onp.float32),
          mask=onp.array([True, False]),
          axis=None,
          exp_mean=[0],
      ),
      dict(
          testcase_name='2dim_div_by_0_axis_0_no_mask',
          x=onp.array([[0, 1], [0, 0]]).astype(onp.float32),
          mask=None,
          axis=(0,),
          exp_mean=onp.array([[0, (1 + 0) / 1]]),
      ),
      # Because mean[0] of x will have div by 0, it's an invalid mean, so
      # it should not update the initial value of 0.
      dict(
          testcase_name='2dim_div_by_0_axis_0_with_mask',
          x=onp.array([[0, 1], [2, 3]]).astype(onp.float32),
          mask=onp.array([[True, True], [False, True]]),
          axis=(0,),
          exp_mean=onp.array([[0, (1 + 3) / 2]]),
      ),
  )
  def test_exclude_zeros(self, x, mask, axis, exp_mean):
    """Stats should be different when excluding zeros."""
    stats = Stats.stats_initializer(shape=())
    stats = Stats.create_updated_stats(
        stats, x, axis=axis, mask=mask, exclude_zeros=True)
    onp.testing.assert_allclose(stats.mean, exp_mean)


if __name__ == '__main__':
  absltest.main()
