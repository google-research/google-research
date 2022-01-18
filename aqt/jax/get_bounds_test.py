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

"""Tests for aqt.jax.get_bounds."""

from absl.testing import absltest
from absl.testing import parameterized
import flax
from jax import random
import jax.numpy as jnp
import numpy as onp

from aqt.jax import get_bounds
from aqt.jax import quant_config
from aqt.jax import test_utils

test_utils.configure_jax()


class GetBoundsTest(parameterized.TestCase):

  def setUp(self):
    super(GetBoundsTest, self).setUp()
    self.rng = random.PRNGKey(0)
    key1, key2 = random.split(self.rng)
    self.key2 = key2
    self.x = random.normal(key1, (4, 3, 2))
    self.x2 = jnp.ones((4, 3, 2))
    self.hyperparam = get_bounds.GetBounds.Hyper(
        initial_bound=6.0,
        stddev_coeff=2.0,
        absdev_coeff=1.5,
        mix_coeff=0.7,
        reset_stats=False,
        granularity=quant_config.QuantGranularity.per_channel)

  def init_model(self,
                 update_bounds,
                 update_stats=True,
                 reset_stats=False,
                 use_cams=False,
                 granularity=quant_config.QuantGranularity.per_tensor,
                 ema_coeff=None):
    self.hyperparam = get_bounds.GetBounds.Hyper(
        initial_bound=self.hyperparam.initial_bound,
        stddev_coeff=self.hyperparam.stddev_coeff,
        absdev_coeff=self.hyperparam.absdev_coeff,
        mix_coeff=self.hyperparam.mix_coeff,
        reset_stats=reset_stats,
        use_cams=use_cams,
        ema_coeff=ema_coeff,
        granularity=granularity)
    gb_bounds_params = get_bounds.GetBounds.Params(
        update_bounds=update_bounds, update_stats=update_stats)
    bounds_module = get_bounds.GetBounds(hyper=self.hyperparam)
    init_state = bounds_module.init(
        self.key2, self.x, bounds_params=gb_bounds_params)
    return bounds_module, init_state, gb_bounds_params

  # TODO(shivaniagrawal): parametrize test for different values of axis
  @parameterized.named_parameters(
      dict(testcase_name='update_bound', update_bound=True),
      dict(testcase_name='do_not_update', update_bound=False),
  )
  def test_get_bounds_init(self, update_bound):
    _, init_state, _ = self.init_model(update_bound)
    init_state_stats = init_state['get_bounds']['stats']
    onp.testing.assert_array_equal(init_state_stats.n, 0)
    onp.testing.assert_array_equal(init_state_stats.mean, 0)
    onp.testing.assert_array_equal(init_state_stats.mean_abs, 0)
    onp.testing.assert_array_equal(init_state_stats.mean_sq, 0)
    onp.testing.assert_array_equal(init_state['get_bounds']['bounds'], 6.)

  # TODO(shivaniagrawal): more elaborate testing here as follows:
  # - run with (update_stats, update_bounds) = (False, False)
  # check that neither state changed
  # - run with (update_stats, update_bounds) = (True, False)
  # check that stats.n increased, bound unchanged
  # - run with (update_stats, update_bounds) = (False, True)
  # check that stats.n unchanged but bound updated.
  # - run again with (update_stats, update_bounds) = (False, True)
  # check that both unchanged (update_bounds is idempotent)
  # - run again with (update_stats, update_bounds) = (True, True)
  # check that both changed.

  @parameterized.named_parameters(
      dict(
          testcase_name='update_bound_reset_stats',
          update_bound=True,
          reset_stats=True),
      dict(
          testcase_name='no_update_bound_reset_stats',
          update_bound=False,
          reset_stats=True),
      dict(
          testcase_name='update_bound_no_reset_stats',
          update_bound=True,
          reset_stats=False),
      dict(
          testcase_name='no_update_bound_no_reset_stats',
          update_bound=False,
          reset_stats=False),
  )
  def test_update_stats(self, update_bound, reset_stats):
    model, init_state, params = self.init_model(
        update_bound,
        reset_stats=reset_stats,
        granularity=quant_config.QuantGranularity.per_tensor)
    _, state_0 = model.apply(
        init_state, self.x, bounds_params=params, mutable='get_bounds')

    stats_0_stats = state_0['get_bounds']['stats']
    if reset_stats and update_bound:
      onp.testing.assert_array_equal(stats_0_stats.n, 0)
    else:
      onp.testing.assert_array_equal(stats_0_stats.n, 1)

    _, state = model.apply(
        state_0, self.x2, bounds_params=params, mutable='get_bounds')
    stats = state['get_bounds']['stats']
    if reset_stats and update_bound:
      onp.testing.assert_array_equal(stats.n, 0)
      expected_updated_mean = 0.
      onp.testing.assert_array_equal(expected_updated_mean, stats.mean)
    else:
      onp.testing.assert_array_equal(stats.n, 2)
      expected_updated_mean = 1 / 2 * (1 + (stats_0_stats.mean))
      onp.testing.assert_array_equal(expected_updated_mean, stats.mean)

  @parameterized.named_parameters(
      dict(
          testcase_name='update_bound_reset_stats',
          update_bound=True,
          reset_stats=True),
      dict(
          testcase_name='no_update_bound_reset_stats',
          update_bound=False,
          reset_stats=True),
      dict(
          testcase_name='update_bound_no_reset_stats',
          update_bound=True,
          reset_stats=False),
      dict(
          testcase_name='no_update_bound_no_reset_stats',
          update_bound=False,
          reset_stats=False),
  )
  def test_update_stats_false(self, update_bound, reset_stats):
    model, init_state, params = self.init_model(
        update_bound, update_stats=False, reset_stats=reset_stats)
    _, state_0 = model.apply(
        init_state, self.x, bounds_params=params, mutable='get_bounds')

    stats_0_stats = state_0['get_bounds']['stats']
    onp.testing.assert_array_equal(stats_0_stats.n, 0)

    _, state = model.apply(
        state_0, self.x2, bounds_params=params, mutable='get_bounds')
    onp.testing.assert_array_equal(state['get_bounds']['stats'].n, 0)
    expected_updated_mean = 0.
    onp.testing.assert_array_equal(expected_updated_mean,
                                   state['get_bounds']['stats'].mean)

  @parameterized.named_parameters(
      dict(
          testcase_name='update_bounds_true',
          update_stats=False,
          update_bounds=True),
      dict(
          testcase_name='update_stats_true',
          update_stats=True,
          update_bounds=False),
      dict(testcase_name='both_true', update_stats=True, update_bounds=True),
  )
  def test_update_state_with_mutable_false_context_raises_error(
      self, update_stats, update_bounds):
    model, init_state, _ = self.init_model(True)

    with self.assertRaises(flax.errors.ModifyScopeVariableError):
      model.apply(
          init_state,
          self.x2,
          bounds_params=get_bounds.GetBounds.Params(
              update_stats=update_stats, update_bounds=update_bounds),
          mutable=False)

  @parameterized.named_parameters(
      dict(
          testcase_name='update_bound_no_ucb',
          update_bound=True,
          use_cams=False),
      dict(
          testcase_name='update_bound_with_ucb',
          update_bound=True,
          use_cams=True),
      dict(testcase_name='do_not_update', update_bound=False, use_cams=False),
  )
  def test_get_bounds_update_bounds(self, update_bound, use_cams=False):
    model, init_state, params = self.init_model(update_bound, use_cams=use_cams)
    y, state_0 = model.apply(
        init_state, self.x, bounds_params=params, mutable='get_bounds')
    if not update_bound:
      onp.testing.assert_array_equal(state_0['get_bounds']['bounds'], 6.)
      onp.testing.assert_array_equal(y, 6.)
    else:
      stats_0_stats = state_0['get_bounds']['stats']
      if use_cams:
        expected_y = onp.abs(onp.mean(
            self.x)) + self.hyperparam.stddev_coeff * onp.std(self.x)
      else:
        expected_y = (
            self.hyperparam.stddev_coeff * self.hyperparam.mix_coeff *
            jnp.sqrt(stats_0_stats.mean_sq) + self.hyperparam.absdev_coeff *
            (1 - self.hyperparam.mix_coeff) * stats_0_stats.mean_abs)
      onp.testing.assert_array_equal(state_0['get_bounds']['bounds'], y)
      onp.testing.assert_allclose(expected_y, y)

    y2, state = model.apply(
        state_0, self.x2, bounds_params=params, mutable='get_bounds')
    onp.testing.assert_array_equal(state['get_bounds']['bounds'], y2)

  @parameterized.named_parameters(
      dict(testcase_name='no_ema', ema_coeff=None),
      dict(testcase_name='ema_.8', ema_coeff=0.8),
      dict(testcase_name='ema_.1', ema_coeff=0.1))
  def test_ema_coeff(self, ema_coeff):
    x1 = jnp.array(1.0)
    x2 = jnp.array(-2.0)
    model, state, params = self.init_model(False, ema_coeff=ema_coeff)
    _, state1 = model.apply(
        state, x1, bounds_params=params, mutable='get_bounds')
    _, state2 = model.apply(
        state1, x2, bounds_params=params, mutable='get_bounds')
    stats = state2['get_bounds']['stats']

    def compute_ema_two_steps(x1, x2, alpha):
      initial_value = 0.0
      ema_step_1 = initial_value + alpha * (x1 - initial_value)
      ema_step_2 = ema_step_1 + alpha * (x2 - ema_step_1)
      return ema_step_2

    if ema_coeff is None:
      exp_mean = (x1 + x2) / 2
      exp_mean_sq = (x1**2 + x2**2) / 2
      exp_mean_abs = (jnp.abs(x1) + jnp.abs(x2)) / 2
    else:
      exp_mean = compute_ema_two_steps(x1, x2, ema_coeff)
      exp_mean_sq = compute_ema_two_steps(x1**2, x2**2, ema_coeff)
      exp_mean_abs = compute_ema_two_steps(jnp.abs(x1), jnp.abs(x2), ema_coeff)
    onp.testing.assert_allclose(stats.mean, exp_mean)
    onp.testing.assert_allclose(stats.mean_sq, exp_mean_sq)
    onp.testing.assert_allclose(stats.mean_abs, exp_mean_abs)
    print(stats)
    return


if __name__ == '__main__':
  absltest.main()
