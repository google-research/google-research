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

"""Tests for risk metrics estimators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.stats
import tensorflow as tf  # tf

from simulation_research.tf_risk import dynamics
from simulation_research.tf_risk import monte_carlo_manager
from simulation_research.tf_risk import risk_metrics


class RiskMetricsTest(tf.test.TestCase):

  def test_var_has_increasing_estimates_with_increasing_levels(self):
    np.random.seed(0)

    var_levels = [0.95, 0.98, 0.999]

    drift = np.asarray([0.1, 0.3, -0.05], dtype=np.float32)
    vol_matrix = 0.2 * np.asarray(
        [[1.5, 0.2, 0.3], [0.2, 1.1, -0.1], [0.3, -0.1, 0.8]], dtype=np.float32)
    maturity = 1.0
    dt = 0.01
    num_samples = 8
    num_dims = drift.shape[0]

    key_placeholder = tf.placeholder(shape=(), dtype=tf.int32)

    def _dynamics_op(s, t, dt):
      return dynamics.gbm_euler_step_nd(s, drift, vol_matrix, t, dt)

    initial_state = tf.ones([num_dims]) * 100.0
    initial_pf_value = tf.reduce_sum(initial_state)

    payoff_fn = lambda s: tf.reduce_sum(s, axis=-1) - initial_pf_value

    _, _, portfolio_returns = monte_carlo_manager.non_callable_price_mc(
        initial_state=initial_state,
        dynamics_op=_dynamics_op,
        payoff_fn=payoff_fn,
        maturity=maturity,
        num_samples=num_samples,
        dt=dt)

    var_ests, cvar_ests = risk_metrics.var_and_cvar(portfolio_returns,
                                                    var_levels, key_placeholder,
                                                    {})

    self.assertLen(var_ests, len(var_levels))
    self.assertLen(cvar_ests, len(var_levels))

    for i in range(len(var_ests) - 1):
      self.assertLessEqual(var_ests[i], var_ests[i + 1])
      self.assertLessEqual(cvar_ests[i], cvar_ests[i + 1])

  def test_var_in_normal_case(self):
    np.random.seed(0)

    var_levels = [0.95, 0.98, 0.999]

    key_placeholder = tf.placeholder(shape=(), dtype=tf.int32)

    num_samples = int(1e6)
    maturity = 1.0
    dt = 1.0

    def _dynamics_op(unused_s, t, dt):
      return dynamics.random_normal([num_samples], t, dt)

    payoff_fn = tf.identity

    initial_state = tf.constant(0.0)

    _, _, portfolio_returns = monte_carlo_manager.non_callable_price_mc(
        initial_state=initial_state,
        dynamics_op=_dynamics_op,
        payoff_fn=payoff_fn,
        maturity=maturity,
        num_samples=num_samples,
        dt=dt)

    var_ests, cvar_ests = risk_metrics.var_and_cvar(portfolio_returns,
                                                    var_levels, key_placeholder,
                                                    {})

    self.assertLen(var_ests, len(var_levels))
    self.assertLen(cvar_ests, len(var_levels))

    for i in range(len(var_ests) - 1):
      self.assertLessEqual(var_ests[i], var_ests[i + 1])
      self.assertLessEqual(cvar_ests[i], cvar_ests[i + 1])

    for i, var_level in enumerate(var_levels):
      self.assertAlmostEqual(
          var_ests[i], -scipy.stats.norm.isf(var_level), delta=1e-2)


if __name__ == "__main__":
  tf.test.main()
