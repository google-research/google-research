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

"""Tests for Monte Carlo simulation manager."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import numpy as np
import tensorflow as tf  # tf

from simulation_research.tf_risk import dynamics
from simulation_research.tf_risk import monte_carlo_manager
from simulation_research.tf_risk import payoffs
from simulation_research.tf_risk import util
from tensorflow.contrib import stateless as contrib_stateless


def _prng_key(i, key):
  return tf.stack([tf.to_int32(key), tf.to_int32(i)], axis=0)


def _random_normal(shape, i, key=0):
  return contrib_stateless.stateless_random_normal(
      shape=shape, seed=_prng_key(i, key))


class MonteCarloManagerTest(tf.test.TestCase, parameterized.TestCase):

  def assertAllAlmostEqual(self, a, b, delta=1e-7):
    self.assertEqual(a.shape, b.shape)
    a = a.flatten()
    b = b.flatten()
    self.assertLessEqual(np.max(np.abs(a - b)), delta)

  def test_max_is_correct_with_float(self):
    x = 0.1
    max_x = monte_carlo_manager._max(x)
    self.assertEqual(x, max_x)

  def test_max_is_correct_with_array(self):
    np.random.seed(0)
    x = np.random.uniform(size=[8])
    max_x = monte_carlo_manager._max(x)
    self.assertEqual(np.max(x), max_x)

  def test_min_is_correct_with_float(self):
    x = 0.1
    min_x = monte_carlo_manager._min(x)
    self.assertEqual(x, min_x)

  def test_min_is_correct_with_array(self):
    np.random.seed(0)
    x = np.random.uniform(size=[8])
    min_x = monte_carlo_manager._min(x)
    self.assertEqual(np.min(x), min_x)

  def test_maybe_tile_tensor_for_mc_tiles_scalar_tensor(self):
    num_samples = 16
    x = 3.14
    tensor_in = tf.constant(x, dtype=tf.float32)
    tensor_out = monte_carlo_manager._maybe_tile_tensor_for_mc(
        tensor_in, num_samples)
    self.assertEqual(tensor_out.shape.as_list(), [num_samples])
    with self.session() as session:
      tensor_out_eval = session.run(tensor_out)
    self.assertAllEqual(tensor_out_eval,
                        np.asarray([x] * num_samples, dtype=np.float32))

  def test_maybe_tile_tensor_for_mc_tiles_1d_tensor(self):
    np.random.seed(0)
    num_samples = 16
    num_dims = 7
    x = np.random.uniform(size=[num_dims])
    tensor_in = tf.constant(x)
    tensor_out = monte_carlo_manager._maybe_tile_tensor_for_mc(
        tensor_in, num_samples)
    self.assertAllEqual(tensor_out.shape.as_list(), [num_samples, num_dims])
    with self.session() as session:
      tensor_out_eval = session.run(tensor_out)
    self.assertAllEqual(tensor_out_eval, np.tile(x, [num_samples, 1]))

  def test_maybe_tile_tensor_for_mc_does_not_tile_2d_tensor(self):
    np.random.seed(0)
    num_samples = 16
    num_dims = 7
    x = np.random.uniform(size=[num_samples, num_dims])
    tensor_in = tf.constant(x)
    tensor_out = monte_carlo_manager._maybe_tile_tensor_for_mc(
        tensor_in, num_samples)
    self.assertAllEqual(tensor_out.shape.as_list(), [num_samples, num_dims])
    with self.session() as session:
      tensor_out_eval = session.run(tensor_out)
    self.assertAllEqual(tensor_out_eval, x)

  def test_reshape_initial_state_for_mc_reshapes_singleton(self):
    np.random.seed(0)
    num_samples = 16
    num_dims = 7
    tensor_in = tf.random_uniform(shape=[num_dims])
    tensor_out = monte_carlo_manager._reshape_initial_state_for_mc(
        tensor_in, num_samples)
    expected_tensor_out = tf.tile(
        tf.expand_dims(tensor_in, 0), [num_samples, 1])
    self.assertAllEqual(tensor_out.shape.as_list(),
                        expected_tensor_out.shape.as_list())
    with self.session() as session:
      (tensor_out_eval, expected_tensor_out_eval) = session.run(
          (tensor_out, expected_tensor_out))
    self.assertAllEqual(tensor_out_eval, expected_tensor_out_eval)

  def test_reshape_initial_state_for_mc_reshapes_tuple(self):
    np.random.seed(0)
    num_samples = 16
    tensors_in = (tf.random_uniform(shape=[3]), tf.random_uniform(shape=[7]),
                  tf.random_uniform(shape=()))
    tensors_out = monte_carlo_manager._reshape_initial_state_for_mc(
        tensors_in, num_samples)
    expected_tensors_out = (monte_carlo_manager._maybe_tile_tensor_for_mc(
        tensors_in[0], num_samples),
                            monte_carlo_manager._maybe_tile_tensor_for_mc(
                                tensors_in[1], num_samples),
                            monte_carlo_manager._maybe_tile_tensor_for_mc(
                                tensors_in[2], num_samples))
    for i in range(len(expected_tensors_out)):
      self.assertAllEqual(tensors_out[i].shape.as_list(),
                          expected_tensors_out[i].shape.as_list())
    with self.session() as session:
      (tensors_out_eval, expected_tensors_out_eval) = session.run(
          (tensors_out, expected_tensors_out))
    for i in range(len(expected_tensors_out)):
      self.assertAllEqual(tensors_out_eval[i], expected_tensors_out_eval[i])

  @parameterized.named_parameters(
      ("unrolled_tf", monte_carlo_manager.multistep_runner_unrolled_tf, {}),
      ("tf_while", monte_carlo_manager.multistep_runner_tf_while, {
          "back_prop": True
      }),
      ("tf_while_no_back_prop", monte_carlo_manager.multistep_runner_tf_while, {
          "back_prop": False
      }),
  )
  def test_multistep_runners_are_correct(self, runner, runner_kwargs):
    """Example: Euler discretization of y'=y, y(0)=1."""
    initial_state = tf.constant(1.0, dtype=tf.float32)
    dynamics_op = lambda x, t, dt: x + (t + dt) + (1.0 * dt)
    dt = 0.05
    duration = 1.0

    multistep_runner = runner(**runner_kwargs)

    while_loop_end_state = multistep_runner(
        dynamics_op=dynamics_op, states=initial_state, dt=dt, duration=duration)

    expected_while_loop_end_state = initial_state
    t = 0.0
    while t < duration:
      expected_while_loop_end_state = dynamics_op(expected_while_loop_end_state,
                                                  t, dt)
      t += dt

    with self.test_session() as session:
      (while_loop_end_state_eval,
       expected_while_loop_end_state_eval) = session.run(
           (while_loop_end_state, expected_while_loop_end_state))

    self.assertAlmostEqual(
        while_loop_end_state_eval,
        expected_while_loop_end_state_eval,
        delta=1e-5)

  def test_multistep_runner_unrolled_tf_is_correct_with_trajectories(self):
    initial_state = tf.constant(1.0, dtype=tf.float32)
    dynamics_op = lambda x, t, dt: x + (t + dt) + (1.0 * dt)
    dt = 0.05
    duration = 1.0

    multistep_runner = monte_carlo_manager.multistep_runner_unrolled_tf(
        return_trajectories=True)

    while_loop_trajectories = multistep_runner(
        dynamics_op=dynamics_op, states=initial_state, dt=dt, duration=duration)

    expected_while_loop_trajectories = [initial_state]
    t = 0.0
    while t < duration:
      expected_while_loop_trajectories.append(
          dynamics_op(expected_while_loop_trajectories[-1], t, dt))
      t += dt

    with self.test_session() as session:
      (while_loop_trajectories_eval,
       expected_while_loop_trajectories_eval) = session.run(
           (while_loop_trajectories, expected_while_loop_trajectories))

    self.assertAlmostEqual(
        while_loop_trajectories_eval,
        expected_while_loop_trajectories_eval,
        delta=1e-5)

  @parameterized.named_parameters(
      ("unrolled_loop", monte_carlo_manager.multistep_runner_unrolled_tf()),
      ("tf_while_loop", monte_carlo_manager.multistep_runner_tf_while()),
  )
  def test_runners_are_deterministic_with_stateless_prng(self, runner):
    """Example with a Brownian Motion, checks issues with prng inside loop."""
    num_samples = 32
    initial_state = tf.zeros(num_samples)
    dynamics_op = lambda x, t, dt: x + _random_normal([num_samples], t)
    dt = 1.0
    duration = 10.0

    while_loop_end_state = runner(
        dynamics_op=dynamics_op, states=initial_state, dt=dt, duration=duration)

    expected_while_loop_end_state = initial_state
    t = 0.0
    while t < duration:
      expected_while_loop_end_state = dynamics_op(expected_while_loop_end_state,
                                                  t, dt)
      t += dt

    with self.test_session() as session:
      (while_loop_end_state_eval,
       expected_while_loop_end_state_eval) = session.run(
           (while_loop_end_state, expected_while_loop_end_state))

    self.assertAllAlmostEqual(
        while_loop_end_state_eval,
        expected_while_loop_end_state_eval,
        delta=1e-5)

  def test_european_call_euler_mc_close_to_black_scholes(self):
    current_price = 100.0
    r = interest_rate = 0.05
    vol = 0.2
    strike = 120.0
    maturity = 0.5
    dt = 0.01
    discount = tf.exp(-r * maturity)

    bs_call_price = util.black_scholes_call_price(current_price, interest_rate,
                                                  vol, strike, maturity)

    num_samples = int(1e4)
    initial_state = tf.constant(current_price)

    dynamics_op = lambda s, t, dt: dynamics.gbm_euler_step(s, r, vol, t, dt)
    payoff_fn = lambda s: discount * payoffs.call_payoff(s, strike)

    (mean_outcome, mean_sq_outcome,
     _) = monte_carlo_manager.non_callable_price_mc(initial_state, dynamics_op,
                                                    payoff_fn, maturity,
                                                    num_samples, dt)

    std_outcomes = util.stddev_est(mean_outcome, mean_sq_outcome)

    with self.test_session() as session:
      bs_call_price_eval = session.run(bs_call_price)
      mean_outcome_eval, std_outcomes_eval = session.run(
          (mean_outcome, std_outcomes))

    self.assertLessEqual(
        mean_outcome_eval,
        bs_call_price_eval + 3.0 * std_outcomes_eval / np.sqrt(num_samples))
    self.assertGreaterEqual(
        mean_outcome_eval,
        bs_call_price_eval - 3.0 * std_outcomes_eval / np.sqrt(num_samples))

  def test_european_call_log_euler_mc_close_to_black_scholes(self):
    current_price = 100.0
    r = interest_rate = 0.05
    vol = 0.2
    strike = 120.0
    maturity = 0.5
    dt = 0.01
    discount = tf.exp(-r * maturity)

    bs_call_price = util.black_scholes_call_price(current_price, interest_rate,
                                                  vol, strike, maturity)

    num_samples = int(1e4)
    initial_state = tf.constant(current_price)

    dynamics_op = lambda s, t, dt: dynamics.gbm_log_euler_step(s, r, vol, t, dt)
    payoff_fn = lambda s: discount * payoffs.call_payoff(tf.exp(s), strike)

    (mean_outcome, mean_sq_outcome,
     _) = monte_carlo_manager.non_callable_price_mc(
         tf.log(initial_state), dynamics_op, payoff_fn, maturity, num_samples,
         dt)

    std_outcomes = util.stddev_est(mean_outcome, mean_sq_outcome)

    with self.test_session() as session:
      bs_call_price_eval = session.run(bs_call_price)
      mean_outcome_eval, std_outcomes_eval = session.run(
          (mean_outcome, std_outcomes))

    self.assertLessEqual(
        mean_outcome_eval,
        bs_call_price_eval + 3.0 * std_outcomes_eval / np.sqrt(num_samples))
    self.assertGreaterEqual(
        mean_outcome_eval,
        bs_call_price_eval - 3.0 * std_outcomes_eval / np.sqrt(num_samples))

  def test_european_derivative_price_delta_mc_close_to_black_scholes(self):
    current_price = tf.constant(100.0, dtype=tf.float32)
    r = 0.05
    vol = tf.constant([[0.2]], dtype=tf.float32)
    strike = 100.0
    maturity = 0.1
    dt = 0.01

    bs_call_price = util.black_scholes_call_price(current_price, r, vol, strike,
                                                  maturity)
    bs_delta = monte_carlo_manager.sensitivity_autodiff(bs_call_price,
                                                        current_price)

    num_samples = int(1e3)
    initial_states = tf.ones([num_samples, 1]) * current_price

    def _dynamics_op(log_s, t, dt):
      return dynamics.gbm_log_euler_step_nd(log_s, r, vol, t, dt, key=1)

    def _payoff_fn(log_s):
      return tf.exp(-r * maturity) * payoffs.call_payoff(tf.exp(log_s), strike)

    (_, _, outcomes) = monte_carlo_manager.non_callable_price_mc(
        tf.log(initial_states), _dynamics_op, _payoff_fn, maturity, num_samples,
        dt)

    mc_deltas = monte_carlo_manager.sensitivity_autodiff(
        outcomes, initial_states)

    mean_mc_deltas = tf.reduce_mean(mc_deltas)
    mc_deltas_std = tf.sqrt(tf.reduce_mean(mc_deltas**2) - (mean_mc_deltas**2))

    with self.test_session() as session:
      bs_delta_eval = session.run(bs_delta)
      mean_mc_deltas_eval, mc_deltas_std_eval = session.run(
          (mean_mc_deltas, mc_deltas_std))

    self.assertLessEqual(
        mean_mc_deltas_eval,
        bs_delta_eval + 3.0 * mc_deltas_std_eval / np.sqrt(num_samples))
    self.assertGreaterEqual(
        mean_mc_deltas_eval,
        bs_delta_eval - 3.0 * mc_deltas_std_eval / np.sqrt(num_samples))

  def test_mc_estimator_converges_with_fixed_estimates(self):
    mean = 1.3
    stddev = 2.0
    batch_size = int(1e5)
    dummy_key_placeholder = tf.placeholder(shape=(), dtype=tf.int32)

    mean_est = tf.constant(mean)
    mean_sq_est = tf.constant(stddev**2)

    tol = 5e-3
    conf_level = 0.99

    max_num_steps = (util.half_clt_conf_interval(conf_level, 1, stddev) /
                     (mean * tol))**2 + 1

    with self.test_session() as session:
      (mean_est_eval, rel_half_conf_interval,
       converged) = monte_carlo_manager.mc_estimator(mean_est, mean_sq_est,
                                                     batch_size,
                                                     dummy_key_placeholder, {},
                                                     tol, conf_level,
                                                     max_num_steps, session)

    self.assertAlmostEqual(mean_est_eval, mean)
    self.assertTrue(converged)
    self.assertGreaterEqual(
        rel_half_conf_interval,
        util.half_clt_conf_interval(conf_level, max_num_steps * batch_size,
                                    stddev) / mean)

  def test_mc_estimator_converges_with_fixed_estimates_nd(self):
    num_dims = 8
    mean = 1.3
    stddev = 2.0
    batch_size = int(1e5)
    dummy_key_placeholder = tf.placeholder(shape=(), dtype=tf.int32)

    mean_est = tf.ones([num_dims]) * mean
    mean_sq_est = tf.ones([num_dims]) * (stddev**2)

    tol = 5e-3
    conf_level = 0.99

    max_num_steps = (util.half_clt_conf_interval(conf_level, 1, stddev) /
                     (mean * tol))**2 + 1

    with self.test_session() as session:
      (mean_est_eval, rel_half_conf_interval,
       converged) = monte_carlo_manager.mc_estimator(mean_est, mean_sq_est,
                                                     batch_size,
                                                     dummy_key_placeholder, {},
                                                     tol, conf_level,
                                                     max_num_steps, session)

    self.assertAllAlmostEqual(mean_est_eval, mean * np.ones([num_dims]))
    self.assertTrue(converged)
    self.assertAllGreaterEqual(
        rel_half_conf_interval,
        util.half_clt_conf_interval(conf_level, max_num_steps * batch_size,
                                    stddev) / mean)

  def test_mc_estimator_converges_with_multiple_fixed_estimates_nd(self):
    num_dims = 8
    mean = np.arange(1.27, 1.349, 0.01)
    stddev = 2.0
    batch_size = int(1e5)
    dummy_key_placeholder = tf.placeholder(shape=(), dtype=tf.int32)

    mean_est = tf.constant(mean)
    mean_sq_est = tf.ones([num_dims]) * (stddev**2)

    tol = 5e-3
    conf_level = 0.99

    max_num_steps = (util.half_clt_conf_interval(conf_level, 1, stddev) /
                     (np.min(mean) * tol))**2 + 1
    with self.test_session() as session:
      (mean_est_eval, rel_half_conf_interval,
       converged) = monte_carlo_manager.mc_estimator(mean_est, mean_sq_est,
                                                     batch_size,
                                                     dummy_key_placeholder, {},
                                                     tol, conf_level,
                                                     max_num_steps, session)

    self.assertAllAlmostEqual(mean_est_eval, mean * np.ones([num_dims]))
    self.assertTrue(converged)
    self.assertAllGreaterEqual(
        rel_half_conf_interval * mean,
        util.half_clt_conf_interval(conf_level, max_num_steps * batch_size,
                                    stddev))

  def test_european_call_estimator_converges_close_to_black_scholes(self):
    current_price = 100.0
    r = interest_rate = 0.05
    vol = 0.2
    strike = 120.0
    maturity = 0.5
    dt = 0.001
    discount = tf.exp(-r * maturity)

    tol = 5e-2
    conf_level = 0.95
    batch_size = int(1e4)
    k = key_placeholder = tf.placeholder(shape=(), dtype=tf.int32)
    max_num_steps = 1e5

    bs_call_price = util.black_scholes_call_price(current_price, interest_rate,
                                                  vol, strike, maturity)

    initial_state = tf.constant(current_price)

    dynamics_op = lambda s, t, dt: dynamics.gbm_euler_step(s, r, vol, t, dt, k)
    payoff_fn = lambda s: discount * payoffs.call_payoff(s, strike)

    (mean_est, mean_sq_est,
     _) = monte_carlo_manager.non_callable_price_mc(initial_state, dynamics_op,
                                                    payoff_fn, maturity,
                                                    batch_size, dt)

    with self.test_session() as session:
      (mean_est_eval, _, converged) = monte_carlo_manager.mc_estimator(
          mean_est, mean_sq_est, batch_size, key_placeholder, {}, tol,
          conf_level, max_num_steps, session)

      bs_call_price_eval = session.run(bs_call_price)

    self.assertTrue(converged)
    # Here the discretization bias would make these asserts fail with larger dt.
    self.assertLessEqual(mean_est_eval, bs_call_price_eval * (1.0 + tol))
    self.assertGreaterEqual(mean_est_eval, bs_call_price_eval * (1.0 - tol))


if __name__ == "__main__":
  tf.test.main()
