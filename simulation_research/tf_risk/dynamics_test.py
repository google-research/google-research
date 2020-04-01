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

"""Tests for stochastic dynamics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf  # tf

from simulation_research.tf_risk import dynamics
from tensorflow.contrib import stateless as contrib_stateless


class DynamicsTest(tf.test.TestCase):

  def assertAllDistinct(self, a, b):
    self.assertEqual(a.shape, b.shape)
    a = a.flatten()
    b = b.flatten()
    for i in range(len(a)):
      self.assertNotEqual(a[i], b[i])

  def test_antithetic_uniform_is_symmetrical(self):
    shape = [512]

    antithetic_uniform_samples = dynamics.random_antithetic_uniform(shape)

    with self.session() as session:
      [samples,
       sym_samples] = session.run(tf.split(antithetic_uniform_samples, 2))

    self.assertAllEqual(samples, 1.0 - sym_samples)

  def test_antithetic_uniform_lowers_variance(self):
    shape = [512]
    num_trials = 128

    key_ph = tf.placeholder(shape=(), dtype=tf.int32)
    uniform_samples = dynamics.random_uniform(shape, key=key_ph)
    antithetic_uniform_samples = dynamics.random_antithetic_uniform(
        shape, key=key_ph)

    mean_estimator = tf.reduce_mean(uniform_samples)
    antithetic_mean_estimator = tf.reduce_mean(antithetic_uniform_samples)

    mean_estimates = []
    antithetic_mean_estimates = []
    with self.session() as session:
      for i in range(num_trials):
        mean_estimates.append(
            session.run(mean_estimator, feed_dict={key_ph: i}))
        antithetic_mean_estimates.append(
            session.run(antithetic_mean_estimator, feed_dict={key_ph: i}))

    self.assertLessEqual(
        np.std(antithetic_mean_estimates), np.std(mean_estimates))

  def test_antithetic_normal_is_symmetrical(self):
    shape = [512]

    antithetic_normal_samples = dynamics.random_antithetic_normal(shape)

    with self.session() as session:
      [samples,
       sym_samples] = session.run(tf.split(antithetic_normal_samples, 2))

    self.assertAllEqual(samples, -sym_samples)

  def test_antithetic_normal_lowers_variance(self):
    shape = [512]
    num_trials = 128

    key_ph = tf.placeholder(shape=(), dtype=tf.int32)
    normal_samples = dynamics.random_normal(shape, key=key_ph)
    antithetic_normal_samples = dynamics.random_antithetic_normal(
        shape, key=key_ph)

    mean_estimator = tf.reduce_mean(normal_samples)
    antithetic_mean_estimator = tf.reduce_mean(antithetic_normal_samples)

    mean_estimates = []
    antithetic_mean_estimates = []
    with self.session() as session:
      for i in range(num_trials):
        mean_estimates.append(
            session.run(mean_estimator, feed_dict={key_ph: i}))
        antithetic_mean_estimates.append(
            session.run(antithetic_mean_estimator, feed_dict={key_ph: i}))

    self.assertLessEqual(
        np.std(antithetic_mean_estimates), np.std(mean_estimates))

  def test_gbm_euler_step_output_is_correct(self):
    np.random.seed(0)
    drift = 0.2
    vol = 0.1
    t = 0.0
    dt = 0.01
    num_samples = 8

    states = tf.ones([num_samples])
    eps_t = np.ndarray.astype(
        np.random.normal(size=[num_samples]), dtype=np.float32)

    next_states = dynamics.gbm_euler_step(
        states, drift, vol, t, dt, random_normal_op=lambda: eps_t)

    with self.session() as session:
      next_states_eval = session.run(next_states)

    self.assertEqual(next_states_eval.shape, (num_samples,))

    # Here the maximum discrepancy is 1.17e-7 due to differences in
    # numerical implementations between tf and np so we set delta to 1.2e-7.
    self.assertAllClose(
        next_states_eval,
        np.ones([num_samples], dtype=np.float32) *
        (1.0 + drift * dt + vol * eps_t * np.sqrt(dt)),
        atol=1.2e-7)

  def test_gbm_euler_step_expects_static_shape(self):
    drift = 0.2
    vol = 0.1
    t = 0.0
    dt = 0.01

    states = tf.placeholder(dtype=tf.float32, shape=[None])

    with self.assertRaises(ValueError):
      dynamics.gbm_euler_step(states, drift, vol, t, dt)

  def test_gbm_euler_step_is_deterministic(self):
    drift = 0.2
    vol = 0.1
    t = 0.2
    dt = 0.01
    num_samples = 8
    key = 1337

    states = tf.ones([num_samples])
    eps_t = contrib_stateless.stateless_random_normal(
        shape=[num_samples], seed=[key, int(t / dt)])

    next_states = dynamics.gbm_euler_step(
        states, drift, vol, t, dt, random_normal_op=lambda: eps_t)
    next_states_bis = dynamics.gbm_euler_step(
        states, drift, vol, t, dt, key=key)

    with self.session() as session:
      next_states_eval, next_states_bis_eval = session.run((next_states,
                                                            next_states_bis))

    self.assertEqual(next_states_eval.shape, (num_samples,))
    self.assertEqual(next_states_bis_eval.shape, (num_samples,))

    self.assertAllClose(next_states_eval, next_states_bis_eval, atol=1e-7)

  def test_gbm_euler_step_output_changes_with_t(self):
    drift = 0.2
    vol = 0.1
    t_0 = 0.2
    dt = 0.01
    num_samples = 8
    t_1 = t_0 + dt

    states = tf.ones([num_samples])

    next_states_0 = dynamics.gbm_euler_step(states, drift, vol, t_0, dt)
    next_states_1 = dynamics.gbm_euler_step(states, drift, vol, t_1, dt)

    with self.session() as session:
      next_states_0_eval, next_states_1_eval = session.run((next_states_0,
                                                            next_states_1))

    self.assertEqual(next_states_0_eval.shape, (num_samples,))
    self.assertEqual(next_states_1_eval.shape, (num_samples,))

    # The step is a bijection w.r.t. dw_t, all terms should be different.
    self.assertAllDistinct(next_states_0_eval, next_states_1_eval)

  def test_gbm_euler_step_output_changes_with_key(self):
    drift = 0.2
    vol = 0.1
    t = 0.2
    dt = 0.01
    num_samples = 8
    key_0 = 74
    key_1 = 75

    states = tf.ones([num_samples])

    next_states_0 = dynamics.gbm_euler_step(
        states, drift, vol, t, dt, key=key_0)
    next_states_1 = dynamics.gbm_euler_step(
        states, drift, vol, t, dt, key=key_1)

    with self.session() as session:
      next_states_0_eval, next_states_1_eval = session.run((next_states_0,
                                                            next_states_1))

    # The step is a bijection w.r.t. dw_t, all terms should be different.
    self.assertAllDistinct(next_states_0_eval, next_states_1_eval)

  def test_gbm_euler_step_running_max_output_is_correct(self):
    np.random.seed(0)
    drift = 0.2
    vol = 0.1
    t = 0.0
    dt = 0.01
    num_samples = 8
    initial_states = np.ones([num_samples], dtype=np.float32)

    states_and_max = [tf.constant(initial_states)] * 2

    eps_t = np.ndarray.astype(
        np.random.normal(size=[num_samples]), dtype=np.float32)

    (next_states, next_max) = dynamics.gbm_euler_step_running_max(
        states_and_max,
        drift,
        vol,
        t,
        dt,
        simulate_bridge=False,
        random_normal_op=lambda: eps_t)

    with self.session() as session:
      (next_states_eval, next_max_eval) = session.run((next_states, next_max))

    self.assertEqual(next_states_eval.shape, (num_samples,))
    self.assertEqual(next_max_eval.shape, (num_samples,))

    expected_next_states = initial_states * (1.0 + drift * dt +
                                             vol * eps_t * np.sqrt(dt))
    expected_next_max = np.maximum(expected_next_states, initial_states)

    # Here the maximum discrepancy is 1.17e-7 due to differences in
    # numerical implementations between tf and np so we set delta to 1.2e-7.
    self.assertAllClose(
        next_states_eval, expected_next_states, atol=1.2e-7)
    self.assertAllClose(next_max_eval, expected_next_max, atol=1.2e-7)

  def test_gbm_euler_step_running_max_is_deterministic(self):
    drift = 0.2
    vol = 0.1
    t = 0.2
    dt = 0.01
    num_samples = 8
    key = 1337

    states_and_max = [tf.ones([num_samples])] * 2
    eps_t = contrib_stateless.stateless_random_normal(
        shape=[num_samples], seed=[key, int(t / dt)])

    next_states_and_max = dynamics.gbm_euler_step_running_max(
        states_and_max,
        drift,
        vol,
        t,
        dt,
        simulate_bridge=False,
        random_normal_op=lambda: eps_t)
    next_states_and_max_bis = dynamics.gbm_euler_step_running_max(
        states_and_max, drift, vol, t, dt, simulate_bridge=False, key=key)

    with self.session() as session:
      next_states_and_max_eval, next_states_and_max_bis_eval = session.run(
          (next_states_and_max, next_states_and_max_bis))

    next_states_eval, next_max_eval = next_states_and_max_eval
    next_states_bis_eval, next_max_bis_eval = next_states_and_max_bis_eval

    self.assertEqual(next_states_eval.shape, (num_samples,))
    self.assertEqual(next_states_bis_eval.shape, (num_samples,))

    self.assertEqual(next_max_eval.shape, (num_samples,))
    self.assertEqual(next_max_bis_eval.shape, (num_samples,))

    self.assertAllClose(next_states_eval, next_states_bis_eval)
    self.assertAllClose(next_max_eval, next_max_bis_eval)

  def test_gbm_euler_step_running_max_expects_static_shape_left_member(self):
    drift = 0.2
    vol = 0.1
    t = 0.2
    dt = 0.01
    num_samples = 8

    states_and_max = [
        tf.placeholder(dtype=tf.float32, shape=[None]), tf.ones([num_samples])]

    with self.assertRaises(ValueError):
      dynamics.gbm_euler_step_running_max(states_and_max, drift, vol, t, dt)

  def test_gbm_euler_step_running_max_expects_static_shape_right_member(self):
    drift = 0.2
    vol = 0.1
    t = 0.2
    dt = 0.01
    num_samples = 8

    states_and_max = [
        tf.ones([num_samples]), tf.placeholder(dtype=tf.float32, shape=[None])]

    with self.assertRaises(ValueError):
      dynamics.gbm_euler_step_running_max(states_and_max, drift, vol, t, dt)

  def test_gbm_euler_step_running_max_expects_static_shape_both_members(self):
    drift = 0.2
    vol = 0.1
    t = 0.2
    dt = 0.01

    states_and_max = [
        tf.placeholder(dtype=tf.float32, shape=[None]),
        tf.placeholder(dtype=tf.float32, shape=[None])]

    with self.assertRaises(ValueError):
      dynamics.gbm_euler_step_running_max(states_and_max, drift, vol, t, dt)

  def test_gbm_euler_step_running_max_changes_with_t(self):
    drift = 0.2
    vol = 0.1
    t_0 = 0.2
    dt = 0.01
    num_samples = 8
    t_1 = t_0 + dt

    states_and_max = [tf.ones([num_samples])] * 2

    next_states_and_max_0 = dynamics.gbm_euler_step_running_max(
        states_and_max, drift, vol, t_0, dt, simulate_bridge=False)
    next_states_and_max_1 = dynamics.gbm_euler_step_running_max(
        states_and_max, drift, vol, t_1, dt, simulate_bridge=False)

    with self.session() as session:
      next_states_and_max_0_eval, next_states_and_max_1_eval = session.run(
          (next_states_and_max_0, next_states_and_max_1))

    next_states_0_eval, next_max_0_eval = next_states_and_max_0_eval
    next_states_1_eval, next_max_1_eval = next_states_and_max_1_eval

    self.assertEqual(next_states_0_eval.shape, (num_samples,))
    self.assertEqual(next_states_1_eval.shape, (num_samples,))

    self.assertEqual(next_max_0_eval.shape, (num_samples,))
    self.assertEqual(next_max_1_eval.shape, (num_samples,))

    # The step is a bijection w.r.t. dw_t, all terms should be different.
    # However there is no such guarantee for the running maxima.
    self.assertAllDistinct(next_states_0_eval, next_states_1_eval)

  def test_gbm_euler_step_running_max_changes_with_key(self):
    drift = 0.2
    vol = 0.1
    t = 0.2
    dt = 0.01
    num_samples = 8
    key_0 = 74
    key_1 = 75

    states_and_max = [tf.ones([num_samples])] * 2

    next_states_and_max_0 = dynamics.gbm_euler_step_running_max(
        states_and_max, drift, vol, t, dt, key=key_0, simulate_bridge=False)
    next_states_and_max_1 = dynamics.gbm_euler_step_running_max(
        states_and_max, drift, vol, t, dt, key=key_1, simulate_bridge=False)

    with self.session() as session:
      next_states_and_max_0_eval, next_states_and_max_1_eval = session.run(
          (next_states_and_max_0, next_states_and_max_1))

    next_states_0_eval, next_max_0_eval = next_states_and_max_0_eval
    next_states_1_eval, next_max_1_eval = next_states_and_max_1_eval

    self.assertEqual(next_states_0_eval.shape, (num_samples,))
    self.assertEqual(next_states_1_eval.shape, (num_samples,))

    self.assertEqual(next_max_0_eval.shape, (num_samples,))
    self.assertEqual(next_max_1_eval.shape, (num_samples,))

    # The step is a bijection w.r.t. dw_t, all terms should be different.
    # However there is no such guarantee for the running maxima.
    self.assertAllDistinct(next_states_0_eval, next_states_1_eval)

  def test_gbm_euler_step_running_max_bridge_output_is_correct(self):
    np.random.seed(0)
    drift = 0.2
    vol = 0.1
    t = 0.0
    dt = 0.01
    num_samples = 8
    initial_states = np.ones([num_samples], dtype=np.float32)

    states_and_max = [tf.constant(initial_states)] * 2

    eps_t = np.ndarray.astype(
        np.random.normal(size=[num_samples]), dtype=np.float32)
    u_t = np.ndarray.astype(
        np.random.uniform(size=[num_samples]), dtype=np.float32)

    (next_states, next_max) = dynamics.gbm_euler_step_running_max(
        states_and_max,
        drift,
        vol,
        t,
        dt,
        simulate_bridge=True,
        random_normal_op=lambda: eps_t,
        random_uniform_op=lambda: u_t)

    with self.session() as session:
      (next_states_eval, next_max_eval) = session.run((next_states, next_max))

    self.assertEqual(next_states_eval.shape, (num_samples,))
    self.assertEqual(next_max_eval.shape, (num_samples,))

    expected_next_states = initial_states * (1.0 + drift * dt +
                                             vol * eps_t * np.sqrt(dt))

    expected_bridge_max = 0.5 * (
        initial_states + expected_next_states +
        np.sqrt((initial_states - expected_next_states)**2 - 2.0 * dt *
                (vol * initial_states)**2 * np.log(u_t)))
    expected_next_max = np.maximum(expected_next_states, expected_bridge_max)

    # Here the maximum discrepancy is 1.17e-7 due to differences in
    # numerical implementations between tf and np so we set delta to 1.2e-7.
    self.assertAllClose(
        next_states_eval, expected_next_states, atol=1.2e-7)
    self.assertAllClose(next_max_eval, expected_next_max, atol=1.2e-7)

  def test_gbm_euler_step_running_max_bridge_is_deterministic(self):
    drift = 0.2
    vol = 0.1
    t = 0.2
    dt = 0.01
    num_samples = 8
    key = 1337

    states_and_max = [tf.ones([num_samples])] * 2
    eps_t = contrib_stateless.stateless_random_normal(
        shape=[num_samples], seed=[2 * key, int(t / dt)])
    u_t = contrib_stateless.stateless_random_uniform(
        shape=[num_samples], seed=[2 * key + 1, int(t / dt)])

    next_states_and_max = dynamics.gbm_euler_step_running_max(
        states_and_max,
        drift,
        vol,
        t,
        dt,
        simulate_bridge=True,
        random_normal_op=lambda: eps_t,
        random_uniform_op=lambda: u_t)
    next_states_and_max_bis = dynamics.gbm_euler_step_running_max(
        states_and_max, drift, vol, t, dt, simulate_bridge=True, key=key)

    with self.session() as session:
      next_states_and_max_eval, next_states_and_max_bis_eval = session.run(
          (next_states_and_max, next_states_and_max_bis))

    next_states_eval, next_max_eval = next_states_and_max_eval
    next_states_bis_eval, next_max_bis_eval = next_states_and_max_bis_eval

    self.assertEqual(next_states_eval.shape, (num_samples,))
    self.assertEqual(next_states_bis_eval.shape, (num_samples,))

    self.assertEqual(next_max_eval.shape, (num_samples,))
    self.assertEqual(next_max_bis_eval.shape, (num_samples,))

    self.assertAllClose(next_states_eval, next_states_bis_eval)
    self.assertAllClose(next_max_eval, next_max_bis_eval)

  def test_gbm_euler_step_running_max_bridge_changes_with_t(self):
    drift = 0.2
    vol = 0.1
    t_0 = 0.2
    dt = 0.01
    num_samples = 8
    t_1 = t_0 + dt

    states_and_max = [tf.ones([num_samples])] * 2

    next_states_and_max_0 = dynamics.gbm_euler_step_running_max(
        states_and_max, drift, vol, t_0, dt, simulate_bridge=True)
    next_states_and_max_1 = dynamics.gbm_euler_step_running_max(
        states_and_max, drift, vol, t_1, dt, simulate_bridge=True)

    with self.session() as session:
      next_states_and_max_0_eval, next_states_and_max_1_eval = session.run(
          (next_states_and_max_0, next_states_and_max_1))

    next_states_0_eval, next_max_0_eval = next_states_and_max_0_eval
    next_states_1_eval, next_max_1_eval = next_states_and_max_1_eval

    self.assertEqual(next_states_0_eval.shape, (num_samples,))
    self.assertEqual(next_states_1_eval.shape, (num_samples,))

    self.assertEqual(next_max_0_eval.shape, (num_samples,))
    self.assertEqual(next_max_1_eval.shape, (num_samples,))

    # The step is a bijection w.r.t. dw_t, all terms should be different.
    # However there is no such guarantee for the running maxima.
    self.assertAllDistinct(next_states_0_eval, next_states_1_eval)

  def test_gbm_euler_step_running_max_bridge_changes_with_key(self):
    drift = 0.2
    vol = 0.1
    t = 0.2
    dt = 0.01
    num_samples = 8
    key_0 = 74
    key_1 = 75

    states_and_max = [tf.ones([num_samples])] * 2

    next_states_and_max_0 = dynamics.gbm_euler_step_running_max(
        states_and_max, drift, vol, t, dt, key=key_0, simulate_bridge=True)
    next_states_and_max_1 = dynamics.gbm_euler_step_running_max(
        states_and_max, drift, vol, t, dt, key=key_1, simulate_bridge=True)

    with self.session() as session:
      next_states_and_max_0_eval, next_states_and_max_1_eval = session.run(
          (next_states_and_max_0, next_states_and_max_1))

    next_states_0_eval, next_max_0_eval = next_states_and_max_0_eval
    next_states_1_eval, next_max_1_eval = next_states_and_max_1_eval

    self.assertEqual(next_states_0_eval.shape, (num_samples,))
    self.assertEqual(next_states_1_eval.shape, (num_samples,))

    self.assertEqual(next_max_0_eval.shape, (num_samples,))
    self.assertEqual(next_max_1_eval.shape, (num_samples,))

    # The step is a bijection w.r.t. dw_t, all terms should be different.
    # However there is no such guarantee for the running maxima.
    self.assertAllDistinct(next_states_0_eval, next_states_1_eval)

  def test_gbm_euler_step_running_sum_output_is_correct(self):
    np.random.seed(0)
    drift = 0.2
    vol = 0.1
    t = 0.0
    dt = 0.01
    num_samples = 8
    initial_states = np.ones([num_samples], dtype=np.float32)

    states_and_sums = [tf.constant(initial_states)] * 2

    eps_t = np.ndarray.astype(
        np.random.normal(size=[num_samples]), dtype=np.float32)

    (next_states, next_sums) = dynamics.gbm_euler_step_running_sum(
        states_and_sums, drift, vol, t, dt, random_normal_op=lambda: eps_t)

    with self.session() as session:
      (next_states_eval, next_sums_eval) = session.run((next_states, next_sums))

    self.assertEqual(next_states_eval.shape, (num_samples,))
    self.assertEqual(next_sums_eval.shape, (num_samples,))

    expected_next_states = initial_states * (1.0 + drift * dt +
                                             vol * eps_t * np.sqrt(dt))
    expected_next_sums = expected_next_states + initial_states

    # Here the maximum discrepancy is 1.17e-7 due to differences in
    # numerical implementations between tf and np so we set delta to 1.2e-7.
    self.assertAllClose(next_states_eval, expected_next_states, atol=1.2e-7)
    self.assertAllClose(next_sums_eval, expected_next_sums, atol=1.2e-7)

  def test_gbm_euler_step_running_sum_expects_static_shape_left_member(self):
    drift = 0.2
    vol = 0.1
    t = 0.2
    dt = 0.01
    num_samples = 8

    states_and_sums = [
        tf.placeholder(dtype=tf.float32, shape=[None]), tf.ones([num_samples])]

    with self.assertRaises(ValueError):
      dynamics.gbm_euler_step_running_sum(states_and_sums, drift, vol, t, dt)

  def test_gbm_euler_step_running_sum_expects_static_shape_right_member(self):
    drift = 0.2
    vol = 0.1
    t = 0.2
    dt = 0.01
    num_samples = 8

    states_and_sums = [
        tf.ones([num_samples]), tf.placeholder(dtype=tf.float32, shape=[None])]

    with self.assertRaises(ValueError):
      dynamics.gbm_euler_step_running_sum(states_and_sums, drift, vol, t, dt)

  def test_gbm_euler_step_running_sum_expects_static_shape_both_members(self):
    drift = 0.2
    vol = 0.1
    t = 0.2
    dt = 0.01

    states_and_sums = [
        tf.placeholder(dtype=tf.float32, shape=[None]),
        tf.placeholder(dtype=tf.float32, shape=[None])]

    with self.assertRaises(ValueError):
      dynamics.gbm_euler_step_running_sum(states_and_sums, drift, vol, t, dt)

  def test_gbm_euler_step_running_sum_is_deterministic(self):
    drift = 0.2
    vol = 0.1
    t = 0.2
    dt = 0.01
    num_samples = 8
    key = 1337

    states_and_sums = [tf.ones([num_samples])] * 2
    eps_t = contrib_stateless.stateless_random_normal(
        shape=[num_samples], seed=[key, int(t / dt)])

    next_states_and_sums = dynamics.gbm_euler_step_running_sum(
        states_and_sums, drift, vol, t, dt, random_normal_op=lambda: eps_t)
    next_states_and_sums_bis = dynamics.gbm_euler_step_running_sum(
        states_and_sums, drift, vol, t, dt, key=key)

    with self.session() as session:
      next_states_and_sums_eval, next_states_and_sums_bis_eval = session.run(
          (next_states_and_sums, next_states_and_sums_bis))

    next_states_eval, next_sums_eval = next_states_and_sums_eval
    next_states_bis_eval, next_sums_bis_eval = next_states_and_sums_bis_eval

    self.assertEqual(next_states_eval.shape, (num_samples,))
    self.assertEqual(next_states_bis_eval.shape, (num_samples,))

    self.assertEqual(next_sums_eval.shape, (num_samples,))
    self.assertEqual(next_sums_bis_eval.shape, (num_samples,))

    self.assertAllClose(next_states_eval, next_states_bis_eval)
    self.assertAllClose(next_sums_eval, next_sums_bis_eval)

  def test_gbm_euler_step_running_sum_changes_with_t(self):
    drift = 0.2
    vol = 0.1
    t_0 = 0.2
    dt = 0.01
    num_samples = 8
    t_1 = t_0 + dt

    states_and_sums = [tf.ones([num_samples])] * 2

    next_states_and_sums_0 = dynamics.gbm_euler_step_running_sum(
        states_and_sums, drift, vol, t_0, dt)
    next_states_and_sums_1 = dynamics.gbm_euler_step_running_sum(
        states_and_sums, drift, vol, t_1, dt)

    with self.session() as session:
      next_states_and_sums_0_eval, next_states_and_sums_1_eval = session.run(
          (next_states_and_sums_0, next_states_and_sums_1))

    next_states_0_eval, next_sums_0_eval = next_states_and_sums_0_eval
    next_states_1_eval, next_sums_1_eval = next_states_and_sums_1_eval

    self.assertEqual(next_states_0_eval.shape, (num_samples,))
    self.assertEqual(next_states_1_eval.shape, (num_samples,))

    self.assertEqual(next_sums_0_eval.shape, (num_samples,))
    self.assertEqual(next_sums_1_eval.shape, (num_samples,))

    # The step is a bijection w.r.t. dw_t, all terms should be different.
    # However there is no such guarantee for the running maxima.
    self.assertAllDistinct(next_states_0_eval, next_states_1_eval)

  def test_gbm_euler_step_running_sum_changes_with_key(self):
    drift = 0.2
    vol = 0.1
    t = 0.2
    dt = 0.01
    num_samples = 8
    key_0 = 74
    key_1 = 75

    states_and_sums = [tf.ones([num_samples])] * 2

    next_states_and_sums_0 = dynamics.gbm_euler_step_running_sum(
        states_and_sums, drift, vol, t, dt, key=key_0)
    next_states_and_sums_1 = dynamics.gbm_euler_step_running_sum(
        states_and_sums, drift, vol, t, dt, key=key_1)

    with self.session() as session:
      next_states_and_sums_0_eval, next_states_and_sums_1_eval = session.run(
          (next_states_and_sums_0, next_states_and_sums_1))

    next_states_0_eval, next_sums_0_eval = next_states_and_sums_0_eval
    next_states_1_eval, next_sums_1_eval = next_states_and_sums_1_eval

    self.assertEqual(next_states_0_eval.shape, (num_samples,))
    self.assertEqual(next_states_1_eval.shape, (num_samples,))

    self.assertEqual(next_sums_0_eval.shape, (num_samples,))
    self.assertEqual(next_sums_1_eval.shape, (num_samples,))

    # The step is a bijection w.r.t. dw_t, all terms should be different.
    # However there is no such guarantee for the running maxima.
    self.assertAllDistinct(next_states_0_eval, next_states_1_eval)

  def test_gbm_euler_step_nd_output_is_correct(self):
    np.random.seed(0)
    drift = np.asarray([0.1, 0.3, -0.05], dtype=np.float32)
    vol_matrix = 0.2 * np.asarray(
        [[1.5, 0.2, 0.3], [0.2, 1.1, -0.1], [0.3, -0.1, 0.8]], dtype=np.float32)
    t = 0.0
    dt = 0.01
    num_samples = 8
    num_dims = drift.shape[0]

    states = tf.ones([num_samples, num_dims])
    eps_t = np.ndarray.astype(
        np.random.normal(size=[num_samples, num_dims]), dtype=np.float32)

    next_states = dynamics.gbm_euler_step_nd(
        states, drift, vol_matrix, t, dt, random_normal_op=lambda: eps_t)

    with self.session() as session:
      next_states_eval = session.run(next_states)

    self.assertEqual(next_states_eval.shape, (num_samples, num_dims))

    for i in range(num_samples):
      self.assertAllClose(
          next_states_eval[i],
          np.ones([num_dims], dtype=np.float32) *
          (1.0 + drift * dt + np.matmul(vol_matrix, eps_t[i] * np.sqrt(dt))))

  def test_gbm_euler_step_nd_expects_static_shape(self):
    drift = np.asarray([0.1, 0.3, -0.05], dtype=np.float32)
    vol_matrix = 0.2 * np.asarray(
        [[1.5, 0.2, 0.3], [0.2, 1.1, -0.1], [0.3, -0.1, 0.8]], dtype=np.float32)
    t = 0.0
    dt = 0.01
    num_dims = drift.shape[0]

    states = tf.placeholder(dtype=tf.float32, shape=[None, num_dims])

    with self.assertRaises(ValueError):
      dynamics.gbm_euler_step_nd(states, drift, vol_matrix, t, dt)

  def test_gbm_euler_step_nd_is_deterministic(self):
    drift = np.asarray([0.1, 0.3, -0.05], dtype=np.float32)
    vol_matrix = 0.2 * np.asarray(
        [[1.5, 0.2, 0.3], [0.2, 1.1, -0.1], [0.3, -0.1, 0.8]], dtype=np.float32)
    t = 0.3
    dt = 0.01
    num_samples = 8
    num_dims = drift.shape[0]
    key = 42

    states = tf.ones([num_samples, num_dims])
    eps_t = contrib_stateless.stateless_random_normal(
        shape=[num_samples, num_dims], seed=[key, int(t / dt)])

    next_states = dynamics.gbm_euler_step_nd(
        states, drift, vol_matrix, t, dt, random_normal_op=lambda: eps_t)
    next_states_bis = dynamics.gbm_euler_step_nd(
        states, drift, vol_matrix, t, dt, key=key)

    with self.session() as session:
      next_states_eval, next_states_bis_eval = session.run((next_states,
                                                            next_states_bis))

    self.assertEqual(next_states_eval.shape, (num_samples, num_dims))
    self.assertEqual(next_states_bis_eval.shape, (num_samples, num_dims))

    self.assertAllClose(next_states_eval, next_states_bis_eval)

  def test_gbm_euler_step_nd_output_changes_with_t(self):
    drift = np.asarray([0.1, 0.3, -0.05], dtype=np.float32)
    vol_matrix = 0.2 * np.asarray(
        [[1.5, 0.2, 0.3], [0.2, 1.1, -0.1], [0.3, -0.1, 0.8]], dtype=np.float32)
    t_0 = 0.3
    dt = 0.01
    num_samples = 8
    num_dims = drift.shape[0]
    t_1 = t_0 + dt

    states = tf.ones([num_samples, num_dims])
    next_states_0 = dynamics.gbm_euler_step_nd(states, drift, vol_matrix, t_0,
                                               dt)
    next_states_1 = dynamics.gbm_euler_step_nd(states, drift, vol_matrix, t_1,
                                               dt)

    with self.session() as session:
      next_states_0_eval, next_states_1_eval = session.run((next_states_0,
                                                            next_states_1))

    self.assertEqual(next_states_0_eval.shape, (num_samples, num_dims))
    self.assertEqual(next_states_1_eval.shape, (num_samples, num_dims))

    # The step is a bijection w.r.t. dw_t, all terms should be different.
    self.assertAllDistinct(next_states_0_eval, next_states_1_eval)

  def test_gbm_euler_step_nd_output_changes_with_key(self):
    drift = np.asarray([0.1, 0.3, -0.05], dtype=np.float32)
    vol_matrix = 0.2 * np.asarray(
        [[1.5, 0.2, 0.3], [0.2, 1.1, -0.1], [0.3, -0.1, 0.8]], dtype=np.float32)
    t = 0.3
    dt = 0.01
    num_samples = 8
    num_dims = drift.shape[0]
    key_0 = 42
    key_1 = 77

    states = tf.ones([num_samples, num_dims])
    next_states_0 = dynamics.gbm_euler_step_nd(
        states, drift, vol_matrix, t, dt, key=key_0)
    next_states_1 = dynamics.gbm_euler_step_nd(
        states, drift, vol_matrix, t, dt, key=key_1)

    with self.session() as session:
      next_states_0_eval, next_states_1_eval = session.run((next_states_0,
                                                            next_states_1))

    self.assertEqual(next_states_0_eval.shape, (num_samples, num_dims))
    self.assertEqual(next_states_1_eval.shape, (num_samples, num_dims))

    # The step is a bijection w.r.t. dw_t, all terms should be different.
    self.assertAllDistinct(next_states_0_eval, next_states_1_eval)

  def test_gbm_log_euler_step_output_is_correct(self):
    np.random.seed(0)
    drift = 0.2
    vol = 0.1
    t = 0.0
    dt = 0.01
    num_samples = 8

    log_states = tf.zeros([num_samples])
    eps_t = np.ndarray.astype(
        np.random.normal(size=[num_samples]), dtype=np.float32)

    next_log_states = dynamics.gbm_log_euler_step(
        log_states, drift, vol, t, dt, random_normal_op=lambda: eps_t)

    with self.session() as session:
      next_log_states_eval = session.run(next_log_states)

    self.assertEqual(next_log_states_eval.shape, (num_samples,))

    self.assertAllClose(
        next_log_states_eval,
        np.zeros([num_samples], dtype=np.float32) +
        (drift - 0.5 * (vol**2)) * dt + vol * eps_t * np.sqrt(dt))

  def test_gbm_log_euler_step_expects_static_shape(self):
    drift = 0.2
    vol = 0.1
    t = 0.0
    dt = 0.01

    log_states = tf.placeholder(dtype=tf.float32, shape=[None])

    with self.assertRaises(ValueError):
      dynamics.gbm_log_euler_step(log_states, drift, vol, t, dt)

  def test_gbm_log_euler_step_is_deterministic(self):
    drift = 0.2
    vol = 0.1
    t = 0.0
    dt = 0.01
    num_samples = 8
    key = 13

    log_states = tf.zeros([num_samples])
    eps_t = contrib_stateless.stateless_random_normal(
        shape=[num_samples], seed=[key, int(t / dt)])

    next_log_states = dynamics.gbm_log_euler_step(
        log_states, drift, vol, t, dt, random_normal_op=lambda: eps_t)
    next_log_states_bis = dynamics.gbm_log_euler_step(
        log_states, drift, vol, t, dt, key=key)

    with self.session() as session:
      next_log_states_eval, next_log_states_bis_eval = session.run(
          (next_log_states, next_log_states_bis))

    self.assertEqual(next_log_states_eval.shape, (num_samples,))
    self.assertEqual(next_log_states_bis_eval.shape, (num_samples,))

    self.assertAllClose(next_log_states_eval, next_log_states_bis_eval)

  def test_gbm_log_euler_step_output_changes_with_t(self):
    drift = 0.2
    vol = 0.1
    t_0 = 0.0
    dt = 0.01
    num_samples = 8
    t_1 = t_0 + dt

    log_states = tf.zeros([num_samples])

    next_log_states_0 = dynamics.gbm_log_euler_step(log_states, drift, vol, t_0,
                                                    dt)
    next_log_states_1 = dynamics.gbm_log_euler_step(log_states, drift, vol, t_1,
                                                    dt)

    with self.session() as session:
      next_log_states_0_eval, next_log_states_1_eval = session.run(
          (next_log_states_0, next_log_states_1))

    self.assertEqual(next_log_states_0_eval.shape, (num_samples,))
    self.assertEqual(next_log_states_1_eval.shape, (num_samples,))

    # The step is a bijection w.r.t. dw_t, all terms should be different.
    self.assertAllDistinct(next_log_states_0_eval, next_log_states_1_eval)

  def test_gbm_log_euler_step_output_changes_with_key(self):
    drift = 0.2
    vol = 0.1
    t = 0.0
    dt = 0.01
    num_samples = 8
    key_0 = 1137
    key_1 = 0

    log_states = tf.zeros([num_samples])

    next_log_states_0 = dynamics.gbm_log_euler_step(
        log_states, drift, vol, t, dt, key=key_0)
    next_log_states_1 = dynamics.gbm_log_euler_step(
        log_states, drift, vol, t, dt, key=key_1)

    with self.session() as session:
      next_log_states_0_eval, next_log_states_1_eval = session.run(
          (next_log_states_0, next_log_states_1))

    self.assertEqual(next_log_states_0_eval.shape, (num_samples,))
    self.assertEqual(next_log_states_1_eval.shape, (num_samples,))

    # The step is a bijection w.r.t. dw_t, all terms should be different.
    self.assertAllDistinct(next_log_states_0_eval, next_log_states_1_eval)

  def test_gbm_log_euler_step_nd_output_is_correct(self):
    drift = np.asarray([0.1, 0.3, -0.05], dtype=np.float32)
    vol_matrix = 0.2 * np.asarray(
        [[1.5, 0.2, 0.3], [0.2, 1.1, -0.1], [0.3, -0.1, 0.8]], dtype=np.float32)
    dt = 0.01
    t = 0.0
    num_samples = 8
    num_dims = drift.shape[0]

    log_states = tf.zeros([num_samples, num_dims])
    eps_t = np.ndarray.astype(
        np.random.normal(size=[num_samples, num_dims]), dtype=np.float32)

    next_log_states = dynamics.gbm_log_euler_step_nd(
        log_states, drift, vol_matrix, t, dt, random_normal_op=lambda: eps_t)

    with self.session() as session:
      next_log_states_eval = session.run(next_log_states)

    self.assertEqual(next_log_states_eval.shape, (num_samples, num_dims))

    for i in range(num_samples):
      self.assertAllClose(
          next_log_states_eval[i],
          np.zeros([num_dims], dtype=np.float32) +
          (drift - 0.5 * np.sum(vol_matrix**2, axis=0)) * dt +
          np.matmul(vol_matrix, eps_t[i] * np.sqrt(dt)))

  def test_gbm_log_euler_step_nd_expects_static_shape(self):
    drift = np.asarray([0.1, 0.3, -0.05], dtype=np.float32)
    vol_matrix = 0.2 * np.asarray(
        [[1.5, 0.2, 0.3], [0.2, 1.1, -0.1], [0.3, -0.1, 0.8]], dtype=np.float32)
    dt = 0.01
    t = 0.0
    num_dims = drift.shape[0]

    log_states = tf.placeholder(dtype=tf.float32, shape=[None, num_dims])
    with self.assertRaises(ValueError):
      dynamics.gbm_log_euler_step_nd(log_states, drift, vol_matrix, t, dt)

  def test_gbm_log_euler_step_nd_is_deterministic(self):
    drift = np.asarray([0.1, 0.3, -0.05], dtype=np.float32)
    vol_matrix = 0.2 * np.asarray(
        [[1.5, 0.2, 0.3], [0.2, 1.1, -0.1], [0.3, -0.1, 0.8]], dtype=np.float32)
    dt = 0.01
    t = 0.0
    num_samples = 8
    num_dims = drift.shape[0]
    key = 128

    log_states = tf.zeros([num_samples, num_dims])
    eps_t = contrib_stateless.stateless_random_normal(
        shape=[num_samples, num_dims], seed=[key, int(t / dt)])

    next_log_states = dynamics.gbm_log_euler_step_nd(
        log_states, drift, vol_matrix, t, dt, random_normal_op=lambda: eps_t)
    next_log_states_bis = dynamics.gbm_log_euler_step_nd(
        log_states, drift, vol_matrix, t, dt, key=key)

    with self.session() as session:
      next_log_states_eval, next_log_states_bis_eval = session.run(
          (next_log_states, next_log_states_bis))

    self.assertEqual(next_log_states_eval.shape, (num_samples, num_dims))
    self.assertEqual(next_log_states_bis_eval.shape, (num_samples, num_dims))

    self.assertAllClose(next_log_states_eval, next_log_states_bis_eval)

  def test_gbm_log_euler_step_nd_output_changes_with_t(self):
    drift = np.asarray([0.1, 0.3, -0.05], dtype=np.float32)
    vol_matrix = 0.2 * np.asarray(
        [[1.5, 0.2, 0.3], [0.2, 1.1, -0.1], [0.3, -0.1, 0.8]], dtype=np.float32)
    dt = 0.01
    t_0 = 0.0
    num_samples = 8
    num_dims = drift.shape[0]
    t_1 = t_0 + dt

    log_states = tf.zeros([num_samples, num_dims])

    next_log_states_0 = dynamics.gbm_log_euler_step_nd(log_states, drift,
                                                       vol_matrix, t_0, dt)
    next_log_states_1 = dynamics.gbm_log_euler_step_nd(log_states, drift,
                                                       vol_matrix, t_1, dt)

    with self.session() as session:
      next_log_states_0_eval, next_log_states_1_eval = session.run(
          (next_log_states_0, next_log_states_1))

    self.assertEqual(next_log_states_0_eval.shape, (num_samples, num_dims))
    self.assertEqual(next_log_states_1_eval.shape, (num_samples, num_dims))

    # The step is a bijection w.r.t. dw_t, all terms should be different.
    self.assertAllDistinct(next_log_states_0_eval, next_log_states_1_eval)

  def test_gbm_log_euler_step_nd_output_changes_with_key(self):
    drift = np.asarray([0.1, 0.3, -0.05], dtype=np.float32)
    vol_matrix = 0.2 * np.asarray(
        [[1.5, 0.2, 0.3], [0.2, 1.1, -0.1], [0.3, -0.1, 0.8]], dtype=np.float32)
    dt = 0.01
    t = 0.0
    num_samples = 8
    num_dims = drift.shape[0]
    key_0 = 50
    key_1 = 99

    log_states = tf.zeros([num_samples, num_dims])

    next_log_states_0 = dynamics.gbm_log_euler_step_nd(
        log_states, drift, vol_matrix, t, dt, key=key_0)
    next_log_states_1 = dynamics.gbm_log_euler_step_nd(
        log_states, drift, vol_matrix, t, dt, key=key_1)

    with self.session() as session:
      next_log_states_0_eval, next_log_states_1_eval = session.run(
          (next_log_states_0, next_log_states_1))

    self.assertEqual(next_log_states_0_eval.shape, (num_samples, num_dims))
    self.assertEqual(next_log_states_1_eval.shape, (num_samples, num_dims))

    # The step is a bijection w.r.t. dw_t, all terms should be different.
    self.assertAllDistinct(next_log_states_0_eval, next_log_states_1_eval)

  @tf.test.mock.patch.object(tf.random, 'stateless_normal')
  def test_random_normal(self, mock_stateless_random_normal):
    _ = dynamics.random_normal(shape=[3, 1], i=41 / 5, key=9)
    _, call_args = mock_stateless_random_normal.call_args
    assert_ops = [
        tf.assert_equal(tf.stack([9, 8]), call_args['seed']),
        tf.assert_equal([3, 1], call_args['shape'])
    ]
    with self.session() as sess:
      sess.run(assert_ops)


if __name__ == '__main__':
  tf.test.main()
