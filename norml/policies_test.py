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

"""Tests for policies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import scipy.stats
import tensorflow.compat.v1 as tf

from norml import networks
from norml import policies


class PoliciesTest(tf.test.TestCase):

  def setUp(self):
    np.random.seed(12345)

  def test_sample(self):
    """Verify whether samples of policy are approximately Gaussian."""
    net_in = tf.placeholder(tf.float64, shape=(None, 1), name='input')
    log_std = tf.placeholder(tf.float64, shape=(), name='log_std')
    net_out = net_in
    policy = policies.GaussianPolicy(net_in, net_out, 1, log_std)
    with self.session() as sess:
      init = tf.global_variables_initializer()
      sess.run(init)
      test_in = np.array([[np.random.randn(),]])
      test_log_std = np.random.uniform(-1, 0.5)
      test_in_mult = np.repeat(test_in, 10000, axis=0)
      # Test various ways to create samples
      samples_1 = policy.sample(test_in_mult, sess, {log_std: test_log_std})
      samples_2 = sess.run(policy.sample_op()[0], {net_in: test_in_mult,
                                                   log_std: test_log_std})

      samples_3 = sess.run(policy.sample_op(10000)[0], {
          net_in: test_in, log_std: test_log_std
      }).ravel()
      # Check if approximately normally distributed
      samples = [samples_1, samples_2, samples_3]
      pvals = [scipy.stats.normaltest(
          (s - test_in[0, 0]) / test_log_std)[1] for s in samples]
      means = [s.mean() for s in samples]
      for pval in pvals:
        self.assertGreater(pval, 1e-4)

      # Verify shapes
      self.assertEqual(samples[0].shape[0], test_in_mult.shape[0])
      self.assertEqual(samples[0].shape[1], test_in_mult.shape[1])
      self.assertEqual(samples[1].shape[0], test_in_mult.shape[0])
      self.assertEqual(samples[1].shape[1], test_in_mult.shape[1])
      # Check if sample mean corresponds to actual mean
      for mean in means:
        self.assertAlmostEqual(mean, test_in[0, 0], places=1)

  def test_mean_std(self):
    """Simple check of the mean and standard deviation."""
    net_in = tf.placeholder(tf.float64, shape=(None, 1), name='input')
    log_std = tf.placeholder(tf.float64, shape=(), name='log_std')
    net_out = net_in
    policy = policies.GaussianPolicy(net_in, net_out, 1, log_std)
    with self.session() as sess:
      init = tf.global_variables_initializer()
      sess.run(init)
      for _ in range(10):
        test_in = np.random.randn(1, 1)
        log_std_in = np.random.randn()
        mean, std, log_std_out = sess.run(policy.mean_std_log_std_op(), {
            net_in: test_in,
            log_std: log_std_in
        })
        self.assertAlmostEqual(mean[0, 0], test_in[0, 0])
        self.assertAlmostEqual(log_std_out, log_std_in)
        self.assertAlmostEqual(std, np.exp(log_std_in))

  def test_likelihood(self):
    """Check likelihood and log-likelihood computation."""
    net_in = tf.placeholder(tf.float64, shape=(None, 2), name='input')
    log_std = tf.placeholder(tf.float64, shape=(), name='log_std')
    net_out = net_in
    policy = policies.GaussianPolicy(net_in, net_out, 2, log_std)
    with self.session() as sess:
      init = tf.global_variables_initializer()
      sess.run(init)
      for _ in range(100):
        test_in = np.random.randn(1, 2)
        log_std_in = np.random.randn()
        test_actions = np.random.randn(1, 2)
        lik = sess.run(policy.likelihood_op(test_actions),
                       {net_in: test_in, log_std: log_std_in})
        log_lik = sess.run(policy.log_likelihood_op(test_actions),
                           {net_in: test_in, log_std: log_std_in})

        # Likelihood
        lik_test = np.zeros(2)
        lik_test[0] = scipy.stats.multivariate_normal.pdf(
            test_actions[0, 0],
            mean=test_in.ravel()[0], cov=np.exp(log_std_in)**2)
        lik_test[1] = scipy.stats.multivariate_normal.pdf(
            test_actions[0, 1],
            mean=test_in.ravel()[1], cov=np.exp(log_std_in)**2)
        self.assertAllClose(lik_test[0]*lik_test[1], lik[0, 0])

        # Log likelihood
        log_lik_test = np.zeros(2)
        log_lik_test[0] = scipy.stats.multivariate_normal.logpdf(
            test_actions[0, 0],
            mean=test_in.ravel()[0], cov=np.exp(log_std_in)**2)
        log_lik_test[1] = scipy.stats.multivariate_normal.logpdf(
            test_actions[0, 1],
            mean=test_in.ravel()[1], cov=np.exp(log_std_in)**2)
        self.assertAllClose(log_lik_test[0]+log_lik_test[1], log_lik[0, 0])

  def test_learn_simple_policy(self):
    """Train a gaussian "policy" to react differently to various inputs.

    Inputs are sampled from a 2D Gaussian distribution.
    Outputs are one dimensional.
    """
    input_means = np.array([[-1., -1], [-1, 1], [1, -1], [1, 1]])
    input_std = .1
    output_means = np.array([[0.], [1], [2], [3]])

    network_generator = networks.FullyConnectedNetworkGenerator(
        2, 1, (
            64,
            64,
        ), tf.nn.relu)
    weights = network_generator.construct_network_weights()
    net_in = tf.placeholder(tf.float32, shape=(None, 2), name='input')
    net_out = network_generator.construct_network(net_in, weights)
    policy = policies.GaussianPolicy(net_in, net_out, 1, -5.)

    actions = tf.placeholder(tf.float32, shape=(None, 1), name='actions')
    log_lik = policy.log_likelihood_op(actions)
    optimizer = tf.train.AdamOptimizer(0.001)
    minimizer = optimizer.minimize(-tf.reduce_mean(log_lik))

    pol_mean, _ = policy.mean_op()

    with self.session() as sess:
      init = tf.global_variables_initializer()
      sess.run(init)
      for _ in range(1000):
        sample_input = np.repeat(input_means, 100, axis=0)
        sample_input += np.random.normal(0, input_std, sample_input.shape)
        sample_output = np.repeat(output_means, 100, axis=0)
        sess.run(minimizer, {net_in: sample_input, actions: sample_output})
      output_means_res = sess.run(pol_mean, {net_in: input_means})
      mae = np.mean(np.abs(output_means - output_means_res))
      self.assertAlmostEqual(mae, 0, places=1)


if __name__ == '__main__':
  tf.test.main()
