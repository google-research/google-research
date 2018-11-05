# coding=utf-8
# Copyright 2018 The Google Research Authors.
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

"""Tests for program transformations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp
import simple_probabilistic_programming as ed

tfd = tfp.distributions
tfe = tf.contrib.eager


class ProgramTransformationsTest(tf.test.TestCase):

  @tfe.run_test_in_graph_and_eager_modes
  def testMakeLogJointFnUnconditional(self):
    """Test `make_log_joint_fn` on unconditional Edward program."""
    def normal_with_unknown_mean():
      loc = ed.Normal(loc=0., scale=1., name="loc")
      x = ed.Normal(loc=loc, scale=0.5, sample_shape=5, name="x")
      return x

    def true_log_joint(loc, x):
      log_prob = tf.reduce_sum(tfd.Normal(loc=0., scale=1.).log_prob(loc))
      log_prob += tf.reduce_sum(tfd.Normal(loc=loc, scale=0.5).log_prob(x))
      return log_prob

    loc_value = 0.3
    x_value = tf.random_normal([5])

    log_joint = ed.make_log_joint_fn(normal_with_unknown_mean)
    actual_log_prob = true_log_joint(loc_value, x_value)
    expected_log_prob = log_joint(
        loc=loc_value, x=x_value,
        f="https://github.com/tensorflow/probability/issues/160")

    with self.assertRaises(LookupError):
      _ = log_joint(loc=loc_value)

    actual_log_prob_, expected_log_prob_ = self.evaluate(
        [actual_log_prob, expected_log_prob])
    self.assertEqual(actual_log_prob_, expected_log_prob_)

  @tfe.run_test_in_graph_and_eager_modes
  def testMakeLogJointFnConditional(self):
    """Test `make_log_joint_fn` on conditional Edward program."""
    def linear_regression(features, prior_precision):
      w = ed.Normal(loc=0.,
                    scale=tf.rsqrt(prior_precision),
                    sample_shape=features.shape[1],
                    name="w")
      y = ed.Normal(loc=tf.tensordot(features, w, [[1], [0]]),
                    scale=1.,
                    name="y")
      return y

    features = tf.random_normal([3, 2])
    prior_precision = 0.5
    w_value = tf.random_normal([2])
    y_value = tf.random_normal([3])

    def true_log_joint(features, prior_precision, w, y):
      log_prob = tf.reduce_sum(tfd.Normal(
          loc=0.,
          scale=tf.rsqrt(prior_precision)).log_prob(w))
      log_prob += tf.reduce_sum(tfd.Normal(
          loc=tf.tensordot(features, w, [[1], [0]]),
          scale=1.).log_prob(y))
      return log_prob

    log_joint = ed.make_log_joint_fn(linear_regression)
    actual_log_prob = true_log_joint(
        features, prior_precision, w_value, y_value)
    expected_log_prob = log_joint(
        features, prior_precision, y=y_value, w=w_value)

    with self.assertRaises(LookupError):
      _ = log_joint(features, prior_precision, w=w_value)

    actual_log_prob_, expected_log_prob_ = self.evaluate(
        [actual_log_prob, expected_log_prob])
    self.assertEqual(actual_log_prob_, expected_log_prob_)

  @tfe.run_test_in_graph_and_eager_modes
  def testMakeLogJointFnDynamic(self):
    """Test `make_log_joint_fn` on Edward program with stochastic control flow.

    This verifies that Edward's program transformation is done by tracing the
    execution at runtime (and not purely by static analysis). In particular,
    the execution is controlled by random variable outcomes, which in turn is
    controlled by the log-joint's inputs.
    """
    if not tf.executing_eagerly():
      # Don't run test in graph mode.
      return

    def mixture_of_real_and_int():
      loc = ed.Normal(loc=0., scale=1., name="loc")
      flip = ed.Bernoulli(probs=0.5, name="flip")
      if tf.equal(flip, 1):
        x = ed.Normal(loc=loc, scale=0.5, sample_shape=5, name="x")
      else:
        x = ed.Poisson(rate=tf.nn.softplus(loc), sample_shape=3, name="x")
      return x

    def true_log_joint(loc, flip, x):
      log_prob = tf.reduce_sum(tfd.Normal(loc=0., scale=1.).log_prob(loc))
      log_prob += tf.reduce_sum(tfd.Bernoulli(probs=0.5).log_prob(flip))
      if tf.equal(flip, 1):
        log_prob += tf.reduce_sum(tfd.Normal(loc=loc, scale=0.5).log_prob(x))
      else:
        log_prob += tf.reduce_sum(
            tfd.Poisson(rate=tf.nn.softplus(loc)).log_prob(x))
      return log_prob

    loc_value = 0.3
    flip_value = tf.constant(1)
    x_value = tf.random_normal([5])

    log_joint = ed.make_log_joint_fn(mixture_of_real_and_int)
    actual_log_prob = true_log_joint(loc_value, flip_value, x_value)
    expected_log_prob = log_joint(loc=loc_value, flip=flip_value, x=x_value)

    actual_log_prob_, expected_log_prob_ = self.evaluate(
        [actual_log_prob, expected_log_prob])
    self.assertEqual(actual_log_prob_, expected_log_prob_)

    loc_value = 1.2
    flip_value = tf.constant(0)
    x_value = tf.random_normal([3])

    actual_log_prob = true_log_joint(loc_value, flip_value, x_value)
    expected_log_prob = log_joint(loc=loc_value, flip=flip_value, x=x_value)

    actual_log_prob_, expected_log_prob_ = self.evaluate(
        [actual_log_prob, expected_log_prob])
    self.assertEqual(actual_log_prob_, expected_log_prob_)

  def testMakeLogJointFnTemplate(self):
    """Test `make_log_joint_fn` on program returned by tf.make_template."""
    def variational():
      loc = tf.get_variable("loc", [])
      qz = ed.Normal(loc=loc, scale=0.5, name="qz")
      return qz

    def true_log_joint(loc, qz):
      log_prob = tf.reduce_sum(tfd.Normal(loc=loc, scale=0.5).log_prob(qz))
      return log_prob

    qz_value = 1.23
    variational_template = tf.make_template("variational", variational)

    log_joint = ed.make_log_joint_fn(variational_template)
    expected_log_prob = log_joint(qz=qz_value)
    loc = tf.trainable_variables("variational")[0]
    actual_log_prob = true_log_joint(loc, qz_value)

    with self.cached_session() as sess:
      sess.run(tf.initialize_all_variables())
      actual_log_prob_, expected_log_prob_ = sess.run(
          [actual_log_prob, expected_log_prob])
      self.assertEqual(actual_log_prob_, expected_log_prob_)

  @tfe.run_test_in_graph_and_eager_modes
  def testMakeLogJointFnError(self):
    """Test `make_log_joint_fn` raises errors when `name`(s) not supplied."""
    def normal_with_unknown_mean():
      loc = ed.Normal(loc=0., scale=1., name="loc")
      x = ed.Normal(loc=loc, scale=0.5, sample_shape=5)
      return x

    loc_value = 0.3
    x_value = tf.random_normal([5])

    log_joint = ed.make_log_joint_fn(normal_with_unknown_mean)

    with self.assertRaises(KeyError):
      _ = log_joint(loc=loc_value, x=x_value)


if __name__ == "__main__":
  tf.test.main()
