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

"""Tests for generated random variables."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_probability import distributions as tfd
import simple_probabilistic_programming as ed

tfe = tf.contrib.eager


class GeneratedRandomVariablesTest(parameterized.TestCase, tf.test.TestCase):

  @tfe.run_test_in_graph_and_eager_modes
  def testBernoulliDoc(self):
    self.assertGreater(len(ed.Bernoulli.__doc__), 0)
    self.assertIn(inspect.cleandoc(tfd.Bernoulli.__init__.__doc__),
                  ed.Bernoulli.__doc__)
    self.assertEqual(ed.Bernoulli.__name__, "Bernoulli")

  @parameterized.named_parameters(
      {"testcase_name": "1d_rv_1d_event", "logits": np.zeros(1), "n": [1]},
      {"testcase_name": "1d_rv_5d_event", "logits": np.zeros(1), "n": [5]},
      {"testcase_name": "5d_rv_1d_event", "logits": np.zeros(5), "n": [1]},
      {"testcase_name": "5d_rv_5d_event", "logits": np.zeros(5), "n": [5]},
  )
  @tfe.run_test_in_graph_and_eager_modes
  def testBernoulliLogProb(self, logits, n):
    rv = ed.Bernoulli(logits)
    dist = tfd.Bernoulli(logits)
    x = rv.distribution.sample(n)
    rv_log_prob, dist_log_prob = self.evaluate(
        [rv.distribution.log_prob(x), dist.log_prob(x)])
    self.assertAllEqual(rv_log_prob, dist_log_prob)

  @parameterized.named_parameters(
      {"testcase_name": "0d_rv_0d_sample",
       "logits": 0.,
       "n": 1},
      {"testcase_name": "0d_rv_1d_sample",
       "logits": 0.,
       "n": [1]},
      {"testcase_name": "1d_rv_1d_sample",
       "logits": np.array([0.]),
       "n": [1]},
      {"testcase_name": "1d_rv_5d_sample",
       "logits": np.array([0.]),
       "n": [5]},
      {"testcase_name": "2d_rv_1d_sample",
       "logits": np.array([-0.2, 0.8]),
       "n": [1]},
      {"testcase_name": "2d_rv_5d_sample",
       "logits": np.array([-0.2, 0.8]),
       "n": [5]},
  )
  @tfe.run_test_in_graph_and_eager_modes
  def testBernoulliSample(self, logits, n):
    rv = ed.Bernoulli(logits)
    dist = tfd.Bernoulli(logits)
    self.assertEqual(rv.distribution.sample(n).shape, dist.sample(n).shape)

  @parameterized.named_parameters(
      {"testcase_name": "0d_bernoulli",
       "rv": ed.Bernoulli(probs=0.5),
       "sample_shape": [],
       "batch_shape": [],
       "event_shape": []},
      {"testcase_name": "2d_bernoulli",
       "rv": ed.Bernoulli(tf.zeros([2, 3])),
       "sample_shape": [],
       "batch_shape": [2, 3],
       "event_shape": []},
      {"testcase_name": "2x0d_bernoulli",
       "rv": ed.Bernoulli(probs=0.5, sample_shape=2),
       "sample_shape": [2],
       "batch_shape": [],
       "event_shape": []},
      {"testcase_name": "2x1d_bernoulli",
       "rv": ed.Bernoulli(probs=0.5, sample_shape=[2, 1]),
       "sample_shape": [2, 1],
       "batch_shape": [],
       "event_shape": []},
      {"testcase_name": "3d_dirichlet",
       "rv": ed.Dirichlet(tf.zeros(3)),
       "sample_shape": [],
       "batch_shape": [],
       "event_shape": [3]},
      {"testcase_name": "2x3d_dirichlet",
       "rv": ed.Dirichlet(tf.zeros([2, 3])),
       "sample_shape": [],
       "batch_shape": [2],
       "event_shape": [3]},
      {"testcase_name": "1x3d_dirichlet",
       "rv": ed.Dirichlet(tf.zeros(3), sample_shape=1),
       "sample_shape": [1],
       "batch_shape": [],
       "event_shape": [3]},
      {"testcase_name": "2x1x3d_dirichlet",
       "rv": ed.Dirichlet(tf.zeros(3), sample_shape=[2, 1]),
       "sample_shape": [2, 1],
       "batch_shape": [],
       "event_shape": [3]},
  )
  @tfe.run_test_in_graph_and_eager_modes
  def testShape(self, rv, sample_shape, batch_shape, event_shape):
    self.assertEqual(rv.shape, sample_shape + batch_shape + event_shape)
    self.assertEqual(rv.sample_shape, sample_shape)
    self.assertEqual(rv.distribution.batch_shape, batch_shape)
    self.assertEqual(rv.distribution.event_shape, event_shape)

  def _testValueShapeAndDtype(self, cls, value, **kwargs):
    rv = cls(value=value, **kwargs)
    value_shape = rv.value.shape
    expected_shape = rv.sample_shape.concatenate(
        rv.distribution.batch_shape).concatenate(rv.distribution.event_shape)
    self.assertEqual(value_shape, expected_shape)
    self.assertEqual(rv.distribution.dtype, rv.value.dtype)

  @parameterized.parameters(
      {"cls": ed.Normal, "value": 2, "kwargs": {"loc": 0.5, "scale": 1.0}},
      {"cls": ed.Normal, "value": [2],
       "kwargs": {"loc": [0.5], "scale": [1.0]}},
      {"cls": ed.Poisson, "value": 2, "kwargs": {"rate": 0.5}},
  )
  @tfe.run_test_in_graph_and_eager_modes
  def testValueShapeAndDtype(self, cls, value, kwargs):
    self._testValueShapeAndDtype(cls, value, **kwargs)

  @tfe.run_test_in_graph_and_eager_modes
  def testValueMismatchRaises(self):
    with self.assertRaises(ValueError):
      self._testValueShapeAndDtype(ed.Normal, 2, loc=[0.5, 0.5], scale=1.0)
    with self.assertRaises(ValueError):
      self._testValueShapeAndDtype(ed.Normal, 2, loc=[0.5], scale=[1.0])
    with self.assertRaises(ValueError):
      self._testValueShapeAndDtype(
          ed.Normal, np.zeros([10, 3]), loc=[0.5, 0.5], scale=[1.0, 1.0])

  def testValueUnknownShape(self):
    # should not raise error
    ed.Bernoulli(probs=0.5, value=tf.placeholder(tf.int32))

  @tfe.run_test_in_graph_and_eager_modes
  def testMakeRandomVariable(self):
    """Tests that manual wrapping is the same as the built-in solution."""
    custom_normal = ed.make_random_variable(tfd.Normal)

    def model_builtin():
      return ed.Normal(1., 0.1, name="x")

    def model_wrapped():
      return custom_normal(1., 0.1, name="x")

    log_joint_builtin = ed.make_log_joint_fn(model_builtin)
    log_joint_wrapped = ed.make_log_joint_fn(model_wrapped)
    self.assertEqual(self.evaluate(log_joint_builtin(x=7.)),
                     self.evaluate(log_joint_wrapped(x=7.)))

if __name__ == "__main__":
  tf.test.main()
