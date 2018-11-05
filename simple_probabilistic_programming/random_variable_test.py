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

"""Tests for random variable."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import simple_probabilistic_programming as ed

tfd = tfp.distributions
tfe = tf.contrib.eager


class FakeDistribution(tfd.Distribution):
  """Fake distribution class for testing."""

  def __init__(self):
    super(FakeDistribution, self).__init__(
        dtype=None,
        reparameterization_type=tfd.FULLY_REPARAMETERIZED,
        validate_args=False,
        allow_nan_stats=True)


class RandomVariableTest(parameterized.TestCase, tf.test.TestCase):

  @tfe.run_test_in_graph_and_eager_modes
  def testConstructor(self):
    x = ed.RandomVariable(tfd.Poisson(rate=tf.ones([2, 5])),
                          value=tf.ones([2, 5]))
    x_sample, x_value = self.evaluate([tf.convert_to_tensor(x), x.value])
    self.assertAllEqual(x_sample, x_value)
    with self.assertRaises(ValueError):
      _ = ed.RandomVariable(tfd.Bernoulli(probs=0.5),
                            value=tf.zeros([2, 5], dtype=tf.int32))
    x = ed.RandomVariable(FakeDistribution())
    with self.assertRaises(NotImplementedError):
      _ = x.value

  @tfe.run_test_in_graph_and_eager_modes
  def testGradientsFirstOrder(self):
    f = lambda x: 2. * x
    x = ed.RandomVariable(tfd.Normal(0., 1.))
    y = f(x)
    if tfe.in_eager_mode():
      df = tfe.gradients_function(f)
      (z,) = df(x)
    else:
      (z,) = tf.gradients(y, x)
    self.assertEqual(self.evaluate(z), 2.)

  @tfe.run_test_in_graph_and_eager_modes
  def testGradientsSecondOrder(self):
    f = lambda x: 2 * (x ** 2)
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0))
    y = f(x)
    if tfe.in_eager_mode():
      df = tfe.gradients_function(f)
      d2f = tfe.gradients_function(lambda x: df(x)[0])
      (z,) = d2f(x)
    else:
      (z,) = tf.gradients(y, x)
      (z,) = tf.gradients(z, x)
    self.assertEqual(self.evaluate(z), 4.0)

  @tfe.run_test_in_graph_and_eager_modes
  def testStr(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0), value=1.234)
    if tfe.in_eager_mode():
      pattern = "RandomVariable(\"1.234\", shape=(), dtype=float32"
    else:
      pattern = "RandomVariable(\"Normal/\", shape=(), dtype=float32"
    regexp = re.escape(pattern)
    self.assertRegexpMatches(str(x), regexp)

  @tfe.run_test_in_graph_and_eager_modes
  def testRepr(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0), value=1.234)
    if tfe.in_eager_mode():
      string = ("<ed.RandomVariable 'Normal/' shape=() "
                "dtype=float32 numpy=1.234>")
    else:
      string = "<ed.RandomVariable 'Normal/' shape=() dtype=float32>"
    self.assertEqual(repr(x), string)

  @tfe.run_test_in_graph_and_eager_modes
  def testNumpy(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0), value=1.23)
    if tfe.in_eager_mode():
      self.assertEqual(x.numpy(), tf.constant(1.23).numpy())
    else:
      with self.assertRaises(NotImplementedError):
        _ = x.numpy()

  @tfe.run_test_in_graph_and_eager_modes
  def testOperatorsAdd(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0))
    y = 5.0
    z = x + y
    z_value = x.value + y
    z_eval, z_value_eval = self.evaluate([z, z_value])
    self.assertAllEqual(z_eval, z_value_eval)

  @tfe.run_test_in_graph_and_eager_modes
  def testOperatorsRadd(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0))
    y = 5.0
    z = y + x
    z_value = y + x.value
    z_eval, z_value_eval = self.evaluate([z, z_value])
    self.assertAllEqual(z_eval, z_value_eval)

  @tfe.run_test_in_graph_and_eager_modes
  def testOperatorsSub(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0))
    y = 5.0
    z = x - y
    z_value = x.value - y
    z_eval, z_value_eval = self.evaluate([z, z_value])
    self.assertAllEqual(z_eval, z_value_eval)

  @tfe.run_test_in_graph_and_eager_modes
  def testOperatorsRsub(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0))
    y = 5.0
    z = y - x
    z_value = y - x.value
    z_eval, z_value_eval = self.evaluate([z, z_value])
    self.assertAllEqual(z_eval, z_value_eval)

  @tfe.run_test_in_graph_and_eager_modes
  def testOperatorsMul(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0))
    y = 5.0
    z = x * y
    z_value = x.value * y
    z_eval, z_value_eval = self.evaluate([z, z_value])
    self.assertAllEqual(z_eval, z_value_eval)

  @tfe.run_test_in_graph_and_eager_modes
  def testOperatorsRmul(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0))
    y = 5.0
    z = y * x
    z_value = y * x.value
    z_eval, z_value_eval = self.evaluate([z, z_value])
    self.assertAllEqual(z_eval, z_value_eval)

  @tfe.run_test_in_graph_and_eager_modes
  def testOperatorsDiv(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0))
    y = 5.0
    z = x / y
    z_value = x.value / y
    z_eval, z_value_eval = self.evaluate([z, z_value])
    self.assertAllEqual(z_eval, z_value_eval)

  @tfe.run_test_in_graph_and_eager_modes
  def testOperatorsRdiv(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0))
    y = 5.0
    z = y / x
    z_value = y / x.value
    z_eval, z_value_eval = self.evaluate([z, z_value])
    self.assertAllEqual(z_eval, z_value_eval)

  @tfe.run_test_in_graph_and_eager_modes
  def testOperatorsFloordiv(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0))
    y = 5.0
    z = x // y
    z_value = x.value // y
    z_eval, z_value_eval = self.evaluate([z, z_value])
    self.assertAllEqual(z_eval, z_value_eval)

  @tfe.run_test_in_graph_and_eager_modes
  def testOperatorsRfloordiv(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0))
    y = 5.0
    z = y // x
    z_value = y // x.value
    z_eval, z_value_eval = self.evaluate([z, z_value])
    self.assertAllEqual(z_eval, z_value_eval)

  @tfe.run_test_in_graph_and_eager_modes
  def testOperatorsMod(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0))
    y = 5.0
    z = x % y
    z_value = x.value % y
    z_eval, z_value_eval = self.evaluate([z, z_value])
    self.assertAllEqual(z_eval, z_value_eval)

  @tfe.run_test_in_graph_and_eager_modes
  def testOperatorsRmod(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0))
    y = 5.0
    z = y % x
    z_value = y % x.value
    z_eval, z_value_eval = self.evaluate([z, z_value])
    self.assertAllEqual(z_eval, z_value_eval)

  @tfe.run_test_in_graph_and_eager_modes
  def testOperatorsLt(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0))
    y = 5.0
    z = x < y
    z_value = x.value < y
    z_eval, z_value_eval = self.evaluate([z, z_value])
    self.assertAllEqual(z_eval, z_value_eval)

  @tfe.run_test_in_graph_and_eager_modes
  def testOperatorsLe(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0))
    y = 5.0
    z = x <= y
    z_value = x.value <= y
    z_eval, z_value_eval = self.evaluate([z, z_value])
    self.assertAllEqual(z_eval, z_value_eval)

  @tfe.run_test_in_graph_and_eager_modes
  def testOperatorsGt(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0))
    y = 5.0
    z = x > y
    z_value = x.value > y
    z_eval, z_value_eval = self.evaluate([z, z_value])
    self.assertAllEqual(z_eval, z_value_eval)

  @tfe.run_test_in_graph_and_eager_modes
  def testOperatorsGe(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0))
    y = 5.0
    z = x >= y
    z_value = x.value >= y
    z_eval, z_value_eval = self.evaluate([z, z_value])
    self.assertAllEqual(z_eval, z_value_eval)

  @tfe.run_test_in_graph_and_eager_modes
  def testOperatorsGetitem(self):
    x = ed.RandomVariable(tfd.Normal(tf.zeros([3, 4]), tf.ones([3, 4])))
    z = x[0:2, 2:3]
    z_value = x.value[0:2, 2:3]
    z_eval, z_value_eval = self.evaluate([z, z_value])
    self.assertAllEqual(z_eval, z_value_eval)

  @tfe.run_test_in_graph_and_eager_modes
  def testOperatorsPow(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0))
    y = 5.0
    z = x ** y
    z_value = x.value ** y
    z_eval, z_value_eval = self.evaluate([z, z_value])
    self.assertAllEqual(z_eval, z_value_eval)

  @tfe.run_test_in_graph_and_eager_modes
  def testOperatorsRpow(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0))
    y = 5.0
    z = y ** x
    z_value = y ** x.value
    z_eval, z_value_eval = self.evaluate([z, z_value])
    self.assertAllEqual(z_eval, z_value_eval)

  @tfe.run_test_in_graph_and_eager_modes
  def testOperatorsNeg(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0))
    z = -x
    z_value = -x.value
    z_eval, z_value_eval = self.evaluate([z, z_value])
    self.assertAllEqual(z_eval, z_value_eval)

  @tfe.run_test_in_graph_and_eager_modes
  def testOperatorsAbs(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0))
    z = abs(x)
    z_value = abs(x.value)
    z_eval, z_value_eval = self.evaluate([z, z_value])
    self.assertAllEqual(z_eval, z_value_eval)

  @tfe.run_test_in_graph_and_eager_modes
  def testOperatorsHash(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0))
    y = 5.0
    self.assertNotEqual(hash(x), hash(y))
    self.assertEqual(hash(x), id(x))

  @tfe.run_test_in_graph_and_eager_modes
  def testOperatorsEq(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0))
    self.assertEqual(x, x)

  @tfe.run_test_in_graph_and_eager_modes
  def testOperatorsNe(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0))
    y = 5.0
    self.assertNotEqual(x, y)

  @tfe.run_test_in_graph_and_eager_modes
  def testOperatorsBoolNonzero(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0))
    with self.assertRaises(TypeError):
      _ = not x

  @tfe.run_test_in_graph_and_eager_modes
  def testArrayPriority(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 1.0))
    y = np.array(5.0, dtype=np.float32)
    z = y / x
    z_value = y / x.value
    z_eval, z_value_eval = self.evaluate([z, z_value])
    self.assertAllEqual(z_eval, z_value_eval)

  @tfe.run_test_in_graph_and_eager_modes
  def testConvertToTensor(self):
    x = ed.RandomVariable(tfd.Normal(0.0, 0.1))
    with self.assertRaises(ValueError):
      _ = tf.convert_to_tensor(x, dtype=tf.int32)

  def testSessionEval(self):
    with self.cached_session() as sess:
      x = ed.RandomVariable(tfd.Normal(0.0, 0.1))
      x_ph = tf.placeholder(tf.float32, [])
      y = ed.RandomVariable(tfd.Normal(x_ph, 0.1))
      self.assertLess(x.eval(), 5.0)
      self.assertLess(x.eval(sess), 5.0)
      self.assertLess(x.eval(feed_dict={x_ph: 100.0}), 5.0)
      self.assertGreater(y.eval(feed_dict={x_ph: 100.0}), 5.0)
      self.assertGreater(y.eval(sess, feed_dict={x_ph: 100.0}), 5.0)
      self.assertRaises(tf.errors.InvalidArgumentError, y.eval)
      self.assertRaises(tf.errors.InvalidArgumentError, y.eval, sess)

  def testSessionRun(self):
    with self.cached_session() as sess:
      x = ed.RandomVariable(tfd.Normal(0.0, 0.1))
      x_ph = tf.placeholder(tf.float32, [])
      y = ed.RandomVariable(tfd.Normal(x_ph, 0.1))
      self.assertLess(sess.run(x), 5.0)
      self.assertLess(sess.run(x, feed_dict={x_ph: 100.0}), 5.0)
      self.assertGreater(sess.run(y, feed_dict={x_ph: 100.0}), 5.0)
      self.assertRaises(tf.errors.InvalidArgumentError, sess.run, y)

  @parameterized.parameters(
      {"rv": ed.RandomVariable(tfd.Bernoulli(probs=0.5)),
       "sample_shape": [],
       "batch_shape": [],
       "event_shape": []},
      {"rv": ed.RandomVariable(tfd.Bernoulli(tf.zeros([2, 3]))),
       "sample_shape": [],
       "batch_shape": [2, 3],
       "event_shape": []},
      {"rv": ed.RandomVariable(tfd.Bernoulli(probs=0.5), sample_shape=2),
       "sample_shape": [2],
       "batch_shape": [],
       "event_shape": []},
      {"rv": ed.RandomVariable(tfd.Bernoulli(probs=0.5), sample_shape=[2, 1]),
       "sample_shape": [2, 1],
       "batch_shape": [],
       "event_shape": []},
      {"rv": ed.RandomVariable(tfd.Bernoulli(probs=0.5),
                               sample_shape=tf.constant([2])),
       "sample_shape": [2],
       "batch_shape": [],
       "event_shape": []},
      {"rv": ed.RandomVariable(tfd.Bernoulli(probs=0.5),
                               sample_shape=tf.constant([2, 4])),
       "sample_shape": [2, 4],
       "batch_shape": [],
       "event_shape": []},
  )
  @tfe.run_test_in_graph_and_eager_modes
  def testShape(self, rv, sample_shape, batch_shape, event_shape):
    self.assertEqual(rv.shape, sample_shape + batch_shape + event_shape)
    self.assertEqual(rv.shape, rv.shape)
    self.assertEqual(rv.sample_shape, sample_shape)
    self.assertEqual(rv.distribution.batch_shape, batch_shape)
    self.assertEqual(rv.distribution.event_shape, event_shape)

  @tfe.run_test_in_graph_and_eager_modes
  def testRandomTensorSample(self):
    num_samples = tf.cast(tfd.Poisson(rate=5.).sample(), tf.int32)
    _ = ed.RandomVariable(tfd.Normal(loc=0.0, scale=1.0),
                          sample_shape=num_samples)


if __name__ == "__main__":
  tf.test.main()
