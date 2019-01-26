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

"""Tests for sim.py in the exp_framework module."""
from absl.testing import absltest
import numpy as np
from sparse_data.data import sim

NUM_DATASET = 5
NUM_REPLICATE = 5
SEED = 2462723


def setUpModule():
  np.random.seed(SEED)


class TestSim(absltest.TestCase):

  def setUp(self):
    super(TestSim, self).setUp()
    self.init_method = None  # children should define this

  def test_reproducability(self):
    if self.init_method is None:
      return

    # pylint: disable=not-callable
    datasets = [self.init_method() for _ in range(NUM_DATASET)]

    # check reproducability of get() function
    for _ in range(NUM_REPLICATE):
      xs, ys = [], []
      for d in datasets:
        d.reset()
        x, y = d.generate()
        xs.append(x)
        ys.append(y)
        np.random.randn()  # make calls to global np.random RNG

      for i in range(NUM_DATASET - 1):
        self.assertTrue(np.array_equal(xs[i], xs[i + 1]))
        self.assertTrue(np.array_equal(ys[i], ys[i + 1]))

    # check reproducability of generate() function
    for _ in range(NUM_REPLICATE):
      x_trains, y_trains, x_tests, y_tests = [], [], [], []
      for d in datasets:
        d.reset()
        x_train, y_train, x_test, y_test = d.get()
        x_trains.append(x_train)
        y_trains.append(y_train)
        x_tests.append(x_test)
        y_tests.append(y_test)
        np.random.randn()  # make calls to global np.random RNG

      for i in range(NUM_DATASET - 1):
        self.assertTrue(np.array_equal(x_trains[i], x_trains[i + 1]))
        self.assertTrue(np.array_equal(y_trains[i], y_trains[i + 1]))
        self.assertTrue(np.array_equal(x_tests[i], x_tests[i + 1]))
        self.assertTrue(np.array_equal(y_tests[i], y_tests[i + 1]))


class TestLinear(TestSim):

  def setUp(self):
    super(TestLinear, self).setUp()
    self.init_method = sim.LinearSimulation

  def test_shape(self):
    num_sample = np.random.randint(10, 20)
    num_feature = np.random.randint(10, 20)
    problem = 'classification'

    for _ in range(NUM_REPLICATE):
      d = self.init_method(
          num_sample=num_sample, num_feature=num_feature, problem=problem)
      d.reset()
      x, y = d.generate()

      self.assertEqual(x.shape, (num_sample, num_feature))
      self.assertEqual(y.shape, (num_sample,))

  def test_sparsity(self):
    num_sample = 1000
    num_feature = 10
    problem = 'classification'

    for _ in range(NUM_REPLICATE):
      prop_nonzero = np.random.uniform(0.2, 0.8)
      d = self.init_method(
          num_sample=num_sample,
          num_feature=num_feature,
          prop_nonzero=prop_nonzero,
          problem=problem)
      d.reset()
      x, _ = d.generate()
      observed_prop_nonzero = np.true_divide(np.sum(x > 0), np.size(x))
      self.assertLess(
          np.abs(observed_prop_nonzero - prop_nonzero), 0.1 * prop_nonzero)


class TestCardinality(TestLinear):

  def setUp(self):
    super(TestCardinality, self).setUp()
    self.init_method = sim.CardinalitySimulation

  def test_shape(self):
    num_sample = np.random.randint(10, 20)
    num_feature = np.random.randint(10, 20) * 2  # should be even
    problem = 'classification'

    for _ in range(NUM_REPLICATE):
      d = self.init_method(
          num_sample=num_sample, num_feature=num_feature, problem=problem)
      d.reset()
      x, y = d.generate()

      self.assertEqual(x.shape, (num_sample, num_feature))
      self.assertEqual(y.shape, (num_sample,))

  def test_sparsity(self):
    pass


class TestSparsity(TestCardinality):

  def setUp(self):
    super(TestSparsity, self).setUp()
    self.init_method = sim.SparsitySimulation

  def test_sparsity(self):
    num_sample = 1000
    num_feature = 50
    problem = 'classification'

    for _ in range(NUM_REPLICATE):
      prop_nonzero = np.random.uniform(0.2, 0.8)
      d = self.init_method(
          num_sample=num_sample,
          num_feature=num_feature,
          prop_nonzero=prop_nonzero,
          problem=problem)
      d.reset()
      x, _ = d.generate()
      x_inf = x[:, :num_feature / 2]
      observed_prop_nonzero = np.true_divide(np.sum(x_inf > 0), np.size(x_inf))
      self.assertLess(
          np.abs(observed_prop_nonzero - prop_nonzero), 0.1 * prop_nonzero)


class TestMultiplicative(TestLinear):

  def setUp(self):
    super(TestMultiplicative, self).setUp()
    self.init_method = sim.MultiplicativeSimulation

  def test_shape(self):
    orders = range(1, 10)
    problem = 'classification'

    for _ in range(NUM_REPLICATE):
      num_sample = np.random.randint(10, 20)
      num_group_per_order = np.random.randint(10, 20)
      num_feature = np.sum([o * num_group_per_order for o in orders],
                           dtype=np.int)
      d = self.init_method(
          num_sample=num_sample,
          num_feature=num_feature,
          orders=orders,
          problem=problem)

      d.reset()
      x, y = d.generate()

      self.assertEqual(x.shape, (num_sample, num_feature))
      self.assertEqual(y.shape, (num_sample,))

  def test_sparsity(self):
    num_sample = 1000
    num_group_per_order = 10
    orders = range(1, 10)
    problem = 'classification'
    num_feature = np.sum([o * num_group_per_order for o in orders],
                         dtype=np.int)

    for _ in range(NUM_REPLICATE):
      prop_nonzero = np.random.uniform(0.2, 0.8)
      d = self.init_method(
          num_sample=num_sample,
          num_feature=num_feature,
          orders=orders,
          prop_nonzero=prop_nonzero,
          problem=problem)
      d.reset()
      x, _ = d.generate()
      observed_prop_nonzero = np.true_divide(np.sum(x > 0), np.size(x))
      self.assertLess(
          np.abs(observed_prop_nonzero - prop_nonzero), 0.1 * prop_nonzero)


class TestXOR(TestLinear):

  def setUp(self):
    super(TestXOR, self).setUp()
    self.init_method = sim.XORSimulation

  def test_shape(self):
    problem = 'classification'

    for _ in range(NUM_REPLICATE):
      num_sample = np.random.randint(10, 20)
      num_features = 2 * np.random.randint(10, 20)
      d = self.init_method(
          num_sample=num_sample, num_feature=num_features, problem=problem)
      d.reset()
      x, y = d.generate()

      self.assertEqual(x.shape, (num_sample, num_features))
      self.assertEqual(y.shape, (num_sample,))

  def test_sparsity(self):
    num_sample = 1000
    num_pair = 10
    problem = 'classification'

    for _ in range(NUM_REPLICATE):
      prop_nonzero = np.random.uniform(0.2, 0.8)
      d = self.init_method(
          num_sample=num_sample,
          num_feature=num_pair / 2,
          prop_nonzero=prop_nonzero,
          problem=problem)
      d.reset()
      x, _ = d.generate()
      observed_prop_nonzero = np.true_divide(np.sum(x > 0), np.size(x))
      self.assertLess(
          np.abs(observed_prop_nonzero - prop_nonzero), 0.1 * prop_nonzero)


class TestFunctions(absltest.TestCase):

  def test_continuous_to_binary(self):
    # TODO(jisungkim) add more tests here
    y = [0, 1, 2, 3, 4, 5]
    exp_y_squashed = [0, 0, 0, 1, 1, 1]
    self.assertTrue(
        np.array_equal(exp_y_squashed,
                       sim.continuous_to_binary(y, squashing='linear')))


if __name__ == '__main__':
  absltest.main()
