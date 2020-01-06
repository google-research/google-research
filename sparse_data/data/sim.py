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

"""Generates simulated datasets.

The simulated experimental datasets examine a variety of issues: the effect of
sparsity, the effect of cardinality, and the suitability of learning models
for smooth linear and multiplicative functions.
"""

import numpy as np
import tensorflow.compat.v1 as tf  # tf
from tensorflow.contrib import layers as contrib_layers

SEED = 49 + 32 + 67 + 111 + 114 + 32 + 49 + 51 + 58 + 52 + 45 + 56
DEFAULT_NUM_SAMPLE = 5000
DEFAULT_NUM_FEATURE = 1000


class Simulation(object):
  """Base simulation experiment class."""

  def __init__(self):
    self._rng = np.random.RandomState()
    self._rng.seed(SEED)
    self.was_reset = False

  def generate(self):
    if not self.was_reset:
      raise RuntimeError('Dataset has not been reset. Must call reset() at '
                         'least once before calling generate()')
    raise NotImplementedError

  def reset(self):
    self.was_reset = True

  def get(self):
    x_train, y_train = self.generate()
    x_test, y_test = self.generate()
    return x_train, y_train, x_test, y_test

  def get_feature_columns(self):
    raise NotImplementedError


class LinearSimulation(Simulation):
  """Synthetic dataset with linear relationships between target and features."""

  def __init__(self,
               num_sample=DEFAULT_NUM_SAMPLE,
               num_feature=DEFAULT_NUM_FEATURE,
               prop_nonzero=0.05,
               problem='classification'):
    """Initializes the dataset.

    Args:
      num_sample: number of samples
      num_feature: number of features
      prop_nonzero: float proportion of all features which should be non-zero
      problem: str type of learning problem; values = 'classification',
        'regression'
    """
    assert num_sample > 0
    assert num_feature > 0
    assert problem in ['classification', 'regression']

    super(LinearSimulation, self).__init__()

    self._num_feature = num_feature
    self._num_sample = num_sample
    self._prop_nonzero = prop_nonzero
    self._problem = problem

  def reset(self):
    """Resets dataset so that the next call to get() is a different problem."""
    self._coefficients = 10 * (self._rng.rand(self._num_feature) - 0.5)
    self.was_reset = True

  def generate(self):
    """Generates the dataset.

    Returns:
      x: np.array
        array of features
      y: np.array
        1-D array of targets

    Raises:
      RuntimeError: if ...
    """
    if not self.was_reset:
      raise RuntimeError('Dataset has not been reset. Must call reset() at '
                         'least once before calling generate()')

    x = self._generate_x()

    y = x * np.tile(self._coefficients, (self._num_sample, 1))
    y = np.sum(y, axis=1, dtype=np.float)

    noise_coef = np.std(y)
    noise = noise_coef * (self._rng.rand(self._num_sample) - 0.5)
    y += noise

    if self._problem == 'classification':
      y = continuous_to_binary(y)
    else:
      y = y.astype(np.float32)

    return x.astype(np.float32), y

  def _generate_x(self):
    x = self._rng.rand(self._num_sample, self._num_feature)
    x_mask = self._rng.rand(*x.shape) < self._prop_nonzero
    x *= x_mask
    return x

  def get_feature_columns(self):
    """Get a list of feature column names."""
    feature_columns = [
        'idx_{}.coef_{:.3f}'.format(i, self._coefficients[i])
        for i in range(self._num_feature)
    ]
    return [contrib_layers.real_valued_column(fc) for fc in feature_columns]

  def oracle_predict(self, x):
    """Predicts targets of given data with the perfect oracle.

    Args:
      x: np.array array of features

    Returns:
      y: np.array
        1-D array of predicted targets
    """
    y = x * np.tile(self._coefficients, (np.size(x, 0), 1))
    y = np.sum(y, axis=1)

    if self._problem == 'classification':
      y = continuous_to_binary(y)
    else:
      y = y.astype(np.float32)

    return y


class SparsitySimulation(LinearSimulation):
  """Synthetic dataset examining the effect of sparsity.

  Harder to learn than the EasySparsitySimulation dataset.
  """

  def __init__(self,
               num_sample=DEFAULT_NUM_SAMPLE,
               num_feature=DEFAULT_NUM_FEATURE,
               coef_factor=150,
               prop_nonzero=0.005,
               which_features='all',
               alternate=False,
               problem='classification'):
    """Initializes the dataset.

    There are two types of features: sparse features which have large
    coefficients (and are more informative), and dense features which have small
    coefficients (and are less informative).

    Args:
      num_sample: number of samples
      num_feature: number of features; get's rounded (down) to an even number
      coef_factor: int factor by which sparse feature coeffcients are larger
        than dense feature coefficients
      prop_nonzero: float proportion of informative features which should be
        non-zero
      which_features: str type of features to use; values = 'all', 'inform',
        'uninform'
      alternate: bool whether to perform the alternate experiment where the
        sparsity is the same for all features (for both features with large and
        small coefficients)
      problem: str type of learning problem; values = 'classification',
        'regression'
    """
    assert coef_factor > 0
    assert which_features in ['all', 'inform', 'uninform']
    if alternate:
      assert which_features == 'all'

    super(SparsitySimulation, self).__init__(
        num_sample=num_sample,
        num_feature=num_feature,
        prop_nonzero=prop_nonzero,
        problem=problem)

    self._num_feature = num_feature / 2 * 2
    self._num_inf_feature = self._num_feature / 2
    self._coef_factor = coef_factor
    self._which_features = which_features
    self._alternate = alternate

  def reset(self):
    """Resets dataset so that the next call to get() is a different problem."""
    # small coefficients \in [-1, 1]
    small_coef = 2 * (self._rng.rand(int(self._num_feature / 2)) - 0.5)

    # large coefficients \in [-coef_factor, coef_factor]
    large_coef = self._coef_factor * small_coef

    if self._which_features == 'all':
      self._coefficients = np.hstack([large_coef, small_coef])
    elif self._which_features == 'inform':
      self._coefficients = large_coef
    else:  # self._which_features == 'uninform'
      self._coefficients = small_coef

    self.was_reset = True

  def _generate_x(self):
    x_sparse = self._rng.choice(
        2, (self._num_sample, int(self._num_feature / 2)),
        p=[1 - self._prop_nonzero, self._prop_nonzero])
    if self._alternate:
      x_dense = self._rng.choice(
          2, (self._num_sample, int(self._num_feature / 2)),
          p=[1 - self._prop_nonzero, self._prop_nonzero])
    else:
      x_dense = self._rng.randint(
          2, size=(self._num_sample, int(self._num_feature / 2)))

    if self._which_features == 'all':
      return np.hstack([x_sparse, x_dense])
    elif self._which_features == 'inform':
      return x_sparse
    else:  # self._which_features == 'uninform':
      return x_dense

  def get_feature_columns(self):
    """Get a list of feature column names."""
    inf = [
        'inf.idx_{}.coef_{:.3f}'.format(i, self._coefficients[i])
        for i in range(self._num_inf_feature)
    ]
    uninf = [
        'uninf.idx_{}.coef_{:.3f}'.format(i, self._coefficients[i])
        for i in range(self._num_inf_feature, self._num_feature)
    ]
    return [contrib_layers.real_valued_column(fc) for fc in inf + uninf]


class CardinalitySimulation(SparsitySimulation):
  """Synthetic dataset examining the effect of cardinality."""

  def __init__(self,
               num_sample=DEFAULT_NUM_SAMPLE,
               num_feature=DEFAULT_NUM_FEATURE,
               coef_factor=150,
               prop_nonzero=0.05,
               which_features='all',
               alternate=False,
               problem='classification'):
    """Initializes the dataset.

    There are two types of features: low cardinality features which have
    large coefficients (and are more informative), and high cardinality features
    which have small coefficients (and are less informative).

    Args:
      num_sample: number of samples
      num_feature: number of features; get's rounded (down) to an even number
      coef_factor: int factor by which low cardinality feature coeffcients are
        larger than high cardinality coefficients
      prop_nonzero: float proportion of informative features which should be
        non-zero
      which_features: str type of features to use; values = 'all', 'inform',
        'uninform'
      alternate: bool whether to perform the alternate experiment where the
        cardinality is the same for all features (for both features with large
        and small coefficients)
      problem: str type of learning problem; values = 'classification',
        'regression'
    """
    super(CardinalitySimulation, self).__init__(
        num_sample=num_sample,
        num_feature=num_feature,
        coef_factor=coef_factor,
        alternate=alternate,
        problem=problem)

  def _generate_x(self):
    if self._alternate:  # alternate experiment when cardinality is same
      x_high_card = self._rng.randint(
          2, size=(self._num_sample, self._num_feature / 2))
    else:
      x_high_card = self._rng.rand(self._num_sample, self._num_feature / 2)
    x_low_card = self._rng.randint(
        2, size=(self._num_sample, self._num_feature / 2))

    if self._which_features == 'all':
      return np.hstack([x_low_card, x_high_card])
    elif self._which_features == 'inform':
      return x_low_card
    else:  # self._which_features == 'uninform':
      return x_high_card


class MultiplicativeSimulation(Simulation):
  """Synthetic dataset with multiplicative relationships between features."""

  def __init__(self,  # pylint: disable=dangerous-default-value,
               num_sample=DEFAULT_NUM_SAMPLE,
               num_feature=DEFAULT_NUM_FEATURE,
               orders=[2],
               prop_nonzero=0.05,
               problem='classification'):
    """Initializes the dataset.

    Args:
      num_sample: number of samples
      num_feature: number of features
      orders: [int] order of multiplicative features; for example, an
        order of 3 would represent that the product of 3 features is correlated
        with the target.
      prop_nonzero: float proportion of all features which should be non-zero
      problem: str type of learning problem; values = 'classification',
        'regression'
    """
    assert num_sample > 0
    assert num_feature > 0
    assert np.all(np.greater(orders, 0))
    assert problem in ['classification', 'regression']

    super(MultiplicativeSimulation, self).__init__()

    self._num_sample = num_sample
    self._orders = orders
    self._prop_nonzero = prop_nonzero
    self._problem = problem

    # Rounds number of features and groups per order to closest valid values.
    sum_orders = sum(orders)
    self._num_group_per_order = num_feature / sum_orders
    self._num_feature = self._num_group_per_order * sum_orders
    if num_feature != self._num_feature:
      tf.logging.warn('Number of features is rounded from %d to %d',
                      num_feature, self._num_feature)

  def reset(self):
    """Resets dataset so that the next call to get() is a different problem."""
    self._group_coefficients_by_order = {
        i: 10 * (self._rng.rand(self._num_group_per_order) - 0.5)
        for i in self._orders
    }
    self.was_reset = True

  def generate(self):
    """Generates the dataset.

    Returns:
      x: np.array
        array of features
      y: np.array
        1-D array of targets

    Raises:
      RuntimeError: if reset() is not called before generate() or get()
    """
    if not self.was_reset:
      raise RuntimeError('Dataset has not been reset. Must call reset() at '
                         'least once before calling generate()')

    x = np.zeros((self._num_sample, 0))
    y = np.zeros((self._num_sample,))

    for order in self._orders:
      x_order = self._rng.choice(
          2, (self._num_sample, self._num_group_per_order, order),
          p=[1 - self._prop_nonzero, self._prop_nonzero])
      x_prod = np.prod(x_order, axis=-1)

      coef = self._group_coefficients_by_order[order]

      y += np.sum(x_prod * np.tile(coef, (self._num_sample, 1)), axis=1)
      x = np.hstack([x, x_order.reshape(self._num_sample, -1)])

    noise_coef = np.std(y)
    noise = noise_coef * (self._rng.rand(self._num_sample) - 0.5)
    y += noise

    if self._problem == 'classification':
      y = continuous_to_binary(y)
    else:
      y = y.astype(np.float32)

    return x.astype(np.float32), y

  def get_feature_columns(self):
    """Get a list of feature column names."""
    out = []
    count, group = 0, 0
    for order in self._orders:
      for group_idx in range(self._num_group_per_order):
        for _ in range(order):
          out.append('mult_group_{}.idx_{}.order_{}.group_coef_{:.3}'.format(
              group, count, order,
              self._group_coefficients_by_order[order][group_idx]))
          count += 1
        group += 1
    return [contrib_layers.real_valued_column(fc) for fc in out]

  def oracle_predict(self, x):
    """Predicts targets of given data with the perfect oracle.

    Args:
      x: np.array array of features

    Returns:
      y: np.array
        1-D array of predicted targets
    """
    y = np.zeros((self._num_sample,))

    offset = 0
    for order in self._orders:
      x_order = x[:, offset:offset + order * self._num_group_per_order]
      x_order = x_order.reshape(self._num_sample, self._num_group_per_order,
                                order)
      x_prod = np.prod(x_order, axis=-1)

      coef = self._group_coefficients_by_order[order]

      y += np.sum(x_prod * np.tile(coef, (self._num_sample, 1)), axis=1)
      offset += order * self._num_group_per_order

    if self._problem == 'classification':
      y = continuous_to_binary(y)
    else:
      y = y.astype(np.float32)

    return y


class XORSimulation(Simulation):
  """Synthetic dataset with XOR interactions between features.

  This experiment becomes very easy if the features are sparse; this is because
  the event where two features in an XOR pair are both non-zero is very rare.
  """

  def __init__(self,
               num_sample=DEFAULT_NUM_SAMPLE,
               num_feature=DEFAULT_NUM_FEATURE,
               prop_nonzero=0.05,
               problem='classification'):
    """Initializes the dataset.

    Args:
      num_sample: number of samples
      num_feature: number of features
      prop_nonzero: float proportion of all features which should be non-zero
      problem: str type of learning problem; values = 'classification',
        'regression'
    """
    assert num_sample > 0
    assert num_feature > 0
    assert problem in ['classification', 'regression']

    super(XORSimulation, self).__init__()

    self._num_sample = num_sample
    self._prop_nonzero = prop_nonzero
    self._problem = problem

    # Rounds number of features and pairs to closest valid values.
    self._num_pair = num_feature / 2
    self._num_feature = self._num_pair * 2
    if num_feature != self._num_feature:
      tf.logging.warn('Number of features is rounded from %d to %d',
                      num_feature, self._num_feature)

  def reset(self):
    """Resets dataset so that the next call to get() is a different problem."""
    self._pair_coefficients = 10 * (self._rng.rand(self._num_pair) - 0.5)
    self.was_reset = True

  def generate(self):
    """Generates the dataset.

    Returns:
      x: np.array
        array of features
      y: np.array
        1-D array of targets

    Raises:
      RuntimeError: if reset() is not called before generate() or get()
    """
    if not self.was_reset:
      raise RuntimeError('Dataset has not been reset. Must call reset() at '
                         'least once before calling generate()')

    x1 = self._rng.choice(
        2, (self._num_sample, self._num_pair),
        p=[1 - self._prop_nonzero, self._prop_nonzero])
    x2 = self._rng.choice(
        2, (self._num_sample, self._num_pair),
        p=[1 - self._prop_nonzero, self._prop_nonzero])

    x = np.hstack([x1, x2])
    xor = np.logical_xor(x1, x2)

    y = np.sum(
        xor * np.tile(self._pair_coefficients, (self._num_sample, 1)), axis=1)

    noise_coef = np.std(y)
    noise = noise_coef * (self._rng.rand(self._num_sample) - 0.5)
    y += noise

    if self._problem == 'classification':
      y = continuous_to_binary(y)
    else:
      y = y.astype(np.float32)

    return x.astype(np.float32), y

  def get_feature_columns(self):
    """Get a list of feature column names."""
    num_feature = self._num_pair * 2
    x1_col = ['xorpair_{}.idx_{}'.format(i, i) for i in range(self._num_pair)]
    x2_col = [
        'xorpair_{}.idx_{}'.format(i - self._num_pair, i)
        for i in range(self._num_pair, num_feature)
    ]
    return [contrib_layers.real_valued_column(fc) for fc in x1_col + x2_col]

  def oracle_predict(self, x):
    """Predicts targets of given data with the perfect oracle.

    Args:
      x: np.array array of features

    Returns:
      y: np.array
        1-D array of predicted targets
    """
    pivot = np.size(x, 1) / 2
    x1 = x[:, :pivot]
    x2 = x[:, pivot:]

    xor = np.logical_xor(x1, x2)

    y = np.sum(
        xor * np.tile(self._pair_coefficients, (np.size(x, 0), 1)), axis=1)

    if self._problem == 'classification':
      y = continuous_to_binary(y)
    else:
      y = y.astype(np.float32)

    return y


def continuous_to_binary(y, squashing='linear'):
  """Squash continuous values to binary values.

  Args:
    y: np.array 1-D array of continuous values
    squashing: str type of squashing function; values = 'linear', 'sigmoid',
      'logistic'

  Returns:
    y_squashed: np.array
      1-D array of squashed values
  """
  if squashing == 'linear':
    y -= np.min(y)
    y = np.true_divide(y, np.max(y))
  elif squashing == 'sigmoid' or squashing == 'logistic':
    y = np.true_divide(1, 1 + np.exp(-y))

  # classes should be approximately balanced
  return (y > np.mean(y)).astype(np.int32)
