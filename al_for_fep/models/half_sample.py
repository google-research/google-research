# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Wrapper for running a model in a half sampling environment for Makita."""

import copy
import numpy as np
from sklearn import base


def orthogonal_array(orthogonal_array_size_log2):
  """Output is N by N array called X, where N is 2**orthogonal_array_size_log2.

  Args:
    orthogonal_array_size_log2: An int describing log2 of the number of data
      shards that the orthogonal array can cater for.

  Returns:
    output_array: A square array of size 2**k allocating shards to models.
  """
  n_shards = 2**orthogonal_array_size_log2
  k_intercept = orthogonal_array_size_log2 + 1
  generator_matrix = np.ones((k_intercept, n_shards), dtype=int)

  # Compute values for the generator matrix.
  for j in range(1, k_intercept):
    for i in range(n_shards):
      generator_matrix[j][i] = int(np.floor((i % 2**j) / 2**(j - 1)))

  # Calculate values of the orthogonal array as a modulo 2 outer product.
  output_array = np.matmul(
      np.transpose(generator_matrix[1:]), generator_matrix[1:]) % 2
  output_array[:, 0] = range(n_shards)
  assert np.sum(output_array[:, 1]) <= 0.5 * n_shards

  return output_array


class HalfSampleRegressor(base.RegressorMixin):
  """Model wrapper for half sampling."""

  def __init__(self, subestimator, shards_log2, add_estimators):
    """Initialization.

    Args:
      subestimator: Regressor to train on each subsample of data.
      shards_log2: Integer. Log2 of the number of shards to use for half
        sampling.
      add_estimators: Boolean. If true, estimators will be added to the model at
        each cycle. Random Forests and Gradient Boosted Machines will benefit
        from this.
    """
    self.subestimator = subestimator
    self.shards_log2 = shards_log2
    self.add_estimators = add_estimators

  def fit(self, train_x, train_y):
    """Perform model training on half sampling subsets of the data.

    Args:
      train_x: Numpy array. Input features for the model.
      train_y: Numpy array. Target values for training.
    """

    max_shard = 2**self.shards_log2

    shards = np.array(list(map(lambda x: x % max_shard, range(len(train_y)))))

    self.subestimator.fit(train_x, train_y)
    self.subestimators_ = []

    for split in np.transpose(orthogonal_array(self.shards_log2))[1:]:
      first_half_indices = np.isin(shards, np.where(split == 1)[0])
      second_half_indices = np.isin(shards, np.where(split == 0)[0])

      for indices in [first_half_indices, second_half_indices]:
        warm_model = copy.deepcopy(self.subestimator)
        if self.add_estimators:
          warm_model.n_estimators += warm_model.n_estimators
        warm_model.fit(train_x[indices], train_y[indices])

        self.subestimators_.append(warm_model)

  def predict(self, features):
    return np.array([model.predict(features) for model in self.subestimators_])

  def get_model(self):
    return self
