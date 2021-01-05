# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Matrix connection utilities."""
import abc
import numpy as np
import six


class Connector(six.with_metaclass(abc.ABCMeta)):
  """Defines API for a weight connector."""

  def __init__(self, sparsity, round_to=1):
    """Initialization for weight connector.

    This method can be overridden to save input keyword arguments
    for the specific conenctor.

    Args:
      sparsity: Desired sparsity for the weight matrix.
      round_to: The number of nonzeros to round up to.
    """
    if sparsity < 0.0 or sparsity >= 1.0:
      raise ValueError("Sparsity should be >= 0 and < 1.0.")
    self.sparsity = sparsity
    self.round_to = round_to

  @abc.abstractmethod
  def __call__(self, dense_matrix):
    pass


class Uniform(Connector):
  """Uniformly samples which weights should be nonzero."""

  def __call__(self, dense_weights):
    """Masks weights selected uniformly from `dense_weights`.

    Args:
      dense_weights: Numpy array of the dense weight matrix.

    Returns:
      A numpy array with a proportion of the weights set to
      zero.
    """
    if self.sparsity == 0.0:
      return dense_weights

    # Select (without replacement) the weights that
    # should be set to zero.
    num_dormant = int(round(self.sparsity * dense_weights.size))

    if self.round_to > 1:
      nnz = dense_weights.size - num_dormant
      nnz = (nnz + self.round_to - 1) // self.round_to * self.round_to
      num_dormant = dense_weights.size - nnz

    dormant_mask = np.random.choice(
        dense_weights.size, num_dormant, replace=False)

    weights_shape = dense_weights.shape
    dense_weights = np.reshape(dense_weights, [-1])
    dense_weights[dormant_mask] = 0.0
    sparse_weights = np.reshape(dense_weights, weights_shape)
    return sparse_weights
