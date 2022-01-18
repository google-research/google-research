# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

# Lint as: python3
"""Implementation of Ising models. Try to find maximum energy configuration."""

import gin
import numpy as np
from six.moves import range
import tensorflow.compat.v1 as tf

from amortized_bo import base_problem
from amortized_bo import domains


def _one_hot(inputs, depth):
  """Wrapper around tf.one_hot that passes through one-hot inputs unchanged."""

  if len(inputs.shape) == 3:
    return inputs
  else:
    return tf.one_hot(inputs, depth=depth)


def _fully_connected_ising_model_energy(variables, potentials):
  """Ising model with full-connected coupling graph.

  Args:
    variables: [batch_size, sequence_length] int array (np or Tensor) or
      [batch_size, sequence_length, vocab_size] array (corresponding to one-hot
      vectors).
    potentials: [sequence_length, sequence_length, vocab_size, vocab_size] float
      array (np or Tensor).

  Returns:
    [batch_size] Tensor of energy.
  """
  variables = np.asarray(variables, dtype=int)
  vocab_size = potentials.shape[-1]
  onehot = _one_hot(variables, depth=vocab_size)
  return tf.einsum('bim,bjn,ijmn->b', onehot, onehot, potentials)


def _locally_connected_ising_model_energy(variables, potentials):
  """1D Ising model with couplings between adjacent variables.

  Args:
    variables: [batch_size, sequence_length] int array (np or Tensor) or
      [batch_size, sequence_length, vocab_size] array (corresponding to one-hot
      vectors).
    potentials: [sequence_length - 1, vocab_size, vocab_size]

  Returns:
    [batch_size] array of energy
  """
  variables = np.asarray(variables, dtype=int)
  vocab_size = potentials.shape[-1]
  oh = _one_hot(variables, depth=vocab_size)
  return tf.einsum('bim,bin,imn->b', oh[:, :-1, :], oh[:, 1:, :], potentials)


@gin.configurable
class IsingModel(base_problem.BaseProblem):
  """Maximize energy of a hidden Ising model.

  Attributes:
    length: Number of nodes.
    vocab_size: Number of possible values for each position.
    seed: Seed for generating potentials between positions.
    fully_connected: If true, simulate fully connected ising model. Otherwise
      locally connected.
    weight_variance: std of normal distribution used to generate potentials.
    **kwargs: additional args passed to the BaseProblem constructor.
  """

  def __init__(self,
               length=8,
               vocab_size=2,
               seed=0,
               fully_connected=False,
               weight_variance=1.0,
               **kwargs):
    self._domain = domains.FixedLengthDiscreteDomain(
        vocab_size=vocab_size, length=length)
    super(IsingModel, self).__init__(**kwargs)

    random = np.random.RandomState(seed=seed)
    if fully_connected:
      self.potentials = tf.constant(
          random.normal(
              size=[length, length, vocab_size, vocab_size],
              scale=weight_variance),
          dtype=tf.float32)
    else:
      self.potentials = tf.constant(
          random.normal(
              size=[length - 1, vocab_size, vocab_size], scale=weight_variance),
          dtype=tf.float32)

    self._fully_connected = fully_connected

  def compute_output_shape(self, input_shape):
    return (input_shape[0],)

  def __call__(self, sequences):
    if self._fully_connected:
      return _fully_connected_ising_model_energy(sequences, self.potentials)
    else:
      return _locally_connected_ising_model_energy(sequences, self.potentials)


def _alternating_sequence(token1, token2, length):
  """Make alternating sequence of token1 and token2 with specified length."""

  return [(token2 if i % 2 else token1) for i in range(length)]


@gin.configurable
class AlternatingChainIsingModel(base_problem.BaseProblem):
  """Ising model with a controllable number of isolated local optima.

  Suppose the model's vocabulary is {A, B, C, D, ...}. We break the vocabulary
  into pairs {(A, B), (C, D), ...}. The energy function counts the number of
  times elements of a pair are next to each other (e.g., the number of times a B
  is next to an A or a C is next to a D) in the sequence.

  Note that there are multiple isolated local optima:
  ABABAB, BABABA, CDCDCD, DCDCDC.

  This problem serves as a benchmark for solvers' ability to find all local
  optima.

  Attributes:
    length: Number of nodes.
    vocab_size: Number of possible values for each position. Must be even.
    **kwargs: additional args passed to the BaseProblem constructor.
  """

  def __init__(self, length=4, vocab_size=2, **kwargs):
    self._domain = domains.FixedLengthDiscreteDomain(
        vocab_size=vocab_size, length=length)
    super(AlternatingChainIsingModel, self).__init__(**kwargs)

    if vocab_size % 2:
      raise ValueError('vocab_size must be even for '
                       'AlternatingChainIsingModel.')

    potentials = np.zeros(shape=[vocab_size, vocab_size], dtype=np.float32)
    for i in range(0, vocab_size, 2):
      potentials[i][i + 1] = 1.
      potentials[i + 1][i] = 1.

    self.potentials = tf.tile(potentials[tf.newaxis, :, :], [length - 1, 1, 1])

    self._global_optima = self._get_global_optima()

  def _get_global_optima(self):
    optima = []
    for i in range(0, self._domain.vocab_size, 2):
      optima.append(
          _alternating_sequence(
              token1=i, token2=(i + 1), length=self._domain.length))
      optima.append(
          _alternating_sequence(
              token1=(i + 1), token2=i, length=self._domain.length))
    return optima

  def compute_metrics(self, population, fast_only=False):
    del fast_only
    return {
        'fraction_of_global_optima_found':
            np.mean(
                np.float32(population.contains_structures(self._global_optima)))
    }

  def __call__(self, sequences):
    return _locally_connected_ising_model_energy(sequences, self.potentials)
