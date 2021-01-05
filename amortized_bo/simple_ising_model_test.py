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

# Lint as: python3
"""Tests for simple ising models."""

from absl.testing import parameterized
import numpy as np
from six.moves import range
import tensorflow.compat.v1 as tf

from amortized_bo import simple_ising_model


tf.enable_eager_execution()


def _manual_full_model(variables, potentials):
  """Manual loop to compute energy of a fully connected ising model."""
  variables = np.asarray(variables)
  potentials = np.asarray(potentials)
  total_energy = np.zeros(len(variables))
  for bidx, vector in enumerate(variables):
    for i in range(len(vector)):
      for j in range(len(vector)):
        a = vector[i]
        b = vector[j]
        energy = potentials[i][j][a][b]
        total_energy[bidx] += energy
  return total_energy


def _manual_local_model(variables, potentials):
  """Manual loop to compute energy of a locally connected ising model."""
  variables = np.asarray(variables)
  potentials = np.asarray(potentials)
  total_energy = np.zeros(len(variables))
  for bidx, vector in enumerate(variables):
    for i in range(len(vector) - 1):
      a = vector[i]
      b = vector[i + 1]
      energy = potentials[i][a][b]
      total_energy[bidx] += energy
  return total_energy


class SimpleIsingModelTest(tf.test.TestCase, parameterized.TestCase):

  def test_full_ising_model_single(self):
    connected = simple_ising_model.IsingModel(
        length=5, vocab_size=3, fully_connected=True)
    output = connected(np.array([[1, 1, 1, 1, 1]]))
    self.assertEqual(3, connected.domain.vocab_size)
    self.assertEqual(5, connected.domain.length)
    self.assertAlmostEqual(-2.47013903, output[0].numpy(), delta=1e-5)

  def test_local_ising_model_single(self):
    local = simple_ising_model.IsingModel(
        length=5, vocab_size=3, fully_connected=False)
    output = local(np.array([[1, 1, 1, 1, 1]]))
    self.assertAlmostEqual(3.23183179, output[0].numpy(), delta=1e-5)

  @parameterized.named_parameters(
      dict(testcase_name='small', batch_size=1, length=4, vocab_size=4),
      dict(testcase_name='medium', batch_size=4, length=4, vocab_size=4),
      dict(testcase_name='large', batch_size=16, length=8, vocab_size=10),
  )
  def test_local_ising_model(self, batch_size, length, vocab_size):
    local = simple_ising_model.IsingModel(
        length=length, vocab_size=vocab_size, fully_connected=False)

    variables = tf.random_uniform(
        shape=[batch_size, length], minval=0, maxval=vocab_size, dtype=tf.int32)
    output = local(variables)
    expected = _manual_local_model(variables, local.potentials)
    self.assertAllClose(expected, output.numpy())

    # Check that it can run with one-hot inputs.
    output = local(tf.one_hot(variables, depth=vocab_size))
    self.assertAllClose(expected, output.numpy())

  @parameterized.named_parameters(
      dict(testcase_name='small', batch_size=1, length=4, vocab_size=4),
      dict(testcase_name='medium', batch_size=4, length=4, vocab_size=4),
      dict(testcase_name='large', batch_size=16, length=8, vocab_size=10),
  )
  def test_full_ising_model(self, batch_size, length, vocab_size):
    full = simple_ising_model.IsingModel(
        length=length, vocab_size=vocab_size, fully_connected=True)
    variables = tf.random_uniform(
        shape=[batch_size, length], minval=0, maxval=vocab_size, dtype=tf.int32)
    output = full(variables)

    expected = _manual_full_model(variables, full.potentials)
    self.assertAllClose(expected, output.numpy())

    # Check that it can run with one-hot inputs.
    output = full(tf.one_hot(variables, depth=vocab_size))
    self.assertAllClose(expected, output.numpy())

  def test_alternating_sequence(self):
    actual_seq = simple_ising_model._alternating_sequence(
        token1='a', token2='b', length=4)
    expected_seq = ['a', 'b', 'a', 'b']
    self.assertAllEqual(actual_seq, expected_seq)

    actual_seq = simple_ising_model._alternating_sequence(
        token1='a', token2='b', length=5)
    expected_seq = ['a', 'b', 'a', 'b', 'a']
    self.assertAllEqual(actual_seq, expected_seq)

    actual_seq = simple_ising_model._alternating_sequence(
        token1='a', token2='b', length=0)
    expected_seq = []
    self.assertAllEqual(actual_seq, expected_seq)

  def test_alternating_chain_model_energy(self):
    model = simple_ising_model.AlternatingChainIsingModel(
        length=4, vocab_size=2)
    inputs = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1]])
    energy = model(inputs)
    self.assertAllClose(energy, [3., 3., 2.])

  def test_alternating_chain_model_optima(self):
    model = simple_ising_model.AlternatingChainIsingModel(
        length=4, vocab_size=2)
    actual_optima = model._get_global_optima()
    expected_optima = [[0, 1, 0, 1], [1, 0, 1, 0]]
    self.assertAllClose(actual_optima, expected_optima)

    model = simple_ising_model.AlternatingChainIsingModel(
        length=5, vocab_size=2)
    actual_optima = model._get_global_optima()
    expected_optima = [[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]]
    self.assertAllClose(actual_optima, expected_optima)

    model = simple_ising_model.AlternatingChainIsingModel(
        length=4, vocab_size=4)
    actual_optima = model._get_global_optima()
    expected_optima = [[0, 1, 0, 1], [1, 0, 1, 0], [2, 3, 2, 3], [3, 2, 3, 2]]
    self.assertAllClose(actual_optima, expected_optima)


if __name__ == '__main__':
  tf.test.main()
