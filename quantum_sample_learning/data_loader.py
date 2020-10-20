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

# Lint as: python3
"""Module to generate training data."""

from absl import logging

import numpy as np
import tensorflow.compat.v2 as tf


def load_probabilities_binary(path):
  """Loads probabilities from the binary output of iserge@'s simulator.

  Each row has one float32 number.

  Args:
    path: String, the path to the probabilities of a circuit.

  Returns:
    Float numpy array with shape (num_states,).
  """
  with tf.io.gfile.GFile(path, 'rb') as f:
    return np.frombuffer(f.read(), dtype=np.float32)


def load_probabilities(path):
  """Loads probabilities from file.

  Args:
    path: String, the path to the probabilities of a circuit.

  Returns:
    Float numpy array with shape (num_states,).
  """
  with tf.io.gfile.GFile(path) as f:
    return np.loadtxt(f)


def convert_binary_digits_array_to_bitstrings(array, dtype=np.int64):
  """Converts binary digits array to integers.

  Usually binary digits array are preferred for human readability and energy
  computation. However, integer representation is easier for storing.

  For example, array = [[0, 0, 1, 1], [0, 1, 0, 0]] -> [3, 4]

  Args:
    array: Integer numpy array with shape (batch_size, num_qubits).
    dtype: Integer type to cast the output results to, the range need to cover
        num_states=2 ** num_qubits.

  Returns:
    Integer numpy array with shape (batch_size,).
  """
  _, num_states = array.shape
  return np.dot(array, np.power(2, np.arange(num_states - 1, -1, -1))).astype(
      dtype)


def count_set_bits(n, k):
  """Counts the number of ones in the first k digits of binary representation.

  Args:
    n: Integer, the integer whose binary representation is considered.
    k: Integer, the number of binary digits that the subset parity acts on.

  Returns:
    A subset parity value specified by a binary string of length k.
  """
  count = 0
  digits = 1
  while n and digits <= k:
    count += n & 1
    n >>= 1
    digits += 1
  return count


def count_random_set_bits(n, k, random_set):
  """Counts the number of ones in the randomly chosen k digits of binary.

  Args:
    n: Integer, the integer of interest.
    k: Integer, the number of binary digits that the subset parity acts on.
    random_set: List of integers, random binary string with ones in k locations.

  Returns:
    A subset parity value specified by a binary string of length k.
  """
  count = 0
  digits = 1
  bit_index = 0
  while n and digits <= k:
    n >>= 1
    if random_set[bit_index] > 0:
      count += n & 1
      digits += 1
    bit_index += 1
  return count


def reorder_subset_parity(probabilities, subset_parity_size):
  """Reorders bit strings probabilities according to a subset parity.

  The subset parity is specified by a subset_parity_size-bit string.

  Args:
    probabilities: FLoat numpy array, probability distribution over 2^n bit
        strings.
    subset_parity_size: Integer, the number of binary digits that the subset
        parity acts on.

  Returns:
    Float numpy array with the same shape of the input probabilities. A subset
    parity value specified by a binary string of length subset_parity_size.
  """
  ordered_probabilities = np.sort(probabilities / np.sum(probabilities))
  if subset_parity_size == 0:
    return ordered_probabilities
  else:
    return np.array([
        x[1] for x in
        sorted(enumerate(ordered_probabilities),
               key=lambda y: count_set_bits(y[0], subset_parity_size))])


def reorder_random_subset_parity(probabilities, subset_parity_size, random_set):
  """Reorders bit strings probabilities according to a subset parity.

  The subset parity is specified by a subset_parity_size-bit string.

  Args:
    probabilities: FLoat numpy array, probability distribution over 2^n bit
        strings.
    subset_parity_size: Integer, the number of binary digits that the subset
        parity acts on.
    random_set: List of binary integers, with k nonzero values denoting which
        k of n bits belong to the subset.

  Returns:
    Float numpy array with the same shape of the input probabilities. A subset
    parity value specified by a binary string of length subset_parity_size.
  """
  ordered_probabilities = np.sort(probabilities / np.sum(probabilities))
  if subset_parity_size == 0:
    return ordered_probabilities
  else:
    key = lambda y: count_random_set_bits(y[0], subset_parity_size, random_set)
    return np.array([
        x[1] for x in sorted(enumerate(ordered_probabilities), key=key)])


def load_data(
    num_qubits,
    use_theoretical_distribution,
    probabilities_path,
    subset_parity_size,
    random_subset,
    porter_thomas,
    experimental_bitstrings_path=None,
    train_size=None,
    ):
  """Loading the probability distribution and experimental bitstring."""
  logging.info('Start loading data from %s', probabilities_path)
  probabilities = load_probabilities(probabilities_path)
  if use_theoretical_distribution:
    logging.info('Use theoretical distribution.')
    probabilities = probabilities / np.sum(probabilities)

    if porter_thomas:
      np.random.shuffle(probabilities)
    if subset_parity_size >= 0:
      logging.info('use subset parity as ordering')
      if random_subset:
        random_set = np.zeros(num_qubits)
        random_set[:subset_parity_size] = 1
        np.random.shuffle(random_set)
        probabilities = reorder_random_subset_parity(
            probabilities, subset_parity_size, random_set)
        logging.info('finish loading random subeset parity distribution')
      else:
        probabilities = reorder_subset_parity(probabilities, subset_parity_size)

    sample_data = np.random.choice(
        np.arange(0, 2 ** num_qubits),
        train_size,
        p=probabilities / np.sum(probabilities))
    train_data = sample_data[:, np.newaxis] >> np.arange(num_qubits)[::-1] & 1
    logging.info('train_data shape')
    logging.info(train_data.shape)
    train_array = np.zeros((2 ** num_qubits))
    for i in convert_binary_digits_array_to_bitstrings(train_data):
      train_array[i] += 1
    train_array = train_array / np.sum(train_array)
  else:
    logging.info('Load experimental data.')
    with tf.io.gfile.GFile(experimental_bitstrings_path) as f:
      raw_data = np.genfromtxt(f, dtype='S%d' % num_qubits)
    train_size = int(raw_data.shape[0])
    train_data = np.zeros(((train_size, num_qubits)), np.int32)
    # Convert strings to array.
    for j in range(train_size):
      x = raw_data[j].decode('utf-8')
      for k in range(num_qubits):
        train_data[j, k] = int(x[k])
  def _convert_bitstring_array_to_probabilities(array):
    return probabilities[convert_binary_digits_array_to_bitstrings(array)]

  sampled_probabilities = _convert_bitstring_array_to_probabilities(train_data)
  mean_p_theory = np.mean(sampled_probabilities)
  theory_fidelity = probabilities.size * mean_p_theory - 1
  theory_logistic_fidelity = (
      np.log(probabilities.size) + np.euler_gamma
      + np.mean(np.log(sampled_probabilities)))
  logging.info('normalization of linear fidelity %f', theory_fidelity)
  return train_data, probabilities, theory_fidelity, theory_logistic_fidelity
