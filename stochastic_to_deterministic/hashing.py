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

"""Hashing function to make a stochastic classifier deterministic."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
from absl import app
import numpy as np


def compute_hash(features, hash_matrix, hash_vector):
  """Compute hash values for features using the hash function (A * x + c) mod 2.

  Args:
    features: NumPy float array of shape (n, d), the features to hash.
    hash_matrix: NumPy float array of shape (num_feature_bits, num_hash_bits),
      a random matrix A to construct the hash function.
    hash_vector: NumPy float array of shape (1, num_hash_bits),
      a random vector c to construct the hash function.

  Returns:
    NumPy float array of shape (n, 1) containing the hashed values in [0, 1].
  """
  # Helper function to convert an int array to a bit string array.
  def convert_int_to_bin(x, dimension):
    # Converts x to an array of bit strings of size dimension.
    return '{:b}'.format(x).zfill(dimension)[-dimension:]
  convert_int_to_bin = np.vectorize(convert_int_to_bin)

  # Helper function to convert a bit string array to an into array.
  convert_bin_to_int = np.vectorize(lambda x: int(x, 2))

  # Number of features and hash bits.
  num_features = features.shape[0]
  num_feature_bits, num_hash_bits = hash_matrix.shape

  # Concatenate features and apply MD5 hash to get a fixed length encoding.
  feature_sum_str = [''.join(x) for x in features.astype('str')]
  feature_sum_hex = [hashlib.md5(s.encode()).hexdigest() for s in feature_sum_str]
  feature_sum_int = [int(h, 16) for h in feature_sum_hex]

  # Binarize features
  feature_sum_bin = convert_int_to_bin(
      feature_sum_int, dimension=num_feature_bits)
  feature_sum_bin_matrix = np.array(
      [[int(c) for c in s] for s in feature_sum_bin])

  # Compute hash (Ax + c) mod 2.
  feature_hashed = (
      np.dot(feature_sum_bin_matrix, hash_matrix) +
      np.repeat(hash_vector, repeats=num_features, axis=0))
  feature_hashed_bits = np.mod(feature_hashed, 2)

  # Convert hash to bit string.
  feature_hashed_bit_char = convert_int_to_bin(feature_hashed_bits, 1)
  feature_hashed_bit_str = [''.join(s) for s in feature_hashed_bit_char]
  feature_hashed_int = convert_bin_to_int(feature_hashed_bit_str)
  hashed_val = feature_hashed_int * 1. / 2 ** num_hash_bits

  # Return normalized hashed values in [0, 1].
  return hashed_val.reshape(-1, 1)


def main(argv):
  """Example usage of hash function."""
  del argv

  num_feature_bits = 128
  num_hash_bits = 32

  # Random hash matrix and vector to construct hash function.
  hash_matrix = (np.random.rand(
      num_feature_bits, num_hash_bits) > 0.5).astype('int')
  hash_vector = (np.random.rand(1, num_hash_bits) > 0.5).astype('int')

  # Generate random features.
  num_examples = 10
  dimension = 4
  features = np.random.normal(size=(num_examples, dimension)).astype(np.float32)

  # Compute hash.
  hash_val = compute_hash(features, hash_matrix, hash_vector)

  print('Feature matrix:')
  print(features)
  print('\nHashed values:')
  print(hash_val)


if __name__ == '__main__':
  app.run(main)

