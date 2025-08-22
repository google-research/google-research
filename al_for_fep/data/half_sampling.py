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

"""Implementation of half sampling data splitting technique."""
from typing import Iterable, Tuple
import numpy as np

from al_for_fep.data import data_splitting_strategy


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


class HalfSamplingSplit(data_splitting_strategy.DataSplittingStrategy):
  """Class for half-sampling approach to data sampling."""

  def __init__(self, shards_log2):
    """Initialization.

    Args:
      shards_log2: Log (base 2) of number of shards to use for half sampling.
    """
    self._shards_log2 = shards_log2

  def split(self, example_pool,
            target_pool):
    """Generates training sets in accordance with the half sampling approach.

    See

    Heavlin, W.D. On Ensembles, I-Optimality, and Active Learning. J Stat
    Theory Pract 15, 66 (2021). https://doi.org/10.1007/s42519-021-00200-4

    for in depth details.

    Args:
      example_pool: List of feature lists (training inputs) to construct data
        sets from.
      target_pool: List of target values associated with examples. The i^th
        value in target_pool will be associated with the i^th value in
        example_pool.

    Yields:
      Generator that iterates through subsets of half the data as prescribed
      by the half sampling approach.
    """
    if len(target_pool) != len(example_pool):
      raise ValueError('Example list and target list should have the same '
                       f'length ({len(target_pool)} vs {len(example_pool)}).')

    max_shard = 2**self._shards_log2

    shards = np.array(
        list(map(lambda x: x % max_shard, range(len(target_pool)))))

    yield example_pool, target_pool

    for split in np.transpose(orthogonal_array(self._shards_log2))[1:]:
      first_half_indices = np.isin(shards, np.where(split == 1)[0])
      second_half_indices = np.isin(shards, np.where(split == 0)[0])

      yield (example_pool[first_half_indices], target_pool[first_half_indices])
      yield (example_pool[second_half_indices],
             target_pool[second_half_indices])
