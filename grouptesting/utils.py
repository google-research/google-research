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
"""Some useful array manipulations for sampling."""

import jax
import jax.numpy as np


def unique(rng, binary_vectors):
  """Computes the number of unique binary columns."""
  alpha = jax.random.normal(rng, shape=((1, binary_vectors.shape[1])))
  return 1 + np.count_nonzero(
      np.diff(np.sort(np.sum(binary_vectors * alpha, axis=-1))))


def select_from_sizes(values, sizes):
  """Selects using indices group_sizes the relevant values for a parameter.

  Given a parameter vector (or possibly constant) that describes values
  for groups of size 1,2,...., k_max selects values according to vector
  group_sizes. When an item in group_sizes is larger than the size of
  the vector, we revert to the last element of the vector by default.

  Note that the values array is 0-indexed, therefore the values corresponding
  to size 1 is values[0], to size 2 values[1] and more generally, the value for
  a group of size i is values[i-1].

  Args:
    values: a np.ndarray that can be of size 1 or more, from which to seleect
     the values from.
    sizes: np.array[int] representing the group sizes we want to extract the
     values of.

  Returns:
    vector of parameter values, chosen at corresponding group sizes,
    of the same size of group_sizes.

  Raises:
   ValueError when the size array is not one dimensional.
  """
  values = np.asarray(values)
  dim = np.ndim(values)
  if dim > 1:
    raise ValueError(f"sizes argument has dimension {dim} > 1.")

  # The values are 0-indexed, but sizes are strictly positives.
  indices = np.minimum(sizes, np.size(values)) - 1
  return np.squeeze(values[indices])
