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

"""Utils for two sample hypothesis tests."""
from typing import Any

import flax
import jax
import jax.numpy as jnp


@flax.struct.dataclass
class TestResult:
  all_statistics: jnp.ndarray
  p_val: jnp.ndarray
  result_test: Any
  additional_info: dict[str, Any]


def get_permutations(
    key, num_permutations, m, n
):
  """Returns permutations of indices and corresponding signs.

  This function generates `num_permutations+1` random permutations of indices
  from two sets of sampleswith sizes m and n, along with corresponding signs (+1
  for m samples, -1 for n samples). The last row is the original order.

  Args:
    key: JAX PRNG key.
    num_permutations: Number of permutations to generate.
    m: Size of the first set.
    n: Size of the second set.

  Returns:
    A tuple containing:
      - V11: A matrix of shape (num_permutations + 1, m + n) containing the
        signs of the samples after permutation. The last row is the original
        order.
      - sorted_indices: A matrix of shape (num_permutations + 1, m + n)
        containing the indices of the samples after permutation. The last row
        is the original order.
  """
  _, subkey = jax.random.split(key)
  idx = jax.random.permutation(
      key=subkey,
      x=jnp.array([list(range(m + n))] * num_permutations),
      axis=1,
      independent=True,
  )
  # Append the original order.
  idx = jnp.vstack([idx, jnp.reshape(jnp.arange(m + n), (1, m + n))])

  true_assignments = jnp.concatenate((jnp.ones(m), -jnp.ones(n)))
  permuted_assignments = jnp.tile(true_assignments, (num_permutations + 1, 1))
  permuted_assignments = jnp.take_along_axis(permuted_assignments, idx, axis=1)
  sorted_indices = jnp.argsort(idx, axis=1)
  return permuted_assignments, sorted_indices
