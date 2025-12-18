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

"""Aggregation of hypothesis tests."""

import functools
import jax
import jax.numpy as jnp


internal_p_val_temp = lambda x, y: jnp.mean(x >= y)
internal_p_val_vmap_temp = jax.vmap(internal_p_val_temp, in_axes=(None, 0))
compute_p_val_all = lambda x: internal_p_val_vmap_temp(x, x)
compute_p_val_all_kernels = jax.vmap(compute_p_val_all)
compute_p_val_single = lambda x: internal_p_val_temp(x, x[0])


@functools.partial(jax.jit, static_argnames=['significance'])
def multiple_test(
    key,
    all_statistics,  # (K, B+1)
    significance=0.05,
):
  """Aggregates results from multiple hypothesis tests.

  This function takes statistics from multiple two-sample permutation tests and
  computes an aggregated p-value to determine if at least one null hypothesis
  can be rejected.

  Args:
    key: A JAX random key. Randomness is used to break ties uniformly.
    all_statistics: A jnp.ndarray of shape (K, B+1), where K is the number of
      tests and B is the number of permutations. The first column (index 0) must
      contain the statistic from the original data, and the remaining columns
      contain statistics from permutations.
    significance: The significance level (alpha) for the test.

  Returns:
    A boolean indicating whether the aggregated test rejects the null hypothesis
    (True if rejected, False otherwise).
  """
  p_vals_all = compute_p_val_all_kernels(all_statistics)  # (K, B+1)
  p_vals_all_min = jnp.min(p_vals_all, 0)  # (B+1,)
  # Breaking ties uniformly increases power
  break_ties = True
  if break_ties:
    key, subkey = jax.random.split(key)
    p_vals_all_min = p_vals_all_min + jax.random.uniform(
        subkey,
        shape=(len(p_vals_all_min),),
        minval=0.0,
        maxval=1 / (2 * len(p_vals_all_min)),
    )
  p_val = compute_p_val_single(1 - p_vals_all_min)  # (1,)
  return p_val <= significance, p_val


vmultiple_test = jax.vmap(multiple_test, in_axes=(0, 0))
