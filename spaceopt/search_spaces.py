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

"""Generating random search spaces given a base search space."""
import jax
import jax.numpy as jnp

# pylint: disable=g-doc-return-or-yield
# pylint: disable=line-too-long


def random_subinterval(key, interval, reduce_rate):
  """Generate a reduced interval from a base interval.

  Args:
    key: PRNG key for jax.random.
    interval: (2,) shaped array of min and max values of the interval.
    reduce_rate: interval reduction rate in (0, 1].

  Returns: (2,) shaped array of min and max values of the new interval.
  """
  lower = interval[0]
  upper = interval[1]
  target_length = reduce_rate * (upper - lower)
  lower_new = jax.random.uniform(
      key, minval=lower, maxval=upper - target_length)
  upper_new = lower_new + target_length
  return jnp.array([lower_new, upper_new])


def generate_search_space_reduce_vol(key, search_space, reduce_rate=1/2):
  """Generate a reduced volumed search space from a base search space.

  Args:
    key: PRNG key for jax.random.
    search_space: (d,2) shaped array of min and max values.
    reduce_rate: volume reduction rate in (0, 1].

  Returns: (d,2) shaped array of min and max values of the new search space.
  """
  reduce_rate_dim = reduce_rate**(1/search_space.shape[0])
  keys = jax.random.split(key, search_space.shape[0])
  search_space_reduced = jax.vmap(
      random_subinterval, in_axes=(0, 0, None))(keys, search_space,
                                                reduce_rate_dim)
  condition = reduce_rate == 1
  return jnp.where(condition, search_space, search_space_reduced)


def eval_vol(search_space):
  """Compute volume of a hyperrectangular search space.

  Args:
    search_space: (d,2) shaped array of min and max values.

  Returns: volume of the search space.
  """
  return jnp.prod(search_space[:, 1]-search_space[:, 0])
