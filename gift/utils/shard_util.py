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

"""Utility functions for sharing and unsharding."""

import jax


def unshard_env_batch(pytree):
  """Reshapes all arrays in the pytree.

   `[ndev, env_s, bs, ...]` --> `[host_bs, env_s, ...]

  Args:
    pytree: A pytree of arrays to be sharded.

  Returns:
    Sharded data.
  """

  def _unshard_array(array):
    ndev, envs, bs = array.shape[:3]
    new_shape = (envs, ndev * bs) + array.shape[3:]
    return array.reshape(new_shape)

  return jax.tree_map(_unshard_array, pytree)


def unshard(pytree):
  """Reshapes all arrays in the pytree from `[ndev, bs, ..]` to `[host_bs, ..].

  Args:
    pytree: A pytree of arrays to be sharded.

  Returns:
    Sharded data.
  """

  def _unshard_array(array):
    ndev, bs = array.shape[:2]
    new_shape = (ndev * bs,) + array.shape[2:]
    return array.reshape(new_shape)

  return jax.tree_map(_unshard_array, pytree)
