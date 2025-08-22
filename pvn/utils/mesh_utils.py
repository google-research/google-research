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

"""Mesh Utils."""
from typing import Optional, Sequence, Tuple

import chex
import jax
from jax import lax
from jax.experimental import maps
from jax.experimental import mesh_utils


def create_partition_spec(*args):
  return jax.sharding.PartitionSpec(*args)  # pytype: disable=wrong-arg-count


def with_sharding_constraint(
    x, axis_resources
):
  is_cpu = jax.devices()[0].platform == 'cpu'
  is_single_device = jax.device_count() == 1
  has_global_mesh = global_mesh_defined()
  if is_cpu or is_single_device or not has_global_mesh:
    return x
  else:
    return lax.with_sharding_constraint(x, axis_resources)


def map_leading_axis_to_pspec(
    leaf, mesh_axis_name
):
  return create_partition_spec(
      mesh_axis_name, *(None for _ in range(len(leaf.shape) - 1))
  )


def map_trailing_axis_to_pspec(
    leaf, mesh_axis_name
):
  return create_partition_spec(
      *(None for _ in range(len(leaf.shape) - 1)), mesh_axis_name
  )


def create_global_mesh(
    global_mesh
):
  """Create global mesh."""
  is_cpu = jax.devices()[0].platform == 'cpu'
  is_single_device = jax.device_count() == 1
  if is_cpu or is_single_device or not global_mesh:
    return None
  mesh_axes, mesh_shape = zip(*global_mesh)
  devices = mesh_utils.create_device_mesh(mesh_shape)
  return jax.sharding.Mesh(devices, mesh_axes)


def global_mesh_defined():
  maps_env = maps.thread_resources.env
  return not maps_env.physical_mesh.empty
