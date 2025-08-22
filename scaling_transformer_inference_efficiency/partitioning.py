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

"""Parallel partitioning functions."""

import contextlib
from dataclasses import dataclass  # pylint: disable = g-importing-member
from enum import Enum  # pylint: disable = g-importing-member
import functools
import math
import threading
from typing import Any, List, Optional, Sequence, Tuple, Union, cast

import jax
from jax import core
from jax import lax
from jax.experimental import mesh_utils
from jax.experimental import pjit
from jax.experimental.array_serialization import serialization as jax_array_serialization
from jax.interpreters import pxla
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
import numpy as np
import tensorstore



class AttnAllToAll(Enum):
  """How much of an alltoall to use for attention."""
  NONE = 0  # [batch.B, heads.YZX]
  AXIS_Z = 1  # [batch.ZB, heads.YX]
  AXES_YZ = 2  # [batch.YZB, heads.X]
  AXES_YZX = 3  # [batch.YZXB, heads]


@dataclass
class ShardingConfig:
  """Class to contain useful objects to shard objects in lower layers."""

  mesh: Mesh
  # TODO(sholto): Infer what we can from the rules
  attn_all_to_all: AttnAllToAll
  latency_collectives: bool
  shard_seqlen_vs_batch: bool
  batch_unsharded: bool


def attn_sharding_to_axes(attn_batch_sharding):
  if attn_batch_sharding == AttnAllToAll.NONE:
    return None
  elif attn_batch_sharding == AttnAllToAll.AXIS_Z:
    return 'z'
  elif attn_batch_sharding == AttnAllToAll.AXES_YZ:
    return ('y', 'z')
  elif attn_batch_sharding == AttnAllToAll.AXES_YZX:
    return ('y', 'z', 'x')


def make_rules_two_d(attn_batch_sharding=AttnAllToAll.NONE,
                     shard_seqlen_vs_batch=False,
                     batch_unsharded=False):

  return [
      ('prefix_time', None),
      ('prefix_layers', None),
      ('prefix_qkv', None),
      ('batch', None),
      (
          'residual_batch',
          None if (shard_seqlen_vs_batch or batch_unsharded) else 'z',
      ),
      (
          'logit_batch',
          None if batch_unsharded else 'x',
      ),  # don't shard batch generally
      ('residual_embed', ('x', 'y', 'z') if batch_unsharded else ('x', 'y')),
      ('residual_time', 'z' if shard_seqlen_vs_batch else None),
      ('post_norm_batch', None),
      ('post_norm_embed', 'x'),
      ('heads', ('y', 'z', 'x')),
      ('qkv', None),
      ('params_heads', ('y', 'z')),
      ('params_embed', 'x'),
      ('params_kv_embed', 'x'),
      ('params_vocab', ('y', 'z')),
      ('embedding_embed', 'x'),
      ('vocab', ('y', 'z', 'x') if batch_unsharded else ('y', 'z')),
      ('attn_batch', attn_sharding_to_axes(attn_batch_sharding)),
      ('weight_load_embed', ('x', 'y', 'z')),
      ('weight_load_heads', None),
  ]


def make_rules_one_d():
  return [
      ('prefix_time', None),
      ('prefix_layers', None),
      ('prefix_qkv', None),
      ('batch', None),
      ('residual_batch', None),
      ('logit_batch', None),
      ('residual_embed', 'x'),
      ('residual_time', None),
      ('post_norm_batch', None),
      ('post_norm_embed', 'x'),
      ('heads', ('x', 'y', 'z')),
      ('qkv', None),
      ('params_heads', ('x', 'y', 'z')),
      ('params_embed', None),
      ('params_kv_embed', 'x'),
      ('params_vocab', ('y', 'z')),
      ('vocab', ('y', 'z', 'x')),
      ('embedding_embed', 'x'),
      ('attn_batch', None),
      ('weight_load_embed', ('x', 'y', 'z')),
      ('weight_load_heads', None),
  ]


def make_rules_weight_gathered():

  return [
      ('prefix_time', None),
      ('prefix_layers', None),
      ('prefix_qkv', None),
      ('batch', None),
      ('residual_batch', ('x', 'y', 'z')),
      ('logit_batch', 'x'),
      ('residual_embed', None),
      ('post_norm_batch', ('x', 'y', 'z')),
      ('post_norm_embed', None),
      ('heads', None),
      ('qkv', None),
      ('params_heads', ('y', 'z')),
      ('params_embed', 'x'),
      ('params_kv_embed', None),
      ('params_vocab', ('y', 'z')),
      ('vocab', ('y', 'z')),
      ('embedding_embed', 'x'),
      ('attn_batch', ('x', 'y', 'z')),
      ('weight_load_embed', ('x', 'y', 'z')),
      ('weight_load_heads', None),
  ]


class _ThreadResourcesLocalState(threading.local):

  def __init__(self):
    self.stack = [[]]  # empty rules

  @property
  def rules(self):
    return self.stack[-1]


thread_resources = _ThreadResourcesLocalState()


class PartitioningRules(contextlib.ContextDecorator):
  """Creates a new set of rules in a context manager.

  Usage:
  rules = partitioning.PartitioningRules(
        partitioning.make_rules_two_d(attn_sharding))
  with rules:
    x = logical_to_physical(y)  # no need thread rules everywhere
  """

  def __init__(self, rules):
    self.rules = rules

  def __enter__(self):
    thread_resources.stack.append(self.rules)
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    thread_resources.stack.pop()
    return False


def logical_to_physical(logical_axes):
  """Converts logical to physical axes for a layer using rule priority."""
  # Priority order of logical to physical axes mapping

  result = [None] * len(logical_axes)
  for logical_axis, physical_axis in thread_resources.rules:
    if logical_axis in logical_axes:
      pos = logical_axes.index(logical_axis)
      # Only map that logical axis against the physical if it hasn't already
      # been mapped - therefore earlier rules have priority over later ones.
      if physical_axis not in result:
        result[pos] = result[pos] or physical_axis
  return P(*result)


def make_mesh(devices = None, one_d=False):
  """Creates a device mesh for use with xmap over x/y/z axes."""
  if devices is None:
    devices = jax.devices()

  if 'TPU v5 lite' in devices[0].device_kind:
    # TPUV5 are 2D
    if len(devices) == 8:
      if one_d:
        return Mesh(
            mesh_utils.create_device_mesh((8, 1), devices)[:, :, np.newaxis],
            ('x', 'y', 'z'),
        )
      else:
        return Mesh(
            mesh_utils.create_device_mesh((2, 4), devices)[:, :, np.newaxis],
            ('x', 'y', 'z'),
        )
    else:
      raise NotImplementedError
  if one_d:
    if len(devices) == 8:
      return Mesh(
          mesh_utils.create_device_mesh((8, 1, 1), devices), ('x', 'y', 'z')
      )
    else:
      raise NotImplementedError
  if len(devices) == 1:
    x, y, z = 1, 1, 1  # TODO(sholto): test
  elif len(devices) == 4:
    x, y, z = 2, 1, 2  # TODO(sholto): test
  elif len(devices) == 8:
    x, y, z = 2, 2, 2  # TODO(sholto): test - always appropriate for B=1?
  elif len(devices) == 16:
    # 2,4,2 or 4,2,2 is good
    x, y, z = 2, 4, 2
  elif len(devices) == 32:
    x, y, z = 4, 2, 4
  elif len(devices) == 64:
    x, y, z = 4, 4, 4
  elif len(devices) == 128:
    x, y, z = 8, 4, 4
    # x, y, z = 4, 4, 8
  elif len(devices) == 256:
    # x, y, z = 8, 4, 8
    x, y, z = 4, 8, 8
  elif len(devices) == 512:
    x, y, z = 8, 8, 8
  else:
    raise NotImplementedError

  return Mesh(
      mesh_utils.create_device_mesh((x, y, z), devices), ('x', 'y', 'z'))


def copy_to_device(x, sharding,
                   expected):
  """Copies the input to the device, however is appropriate for the input.

  If it's an np.ndarray, copies from host memory to device memory. If it's a
  core.ShapedArray, creates a jnp.zeros() of the appropriate shape in device
  memory. If it's a tensorstore.Spec, fetches the data from tensorstore to
  device memory using JAX or Pathways, as appropriate for the current JAX
  backend.

  Args:
    x: The input array.
    sharding: The sharding to use for the array.
    expected: Expected shape and type of the output array.

  Returns:
    The array in sharded device memory.
  """
  # If it's a tensorstore spec with an array() driver, it's already in host
  # memory. Convert it to np.ndarray and use that.
  if isinstance(x, tensorstore.Spec):
    spec = cast(tensorstore.Spec, x)
    json = spec.to_json()
    if json.get('driver') == 'array':
      x = tensorstore.open(spec).result().read().result()

  assert x.shape == expected.shape, f'{x.shape} != {expected.shape}'

  if isinstance(x, np.ndarray) or isinstance(x, jnp.ndarray):

    def cb(i):
      return jax.lax.convert_element_type(x[i], expected.dtype)

    return jax.make_array_from_callback(x.shape, sharding, cb)
  elif isinstance(x, core.ShapedArray):

    def sharded_zeros():
      return jnp.zeros(x.shape, expected.dtype)

    assert isinstance(sharding.mesh, jax.sharding.Mesh)
    with sharding.mesh:
      return pjit.pjit(
          sharded_zeros, in_shardings=(), out_shardings=sharding.spec
      )()
  elif isinstance(x, tensorstore.Spec):
    if jax.config.read('jax_xla_backend') == 'pathways':

      # Read from tensorstore using pathways.
      ts = x.to_json()
      # Further code is internal
    else:
      # Read from tensorstore using jax gda_serialization
      (tensor,) = jax_array_serialization.run_deserialization(
          [sharding], [x], [expected.shape], [expected.dtype], concurrent_gb=64
      )
      return tensor
  else:
    raise ValueError(f'Unsupported type: {type(x)}')


_ALLOW_UNEVEN_SHARDING = True


def _with_sharding_constraint(t,
                              spec):
  """Applies a logical sharding constraint to a tensor."""
  axes = logical_to_physical(spec)
  # First check that the sharding is equally sized on all chips. While the SPMD
  # partitioner is _designed_ to support unequal sharding on chips, in practice
  # this seems to be a fertile ground for XLA bugs such as b/245966065 and
  # possibly the underlying bug for cr/455700040. So we just ban it, and push
  # the padding complexity on the caller.
  mesh = pxla.thread_resources.env.physical_mesh
  name_to_size = dict(zip(mesh.axis_names, mesh.devices.shape))
  for size, axis in zip(t.shape, axes):
    if axis is None or axis not in name_to_size:
      continue
    axis_size = name_to_size[axis]
    assert size % axis_size == 0 or _ALLOW_UNEVEN_SHARDING, (
        f'Uneven sharding. Shape: {t.shape}, spec: {spec}, axis: {axis}, axis'
        f' size: {axis_size}'
    )
  return lax.with_sharding_constraint(t, axes)


def get_sharding_divisor(logical):
  """Returns how many shards will be along a given logical axis."""
  sharding_axis = logical_to_physical(logical)
  if sharding_axis == P(None,):
    sharding_axis_size = 1
  else:
    sharding_axis_size = np.prod([jax.lax.psum(1, a) for a in sharding_axis])
  return sharding_axis_size


@functools.cache
def get_sharding_divisor_outside_manual(mesh, dimensions):
  """Usable outside xmap, shardmap."""
  with mesh:
    sizes = {k: v for k, v in mesh.shape.items()}
    sizes[None] = 1
    if dimensions is None:
      return 1
    else:
      return math.prod([sizes[k] for k in dimensions])


def safe_sharding(tensor, sharding, mesh):
  """If something is to be sharded by more than it's size, do not shard."""
  if sharding is None: return None
  if sharding == P(
      None,
  ):
    return P()
  if sharding == P(None):
    return sharding
  if sharding == P():
    return sharding
  shape = tensor.shape
  sharding_size = [
      get_sharding_divisor_outside_manual(mesh, dim) for dim in sharding
  ]
  new_sharding = []

  for i, (tensor_dim, sharding_dim) in enumerate(zip(shape, sharding_size)):
    if tensor_dim >= sharding_dim:
      new_sharding.append(sharding[i])
      assert tensor_dim % sharding_dim == 0
    else:
      # assert to prevent weird tiling
      # TODO(sholto): Re-insert after shard_map updated to use P().
      # assert sharding_dim % tensor_dim == 0
      new_sharding.append(None)
  return P(*new_sharding)
