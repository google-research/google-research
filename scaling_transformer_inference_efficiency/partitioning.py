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

"""Parallel partitioning functions."""

import functools
from typing import Union

import jax
from jax.experimental import mesh_utils
from jax.experimental import pjit
from jax.experimental.gda_serialization import serialization as jax_gda_serialization
from jax.experimental.global_device_array import GlobalDeviceArray
from jax.experimental.maps import Mesh
from jax.experimental.pjit import PartitionSpec as P
import jax.numpy as jnp
from jax.sharding import NamedSharding
import numpy as np
import tensorstore


jax.config.update('jax_parallel_functions_output_gda', True)


def logical_to_physical(logical_axes):
  """Converts logical to physical axes for a layer using rule priority."""
  # Priority order of logical to physical axes mapping
  # TODO(reinerp): Are Flax rules sufficient for this?
  rules = [
      ('heads', 'long'),
      ('ff', 'long'),
      ('embed', 'short'),
      ('batch', 'long'),
      ('table_vocab', ('long', 'short')),
  ]
  result = [None] * len(logical_axes)
  for logical_axis, physical_axis in rules:
    if logical_axis in logical_axes:
      pos = logical_axes.index(logical_axis)
      # Only map that logical axis against the physical if it hasn't already
      # been mapped - therefore earlier rules have priority over later ones.
      if physical_axis not in result:
        result[pos] = result[pos] or physical_axis
  return P(*result)


@functools.cache
def make_mesh():
  """Creates a device mesh for use with pjit."""
  devices = jax.devices()
  if len(devices) == 4:
    long, short = 2, 2
  elif len(devices) == 8:
    long, short = 4, 2
  elif len(devices) == 1:
    long, short = 1, 1
  elif len(devices) == 64:
    long, short = 16, 4
  else:
    raise NotImplementedError

  return Mesh(mesh_utils.create_device_mesh((long, short)), ('long', 'short'))


def copy_to_device(x, sharding,
                   expected):
  """Copies the input to the device, however is appropriate for the input.

  If it's an np.ndarray, copies from host memory to device memory. If it's a
  jax.ShapedArray, creates a jnp.zeros() of the appropriate shape in device
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
    json = x.to_json()
    if json.get('driver') == 'array':
      x = tensorstore.open(x).result().read().result()

  assert x.shape == expected.shape, f'{x.shape} != {expected.shape}'

  if isinstance(x, np.ndarray) or isinstance(x, jnp.ndarray):

    def cb(i):
      return jax.lax.convert_element_type(x[i], expected.dtype)

    if jax.config.jax_array:
      return jax.make_array_from_callback(x.shape, sharding, cb)
    else:
      result = GlobalDeviceArray.from_callback(x.shape, sharding.mesh,
                                               sharding.spec, cb)
      return result  # pytype: disable=bad-return-type
  elif isinstance(x, jax.ShapedArray):

    def sharded_zeros():
      return jnp.zeros(x.shape, expected.dtype)

    with sharding.mesh:
      return pjit.pjit(
          sharded_zeros, in_axis_resources=(),
          out_axis_resources=sharding.spec)()
  elif isinstance(x, tensorstore.Spec):
    if jax.config.read('jax_xla_backend') == 'pathways':

      # Read from tensorstore using pathways.
      ts = x.to_json()
      # Further code is internal
    else:
      # Read from tensorstore using jax gda_serialization
      tensor, = jax_gda_serialization.run_deserialization([sharding], [x],
                                                          [expected.shape],
                                                          [expected.dtype])
      return tensor
  else:
    raise ValueError(f'Unsupported type: {type(x)}')
