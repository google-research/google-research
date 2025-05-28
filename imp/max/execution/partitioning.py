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

"""Tools for partitioning and parallel training."""

import re
from typing import Any, Sequence

from absl import logging
from flax.core import scope
import flax.linen as nn
import jax
from jax import numpy as jnp
from jax.experimental import multihost_utils
from jax.experimental.pjit import pjit as jax_pjit  # pylint: disable=g-importing-member
import numpy as np
from t5x import partitioning as t5x_partitioning
from t5x.contrib.moe import partitioning as t5x_moe_partitioning

from imp.max.core import constants
from imp.max.utils import typing


# Jax modules
PartitionSpec = jax.sharding.PartitionSpec

# Flax modules
CollectionFilter = scope.CollectionFilter
DenyList = scope.DenyList


# Constants
PARAMS = constants.FlaxCollection.PARAMS
INTERMEDIATES = constants.FlaxCollection.INTERMEDIATES
PROBES = constants.FlaxCollection.PROBES
AUX_LOSS = constants.FlaxCollection.AUX_LOSS

# T5X local chuck module
LocalChunkInfo = t5x_partitioning.LocalChunkInfo
# T5X DataLayout module
DataLayout = t5x_partitioning.DataLayout

PyTree = Any


def pjit(
    fun,
    in_axis_resources,
    out_axis_resources,
    static_argnums = (),
    donate_argnums = ()):
  """Wrapper for pjit that calls normal jit on cpu."""
  if jax.devices()[0].platform == 'cpu':
    return jax.jit(
        fun, static_argnums=static_argnums, donate_argnums=donate_argnums)
  else:
    return jax_pjit(
        fun,
        in_axis_resources,
        out_axis_resources,
        static_argnums=static_argnums,
        donate_argnums=donate_argnums)


class ParameterAxisMap:
  """Maps parameter names to lists of axis names.

  Names of parameters nested in a PyTree (e.g., an Optimizer) are formed by
  joining the names along the path to the parameter leaf with '/'.
  """

  def __init__(self, rules):
    self._rules = [(re.compile(r), p) for r, p in rules]

  def __getitem__(self, key):
    for r, p in self._rules:
      if r.search(key):
        if r != re.compile('(.*)'):
          # only log if an actual param name is matched
          logging.info('Matching of parameter %s with rule %s', key, r)
        return p
    raise KeyError(f'No partition rule found for parameter: {key}')


class Partitioner(t5x_partitioning.BasePjitPartitioner):
  """Partitioner that uses P5X version of jax.pjit and regex rules."""

  def __init__(
      self,
      num_partitions,
      model_parallel_submesh = None,
      parameter_partitioning_dims = 1,
      params_on_devices = True,
      num_experts = None):
    """Configures the partitioner.

    Args:
      num_partitions: an integer that specifies the size of the model parallel
        submesh to be automatically selected for the current topology. See
        `model_parallel_submesh` for details on how this submesh is used.
        Mutually exclusive with `model_parallel_submesh`.
      model_parallel_submesh: is a 4-tuple that specifies the `(x, y, z, c)`
        submesh model-parallel device tile, an axis of accelerator parallelism
        orthogonal to data parallelism. Array axes in a model's parameters or
        activations can be sharded over this submesh using axis rules (see
        `logical_axis_rules`) that map them to 'model'. The effective number of
        model sub-partitions is equal to `np.prod(model_parallel_submesh)` and
        must evenly divide the total number of devices (i.e.,
        `jax.device_count() % np.prod(model_parallel_submesh) == 0`). The rest
        of the TPU mesh is the data parallel submesh, providing
        `jax.device_count() // np.prod(model_parallel_submesh)` partitions. It
        is used for data (batch) parallelism and to shard other array axes that
        are mapped to 'data'. This argument is mutually exclusive with
        `num_partitions`.
      parameter_partitioning_dims: 1 for 1-D parameter sharding or 2 for 2-D
        parameter sharding. 0 to force all parameters to be unsharded, which
        sidesteps the need to define additional rules for new layer names in the
        data parallelism case.
      params_on_devices: whether to keep the params on devices, if False -
        params stay in the host memory. Note that some partitioners might ignore
        this setting, for example if they don't support storing all params on
        device memory.
      num_experts: Total number of experts across all devices. Only useful if
        training a Mixture-of-Experts variant. If provided, the partitioner
        will create a 3D mesh: ('expert', 'data', 'model'). Only experts'
        weights and states will be partitioned along the 'expert' axis. Data
        will be partitioned along both 'expert' AND 'data' axes.
    """

    if parameter_partitioning_dims == 0 and num_partitions != 1:
      raise ValueError(
          'When parameter_partitioning_dims is 0, num_partitions must be 1, '
          f'but got {num_partitions} instead.')
    super().__init__(
        num_partitions=num_partitions,
        model_parallel_submesh=model_parallel_submesh,
        params_on_devices=params_on_devices,
    )

    self._num_partitions = num_partitions
    self._num_experts = num_experts
    self._parameter_partitioning_dims = parameter_partitioning_dims
    self._state_specs = None
    self._state_shapes = None
    self.states_initialized = False

  @t5x_partitioning.cached_property
  def mesh(self):
    """Constructs and returns default partitioning mesh."""
    if self._num_experts is None:
      return t5x_partitioning.default_mesh(
          self._num_partitions, self._model_parallel_submesh, self._backend)
    else:
      return t5x_moe_partitioning.default_moe_mesh(
          self._num_experts, self._num_partitions,
          self._model_parallel_submesh, self._backend)

  def shard_and_put_on_devices(self, data, data_specs,
                               output_specs):
    """Puts data on devices with sharding specs.

    Args:
      data: Any PyTree that holds jnp.array or np.array in it.
      data_specs: The sharding annotations for the data.
      output_specs: The sharding annotations for the outputs. This could be
        either the same as data_specs or None in most of practical cases.

    Returns:
      A PyTree similar to `data` placed and sharded on devices.
    """

    def _id_fn(x, ix):
      """Identity function for copying data to the devices, sharded."""
      y = jax.random.split(jax.random.key(jnp.array(ix, dtype=jnp.uint32)))
      return x, y

    shard_fn = jax.jit(
        fun=_id_fn,
        in_shardings=(data_specs, None),
        out_shardings=(output_specs, None),
    )
    with self.mesh:
      outputs, _ = shard_fn(data, np.ones((), dtype=np.int32))

    return outputs

  def all_gather_across_data_shards(
      self, data):
    """Shards data with data shards and gets them back aggregated."""
    data_specs = self.get_data_specs()

    # distribute data across processes
    data = multihost_utils.host_local_array_to_global_array(
        local_inputs=data,
        global_mesh=self.mesh,
        pspecs=data_specs,
    )
    data = self.shard_and_put_on_devices(data, data_specs, None)
    return jax.tree.map(lambda v: v.addressable_data(0), data)

  # TODO(b/243716891): replace numpy ops with jnp once migrated to GDA
  def maybe_pad_batches(
      self, data,
      target_batch_size = None,
  ):
    """Zero-pads along the batch axis, if batch_size is not shardable.

    Args:
      data: A nested numpy array, assumed to contain local data.
      target_batch_size: An optional positive integer. If provided, the
        arrays in `data` will be compared against this number. If their
        batch dimension is smaller than this number, they will be padded.
        Otherwise, if they are larger, an error will be raised.

    Returns:
      A tuple of ('data', 'batch_mask') in which data contains the
      potentially padded arrays along the batch size (if their batch dimension
      is not divisible by data shards). If `data` is padded, `batch_mask`
      contains a sequence of 0/1s indicating where the padding is applied
      (marked by 0s). 'batch_mask' is local, similar to `data`.

    Raises:
      ValueError if there are uneven batches in the leaves of the given
      nested array, OR if an unshardable target_batch_size is provided.
    """
    num_data_shards = self._get_num_data_shards()
    batch_sizes = set(jax.tree.leaves(jax.tree.map(lambda v: v.shape[0], data)))
    if len(batch_sizes) > 1:
      raise ValueError('Received uneven batches in the data structure.')
    batch_size = batch_sizes.pop()

    if target_batch_size is None:
      remainder = batch_size % num_data_shards
      num_pads = num_data_shards - remainder
      num_pads %= num_data_shards  # set to 0 if remainder==0

    else:
      if target_batch_size % num_data_shards != 0:
        raise ValueError(
            f'The provided {target_batch_size=} is not shardable along the '
            f'data mesh axis with {num_data_shards=}.')
      if target_batch_size < batch_size:
        raise ValueError(
            f'The provided {target_batch_size=} cannot be smaller than the '
            f'leaves {batch_size=}.')
      num_pads = target_batch_size - batch_size

    batch_mask = None
    if num_pads != 0:
      def _pad_fn(v):
        pad_width = [(0, num_pads)] + [(0, 0)] * (len(v.shape) - 1)
        return np.pad(v, pad_width=pad_width, mode='constant')
      data = jax.tree.map(_pad_fn, data)
      batch_mask = np.ones((batch_size + num_pads,), dtype='int32')
      batch_mask[batch_size:] = 0

    return data, batch_mask

  def maybe_unpad_batches(
      self, data,
      batch_mask):
    """Unpads along the batch axis, if batch_mask is provided.

    Args:
      data: A nested array which probably contains padded content along the
        batch dimension. This array can be either already on devices (hence
        jnp.array), or on local hosts (hence np.array).
      batch_mask: An optional sequence of 0/1s indicating where the 0s indicate
        the positions in which the padded samples exist. If not provided, the
        `data` will be returned untouched. It is assumed that `batch_mask`
        is a numpy array that locally exists on hosts.
    Returns:
      The (potentially) unpadded data.
    """
    if batch_mask is None:
      return data
    else:
      indices = np.argwhere(batch_mask == 1).reshape(-1)
      def _unpad_fn(v):
        if isinstance(v, jax.Array):
          return jnp.take(v, indices, 0)
        else:
          return np.take(v, indices, 0)
      data = jax.tree.map(_unpad_fn, data)
      return data

  def all_gather_slices_across_processes(
      self, slices):
    """Gathers global samples across processes.

    This method receives a list of nested NUMPY arrays (which might have
    different values and different batch sizes) across different processes
    in a SPMD job and aggregates all of them along the batch dimension.
    This is a helper method to meet the same-size array requirement (in
    the JAX-SPMD all-gather ops) by pading along the batch dimension to
    form same-size arrays across all processes. Even different processes
    might hold different number of such nested arrays. Hence, this method
    also creates placeholder arrays in those processes that don't have certain
    samples. This method returns a list of nested NUMPY arrays with global
    shapes, hence replicated across all processes (all of them have the
    all-gathered version).

    Args:
      slices: A list of nested NUMPY arrays with different number of samples
        across different processes in a JAX-SPMD job. Arrays should have same
        size along the non-batch dimension, but could hold different number of
        samples along the batch dimension (across processes).

    Returns:
      A list of nested NUMPY arrays which are all-gathered across all processes.

    Raises:
      ValueError if different leaves in a nested array have different batch
      size on a single process.
    """

    if not any(slices):
      return slices

    def _get_slice_batch_size(data):
      """Fetch batch_size across arrays in the same slice."""
      batch_sizes = set(jax.tree.leaves(
          jax.tree.map(lambda v: v.shape[0], data)))
      if len(batch_sizes) > 1:
        raise ValueError(
            'Different leaves in this tree have different batch sizes. '
            f'Found {batch_sizes}.')
      batch_size = batch_sizes.pop()
      return batch_size

    slices_length = jnp.array(len(slices), dtype=jnp.int32)
    slices_batch_size = jnp.max(
        jnp.array([_get_slice_batch_size(slice) for slice in slices]), axis=0)

    # gather length and batch_size across all processes
    slices_length = multihost_utils.process_allgather(slices_length)
    slices_batch_size = multihost_utils.process_allgather(slices_batch_size)

    # calculate the maximum length and bath_size across processes
    max_length = int(jnp.max(slices_length, axis=0))
    max_batch_size = int(jnp.max(slices_batch_size, axis=0))

    # pad uneven batches across slices (if any)
    logging.info('Padding uneven batches across slices')
    slices_mask = [np.ones((max_batch_size,), dtype='int32')] * len(slices)
    for n in range(len(slices)):
      slices[n], slice_batch_mask = self.maybe_pad_batches(
          slices[n], max_batch_size)
      if slice_batch_mask is not None:
        slices_mask[n] = slice_batch_mask

    remainder_length = max_length - len(slices)
    if remainder_length > 0:
      logging.info('Remainder length is non-zero. Creating empty arrays.')
      empty_slice = jax.tree.map(np.zeros_like, slices[0])
      empty_batch_mask = np.zeros((max_batch_size,), dtype='int32')
      slices += [empty_slice] * remainder_length
      slices_mask += [empty_batch_mask] * remainder_length

    for n in range(len(slices)):
      global_slice = self.all_gather_across_data_shards(slices[n])
      global_batch_mask = self.all_gather_across_data_shards(slices_mask[n])
      global_slice = self.maybe_unpad_batches(global_slice, global_batch_mask)
      multihost_utils.sync_global_devices('partitioner: fetched global slice')
      slices[n] = jax.tree.map(np.asarray, global_slice)

    return slices

  def get_data_layout(self,
                      batch_size = None,
                      host_index = None):
    """Returns filled `DataLayout` based on the partitioned model layout.

    Overrides default data layout for MoE, where we treat 'data' and 'expert'
    axes as "data" axes.

    Args:
      batch_size: If set, indicates the requested batch size. If not set, the
        batch size is inferred from the layout.
      host_index: Indicates the host index to use for the calculations, if not
        set - use JAX-provided one. Should be in [0, num_hosts) interval and the
        order matches the order of corresponding host in `jax.devices()`.

    Returns:
      Filled `DataLayout` structure.
    """

    if self._num_experts is None:
      return super().get_data_layout(batch_size=batch_size,
                                     host_index=host_index)

    else:
      if host_index is not None:
        raise NotImplementedError('Explicit host_index is not yet implemented.')

      num_data_partitions = self._local_chunker.global_mesh.shape['data']
      num_expert_partitions = self._local_chunker.global_mesh.shape['expert']

      data_mesh_size = num_data_partitions * num_expert_partitions
      batch_size = batch_size or data_mesh_size
      if batch_size % data_mesh_size:
        raise ValueError(
            f'{batch_size=} must be divisible by entire {data_mesh_size=}. '
            'Note that for MoE, the data mesh spans '
            'both the "expert" and "data" virtual mesh axes.')

      num_shards = self._local_chunker.num_chunks[
          'data'] * self._local_chunker.num_chunks['expert']
      if batch_size % num_shards:
        raise ValueError(
            f'{batch_size=} must be divisible by total {num_shards=} across '
            '"data" and "expert" mesh axes.')

      # Partition the batch over both of the 'expert' and 'data' axes.
      global_array_shape = (num_expert_partitions,
                            batch_size // num_expert_partitions)
      replica_id = self._local_chunker.get_local_chunk_info(
          global_array_shape, ('expert', 'data')).replica_id

      return DataLayout(
          batch_size=batch_size,
          shard_id=(self._local_chunker.chunk_ids['data'] +
                    self._local_chunker.chunk_ids['expert'] *
                    self._local_chunker.num_chunks['data']),
          num_shards=num_shards,
          is_first_host_in_replica_set=(replica_id == 0))

  def _get_num_data_shards(self):
    """Fetch number of data shards across all devices."""
    mesh_ids = self.mesh.device_ids.shape
    if self._num_experts is None:
      return mesh_ids[0]
    else:
      return mesh_ids[0] * mesh_ids[1]

  def get_state_specs(self):
    if self._state_specs is None:
      raise ValueError(
          'Please initialize the partitioner by: `partitioner.initialize_states'
          '(model, optimizer, init_rngs, init_override)`'
      )
    else:
      return self._state_specs

  def get_state_shapes(self):
    if self._state_shapes is None:
      raise ValueError(
          'Please initialize the partitioner by: `partitioner.initialize_states'
          '(model, optimizer, init_rngs, init_override)`'
      )
    else:
      return self._state_shapes

  def get_data_specs(self):
    if self._num_experts is None:
      return PartitionSpec('data')  # pytype: disable=wrong-arg-count
    else:
      return PartitionSpec(('expert', 'data'),)

  def initialize_states(
      self,
      boxed_state,
      unboxed_state,
  ):

    def _get_shapes(struct):
      return struct.replace(
          step=struct.step.shape,
          params=jax.tree.map(lambda x: x.shape, struct.params),
          opt_state=jax.tree.map(lambda x: x.shape, struct.opt_state),
      )

    self._state_shapes = _get_shapes(unboxed_state)
    self._state_specs = nn.get_partition_spec(boxed_state)
    self.states_initialized = True
