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

"""Common utility functions."""

import collections
import re
import string
from typing import Any, Mapping, Sequence, Type

from absl import logging
from flax import traverse_util
import flax.linen as nn
from flax.training import train_state
import jax
from jax import lax
from jax import numpy as jnp
import numpy as np
import optax
import tensorflow as tf

from imp.max.core import constants
from imp.max.utils import sharding
from imp.max.utils import typing


def _normalize_indices(
    indices,
    batch_dims = ()):
  """Verifies and returns the 'indices' as an array."""

  # We first make sure the `indices` is jnp array
  indices = jnp.asarray(indices, dtype=jnp.int32)

  # If the `indices` is an integer (hence, ndim == 0) and we use it to take
  # slices, the resulting slice is squeezable (since len_axis = 1). Here, we
  # check if this holds with the given indices or not. The `squeezable` flag
  # is only used in `take_along_axis` function.
  squeezable = False
  if indices.ndim == 0:
    indices = indices[jnp.newaxis]
    squeezable = True

  # By default, the expect rank of the `indices` is 1 (since we only operate on
  # one axis in the functions that use `_normalize_indices`). Although in some
  # cases we generalize the op-of-interest to batched ops (see
  # scatter_along_axis below). In this case, an `indices` with rank > 1 is
  # acceptable as far as accurately specifying the expected `batch_dims`.
  expected_rank = len(batch_dims) + 1
  if indices.ndim != expected_rank:
    raise ValueError(
        f'Indices with rank {indices.ndim} and {batch_dims=} are provided. '
        f'Indices should have rank `1 + num_batch_dims` ({expected_rank} here).'
        f' Instead, received a tensor with shape {indices.shape}.')

  return indices, squeezable


def _normalize_axis(axis, rank):
  """Normalizes a (possibly) negative axis to a positive number."""
  # Normalize axis if negative
  if axis < 0:
    axis += rank

  if axis >= rank or axis < 0:
    raise ValueError(f"Axis exceeds inputs' {rank=}")
  return axis


def _get_nested_keys(
    nested_dict,
):
  """Returns the sequence of nested keys as a tuple of tuples."""
  return tuple(traverse_util.flatten_dict(nested_dict, sep=None).keys())


def _verify_mask_shape(token_mask,
                       input_length):
  """Asserts proper shape and rank for the token mask."""
  if token_mask is None:
    return
  mask_rank = token_mask.ndim
  expected_rank = constants.DataFeatureRank.Common.TOKEN_MASK
  if mask_rank != expected_rank:
    raise ValueError(
        f'Token mask should have a rank of {expected_rank}, '
        f'received rank {mask_rank}')
  mask_length = token_mask.shape[2]
  if input_length != mask_length:
    raise ValueError(
        'Only cases where input_length==mask_length are supported, but '
        f'input_length={input_length} and mask_length={mask_length} requested.')


def _verify_bias_shape(attention_bias,
                       input_length):
  """Verifies the attention bias shape based on the sequence length."""
  if attention_bias is None:
    return
  _, q_length, kv_length = attention_bias.shape
  if q_length != kv_length:
    raise ValueError(
        'Only cases where q_length==kv_length are supported, but q_length='
        f'{q_length} and kv_length={kv_length} requested.')
  if input_length != q_length:
    raise ValueError(
        'Only cases where input_length==q_length are supported, but '
        f'input_length={input_length} and q_length={q_length} requested.')


def verify_attention_shapes(attention_bias,
                            token_mask,
                            inputs_shape):
  """Ensures that attention bias and mask match expected shapes."""
  length = np.prod(inputs_shape[2:-1])
  _verify_bias_shape(attention_bias, length)
  _verify_mask_shape(token_mask, length)


def _flatten_dict_fn(
    dictionary,
    parent_key = None,
    sep = None):
  """Flattens a nested dictionary, keeping track of parent keys."""
  flattened_list = []
  for key, value in dictionary.items():
    if sep is None:
      new_key = parent_key + (key,) if parent_key else (key,)
    else:
      new_key = f'{parent_key}{sep}{key}' if parent_key else key

    if isinstance(value, Mapping):
      flattened = _flatten_dict_fn(value, new_key, sep).items()
      flattened_list.extend(flattened)
    elif isinstance(value, Sequence) and not isinstance(value, str):
      value = tuple(value)
      for i, v in enumerate(value):
        flattened = _flatten_dict_fn({str(i): v}, new_key, sep).items()
        flattened_list.extend(flattened)
    else:
      flattened_list.append((new_key, value))
  return dict(flattened_list)


def flatten_dict(
    dictionary,
    sep = None):
  """Flattens a nested dictionary.

  This function is analogous to flax.traverse_util.flatten_dict, but also
  supports flattening nested tuples. Keys for tuples are represented as the
  string representation of their integer index.

  Note that if a dict contains a tuple, this operation is destructive
  and it cannot be recovered from flax.traverse_util.unflatten_dict, as
  key types from dicts or tuples cannot be distinguished.

  Args:
    dictionary: the dict to flatten.
    sep: if specified, then the keys of the returned dictionary will be
      sep-joined strings (if None, then keys will be tuples).

  Returns:
    the flattened dict.
  """

  return _flatten_dict_fn(dictionary, sep=sep)


def deep_update_data(
    data,
    update_data):
  """Updates a data collection (nested dictionary) with new entries.

  Args:
    data: A data collection that could contain any jax arrays and/or metadata.
    update_data: A data collection to update/override the entries in `data`.

  Returns:
    An updated data collection containing the source and updated entries.
  """
  flattened_data = traverse_util.flatten_dict(
      data, keep_empty_nodes=True, sep='/')
  flattened_update_data = traverse_util.flatten_dict(
      update_data, keep_empty_nodes=True, sep='/')
  flattened_data.update(flattened_update_data)
  data = traverse_util.unflatten_dict(flattened_data, sep='/')
  return data


def flatten_itemize_dict(dictionary):
  """Flattens a dictionary and returns as key-value pairs."""
  return tuple(traverse_util.flatten_dict(dictionary).items())


def unflatten_unitemize_dict(items):
  """Creates a nested dictionary based on key-value pairs."""
  return traverse_util.unflatten_dict(dict(items))


def is_sub_np_dtype(data, py_dtype):
  """Checks if a structure is a subdtype of a py_dtype."""
  return np.issubdtype(np.asarray(data).dtype, py_dtype)


def verify_flow_exists_in_data(
    flow,
    data,
    flow_leaf_is_feature_name = False,
    feature_type = None):
  """Checks if a flow exists a data collection."""
  if feature_type is not None:
    if feature_type not in flow:
      raise ValueError(f'The {feature_type=} does not exist in the flow.')
    if feature_type not in data:
      raise ValueError(f'The {feature_type=} does not exist in the data.')

  featureflow = flow[feature_type] if feature_type is not None else flow
  featuredata = data[feature_type] if feature_type is not None else data
  featureflow = traverse_util.flatten_dict(featureflow, sep='/')
  featuredata = traverse_util.flatten_dict(featuredata, sep='/')
  for featurepath in featureflow:
    if flow_leaf_is_feature_name:
      featurepath += f'/{featureflow[featurepath]}'
    if featurepath not in featuredata:
      raise ValueError(
          f'The {featurepath=} does not exist in the data. The existing '
          f'feature paths in the data are {list(featuredata.keys())}.'
      )


def should_write_file():
  """Returns True if this process should write a file (i.e., is host)."""
  # For multihost, ensure only one host saves the config.
  return jax.process_index() == 0


def safe_write(file_path, content):
  """A function to safely write to a file in a multi-process setup.

  The function will attempt to write the file if it is a jax host process. If
  another host process is already writing the file, this function will exit
  without writing. This should be acceptable, as there should be at least one
  process which opens the file first and writes successfully.

  Args:
    file_path: the path the file will be written to.
    content: the full string contents that will be written.
  """
  if not should_write_file():
    return

  try:
    with tf.io.gfile.GFile(file_path, 'w') as f:
      f.write(content)
  except tf.errors.NotFoundError:
    logging.info(
        'Cannot write file %r as another process is writing to the same '
        'file. This is not an issue as the file is only created for '
        'debugging purpose and has the same content among all the workers. ',
        file_path)


def get_patched_shape(
    input_size,
    patch_size,
):
  """Calculates the number of tokens in a patching process.

  Args:
    input_size: An integer or tuple of integers indicating the size of the input
      along each dimension
    patch_size: The patch size along each dimension of the input. This should
      have the same number of dimensions as the `input_size`.

  Returns:
    The expected size of the patched input. This will be a tuple with the same
    number of dimensions as `input_size` and `patch_size`.

  Raises:
    ValueError if `input_size` and `patch_size` have different number of
    dimensions.
  """

  if isinstance(input_size, int) and isinstance(patch_size, int):
    input_size = (input_size,)
    patch_size = (patch_size,)
  elif not isinstance(input_size, tuple) or not isinstance(patch_size, tuple):
    raise ValueError(
        '`input_size` and `patch_size` should both be either int or tuple.')

  if len(input_size) != len(patch_size):
    raise ValueError(
        '`input_size` and `patch_size` should have the same number of '
        f'dimensions. Instead, received {len(input_size)=} and '
        f'{len(patch_size)=}.')
  return tuple(i // p for i, p in zip(input_size, patch_size))


def get_kernel_size_dilated(kernel_size,
                            kernel_dilation):
  """Constructs the dilated kernel size given the kernel size and dilation."""
  kernel_size_dilated = tuple([
      (k - 1) * d + 1 for k, d in zip(kernel_size, kernel_dilation)
  ])
  return kernel_size_dilated


def get_pre_conv_pads(kernel_size_dilated):
  """Constructs the padding locations/sizes for pre-convolution padding."""
  pads = (
      ((0, 0),)
      + tuple([((k - 1) // 2, k // 2) for k in kernel_size_dilated])
      + ((0, 0),)
  )
  return pads


def construct_per_modality_modules(
    module,
    modalities,
    common_kwargs,
    per_modality_kwargs,
    name,
):
  """Constructs modality-specific modules given common and specific kwargs.

  Args:
    module: Any configurable Flax module.
    modalities: A sequence of strings containing the modalities for which we
      construct the modules.
    common_kwargs: A dictionary of args to be used when configuring the modules.
      These kwargs are shared across all modalitites.
    per_modality_kwargs: A mapping between each modality and their kwargs.
      The kwargs are used when constructing each modality-specific module.
    name: The postfix of the name of the modality-specific modules. At the end,
      each module will be constructed by 'modality_name' convension.

  Returns:
    A dictionary that maps each modality to its specific module.
  """
  per_modality_modules = {}
  for modality in modalities:
    per_modality_modules[modality] = module(
        **common_kwargs,
        **per_modality_kwargs[modality],
        name='_'.join([modality, name]),
    )

  return per_modality_modules


def construct_per_modality_per_feature_modules(
    module,
    modalities,
    feature_keys,
    common_kwargs,
    per_modality_per_feature_kwargs,
    name,
):
  """Constructs modality-feature-specific modules given common+specific kwargs.

  Args:
    module: Any configurable Flax module.
    modalities: A sequence of strings containing the modalities for which we
      construct the modules.
    feature_keys: A sequence of strings containing the feature names for which
      we construct the modality-specific modules.
    common_kwargs: A dictionary of args to be used when configuring the modules.
      These kwargs are shared across all modalitites and feature types.
    per_modality_per_feature_kwargs: A mapping between each modality and their
      specific feature names. Each feature name is then mapped to their
      modality-feature-specific kwargs. The kwargs are used when constructing
      each modality-feature-specific module.
    name: The postfix of the name of the modality-feature-specific modules. At
      the end, each module will be constructed by 'modality_feature_name'
      convension.

  Returns:
    A dictionary that maps each modality to its specific feature name and then
    each feature name to its specific module.
  """
  per_modality_per_feature_modules = collections.defaultdict(dict)
  for modality in modalities:
    for feature_key in feature_keys:
      module_kwargs = common_kwargs
      if per_modality_per_feature_kwargs:
        module_kwargs.update(
            per_modality_per_feature_kwargs[modality][feature_key]
        )
      per_modality_per_feature_modules[modality][feature_key] = module(
          **module_kwargs,
          name='_'.join([modality, feature_key, name]),
      )

  return dict(per_modality_per_feature_modules)


def make_dot_general(config):
  """Constructs the dot general op based on the configuration."""
  if config is None:
    return lax.dot_general
  else:
    # We have to re-define a simple callable to make it hashable for jax calls
    def ret_dg(
        lhs,
        rhs,
        dimension_numbers,
        precision=None,
        preferred_element_type=None,
    ):
      return config.__call__(
          lhs=lhs,
          rhs=rhs,
          dimension_numbers=dimension_numbers,
          precision=precision,
          preferred_element_type=preferred_element_type,
      )
    return ret_dg


def index_to_mask(
    indices,
    length):
  """Converts a sequence of mask indices to a zero/one mask.

  Args:
    indices: An integer or a sequence of integer numbers indicating the
      indices of certain elements along an axis of an n-d array. If int
      is passed, this number should be [0, length). If a sequence of int
      is passed, this sequence should have a maximum length of 'length'.
      All numbers in this squence should be unique integers in [0, length).
    length: An integer, the maximum number of valid indices.
  Returns:
    A 1-d mask whose 'indices' positions are '1's and the rest are '0's.
    This mask has a shape of (length,).
  """
  # Normalize and verify indices if int/non-array
  indices, _ = _normalize_indices(indices)

  iota = lax.iota(jnp.int32, length)
  idx_one_hot = jnp.array(
      indices[Ellipsis, jnp.newaxis] == iota,
      dtype=jnp.int32)  # (n_indices, length)
  idx_zero_one = idx_one_hot.sum(axis=0)

  return idx_zero_one


def mask_to_index(mask, length):
  """Converts a mask sequence to a sequence of indices indicating 1s.

  Args:
    mask: A sequence of 0/1 integers, forming a 1D mask together.
    length: The exact expected number of non-zero elements in 'mask'. This
      is mandatory to avoid JIT errors. If this number is less than actual
      non-zero elements, the resulting array will clip the remaining indices.
      If this is larger than the non-zero elements, it will pad the indices
      with -1.
  Returns:
    A 1-d sequence with size (length,) elements indicate the positions that
    the 1s in 'mask' were placed.
  """

  return jnp.argwhere(mask == 1, size=length, fill_value=-1).reshape(-1)


def range_by_iota(start,
                  stop,
                  step = 1,
                  dtype = jnp.int32):
  """An efficient implementation of np.arange using lax.iota."""
  if not all([isinstance(start, int),
              isinstance(stop, int),
              isinstance(step, int)]):
    raise ValueError('Please provide integer values.')
  num_entries = int(np.ceil((stop - start) / step))
  return start + lax.iota(dtype, num_entries) * step


def take_along_axis(
    inputs,
    indices,
    axis,
    precision = None,
    dot_general = lax.dot_general):
  """A generic `take` implementation based on OHE and dot product.

  This is an SPMD-friendly version of jnp.take which has an improved memory
  consumption and runtime over jnp.take and/or any pythonic indexing/
  slicing methods. It constructs a one-hot encoding with shape
  [len(indices), len(inputs[axis])] and applies a dot product between this
  matrix and the inputs (on its 'axis' dimension). The resulting matrix is a
  tensor whose 'axis' dimension only had len(indices) elements.

  Args:
    inputs: An array with rank > 1.
    indices: An integer or a sequence of integer numbers indicating the
      indices of the to-be-fetched elements in inputs' axis dimension.
    axis: An integer indicating the dimension in which we want to fetch
      the elements.
    precision: The precision of dot_general, which we set to None by
      default, which is equivalent to 'bfloat16' for the fastest results.
    dot_general: The function that performs dot product.

  Returns:
    An array whose `axis` dimension only contains certain elements of the
    `axis` dimension of `inputs`. The position of those elements are defined
    by `indices` arg.
  """

  rank = inputs.ndim
  if rank < 1:
    raise ValueError(
        'Input must have rank >= 1. Instead, received a tensor with '
        f'shape {inputs.shape}')

  # Normalize and verify indices if int/non-array
  indices, squeezable = _normalize_indices(indices)

  # Normalize axis if negative
  axis = _normalize_axis(axis, rank)

  axis_length = inputs.shape[axis]
  iota = lax.iota(jnp.int32, axis_length)

  idx_one_hot = jnp.array(
      indices[Ellipsis, jnp.newaxis] == iota,
      dtype=inputs.dtype)  # (n_indices, axis_length)

  lhs_contracting_dims = (axis,)  # The `axis` dimension
  rhs_contracting_dims = (1,)  # The `axis` dimension
  lhs_batch_dims = ()
  rhs_batch_dims = ()
  outputs = dot_general(
      lhs=inputs,  # (..., axis_length, ...)
      rhs=idx_one_hot,  # (n_indices, axis_length)
      dimension_numbers=(
          (lhs_contracting_dims, rhs_contracting_dims),
          (lhs_batch_dims, rhs_batch_dims)),
      precision=precision,
  )  # --> (..., n_indices)

  # If axis is not the last dimension, transpose axes properly
  # (dot_general puts the contracting dim in the last dimension by default)
  if axis != (rank - 1):
    # (..., n_indices) -> (..., n_indices, ...)
    axes_order = list(range(axis)) + [rank-1] + list(range(axis, rank-1))
    outputs = jnp.transpose(outputs, axes_order)

  # Squeeze if there is a single-index
  if squeezable:
    outputs = lax.squeeze(outputs, [axis])

  return outputs


# TODO(b/243045192): merge shared blocks to one helper function
def scatter_along_axis(
    inputs,
    updates,
    indices,
    axis,
    batch_dims = (),
    precision = None,
    dot_general = lax.dot_general):
  """A generic `scatter` implementation based on masking and dot product.

  This is an SPMD-friendly implementation of array assignment in jax.numpy. The
  current implementation e.g. `array.at[..., indices, ...].set(updates)` may
  introduce overhead under a pjitted graph. Hence, we use this function for any
  array assignment in our pipeline.
  We first construct a sparse array out of `updates` with the same shape as
  `inputs`. This sparse array only contains non-zero values in the positions,
  `indices`, along the dimension `axis`. We also construct a mask with the same
  shape as `inputs`. This mask contains 0s in positions `indices` along the
  dimension `axis` and 1s elsewhere. We then construct the final array by
  `outputs = mask * inputs + sparse_updates` which results in an array with
  exactly the same values as `inputs.at[..., indices, ...].set(updates)`. A
  batched assignment is also supported (e.g. in applications where each sample
  in the batch may have its own `indices`, hence resulting in batched assignment
  along certain axis of the inputs).

  Args:
    inputs: An array with rank > 1.
    updates: An array with the same rank as inputs, and a shape whose non-`axis`
      dimensions should be broadcastable to inputs'. For example, if inputs
      has a shape of [batch, instance, length, dim] and we want to update
      its 'length' axis, the 'updates' array should have a shape brodcastable
      to [batch, instance, len(indices), dim].
    indices: An integer OR a sequence of integer numbers OR an array containing
      integer numbers, indicating the indices of the to-be-scattered elements in
      inputs' axis dimension. This arg could hold multiple shapes depending on
      the application. For example, assuming that an input with shape of
      [batch, instance, length, dim] is given, one can feed indices=0 and axis=2
      and expect to scatter the first element along the `length` dimension. One
      can also feed `indices = [0, 5]` and `axis = 2` and fetch the first and
      the sixth elements along the `length` axis. It is also acceptable to feed
      sample-specific indices and expect an ND scatter. For example, the
      `indices` arg could be an array with shape [batch, instance, num_indices]
      and scatter batch/instance-specific samples along `axis = 2`. In this case
      the user should also specify `batch_dims`=(0, 1) and `inputs`, `updates`,
      and `indices` should all have the same `batch` and `instance` dimensions.
    axis: An integer indicating the dimension in which we want to update the
      elements.
    batch_dims: A sequence of integers indicating the batch dims on both inputs
      and indices (if any). If provided, indices should have a rank of
      `len(batch_dims) + 1` and its batch dims should exactly have the same
      value as the `inputs`' and `updates`'.
    precision: The precision of dot_general, which we set to None by
      default, which is equivalent to 'bfloat16' for the fastest results.
    dot_general: The function that performs dot product.

  Returns:
    An array whose `axis` dimension is updated with values from `updates` whose
    positions are defined by `indices`.
  """
  rank = inputs.ndim
  if rank < 1:
    raise ValueError(
        'Input must have rank >= 1. Instead, received a tensor with '
        f'shape {inputs.shape}')

  updates_rank = updates.ndim
  if rank != updates_rank:
    raise ValueError(
        'Updates should have the same rank as inputs. Instead, received a '
        f'tensor with rank {updates_rank}, while a {rank}-rank tensor was '
        'expected.')

  # Normalize and verify indices if int/non-array
  indices, _ = _normalize_indices(indices, batch_dims)

  # Normalize axis if negative
  axis = _normalize_axis(axis, rank)

  if axis in batch_dims:
    raise ValueError(f'Scatter {axis=} cannot be in the {batch_dims=}')

  # Here we first create a one-hot encoding matrix with shape
  # [*batch_dims, n_indices, n_axis]. We first use this to project the array
  # 'updates' with shape [*batch_dims, ..., n_indices, ...] to an array with
  # shape [*batch_dims, ..., n_axis, ...]. The result of this transformation is
  # an expanded 'updates' array whose non-update entries are all zero. This
  # array is broadcastable to 'inputs'.
  n_axis = inputs.shape[axis]
  iota = lax.iota(jnp.int32, n_axis)
  idx_one_hot = jnp.array(
      indices[Ellipsis, jnp.newaxis] == iota,
      dtype=inputs.dtype)  # (*batch_dims, n_indices, n_axis)

  # The 'axis' dimension of 'updates'
  lhs_contracting_dims = (axis,)
  # The 'indices' dimension of 'idx_one_hot'
  rhs_contracting_dims = (len(batch_dims),)
  # The 'batch' dimension of both 'updates' and 'idx_one_hot'
  lhs_batch_dims = batch_dims
  rhs_batch_dims = batch_dims
  expanded_updates = dot_general(
      lhs=updates,  # (*batch_dims, ..., n_indices, ...)
      rhs=idx_one_hot,  # (*batch_dims, n_indices, n_axis)
      dimension_numbers=(
          (lhs_contracting_dims, rhs_contracting_dims),
          (lhs_batch_dims, rhs_batch_dims)),
      precision=precision,
  )  # --> (*batch_dims, ..., n_axis)

  # Once we get the expanded updates, we create a 0/1 mask with a shape
  # broadcastable to both 'inputs' and 'expanded_updates'.
  mask_shape = []
  for ax in range(rank):
    if ax in batch_dims or ax == axis:
      mask_shape.append(updates.shape[ax])
    else:
      mask_shape.append(1)
  mask = jnp.ones(mask_shape, dtype=jnp.int32)
  expanded_mask = dot_general(
      lhs=mask,  # (*batch_dims, ..., n_indices, ...)
      rhs=idx_one_hot.astype(jnp.int32),  # (*batch_dims, n_indices, n_axis)
      dimension_numbers=(
          (lhs_contracting_dims, rhs_contracting_dims),
          (lhs_batch_dims, rhs_batch_dims)),
      precision=precision,
  )  # --> (*batch_dims, ..., n_axis)
  expanded_mask = (1 - expanded_mask).astype(inputs.dtype)

  # If 'axis' is not the last dimension, the output of dot_general
  # automatically swaps the 'axis' dimension with the last dimension
  # here, we undo that for both expanded updates and mask.
  if axis != (rank - 1):
    # (*batch_dims, ..., n_axis) -> (*batch_dims, ..., n_axis, ...)
    axes_order = list(range(axis)) + [rank-1] + list(range(axis, rank-1))
    expanded_updates = jnp.transpose(expanded_updates, axes_order)
    expanded_mask = jnp.transpose(expanded_mask, axes_order)

  # we now create the scattered outputs by first masking the updatable
  # enries and then updating them by the expanded updates.
  outputs = inputs * expanded_mask + expanded_updates

  return outputs


# TODO(hassanak): Investigate if this is doable by matmul
def scatter_nd(indices,
               updates,
               shape):
  """JAX implementation of tf.scatter_nd.

  See https://www.tensorflow.org/api_docs/python/tf/scatter_nd, and
  https://github.com/jax-ml/jax/discussions/3658.

  Notes:
  - If multiple indices point to the same position, the output value at this
    position is accumulated.
  - Indices falling outside of the created array are quietly ignored.

  Args:
    indices: [num_items, n_dims] array of indices to update.
    updates: [num_items, ...] array of new data points.
    shape: Dimensions of the output array.

  Returns:
    An array of shape `shape` and the same type as `updates`, with updated
    values at given indices.
  """
  zeros = jnp.zeros(shape, updates.dtype)
  # Following `tf.scatter_nd`'s API, the inner vectors of `indices` have `n_dim`
  # values which index into `zeros`. We unpack it into arrays for each
  # dimension. This code is equivalent to `tf.unstack(indices, axis=-1)`.
  key = tuple(jnp.moveaxis(indices, -1, 0))
  return zeros.at[key].add(updates)


def fill_by_scatter(
    inputs,
    updates,
    keep_indices,
    fill_indices,
    axis,
    length,
    keep_batch_dims = (),
    fill_batch_dims = (),
    precision = None,
    dot_general = lax.dot_general):
  """An efficient in-place embedding masking based on scatter.

  Args:
    inputs: A truncated array with rank > 1 whose certain axis will be augmented
      by certain updates.
    updates: The values with which we augment the certain axis of the inputs.
    keep_indices: The indices of which we place the original inputs in the
      full final array. These indices should represent the positions along
      the axis-of-interest in the FULL array (as a result of expanding the
      inputs array).
    fill_indices: The indices of which we place the updates in the full final
      array. These indices should represent the positions along
      the axis-of-interest in the FULL array (as a result of expanding the
      inputs array).
    axis: The axis in which we are trying to fill the values.
    length: The resulting length of the axis-of-fill in the full array.
    keep_batch_dims: The batch dimension of the keep indices (if keep_indices
      has a rank > 1).
    fill_batch_dims: The batch dimension of the keep indices (if fill_indices
      has a rank > 1).
    precision: The precision of dot_general, which we set to None by
      default, which is equivalent to 'bfloat16' for the fastest results.
    dot_general: The function that performs dot product.

  Returns:
    An array whose `axis` dimension is filled with `updates`. This dimension is
    expected to have size `length` = num_keep_indices + num_fill_indices.
  """

  rank = inputs.ndim
  axis = _normalize_axis(axis, rank)

  # Create an empty array with full expected shape
  truncated_shape = inputs.shape
  full_shape = tuple(truncated_shape[n] if n != axis else length
                     for n in range(len(truncated_shape)))
  full_array = jnp.zeros(full_shape, dtype=inputs.dtype)

  # Scatter the original inputs (copy over)
  full_array = scatter_along_axis(
      inputs=full_array,
      updates=inputs,
      indices=keep_indices,
      axis=axis,
      batch_dims=keep_batch_dims,
      precision=precision,
      dot_general=dot_general)

  # Scatter the updates (fill in)
  full_array = scatter_along_axis(
      inputs=full_array,
      updates=updates,
      indices=fill_indices,
      axis=axis,
      batch_dims=fill_batch_dims,
      precision=precision,
      dot_general=dot_general)
  return full_array


def matmul(a,
           b,
           *,
           dot_general = lax.dot_general,
           precision = None):
  """Matrix Multiply with user-defined dot_general.

  Args:
    a: The left-hand-side array in the matrix multiplication.
    b: The righ-hand-side array in the matrix multiplication.
    dot_general: The function with which the dot product is performed.
    precision: The precision with which the dot product is performed.

  Returns:
    The matrix multiplication result: a * b assuming the first n-2 axes are
    batch dimensions.
  """
  a, b = jnp.asarray(a), jnp.asarray(b)
  for i, x in enumerate((a, b)):
    if x.ndim < 1:
      msg = (f'matmul input operand {i} must have ndim at least 1, '
             f'but it has ndim {x.ndim}')
      raise ValueError(msg)

  a_is_mat, b_is_mat = (a.ndim > 1), (b.ndim > 1)
  a_batch_dims: tuple[int | None, Ellipsis] = a.shape[:-2] if a_is_mat else ()
  b_batch_dims: tuple[int | None, Ellipsis] = b.shape[:-2] if b_is_mat else ()
  num_batch_dims = max(len(a_batch_dims), len(b_batch_dims))
  a_batch_dims = (None,) * (num_batch_dims - len(a_batch_dims)) + a_batch_dims
  b_batch_dims = (None,) * (num_batch_dims - len(b_batch_dims)) + b_batch_dims

  # Dimensions to squeeze from the inputs.
  a_squeeze: list[int] = []
  b_squeeze: list[int] = []

  # Positions of batch dimensions in squeezed inputs.
  a_batch = []
  b_batch = []

  # Desired index in final output of each kind of dimension, in the order that
  # lax.dot_general will emit them.
  idx_batch: list[int] = []
  idx_a_other: list[int] = []  # other = non-batch, non-contracting.
  idx_b_other: list[int] = []
  for i, (ba, bb) in enumerate(zip(a_batch_dims, b_batch_dims)):
    if ba is None:
      idx_b_other.append(i)
    elif bb is None:
      idx_a_other.append(i)
    elif ba == 1:
      idx_b_other.append(i)
      a_squeeze.append(len(idx_batch) + len(idx_a_other) + len(a_squeeze))
    elif bb == 1:
      idx_a_other.append(i)
      b_squeeze.append(len(idx_batch) + len(idx_b_other) + len(b_squeeze))
    elif ba == bb:
      a_batch.append(len(idx_batch) + len(idx_a_other))
      b_batch.append(len(idx_batch) + len(idx_b_other))
      idx_batch.append(i)
    else:
      raise ValueError('Incompatible shapes for matmul arguments: {} and {}'
                       .format(a.shape, b.shape))

  if a_is_mat: idx_a_other.append(num_batch_dims)
  if b_is_mat: idx_b_other.append(num_batch_dims + a_is_mat)
  perm = np.argsort(np.concatenate([idx_batch, idx_a_other, idx_b_other]))

  a = lax.squeeze(a, tuple(a_squeeze))
  b = lax.squeeze(b, tuple(b_squeeze))
  outputs = dot_general(
      a, b,
      (((a.ndim - 1,), (b.ndim - 1 - b_is_mat,)),
       (a_batch, b_batch)),
      precision=precision)
  outputs = lax.transpose(outputs, perm)
  return outputs


def create_attention_mask(
    query_token_mask,
    key_token_mask,
    elementwise_fn = jnp.multiply,
    dtype = jnp.float32,
):
  """Creates an attention mask based on query and key token masks.

  Args:
    query_token_mask: An array with shape [batch, instance, q_length] containing
      0/1 masks indicating whether a token is valid or not.
    key_token_mask: An array with shape [batch, instance, k_length] containing
      0/1 masks indicating whether a token is valid or not.
    elementwise_fn: Broadcasting elementwise comparison function.
    dtype: The dtype of the mask to be returned.

  Returns:
    A `[batch, instance, 1, q_length, k_length]` shaped mask for self or cross
    attention.
  """
  # [batch, instance, q_length] -> [batch, instance, q_length, 1]
  query_token_mask = jnp.expand_dims(query_token_mask, axis=-1)

  # [batch, instance, k_length] -> [batch, instance, 1, k_length]
  key_token_mask = jnp.expand_dims(key_token_mask, axis=-2)

  # Mask with shape [batch, instance, q_length, k_length]
  mask = elementwise_fn(query_token_mask, key_token_mask)

  # Extend the third dimension to be broadcastable with `heads`
  mask = jnp.expand_dims(mask, axis=-3)
  return mask.astype(dtype)


def create_causal_attention_mask(
    token_mask,
    dtype = jnp.float32,
):
  """Creates a causal attention mask based on a given token mask array.

  Args:
    token_mask: An array with shape [batch, instance, length] containing
      0/1 masks indicating whether a token is valid or not.
    dtype: The dtype of the mask to be returned.

  Returns:
    A `[batch, instance, 1, length, length]` shaped causal mask.
  """
  broadcasted_idx = lax.broadcasted_iota(jnp.int32, token_mask.shape, 2)
  causal_mask = create_attention_mask(
      broadcasted_idx, broadcasted_idx, jnp.greater_equal, dtype=dtype)
  attention_mask = create_attention_mask(
      token_mask, token_mask, jnp.multiply, dtype=dtype)
  causal_attention_mask = causal_mask * attention_mask
  return causal_attention_mask


def create_all_valid_causal_attention_mask(
    length, dtype = jnp.float32
):
  """Creates a causal attention mask from a given sequence length.

  This is a convenience function for when we want batch and instance size to be
  1 and all tokens to be valid.

  Args:
    length: The sequence length.
    dtype: The dtype of the mask to be returned.

  Returns:
    tensor of shape [1, 1, 1, length, length] with zeros for locations to be
    masked out.
  """
  token_mask = jnp.ones((1, 1, length))
  return create_causal_attention_mask(token_mask, dtype)


def create_groupwise_causal_mask(
    length, group_size, dtype = jnp.float32
):
  """Creates a groupwise causual attention mask.

  There is causal masking across groups, but full attention within each group.

  Args:
    length: The sequence length.
    group_size: The local window size. seq_len must be divisible by group_size.
    dtype: The dtype of the mask to be returned.

  Returns:
    Tensor of shape [1, 1, 1, length, length] with zeros for
    locations to be masked out.
  """
  valid_locs = jnp.ones([group_size, group_size], dtype=dtype)
  valid_locs = jnp.kron(jnp.eye(length // group_size), valid_locs)
  valid_locs = jnp.reshape(valid_locs, [1, 1, 1, length, length])

  one = jnp.array(1, dtype=dtype)
  return one - (
      one - create_all_valid_causal_attention_mask(length, dtype=dtype)
  ) * (one - valid_locs)


def construct_1d_positions(temporal_positions,
                           normalize = True):
  """Constructs 1D positional encoding IDs for image/video.

  Args:
    temporal_positions: An integer indicating the total number of temporal
      positions (or length of the sequence).
    normalize: A bool indicating whether the position values should be
      normalized to the maximum length or not. If this is set to True, the
      resulting tensor with the position values will have a dtype=jnp.float32.

  Returns:
    An array with shape (temporal_positions,) that encodes positional IDs
    corresponding to each temporal position. If normalize=True, this tensor
    will have dtype=jnp.float32. Otherwise, it will be dtype=jnp.int32.
  """
  pos_ids = lax.iota(jnp.int32, temporal_positions)  # (t,)

  if normalize:
    max_positions = jnp.asarray(temporal_positions, dtype=jnp.float32)
    pos_ids = jnp.asarray(pos_ids, dtype=jnp.float32) / max_positions

  return pos_ids


def construct_2d_positions(temporal_positions,
                           feature_positions,
                           normalize = True):
  """Constructs 2D positional encoding IDs for 2D signals.

  Args:
    temporal_positions: An integer indicating the total number of temporal
      positions (or length of the video).
    feature_positions: An integer indicating the total number of spectoral
      positions (or width of the video).
    normalize: A bool indicating whether the position values should be
      normalized to the maximum length or not. If this is set to True, the
      resulting tensor with the position values will have a dtype=jnp.float32.

  Returns:
    An array with shape [time * feature, 2] that encodes positional
    IDs corresponding to each spatio-temporal position. If normalize=True, this
    tensor will have dtype=jnp.float32. Otherwise, it will be dtype=jnp.int32.
  """
  temporal_ids, spectoral_ids = jnp.meshgrid(
      lax.iota(jnp.int32, temporal_positions),
      lax.iota(jnp.int32, feature_positions),
      indexing='ij')

  # (t, s, 2)
  pos_ids = jnp.stack([temporal_ids, spectoral_ids], axis=2)
  pos_ids = jnp.reshape(pos_ids, (-1, 2))  # (t*s, 2)

  if normalize:
    max_positions = jnp.asarray([temporal_positions,
                                 feature_positions], dtype=jnp.float32)
    pos_ids = jnp.asarray(pos_ids, dtype=jnp.float32) / max_positions

  return pos_ids


def construct_3d_positions(temporal_positions,
                           vertical_positions,
                           horizontal_positions,
                           normalize = True):
  """Constructs 3D positional encoding IDs for image/video.

  Args:
    temporal_positions: An integer indicating the total number of temporal
      positions (or length of the video).
    vertical_positions: An integer indicating the total number of vertical
      positions (or height of the video).
    horizontal_positions: An integer indicating the total number of horizontal
      positions (or width of the video).
    normalize: A bool indicating whether the position values should be
      normalized to the maximum length or not. If this is set to True, the
      resulting tensor with the position values will have a dtype=jnp.float32.

  Returns:
    An array with shape [time * height * width, 3] that encodes positional
    IDs corresponding to each spatio-temporal position. If normalize=True, this
    tensor will have dtype=jnp.float32. Otherwise, it will be dtype=jnp.int32.
  """
  temporal_ids, vertical_ids, horizontal_ids = jnp.meshgrid(
      lax.iota(jnp.int32, temporal_positions),
      lax.iota(jnp.int32, vertical_positions),
      lax.iota(jnp.int32, horizontal_positions),
      indexing='ij')

  # (t, h, w, 3)
  pos_ids = jnp.stack([temporal_ids, vertical_ids, horizontal_ids], axis=3)
  pos_ids = jnp.reshape(pos_ids, (-1, 3))  # (t*h*w, 3)

  if normalize:
    max_positions = jnp.asarray([temporal_positions,
                                 vertical_positions,
                                 horizontal_positions], dtype=jnp.float32)
    pos_ids = jnp.asarray(pos_ids, dtype=jnp.float32) / max_positions

  return pos_ids


def sample_drop_idx(length,
                    drop_rate,
                    rng):
  """Randomly (with uniform distribution) samples dropping indices.

  Args:
    length: A positive integer indicating the fixed length of the full inputs.
    drop_rate: The dropping rate, which should be a positive number in (0, 1).
    rng: A PRNG key used as the random key.

  Returns:
    A tuple of two 1D arrays containing the indices of the tokens that are
    kept/dropped after sampling.
  """
  if not 0. < drop_rate < 1.:
    raise ValueError('The drop rate should be a positive number in (0, 1). '
                     f'Instead, received {drop_rate=}.')

  if length < 1:
    raise ValueError('Length should be a positive integer. Instead, received '
                     f'{length=}.')

  max_rate = 1 - 1. / length
  if drop_rate >= max_rate:
    raise ValueError(
        f'The configured dropping {drop_rate=} leads to full drop of the entire'
        f' tokens. This specific input contains {length} tokens. Please provide'
        f' a rate in the range of (0., {max_rate})')

  num_tokens_to_keep = int((1 - drop_rate) * length)
  token_idx = lax.iota(jnp.int32, length)
  token_idx_shuffled = jax.random.permutation(rng, token_idx, independent=True)
  keep_idx = jnp.sort(token_idx_shuffled[:num_tokens_to_keep])
  drop_idx = jnp.sort(token_idx_shuffled[num_tokens_to_keep:])

  return keep_idx, drop_idx


def concatenate_cross_modal_tokens(
    token_embeds,
    token_masks,
    concatenated_token_embed_shardings = (),
    contrarenated_token_mask_shardings = (),
):
  """Concatenates tokens of multiple modalities into a single multimodal one.

  Args:
    token_embeds: A sequence of arrays each containing the token embeddings of
      a certain modality.
    token_masks: A sequence of arrays each containing the token masks of a
      certain modality. Token masks are optional and any of them could be None.
      If ALL of token masks are None, a None is returned as the concatenated
      mask.
    concatenated_token_embed_shardings: Sharding annotations for the
      concatenated token embeds.
    contrarenated_token_mask_shardings: Sharding annotations for the
      concatenated token masks.

  Returns:
    The concatenated token embeddings and masks.

  Raises:
    ValueError: if the number of token embeddings and masks do not match.
    ValueError: if at least one of the token embeddings is None.
  """
  if len(token_masks) != len(token_embeds):
    raise ValueError('The number of token masks should match the number of '
                     f'token embeds. However, received {len(token_masks)=} '
                     f'and {len(token_embeds)=}.')
  if any([token_embed is None for token_embed in token_embeds]):
    raise ValueError('All of the token embeddings should be valid arrays.')

  # Concatenate and shard the token embeddings
  concatentated_token_embed = jnp.concatenate(token_embeds, axis=2)
  concatentated_token_embed = sharding.shard_array(
      concatentated_token_embed, concatenated_token_embed_shardings)

  # If all masks in the sequence are None, the concatenated mask should be None
  # If some of the masks are None, we construct an all-one mask since we need
  # valid arrays to concatenate with each other.
  if all([token_mask is None for token_mask in token_masks]):
    concatentated_token_mask = None
  else:
    embed_dtypes = []
    mask_dtypes = []
    for token_embed, token_mask in zip(token_embeds, token_masks):
      embed_dtypes.append(token_embed.dtype)
      if token_mask is not None:
        mask_dtypes.append(token_mask.dtype)
    embed_dtypes = set(embed_dtypes)
    mask_dtypes = set(mask_dtypes)
    if len(embed_dtypes) > 1 or len(mask_dtypes) > 1:
      raise ValueError(
          'Token embeddings or masks with different dtypes where found: '
          f'{embed_dtypes=}, {mask_dtypes=}. Please provide token '
          'masks with the same dtypes.')

    mask_dtype = mask_dtypes.pop()
    valid_token_masks = []
    for token_embed, token_mask in zip(token_embeds, token_masks):
      if token_mask is None:
        batch_size, n_instance, n_tokens = token_embed.shape[0:3]
        token_mask = jnp.ones(
            (batch_size, n_instance, n_tokens),
            dtype=mask_dtype)
        token_mask = sharding.shard_array(
            token_mask, contrarenated_token_mask_shardings)
      valid_token_masks.append(token_mask)

    concatentated_token_mask = jnp.concatenate(valid_token_masks, axis=2)
    concatentated_token_mask = sharding.shard_array(
        concatentated_token_mask, contrarenated_token_mask_shardings)

  return concatentated_token_embed, concatentated_token_mask


def split_cross_modal_tokens(
    concatenated_token_features,
    seq_lengths,
    concatenation_order,
    split_token_features_shardings = (),
    dot_general = lax.dot_general,
):
  """Splits concatenated tokens of multiple modalities into separate ones.

  Args:
    concatenated_token_features: An array containing the concatenated token
      features of different modalities.
    seq_lengths: A nested dictionary that maps the modalities and their
      features (whose embeddings are present in `concatenated_token_features`)
      to their corresponding sequence lengths.
    concatenation_order: A tuple of string pairs indicating the order in which
      different features of different modalities are concatenated. Each pair
      is presented as a tuple of strings as (modality, feature_name). These,
      along with `seq_lengths` are later used to split the features at the
      output of the encoder model.
    split_token_features_shardings: Sharding annotations for the split token
      features.
    dot_general: The function that performs dot product in the `take_along_axis`
      function.

  Returns:
    A nested dictionary that maps the name of the modalities and their specific
    feature names to their corresponding split token features.

  Raise:
    ValueError: if the `concatenation_order` and `seq_lengths` do not match
      the same hieararchy.
  """
  nested_seq_lengths_keys = _get_nested_keys(seq_lengths)
  if len(concatenation_order) != len(nested_seq_lengths_keys):
    raise ValueError(
        'The concatenation order and the sequence lengths should have the same '
        f'number of elements. Instead, received {len(concatenation_order)=} and'
        f' {len(nested_seq_lengths_keys)=}.')

  if set(concatenation_order) != set(nested_seq_lengths_keys):
    raise ValueError(
        'The concatenation order and the sequence lengths should contain the '
        'same modalities and feature names. Instead, received '
        f'{set(concatenation_order)} and {set(seq_lengths.keys())}.')

  if len(nested_seq_lengths_keys) == 1:
    modality = set(seq_lengths.keys()).pop()
    token_feature_name = set(seq_lengths[modality].keys()).pop()
    return {modality: {token_feature_name: concatenated_token_features}}

  split_token_features = collections.defaultdict(dict)
  take_index_offset = 0
  for modality, token_feature_name in concatenation_order:
    # Fetch the modality-specific sequence length
    seq_length = seq_lengths[modality][token_feature_name]
    indices = take_index_offset + lax.iota(jnp.int32, seq_length)

    # Fetch the modality specific features
    split_token_features[modality][token_feature_name] = take_along_axis(
        inputs=concatenated_token_features,
        indices=indices,
        axis=2,
        dot_general=dot_general,
    )

    # Assert the sharding of the embeddings
    split_token_features[modality][token_feature_name] = (
        sharding.shard_array(
            split_token_features[modality][token_feature_name],
            split_token_features_shardings,
        )
    )

    # Update the offset for the next modality
    take_index_offset += seq_length

  return dict(split_token_features)


def extend_token_mask(
    token_mask,
    extension,
    extended_token_mask_shardings = (),
):
  """Extends a given token mask with '1' depending on the extension type.

  This method is useful in the case of audio and vision modalities that special
  tokens are prepended/appended on-the-fly and the data pipeline is not aware
  of such tokens.

  Args:
    token_mask: An array with rank 3 and shape of (batch, instance, length),
      containing 0/1s in certain positions.
    extension: A string with acceptable values 'prepend'/'append'.
    extended_token_mask_shardings: Sharding annotations for the extended token
      mask.

  Returns:
    An extended array with rank 3 and shape of (batch, instance, length + 1).

  Raises:
    ValueError if token_mask has a non-3 rank.
    ValueError if the extension type is not supported.
  """

  token_mask_shape = token_mask.shape
  mask_rank = token_mask.ndim
  expected_rank = constants.DataFeatureRank.Common.TOKEN_MASK
  if mask_rank != expected_rank:
    raise ValueError(
        f'Token mask should have a rank of {expected_rank}, '
        f'received rank {mask_rank}')

  batch_size, n_instance = token_mask_shape[0:2]
  extension_mask = jnp.ones((batch_size, n_instance, 1),
                            dtype=token_mask.dtype)
  if extended_token_mask_shardings:
    extension_mask_shardings = tuple(
        extended_token_mask_shardings[:-1]) + (None,)
  else:
    extension_mask_shardings = extended_token_mask_shardings
  extension_mask = sharding.shard_array(
      extension_mask, extension_mask_shardings)
  if extension == constants.Extension.PREPEND:
    extended_token_mask = jnp.concatenate([extension_mask, token_mask], axis=2)
  elif extension == constants.Extension.APPEND:
    extended_token_mask = jnp.concatenate([token_mask, extension_mask], axis=2)
  else:
    raise ValueError(
        f'Please provide a valid extension type. Received {extension}.')

  return sharding.shard_array(
      extended_token_mask, extended_token_mask_shardings)


def extend_attention_bias(
    attention_bias,
    extension,
    extended_attention_bias_shardings = (),
):
  """Extends a given attention bias with '0' depending on the extension type.

  This method is useful in the case of audio and vision modalities that special
  tokens are prepended/appended on-the-fly and the data pipeline is not aware
  of such tokens.

  Args:
    attention_bias: An array with rank 3 and shape of (head, q_len, kv_len),
      containing 0/1s in certain positions.
    extension: A string with acceptable values 'prepend'/'append'.
    extended_attention_bias_shardings: Sharding annotations for the extended
      attention bias.

  Returns:
    An extended array with rank 3 and shape of (head, q_len + 1, kv_len + 1).

  Raises:
    ValueError if attention_bias has a non-3 rank.
    ValueError if the extension type is not supported.
  """

  attn_bias_rank = attention_bias.ndim
  if attn_bias_rank != 3:
    raise ValueError(
        'Please provide an attention bias with rank 3. Instead, received an '
        f'array with rank {attn_bias_rank}.')

  if extension == constants.Extension.PREPEND:
    attention_bias = jnp.pad(attention_bias, ((0, 0), (1, 0), (1, 0)))
  elif extension == constants.Extension.APPEND:
    attention_bias = jnp.pad(attention_bias, ((0, 0), (0, 1), (0, 1)))
  else:
    raise ValueError(
        f'Please provide a valid extension type. Received {extension}.')

  return sharding.shard_array(attention_bias, extended_attention_bias_shardings)


def extend_token_pos_id(
    token_pos_id,
    extension,
    max_length = None,
    extended_token_pos_id_shardings = (),
):
  """Extends a given token position ID depending on the extension type.

  This method is useful in the case of audio and vision modalities that special
  tokens are prepended/appended on-the-fly and the data pipeline is not aware
  of such tokens.

  Args:
    token_pos_id: An array with rank 3 and shape of (batch, instance, length),
      containing integers indicating the position of the flattened tokens.
    extension: A string with acceptable values 'prepend'/'append'.
    max_length: An optional integer indicating the maximum expected ID (length)
      in the given token_pos_id. This is useful if extension == 'append', since
      we need to know the exact length of the original input to be able to
      append `max_length` at the end of the `token_pos_id`.
    extended_token_pos_id_shardings: Sharding annotations for the extended
      token_pos_id.

  Returns:
    An extended array with rank 3 and shape of (batch, instance, length + 1).

  Raises:
    ValueError if token_mask has a non-3 rank.
    ValueError if the extension type is not supported.
  """

  pos_id_rank = token_pos_id.ndim
  expected_rank = constants.DataFeatureRank.Common.TOKEN_POSITION_ID
  if pos_id_rank != expected_rank:
    raise ValueError(
        f'Token position ID should have a rank of {expected_rank}, '
        f'received rank {pos_id_rank}')

  if extension == constants.Extension.APPEND and max_length is None:
    raise ValueError(
        'When extension is `append`, maximum expected length should be provided'
        '. Instead, received None.')

  if extension == constants.Extension.PREPEND:
    # First shift all positions by one step
    token_pos_id += 1
    # Then fill 0 in the very first position
    token_pos_id = jnp.pad(token_pos_id, ((0, 0), (0, 0), (1, 0)))
  elif extension == constants.Extension.APPEND:
    token_pos_id = jnp.pad(token_pos_id, ((0, 0), (0, 0), (0, 1)),
                           constant_values=max_length)
  else:
    raise ValueError(
        f'Please provide a valid extension type. Received {extension}.')

  return sharding.shard_array(token_pos_id, extended_token_pos_id_shardings)


def top_k(x, k = 1, axis = -1):
  """Returns the indices of the top k items sorted, efficiently.

  Args:
    x: The array to perform the operation on.
    k: The number of items to keep.
    axis: The axis to perform the operation on.

  Returns:
    The indices top k items sorted.
  """
  if k == 1:
    return jnp.argmax(x, axis=axis, keepdims=True)
  else:
    _, indices = lax.approx_max_k(x, k, reduction_dimension=axis)
    return indices


def extract_volume_patches_from_raw_voxels(
    inputs,
    patch_sizes,
    flatten = False,
    volume_patches_shardings = (),
):
  """Transforms raw inputs into volume patches.

  Args:
    inputs: Array of shape `[batch, instance, time, height, width, channels]`
    patch_sizes: tuple of `[time_patch, height_patch, width_patch]`
    flatten: Flattens the spatio-temporal tokens into a single dimension.
    volume_patches_shardings: Sharding annotations for the volume patches.

  Returns:
    Array of shape `[batch, instance, time_patches, height_patches,
        width_patches, time_patch, height_patch, width_patch, pixels]`
  """
  inputs = jnp.reshape(
      inputs,
      (inputs.shape[0], inputs.shape[1], inputs.shape[2] // patch_sizes[0],
       patch_sizes[0], inputs.shape[3] // patch_sizes[1], patch_sizes[1],
       inputs.shape[4] // patch_sizes[2], patch_sizes[2], inputs.shape[5]))

  # Transpose axes:
  #     (batch, instance, patched_time, voxel_time, patched_height,
  #      voxel_height, patched_width, voxel_width, channels) -->
  #     (batch, instance, patched_time, patched_height, patched_width,
  #      voxel_time, voxel_height, voxel_width, channels)
  inputs = jnp.transpose(inputs, (0, 1, 2, 4, 6, 3, 5, 7, 8))
  inputs = jnp.reshape(inputs,
                       (inputs.shape[0], inputs.shape[1], inputs.shape[2],
                        inputs.shape[3], inputs.shape[4], -1))

  if flatten:
    inputs = jnp.reshape(
        inputs, (inputs.shape[0], inputs.shape[1], -1, inputs.shape[-1]))

  return sharding.shard_array(inputs, volume_patches_shardings)


def extract_raw_voxels_from_volume_patches(
    patches,
    patch_sizes,
    flattened = False,
    expected_voxel_shape = None,
    volume_patches_shardings = (),
):
  """Transforms volume patches into raw voxels.

  Args:
    patches: Array of shape `[batch, instance, time_patches, height_patches,
      width_patches, voxels]` (if not flattened) or an array of shape
      `[batch, instance, time_patches * height_patches * width_patches, voxels]`
      (if flattened).
    patch_sizes: tuple of `[time_patch, height_patch, width_patch]`.
    flattened: Indicating whether the spatio-temporal tokens are flattened into
      a single dimension.
    expected_voxel_shape: If flattened, this should be provided to properly
      reconstruct the original raw voxels.
    volume_patches_shardings: Sharding annotations for the volume patches.

  Returns:
    Array of shape `[batch, instance, time, height, width, channels]`
  """
  if flattened:
    if expected_voxel_shape is None or not expected_voxel_shape:
      raise ValueError(
          'The `expected_voxel_shape` should be provided when the patches are '
          f'flattened. Instead, received {expected_voxel_shape=}.')

    num_voxels_in_patches = np.prod(patches.shape[2:])
    num_expected_voxels = np.prod(expected_voxel_shape)
    if num_expected_voxels != num_voxels_in_patches:
      raise ValueError(
          'There is not enough voxels in the patches to reconstruct the '
          f'`expected_voxel_shape`. {num_expected_voxels=} and '
          f'{num_voxels_in_patches=}. The {expected_voxel_shape=} does not '
          f'match patches_shape={patches.shape}.')
    expected_patches_rank = constants.DataFeatureRank.Common.TOKEN_RAW

  else:
    expected_patches_rank = constants.DataFeatureRank.Vision.RAW

  if patches.ndim != expected_patches_rank:
    raise ValueError(
        f'The patches should have a rank of {expected_patches_rank}. Instead, '
        f'received an array with rank={patches.ndim}')

  if flattened:
    patches = jnp.reshape(
        patches,
        (patches.shape[0], patches.shape[1],
         expected_voxel_shape[0] // patch_sizes[0],
         expected_voxel_shape[1] // patch_sizes[1],
         expected_voxel_shape[2] // patch_sizes[2], -1))

  patches = sharding.shard_array(patches, volume_patches_shardings)
  patches = jnp.reshape(
      patches,
      (*(patches.shape[:-1]), *patch_sizes, -1))

  # Transpose axes:
  #     (batch, instance, patched_time, patched_height, patched_width,
  #      voxel_time, voxel_height, voxel_width, channels) -->
  #     (batch, instance, patched_time, voxel_time, patched_height,
  #      voxel_height, patched_width, voxel_width, channels)
  patches = jnp.transpose(patches, (0, 1, 2, 5, 3, 6, 4, 7, 8))
  patches = jnp.reshape(
      patches,
      (patches.shape[0], patches.shape[1], np.prod(patches.shape[2:4]),
       np.prod(patches.shape[4:6]), np.prod(patches.shape[6:8]), -1))
  return patches


def extract_patches_from_raw_waveform(
    inputs,
    patch_size,
    patches_shardings = (),
):
  """Transforms raw waveform inputs into 1D patches.

  Args:
    inputs: Array of shape `[batch, instance, time, channels]`
    patch_size: The temporal patch size.
    patches_shardings: Sharding annotations for the patches.

  Returns:
    Array of shape `[batch, instance, time_patches, samples]`
  """
  inputs = jnp.reshape(
      inputs,
      (inputs.shape[0], inputs.shape[1], inputs.shape[2] // patch_size, -1))

  return sharding.shard_array(inputs, patches_shardings)


def extract_raw_waveform_from_patches(
    patches,
    patch_size,
    patches_shardings = (),
):
  """Transforms 1D patches into raw waveforms.

  Args:
    patches: Array of shape `[batch, instance, time_patches, samples]`.
    patch_size: The temporal patch size.
    patches_shardings: Sharding annotations for the patches.

  Returns:
    Array of shape `[batch, instance, time, channels]`
  """
  patches = sharding.shard_array(patches, patches_shardings)
  patches = jnp.reshape(
      patches,
      (patches.shape[0], patches.shape[1], patches.shape[2] * patch_size, -1))
  return patches


# TODO(hassanak): add support for precise jax or np modes to avoid memory leak
def harmonic_mean(x, eps = 1e-6):
  """Calculates the harmonic mean of an array."""
  return 1. / np.sum(1. / (x + eps))


# TODO(hassanak): add support for precise jax or np modes to avoid memory leak
def geometric_mean(x):
  """Calculates the geometric mean of an array."""
  return np.prod(x) ** (1. / len(x))


def aggregate_metrics(
    all_metrics,
    naming_patterns = (),
    name = 'aggregated'):
  """Applies metric aggregation statistics on the configured patterns.

  Args:
    all_metrics: a dict of all metrics.
    naming_patterns: a sequence of regex patterns applied to all metric keys.
      The metrics that match any of the patterns will be aggregated.
    name: a name for the aggregated metrics.

  Returns:
    a dict with aggregated statistics.
  """
  all_metrics = traverse_util.flatten_dict(all_metrics, sep='/')

  aggregated_metrics = []
  for key, value in all_metrics.items():
    pattern = '|'.join(f'({p})' for p in naming_patterns)
    if re.fullmatch(pattern, key):
      aggregated_metrics.append(value)

  output_metrics = {}
  if aggregated_metrics:
    aggregated_metrics = np.array(aggregated_metrics)
    output_metrics.update({
        f'{name}/mean': np.mean(aggregated_metrics),
        f'{name}/mean_harmonic': harmonic_mean(aggregated_metrics),
        f'{name}/mean_geometric': geometric_mean(aggregated_metrics),
        f'{name}/stddev': np.std(aggregated_metrics),
        f'{name}/median': np.median(aggregated_metrics),
    })
  else:
    logging.info(
        'No metrics found for %s with patterns %s and metrics %s.',
        name, naming_patterns, all_metrics.keys())

  return output_metrics


def replace_train_step(
    state,
    step):
  """Replaces the step counters for the given train state.

  Also replaces any values in `optax.ScaleByScheduleState` so that
  learning rate is set correctly.

  Args:
    state: the flax train state.
    step: the step to replace the current step counter.

  Returns:
    the flax train state with the replaced step counter.
  """
  step = jnp.array(step, jnp.int32)

  def _is_schedule(x):
    return isinstance(x, optax.ScaleByScheduleState)

  def _set_state_counter(x):
    if _is_schedule(x):
      return optax.ScaleByScheduleState(count=step)
    return x

  return state.replace(
      step=step,
      opt_state=jax.tree.map(
          _set_state_counter, state.opt_state, is_leaf=_is_schedule)
  )




# TODO(b/309894030): Add testing
def pre_conv_padding(inputs,
                     kernel_size,
                     kernel_dilation,
                     mode = 'constant',
                     **kwargs):
  """A generic pre-convolution padding.

  Args:
    inputs: An array with shape [batch, ..., features].
    kernel_size: A sequence of integers each defining the kernel size of the
      convolution along each dimension. The number of dimensions in this
      sequence should match the mid axes of the inputs (inputs.ndim - 2).
    kernel_dilation: A sequence with the same length as 'kernel_size' containing
      the dilation rate along each dimension.
    mode: The padding mode as accepted in jnp.pad.
    **kwargs: The rest of the args to be fed to jnp.pad.

  Returns:
    The padded inputs according to the padding configuration.
  """
  kernel_size_dilated = get_kernel_size_dilated(kernel_size, kernel_dilation)
  pads = get_pre_conv_pads(kernel_size_dilated)
  inputs = jnp.pad(inputs, pads, mode=mode, **kwargs)
  return inputs


# TODO(b/309894030): Add testing
def circular_pre_conv_padding(inputs,
                              kernel_size,
                              kernel_dilation):
  """A circular pre-convolution padding for multi-axis convolution modules.

  This function uses jnp.pad to perform padding using mode='wrap'.

  Args:
    inputs: An array with shape [batch, ..., features].
    kernel_size: A sequence of integers each defining the kernel size of the
      convolution along each dimension. The number of dimensions in this
      sequence should match the mid axes of the inputs (inputs.ndim - 2).
    kernel_dilation: A sequence with the same length as 'kernel_size' containing
      the dilation rate along each dimension.

  Returns:
    The padded inputs according to the padding configuration.
  """
  return pre_conv_padding(inputs, kernel_size, kernel_dilation, mode='wrap')


# TODO(b/309894030): Add testing
def circular_post_conv_padding(inputs,
                               pre_deconv_shape,
                               strides,
                               transpose_kernel):
  """A circular post-convolution padding for multi-axis deconvolution modules.

  This function uses jnp.pad to perform padding using mode='constant'.

  Args:
    inputs: An array with shape [batch, ..., features].
    pre_deconv_shape: Shape of the array before deconvolution.
    strides: Stride of the deconvolution op.
    transpose_kernel: If kernel should be transposed too.

  Returns:
    The padded inputs according to the padding configuration.
  """
  # For circular padding, we need to identify the size of the final output
  # ('period') along each spatial dimension, pad each dimension to an
  # integer number of periods, and wrap the array periodically around each
  # dimension. Padding should be done in such a way that the start of the
  # original input data inside the padded array is located at integer
  # number of periods - otherwise the result would be circularly shifted.

  # Compute period along each spatial dimension - it's input size scaled
  # by the stride.
  scaled_x_dims = [
      x_dim * stride
      for x_dim, stride in zip(pre_deconv_shape[1:-1], strides)
  ]
  # Compute difference between the current size of y and the final output
  # size, and complement this difference to 2 * period - that gives how
  # much we need to pad.
  post_deconv_shape = inputs.shape
  size_diffs = [
      -(y_dim - x_dim) % (2 * x_dim)
      for y_dim, x_dim in zip(post_deconv_shape[1:-1], scaled_x_dims)
  ]
  if transpose_kernel:
    # If the kernel is transposed, the '+1' is put on the right to mirror
    # the regular convolution. If the same kernel parameters are used as for
    # Conv, this layer then computes the proper transpose convolution.
    pads = [
        (size_diff // 2, (size_diff + 1) // 2) for size_diff in size_diffs
    ]
  else:
    # Divide the padding equally between left and right. The choice to put
    # '+1' on the left (and not on the right) represents a convention for
    # aligning even-sized kernels.
    pads = [
        ((size_diff + 1) // 2, size_diff // 2) for size_diff in size_diffs
    ]
  inputs = jnp.pad(inputs, [(0, 0)] + pads + [(0, 0)], mode='constant')
  # Wrap the result periodically around each spatial dimension,
  # one by one.
  for i in range(1, inputs.ndim - 1):
    inputs = inputs.reshape(
        inputs.shape[:i] + (-1, scaled_x_dims[i - 1]) + inputs.shape[i + 1 :]
    )
    inputs = inputs.sum(axis=i)
  return inputs


# TODO(b/309894030): Add testing
def causal_1d_pre_conv_padding(inputs,
                               kernel_size,
                               kernel_dilation):
  """A causal pre-convolution padding for 1D convolution modules.

  Args:
    inputs: An array with shape [batch, time, features].
    kernel_size: A tuple of a single integer defining the kernel size of the
      convolution along the temporal dimension.
    kernel_dilation: A sequence with the same length as 'kernel_size' containing
      the dilation rate along the temporal dimension.

  Returns:
    The padded inputs according to the padding configuration.
  """
  if len(kernel_size) != 1:
    raise ValueError(
        'This padding function is only implemented for 1D convolutions.'
    )
  left_pad = kernel_dilation[0] * (kernel_size[0] - 1)
  pads = [(0, 0), (left_pad, 0), (0, 0)]
  inputs = jnp.pad(inputs, pads)
  return inputs


# TODO(b/309894030): Add testing
def causal_pre_conv_padding(inputs,
                            kernel_size,
                            kernel_dilation,
                            temporal_axis = 1):
  """A causal pre-convolution padding for multi-axis convolution modules.

  This function uses jnp.pad to perform padding using mode='edge'.

  Args:
    inputs: An array with shape [batch, ..., features].
    kernel_size: A sequence of integers each defining the kernel size of the
      convolution along each dimension. The number of dimensions in this
      sequence should match the mid axes of the inputs (inputs.ndim - 2).
    kernel_dilation: A sequence with the same length as 'kernel_size' containing
      the dilation rate along each dimension.
    temporal_axis: The axis for which the causal padding is performed.

  Returns:
    The padded inputs according to the padding configuration.
  """
  kernel_size_dilated = get_kernel_size_dilated(kernel_size, kernel_dilation)
  pads = list(get_pre_conv_pads(kernel_size_dilated))
  left_pad = kernel_dilation[temporal_axis - 1] * (
      kernel_size[temporal_axis - 1] - 1)
  pads[temporal_axis] = (left_pad, 0)
  pads = tuple(pads)
  inputs = jnp.pad(inputs, pads, mode='edge')
  return inputs




# TODO(b/309894030): Add testing
def move_outer_to_inner(inputs, rates):
  """Reshapes an nd array by moving outer dim entries to inner dims."""
  rank = inputs.ndim
  if rank < 4:
    raise ValueError('Inputs should be at least rank 4 containing '
                     '[batch, instance, ..., depth]. Instead, received '
                     f'{inputs.shape=}.')

  batch, instance, *inner, outer = inputs.shape
  if len(rates) != len(inner):
    raise ValueError(
        'Number of rates should match the inner dims of the array. Instead, '
        f'received {inner=} and {rates=}.')

  if outer % np.prod(rates) != 0:
    raise ValueError(
        'The outer dimension of the array should be divisible by the total '
        f'transformation rates. Instead, received {inputs.shape[-1]=} and '
        f'{rates=}.')

  # Take a prod(rates) entries from the outer dimension
  new_outer = outer // np.prod(rates)

  # Distribute them along new inner dimensions
  inputs = jnp.reshape(inputs, [batch, instance, *inner, *rates, new_outer])

  # Move them beside their corresponding inner dimension. For example, if we
  # previously had [..., inner[0], inner[1], ..., outer], we now have
  # [..., inner[0], rates[0], inner[1], rates[1], ..., outer // prod(rates)]
  transpose_shape = [0, 1]
  for n in range(len(inner)):
    transpose_shape.append(2 + n)  # index of the old inner dim
    transpose_shape.append(2 + n + len(rates))  # index of the moved-over rate
  transpose_shape.append(len(rates) + rank - 1)  # index of the outer dim
  inputs = jnp.transpose(inputs, transpose_shape)

  # Squeeze in the newly moved-over entries
  new_inner = [i * r for i, r in zip(inner, rates)]
  inputs = jnp.reshape(inputs, [batch, instance, *new_inner, new_outer])
  return inputs
