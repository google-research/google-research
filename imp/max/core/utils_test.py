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

"""Tests for utils."""

import contextlib
import functools
import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
from jax import numpy as jnp
import numpy as np
import tensorflow as tf

from imp.max.core import constants
from imp.max.core import utils

jax.config.update('jax_threefry_partitionable', False)

PROCESS_COUNT = 4
LOCAL_DEVICE_COUNT = 2


# pylint: disable=line-too-long
@contextlib.contextmanager
def mock_multiprocess_context(process_index = 0):
  with mock.patch.object(jax, 'local_device_count', return_value=LOCAL_DEVICE_COUNT), \
    mock.patch.object(jax, 'process_count', return_value=PROCESS_COUNT), \
    mock.patch.object(jax, 'device_count', return_value=PROCESS_COUNT * LOCAL_DEVICE_COUNT), \
    mock.patch.object(jax, 'local_devices', return_value=jax.devices()), \
    mock.patch.object(jax, 'process_index', return_value=process_index):
    assert jax.process_count() == PROCESS_COUNT
    assert jax.local_device_count() == LOCAL_DEVICE_COUNT
    assert jax.device_count() == PROCESS_COUNT * LOCAL_DEVICE_COUNT
    yield
# pylint: enable=line-too-long


class IoUtilsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.tmp_dir = self.create_tempdir()

  def test_should_write_file(self):
    tmp_file = os.path.join(self.tmp_dir, 'config.txt')
    self.assertTrue(utils.should_write_file())

    with tf.io.gfile.GFile(tmp_file, 'w') as f:
      f.write('test')

    self.assertTrue(utils.should_write_file())

  def test_should_write_file_multiprocess(self):
    with mock_multiprocess_context(process_index=1):
      self.assertFalse(utils.should_write_file())

  def test_write_once(self):
    tmp_file = os.path.join(self.tmp_dir, 'config.txt')
    utils.safe_write(tmp_file, 'test')
    utils.safe_write(tmp_file, 'test2')

    with tf.io.gfile.GFile(tmp_file, 'r') as f:
      self.assertEqual(f.read(), 'test2')


class JaxOpsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('base', (4, 5), np.arange(3), 1, 'float32'),
      ('base_bfloat16', (4, 5), np.arange(3), 1, 'bfloat16'),
      ('base_non_array_indices', (4, 5), range(3), 1, 'float32'),
      ('single_non_squeeze', (2, 4, 5), [1], 0, 'float32'),
      ('single_squeezed', (2, 4, 5), 1, 0, 'float32'),
      ('list_of_indices', (2, 4, 5), [1, 3], 1, 'float32'),
      ('negative_axis', (2, 8, 9), range(4), -2, 'float32'),
      ('1d', (4,), range(2), 0, 'float32'),
      ('2d', (4, 6), range(2), 0, 'float32'),
      ('3d', (4, 6, 7), range(4), 2, 'float32'),
      ('4d', (4, 6, 7, 3), range(3), 1, 'float32'),
      ('4d_squeezed', (4, 6, 7, 3), 1, 1, 'float32'),
  )
  def test_valid_take_along_axis(self, input_shape, indices, axis, precision):

    @jax.jit
    def _run_forward(inputs):
      outputs = utils.take_along_axis(
          inputs=inputs, indices=indices, axis=axis, precision=precision)
      return outputs

    inputs = jax.random.uniform(jax.random.key(0), input_shape)
    outputs = _run_forward(inputs)
    outputs_jnp = jnp.take(inputs, jnp.array(indices), axis)

    chex.assert_equal(jnp.linalg.norm(outputs - outputs_jnp), 0.)
    chex.assert_shape(outputs, outputs_jnp.shape)

  @parameterized.named_parameters(
      ('invalid_axis', (4, 5), range(3), 2),
      ('invalid_rank', (), 0, 0),
      ('invalid_indices', (4, 5), np.ones([2, 2], np.int32), 0),
      ('invalid_large_axis', (4, 5), range(3), 2),
      ('invalid_negative_axis', (4, 5), range(3), -3),
  )
  def test_invalid_take_along_axis(self, input_shape, indices, axis):

    @jax.jit
    def _run_forward(inputs):
      outputs = utils.take_along_axis(
          inputs=inputs, indices=indices, axis=axis)
      return outputs

    inputs = jax.random.uniform(jax.random.key(0), input_shape)
    with self.assertRaises(ValueError):
      _run_forward(inputs)

  @parameterized.named_parameters(
      ('base', (4, 5), (4, 3), np.arange(3), 1, (), 'float32'),
      ('base_bfloat16', (4, 5), (4, 3), np.arange(3), 1, (), 'bfloat16'),
      ('non_array_indices', (4, 5), (4, 3), range(3), 1, (), 'float32'),
      ('list_of_indices', (4, 5), (4, 2), [0, 2], 1, (), 'float32'),
      ('single_index', (4, 5), (4, 1), 2, 1, (), 'float32'),
      ('negative_axis', (4, 5), (4, 3), np.arange(3), -1, (), 'float32'),
      ('2d_indices_2d_inputs', (2, 5), (2, 2),
       np.array([[0, 1], [1, 2]]), 1, (0,), 'float32'),
      ('2d_indices_3d_inputs', (2, 4, 5), (2, 2, 5),
       np.array([[0, 1], [1, 2]]), 1, (0,), 'float32'),
      ('3d_indices_3d_inputs', (2, 2, 5), (2, 2, 2),
       np.array([[[0, 1], [1, 2]], [[3, 4], [1, 3]]]), 2, (0, 1), 'float32'),
  )
  def test_valid_scatter_along_axis(self,
                                    input_shape,
                                    update_shape,
                                    indices,
                                    axis,
                                    batch_dims,
                                    precision):

    @jax.jit
    def _run_forward(inputs, updates):
      outputs = utils.scatter_along_axis(
          inputs=inputs,
          updates=updates,
          indices=indices,
          axis=axis,
          batch_dims=batch_dims,
          precision=precision)
      return outputs

    def _np_scatter_along_axis(inputs, updates, indices, axis, batch_dims):
      inputs = np.array(inputs)
      indices = np.asarray(indices)
      updates = np.asarray(updates)
      rank = inputs.ndim
      if axis < 0:
        axis += rank
      if not indices.shape:
        indices = indices.reshape((1,))

      all_axes = range(inputs.ndim)
      update_axes = batch_dims + (axis,)
      non_update_axes = tuple(set(all_axes) - set(update_axes))
      tile_axes = []
      for ax in all_axes:
        if ax in update_axes:
          tile_axes.append(1)
        else:
          tile_axes.append(inputs.shape[ax])

      indices = np.expand_dims(indices, axis=sorted(non_update_axes))
      indices = np.tile(indices, tile_axes)
      assert indices.shape == updates.shape
      np.put_along_axis(inputs, indices, updates, axis=axis)
      return inputs

    inputs = jax.random.uniform(jax.random.key(0), input_shape)
    updates = jax.random.uniform(jax.random.key(1), update_shape)
    outputs = _run_forward(inputs, updates)
    outputs_np = _np_scatter_along_axis(
        inputs, updates, indices, axis, batch_dims)

    np.testing.assert_array_equal(outputs, outputs_np)
    chex.assert_shape(outputs, outputs_np.shape)

  @parameterized.named_parameters(
      ('invalid_axis', (4, 5), (4, 3), range(3), 2, (), ValueError),
      ('invalid_rank', (), (), 0, 0, (), ValueError),
      ('mismatched_updates', (4, 5), (4, 4), np.arange(3), 1, (), TypeError),
      ('repeated_indices', (4, 5), (2, 5), np.ones([2, 2], np.int32),
       0, (), ValueError),
      ('out_of_bound_axis', (4, 5), (4, 3), range(3), 2, (), ValueError),
      ('invalid_negative_axis', (4, 5), (4, 3), range(3), -3, (), ValueError),
      ('2d_indices_w/o_batch_dims', (2, 5), (2, 2),
       np.array([[0, 1], [1, 2]]), 1, (), ValueError),
      ('3d_indices_truncated_batch_dims', (2, 2, 5), (2, 2, 2),
       np.array([[[0, 1], [1, 2]], [[3, 4], [1, 3]]]), 2, (0,), ValueError),
  )
  def test_invalid_scatter_along_axis(self,
                                      input_shape,
                                      update_shape,
                                      indices,
                                      axis,
                                      batch_dims,
                                      error_type):

    @jax.jit
    def _run_forward(inputs, updates):
      outputs = utils.scatter_along_axis(
          inputs=inputs,
          updates=updates,
          indices=indices,
          axis=axis,
          batch_dims=batch_dims)
      return outputs

    inputs = jax.random.uniform(jax.random.key(0), input_shape)
    updates = jax.random.uniform(jax.random.key(1), update_shape)
    with self.assertRaises(error_type):
      _run_forward(inputs, updates)

  def test_create_attention_mask(self):
    @functools.partial(jax.jit, static_argnums=(2, 3))
    def _run_forward(query_token_mask, key_token_mask, elementwise_fn, dtype):
      outputs = utils.create_attention_mask(
          query_token_mask, key_token_mask, elementwise_fn, dtype)
      return outputs

    batch_size = 2
    num_instance = 2
    q_length = 3
    k_length = 4
    query_token_mask = jnp.ones(
        (batch_size, num_instance, q_length), dtype=jnp.int32)
    key_token_mask = jnp.ones(
        (batch_size, num_instance, k_length), dtype=jnp.int32)
    query_token_mask = query_token_mask.at[:, :, -1:].set(0)
    key_token_mask = key_token_mask.at[:, :, -2:].set(0)
    expected_attention_mask = jnp.array(
        [[[[[1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0]]],
          [[[1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0]]]],
         [[[[1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0]]],
          [[[1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0]]]]], dtype=jnp.int32)
    attention_mask = _run_forward(
        query_token_mask, key_token_mask, jnp.multiply, jnp.int32)
    np.testing.assert_array_equal(attention_mask, expected_attention_mask)

  def test_scatter_nd_simple(self):
    indices = jnp.array([[0, 1]])
    updates = jnp.array([[1, -2, 3]], dtype=jnp.float32)

    actual_result = utils.scatter_nd(indices, updates, shape=(1, 2, 3))
    expected_result = jnp.array([[[0, 0, 0], [1, -2, 3]]], dtype=jnp.float32)
    np.testing.assert_allclose(actual_result, expected_result)

  def test_scatter_nd_3d_update(self):
    indices = jnp.array([[[0, 1], [1, 0], [1, 1]]])
    updates = jnp.array([[[1, -1], [2, -2], [3, -3]]], dtype=jnp.int32)

    actual_result = utils.scatter_nd(indices, updates, shape=(2, 2, 2))
    expected_result = jnp.array([[[0, 0], [1, -1]], [[2, -2], [3, -3]]],
                                dtype=jnp.int32)
    np.testing.assert_allclose(actual_result, expected_result)

  def test_scatter_nd_ignore_outside_indices(self):
    indices = jnp.array([[0, 0], [1, 2], [2, 0]])
    updates = jnp.array([1., 2., 3.])

    actual_result = utils.scatter_nd(indices, updates, shape=(3, 2))
    expected_result = jnp.array([[1., 0.], [0., 0], [3., 0.]])
    np.testing.assert_allclose(actual_result, expected_result)

  def test_scatter_nd_cumulative_updates(self):
    indices = jnp.array([[1, 1], [1, 1], [1, 1]])
    updates = jnp.array([1., 2., 3.])

    actual_result = utils.scatter_nd(indices, updates, shape=(3, 2))
    expected_result = jnp.array([[0., 0.], [0., 6.], [0., 0.]])
    np.testing.assert_allclose(actual_result, expected_result)

  @parameterized.named_parameters(
      ('base_vector', (2, 2, 4), (2, 1, 4), np.array([0, 2]),
       np.array([1]), 1, 3, (), 'float32'),
      ('base_scalar', (2, 2), (2, 1), np.array([0, 2]),
       np.array([1]), 1, 3, (), 'float32'),
      ('batched', (2, 2), (2, 1), np.array([[1, 2], [0, 2]]),
       np.array([[0], [1]]), 1, 3, (0,), 'float32'),
  )
  def test_fill_by_scatter(self,
                           input_shape,
                           update_shape,
                           keep_indices,
                           fill_indices,
                           axis,
                           length,
                           batch_dims,
                           precision):

    @jax.jit
    def _run_forward(inputs, updates):
      outputs = utils.fill_by_scatter(
          inputs=inputs,
          updates=updates,
          keep_indices=keep_indices,
          fill_indices=fill_indices,
          axis=axis,
          length=length,
          keep_batch_dims=batch_dims,
          fill_batch_dims=batch_dims,
          precision=precision)
      return outputs

    inputs = jax.random.uniform(jax.random.key(0), input_shape)
    updates = jax.random.uniform(jax.random.key(1), update_shape)

    outputs = _run_forward(inputs, updates)
    if batch_dims:
      take_fn = jnp.take_along_axis
    else:
      take_fn = jnp.take
    retrieved_inputs = take_fn(outputs, keep_indices, axis)
    retrieved_updates = take_fn(outputs, fill_indices, axis)

    np.testing.assert_array_equal(inputs, retrieved_inputs)
    np.testing.assert_array_equal(updates, retrieved_updates)

  def test_create_causal_attention_mask(self):
    @functools.partial(jax.jit, static_argnums=(1,))
    def _run_forward(token_mask, dtype):
      outputs = utils.create_causal_attention_mask(token_mask, dtype)
      return outputs

    batch_size = 2
    num_instance = 2
    length = 4
    token_mask = jnp.ones((batch_size, num_instance, length), dtype=jnp.int32)
    token_mask = token_mask.at[:, :, -2:].set(0)
    expected_attention_mask = jnp.array(
        [[[[[1, 0, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]]],
          [[[1, 0, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]]]],
         [[[[1, 0, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]]],
          [[[1, 0, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]]]]], dtype=jnp.int32)
    attention_mask = _run_forward(token_mask, jnp.int32)
    np.testing.assert_array_equal(attention_mask, expected_attention_mask)

  def test_create_all_valid_causal_attention_mask(self):
    @functools.partial(jax.jit, static_argnums=(0, 1))
    def _run_forward(length, dtype):
      outputs = utils.create_all_valid_causal_attention_mask(length, dtype)
      return outputs

    attention_mask = _run_forward(length=4, dtype=jnp.int32)
    expected_attention_mask = jnp.array(
        [[[[[1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1]]]]],
        dtype=jnp.int32,
    )
    np.testing.assert_array_equal(attention_mask, expected_attention_mask)

  def test_create_groupwise_causal_mask(self):
    @functools.partial(jax.jit, static_argnums=(0, 1, 2))
    def _run_forward(length, group_size, dtype):
      outputs = utils.create_groupwise_causal_mask(length, group_size, dtype)
      return outputs

    attention_mask = _run_forward(length=6, group_size=2, dtype=jnp.int32)
    expected_attention_mask = jnp.array(
        [[[[
            [1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ]]]],
        dtype=jnp.int32,
    )
    np.testing.assert_array_equal(attention_mask, expected_attention_mask)

  def test_extend_token_mask(self):
    @functools.partial(jax.jit, static_argnums=(1,))
    def _run_forward(token_mask, extension):
      outputs = utils.extend_token_mask(token_mask, extension)
      return outputs

    batch_size = 2
    num_instance = 3
    length = 4
    token_mask = jnp.ones((batch_size, num_instance, length), dtype=jnp.int32)
    token_mask = token_mask.at[:, :, -2:].set(0)
    expected_extended_token_mask_prepend = jnp.array(
        [[[1, 1, 1, 0, 0],
          [1, 1, 1, 0, 0],
          [1, 1, 1, 0, 0]],
         [[1, 1, 1, 0, 0],
          [1, 1, 1, 0, 0],
          [1, 1, 1, 0, 0]]], dtype=jnp.int32)
    expected_extended_token_mask_append = jnp.array(
        [[[1, 1, 0, 0, 1],
          [1, 1, 0, 0, 1],
          [1, 1, 0, 0, 1]],
         [[1, 1, 0, 0, 1],
          [1, 1, 0, 0, 1],
          [1, 1, 0, 0, 1]]], dtype=jnp.int32)
    extended_mask_prepend = _run_forward(token_mask,
                                         constants.Extension.PREPEND)
    extended_mask_append = _run_forward(token_mask,
                                        constants.Extension.APPEND)
    np.testing.assert_array_equal(extended_mask_prepend,
                                  expected_extended_token_mask_prepend)
    np.testing.assert_array_equal(extended_mask_append,
                                  expected_extended_token_mask_append)

  def test_extend_attention_bias(self):
    @functools.partial(jax.jit, static_argnums=(1,))
    def _run_forward(attention_bias, extension):
      outputs = utils.extend_attention_bias(attention_bias, extension)
      return outputs

    num_heads = 3
    q_len = 2
    kv_len = 3

    attention_bias = jax.random.uniform(jax.random.key(0),
                                        (num_heads, q_len, kv_len))
    expected_extended_bias_prepend = jnp.array(
        [[[0., 0., 0., 0.],
          [0., 0.19, 0.45, 0.04],
          [0., 0.58, 0.09, 0.26]],
         [[0., 0., 0., 0.],
          [0., 0.83, 0.49, 0.03],
          [0., 0.10, 0.10, 0.41]],
         [[0., 0., 0., 0.],
          [0., 0.40, 0.62, 0.86],
          [0., 0.52, 0.34, 0.13]]], dtype=jnp.float32)
    expected_extended_bias_append = jnp.array(
        [[[0.19, 0.44, 0.04, 0.],
          [0.59, 0.09, 0.26, 0.],
          [0., 0., 0., 0.]],
         [[0.83, 0.49, 0.03, 0.],
          [0.10, 0.10, 0.41, 0.],
          [0., 0., 0., 0.]],
         [[0.40, 0.62, 0.86, 0.],
          [0.52, 0.34, 0.13, 0.],
          [0., 0., 0., 0.]]], dtype=jnp.float32)
    extended_bias_prepend = _run_forward(attention_bias,
                                         constants.Extension.PREPEND)
    extended_bias_append = _run_forward(attention_bias,
                                        constants.Extension.APPEND)
    np.testing.assert_array_almost_equal(
        extended_bias_prepend, expected_extended_bias_prepend, 1)
    np.testing.assert_array_almost_equal(
        extended_bias_append, expected_extended_bias_append, 1)

  def test_extend_token_pos_id(self):
    @functools.partial(jax.jit, static_argnums=(1, 2))
    def _run_forward(token_pos_id, extension, max_length):
      outputs = utils.extend_token_pos_id(token_pos_id, extension, max_length)
      return outputs

    batch_size = 2
    num_instance = 1
    sequence_length = 3
    max_length = 8

    token_pos_id = jnp.tile(jnp.arange(max_length),
                            (batch_size, num_instance, 1))
    token_pos_id = jax.random.permutation(jax.random.key(0),
                                          token_pos_id, axis=-1,
                                          independent=True)
    token_pos_id = token_pos_id[:, :, :sequence_length]

    expected_extended_token_pos_id_prepend = jnp.array(
        [[[0, 2, 7, 4]],
         [[0, 8, 4, 1]]], dtype=jnp.int32)
    expected_extended_token_pos_id_append = jnp.array(
        [[[1, 6, 3, 8]],
         [[7, 3, 0, 8]]], dtype=jnp.int32)

    extended_pos_id_prepend = _run_forward(
        token_pos_id, constants.Extension.PREPEND, None)
    extended_pos_id_append = _run_forward(
        token_pos_id, constants.Extension.APPEND, max_length)

    np.testing.assert_array_equal(extended_pos_id_prepend,
                                  expected_extended_token_pos_id_prepend)
    np.testing.assert_array_equal(extended_pos_id_append,
                                  expected_extended_token_pos_id_append)

    with self.assertRaises(ValueError):
      _run_forward(token_pos_id, constants.Extension.APPEND, None)

  def test_top_k(self):
    inputs = jnp.array([[5, 6, 4, 1, 2]], dtype=jnp.float32)
    expected = jnp.array([[1, 0, 2, 4, 3]], dtype=jnp.float32)

    for k in range(1, expected.shape[-1] + 1):
      outputs = utils.top_k(inputs, k)
      np.testing.assert_array_equal(outputs, expected[:, :k])

  def test_extract_patches(self):
    @functools.partial(jax.jit, static_argnums=(1, 2, 3))
    def _run_forward_3d(inputs, patch_sizes, flatten, original_shape=None):
      patches = utils.extract_volume_patches_from_raw_voxels(
          inputs=inputs, patch_sizes=patch_sizes, flatten=flatten)
      reconstructed = utils.extract_raw_voxels_from_volume_patches(
          patches=patches, patch_sizes=patch_sizes, flattened=flatten,
          expected_voxel_shape=original_shape)
      return patches, reconstructed

    @functools.partial(jax.jit, static_argnums=(1,))
    def _run_forward_1d(inputs, patch_size):
      patches = utils.extract_patches_from_raw_waveform(
          inputs=inputs, patch_size=patch_size)
      reconstructed = utils.extract_raw_waveform_from_patches(
          patches=patches, patch_size=patch_size)
      return patches, reconstructed

    def tf_extract_volume_patches(inputs, patch_sizes, flatten):
      ksize = (1,) + patch_sizes + (1,)
      patches = tf.map_fn(
          fn=lambda t: tf.extract_volume_patches(t, ksize, ksize, 'SAME'),
          elems=inputs)
      if flatten:
        patches = tf.reshape(
            patches,
            (patches.shape[0], patches.shape[1], -1, patches.shape[-1]))
      return patches

    def tf_extract_waveform_patches(inputs, patch_size):
      ksize = (1, 1, patch_size, 1)
      rates = (1, 1, 1, 1)
      return tf.image.extract_patches(inputs, ksize, ksize, rates, 'SAME')

    batch_size = 2
    num_instance = 3
    voxel_size = (8, 16, 14, 3)
    voxel_patch_size = (4, 8, 7)
    waveform_size = 16
    waveform_patch_size = 4
    inputs_voxel = jax.random.uniform(
        jax.random.key(0), (batch_size, num_instance) + voxel_size)
    inputs_waveform = jax.random.uniform(
        jax.random.key(0), (batch_size, num_instance, waveform_size, 2))

    voxel_patches_flattened, voxel_reconstructed_flattened = _run_forward_3d(
        inputs_voxel, voxel_patch_size, True, voxel_size)
    voxel_patches, voxel_reconstructed = _run_forward_3d(
        inputs_voxel, voxel_patch_size, False)
    waveform_patches, waveform_reconstructed = _run_forward_1d(
        inputs_waveform, waveform_patch_size)

    voxel_patches_flattened_tf = tf_extract_volume_patches(
        inputs_voxel, voxel_patch_size, True)
    voxel_patches_tf = tf_extract_volume_patches(
        inputs_voxel, voxel_patch_size, False)
    waveform_patches_tf = tf_extract_waveform_patches(
        inputs_waveform, waveform_patch_size)

    np.testing.assert_array_equal(inputs_voxel, voxel_reconstructed_flattened)
    np.testing.assert_array_equal(inputs_voxel, voxel_reconstructed)
    np.testing.assert_array_equal(inputs_waveform, waveform_reconstructed)
    np.testing.assert_array_equal(voxel_patches_flattened,
                                  voxel_patches_flattened_tf)
    np.testing.assert_array_equal(voxel_patches, voxel_patches_tf)
    np.testing.assert_array_equal(waveform_patches, waveform_patches_tf)

  @parameterized.named_parameters(
      {
          'testcase_name': 'baseline',
          'start_stop_step': (0, 4, 1),
      },
      {
          'testcase_name': 'non_0_start',
          'start_stop_step': (2, 7, 1),
      },
      {
          'testcase_name': 'non_1_step',
          'start_stop_step': (2, 7, 3),
      },
      {
          'testcase_name': 'reverse',
          'start_stop_step': (7, 2, -2),
      },
      {
          'testcase_name': 'empty',
          'start_stop_step': (4, 4, 1),
      },
  )
  def test_range_by_iota(self, start_stop_step):
    start, stop, step = start_stop_step
    iota_range = jax.jit(utils.range_by_iota,
                         static_argnums=(0, 1, 2))(start, stop, step)
    np_range = np.arange(start, stop, step)
    np.testing.assert_array_equal(iota_range, np_range)

  @parameterized.named_parameters(
      {
          'testcase_name': 'no_dilation',
          'kernel_size': (5, 4, 7),
          'kernel_dilation': (1, 1, 1),
          'expected_kernel_size_dilated': (5, 4, 7),
          'expected_pre_conv_pads': ((0, 0), (2, 2), (1, 2), (3, 3), (0, 0)),
      },
      {
          'testcase_name': 'with_dilation',
          'kernel_size': (5, 4, 7),
          'kernel_dilation': (2, 3, 4),
          'expected_kernel_size_dilated': (9, 10, 25),
          'expected_pre_conv_pads': ((0, 0), (4, 4), (4, 5), (12, 12), (0, 0)),
      },
  )
  def test_conv_kernel_sizes(self,
                             kernel_size,
                             kernel_dilation,
                             expected_kernel_size_dilated,
                             expected_pre_conv_pads):
    kernel_size_dilated = utils.get_kernel_size_dilated(kernel_size,
                                                        kernel_dilation)
    pads = utils.get_pre_conv_pads(kernel_size_dilated)
    chex.assert_equal(kernel_size_dilated, expected_kernel_size_dilated)
    chex.assert_equal(pads, expected_pre_conv_pads)



class GeneralUtilsTest(absltest.TestCase):

  def test_flatten_dict(self):
    dictionary = {
        'a': 1,
        'b': (
            {'c': 2},
            {'c': 3}
        ),
        'd': (4, 5)
    }
    flattened_dictionary = utils.flatten_dict(dictionary)
    expected = {
        ('a',): 1,
        ('b', '0', 'c'): 2,
        ('b', '1', 'c'): 3,
        ('d', '0'): 4,
        ('d', '1'): 5
    }
    self.assertEqual(flattened_dictionary, expected)

  def test_flatten_dict_with_sep(self):
    dictionary = {
        'a': 1,
        'b': (
            {'c': 2},
            {'c': 3}
        ),
        'd': (4, 5)
    }
    flattened_dictionary = utils.flatten_dict(dictionary, sep='/')
    expected = {
        'a': 1,
        'b/0/c': 2,
        'b/1/c': 3,
        'd/0': 4,
        'd/1': 5
    }
    self.assertEqual(flattened_dictionary, expected)

if __name__ == '__main__':
  absltest.main()
