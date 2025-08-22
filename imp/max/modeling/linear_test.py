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

"""Tests for linear projection layers."""
import functools

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
from jax.nn import initializers
import jax.numpy as jnp
import numpy as np

from imp.max.core import constants
from imp.max.core import utils
from imp.max.modeling import linear
from imp.max.utils import sharding


_TOKENS_SHARDINGS = ('data', None, None, None)  # (b, n, t, d)


def _create_global_mesh():
  return jax.sharding.Mesh(
      sharding.create_tpu_device_mesh(ici_mesh_shape=(1, 1),
                                      dcn_mesh_shape=(1, 1)),
      ['data', 'model'],
  )


class LinearTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'baseline',
          'batch_size': 1,
          'n_instances': 2,
          'length': 3,
          'dim': 4,
          'features': 5,
          'use_bias': True,
      }, {
          'testcase_name': 'lora',
          'batch_size': 1,
          'n_instances': 2,
          'length': 3,
          'dim': 4,
          'features': 5,
          'use_bias': True,
          'lora_rank': 2,
          'lora_scale': 1.0,
      })
  def test_dense(self,
                 batch_size,
                 n_instances,
                 length,
                 dim,
                 features,
                 use_bias,
                 lora_rank=2,
                 lora_scale=0.):

    kernel_shardings = (None, 'model')
    dense = linear.Dense(
        features=features,
        use_bias=use_bias,
        lora_rank=lora_rank,
        lora_scale=lora_scale,
        kernel_shardings=kernel_shardings,
    )

    @jax.jit
    def _run_forward(inputs):
      variables = dense.init(
          rngs={'params': jax.random.key(1)},
          inputs=inputs)
      outputs = dense.apply(
          variables=variables,
          inputs=inputs)
      return outputs, variables

    with _create_global_mesh():
      inputs = jnp.ones((batch_size, n_instances, length, dim))
      inputs = sharding.shard_array(inputs, _TOKENS_SHARDINGS)
      outputs, variables = _run_forward(inputs)

    # Check kernel(s) shape
    kernel = variables[constants.FlaxCollection.PARAMS]['kernel']
    bias = variables[constants.FlaxCollection.PARAMS]['bias']
    expected_kernel_shape = (dim, features)
    expected_bias_shape = (features,)
    chex.assert_shape(kernel.value, expected_kernel_shape)
    chex.assert_shape(bias.value, expected_bias_shape)

    # Assert shardings are propagated properly
    self.assertEqual(kernel.names, kernel_shardings)
    self.assertEqual(bias.names, kernel_shardings[-1:])

    if lora_scale > 0.:
      kernel_left = variables[constants.FlaxCollection.PARAMS]['kernel_left']
      kernel_right = variables[constants.FlaxCollection.PARAMS]['kernel_right']
      expected_kernel_left_shape = (dim, lora_rank)
      expected_kernel_right_shape = (lora_rank, features)
      chex.assert_shape(kernel_left.value, expected_kernel_left_shape)
      chex.assert_shape(kernel_right.value, expected_kernel_right_shape)

      # Assert shardings are propagated properly
      self.assertEqual(kernel_left.names, kernel_shardings[:-1] + (None,))
      self.assertEqual(kernel_right.names, (None,) + kernel_shardings[1:])

    # Check output shape
    expected_outputs_shape = (batch_size, n_instances, length, features)
    chex.assert_shape(outputs, expected_outputs_shape)

  @parameterized.named_parameters(
      {
          'testcase_name': 'baseline',
          'batch_size': 1,
          'n_instances': 2,
          'length': 3,
          'dim': 4,
          'features': 5,
          'axis': -1,
      }, {
          'testcase_name': 'multi_features',
          'batch_size': 1,
          'n_instances': 2,
          'length': 3,
          'dim': 4,
          'features': (2, 5),
          'axis': -1,
      }, {
          'testcase_name': 'multi_axis',
          'batch_size': 1,
          'n_instances': 2,
          'length': 3,
          'dim': 4,
          'features': 5,
          'axis': (-3, -1),
      }, {
          'testcase_name': 'multi_axis_multi_features',
          'batch_size': 1,
          'n_instances': 2,
          'length': 3,
          'dim': 4,
          'features': (2, 5),
          'axis': (-3, -1),
      }, {
          'testcase_name': 'baseline_lora',
          'batch_size': 1,
          'n_instances': 2,
          'length': 3,
          'dim': 4,
          'features': 5,
          'axis': -1,
          'lora_rank': 2,
          'lora_scale': 1.0,
      }, {
          'testcase_name': 'multi_features_lora',
          'batch_size': 1,
          'n_instances': 2,
          'length': 3,
          'dim': 4,
          'features': (2, 5),
          'axis': -1,
          'lora_rank': 2,
          'lora_scale': 1.0,
      }, {
          'testcase_name': 'multi_axis_lora',
          'batch_size': 1,
          'n_instances': 2,
          'length': 3,
          'dim': 4,
          'features': 5,
          'axis': (-3, -1),
          'lora_rank': 2,
          'lora_scale': 1.0,
      }, {
          'testcase_name': 'multi_axis_multi_features_lora',
          'batch_size': 1,
          'n_instances': 2,
          'length': 3,
          'dim': 4,
          'features': (2, 5),
          'axis': (-3, -1),
          'lora_rank': 2,
          'lora_scale': 1.0,
      })
  def test_dense_general(self,
                         batch_size,
                         n_instances,
                         length,
                         dim,
                         features,
                         axis,
                         lora_rank=2,
                         lora_scale=0.):
    # Construct inputs
    inputs = jnp.ones((batch_size, n_instances, length, dim))

    # Canonalize/normalize args
    features = linear._canonicalize_tuple(features)
    axis = linear._canonicalize_tuple(axis)
    axis = linear._normalize_axes(axis, inputs.ndim)

    kernel_shardings = tuple(
        (len(axis) + len(features) - 1) * [None] + ['model'])
    dense_general = linear.DenseGeneral(
        features=features,
        axis=axis,
        use_bias=True,
        lora_rank=lora_rank,
        lora_scale=lora_scale,
        kernel_shardings=kernel_shardings,
    )

    @jax.jit
    def _run_forward(inputs):
      variables = dense_general.init(
          rngs={'params': jax.random.key(1)},
          inputs=inputs)
      outputs = dense_general.apply(
          variables=variables,
          inputs=inputs)
      return outputs, variables

    # Initialize and perform forward call
    with _create_global_mesh():
      inputs = sharding.shard_array(inputs, _TOKENS_SHARDINGS)
      outputs, variables = _run_forward(inputs)

    # Check kernel(s) shape
    kernel = variables[constants.FlaxCollection.PARAMS]['kernel']
    bias = variables[constants.FlaxCollection.PARAMS]['bias']
    expected_kernel_shape = [inputs.shape[ax] for ax in axis] + list(features)
    expected_bias_shape = features
    chex.assert_shape(kernel.value, expected_kernel_shape)
    chex.assert_shape(bias.value, expected_bias_shape)

    # Assert shardings are propagated properly
    self.assertEqual(kernel.names, kernel_shardings)
    self.assertEqual(bias.names, kernel_shardings[-len(features):])

    if lora_scale > 0.:
      kernel_left = variables[constants.FlaxCollection.PARAMS]['kernel_left']
      kernel_right = variables[constants.FlaxCollection.PARAMS]['kernel_right']
      expected_kernel_left_shape = list(
          expected_kernel_shape[:-len(features)]) + [lora_rank]
      expected_kernel_right_shape = [lora_rank] + list(features)
      chex.assert_shape(kernel_left.value, expected_kernel_left_shape)
      chex.assert_shape(kernel_right.value, expected_kernel_right_shape)

      # Assert shardings are propagated properly
      expected_kernel_left_shardings = (
          kernel_shardings[:len(expected_kernel_left_shape)-1] + (None,))
      expected_kernel_right_shardings = (
          (None,) + kernel_shardings[-len(expected_kernel_right_shape)+1:])
      self.assertEqual(kernel_left.names, expected_kernel_left_shardings)
      self.assertEqual(kernel_right.names, expected_kernel_right_shardings)

    # Check output shape
    expected_output_shape = [
        inputs.shape[ax] for ax in range(inputs.ndim) if ax not in axis
    ] + list(features)
    chex.assert_shape(outputs, expected_output_shape)

  @parameterized.product(use_bias=(True, False))
  def test_conv(self, use_bias):
    rng = dict(params=jax.random.key(0))
    x = jnp.ones((1, 8, 3))
    conv_module = linear.Conv(
        features=4,
        use_bias=use_bias,
        kernel_size=(3,),
        padding='VALID',
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
    )
    y, variables = conv_module.init_with_output(rng, x)
    self.assertEqual(variables['params']['kernel'].value.shape, (3, 3, 4))
    expected = 10.0 if use_bias else 9.0
    np.testing.assert_allclose(y, np.full((1, 6, 4), expected))

  @parameterized.product(use_bias=(True, False))
  def test_multibatch_input_conv(self, use_bias):
    rng = dict(params=jax.random.key(0))
    x = jnp.ones((2, 5, 8, 3))
    conv_module = linear.Conv(
        features=4,
        use_bias=use_bias,
        kernel_size=(3,),
        padding='VALID',
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
    )
    y, variables = conv_module.init_with_output(rng, x)
    self.assertEqual(variables['params']['kernel'].value.shape, (3, 3, 4))
    expected = 10.0 if use_bias else 9.0
    np.testing.assert_allclose(y, np.full((2, 5, 6, 4), expected))

  def test_conv_local(self):
    rng = dict(params=jax.random.key(0))
    x = jnp.ones((1, 8, 2))
    conv_module = linear.ConvLocal(
        features=4,
        kernel_size=(3,),
        padding='VALID',
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
    )
    y, variables = conv_module.init_with_output(rng, x)
    self.assertEqual(variables['params']['kernel'].value.shape, (6, 3 * 2, 4))
    np.testing.assert_allclose(y, np.full((1, 6, 4), 7.0))

  def test_single_input_conv(self):
    rng = dict(params=jax.random.key(0))
    x = jnp.ones((8, 3))
    conv_module = linear.Conv(
        features=4,
        kernel_size=(3,),
        padding='VALID',
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
    )
    y, variables = conv_module.init_with_output(rng, x)
    self.assertEqual(variables['params']['kernel'].value.shape, (3, 3, 4))
    np.testing.assert_allclose(y, np.full((6, 4), 10.0))

  def test_single_input_masked_conv(self):
    rng = dict(params=jax.random.key(0))
    x = jnp.ones((8, 3))
    m = jnp.tril(jnp.ones((3, 3, 4)))
    conv_module = linear.Conv(
        features=4,
        kernel_size=(3,),
        padding='VALID',
        mask=m,
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
    )
    expected = jnp.array(
        [
            [10.0, 7.0, 4.0, 1.0],
            [10.0, 7.0, 4.0, 1.0],
            [10.0, 7.0, 4.0, 1.0],
            [10.0, 7.0, 4.0, 1.0],
            [10.0, 7.0, 4.0, 1.0],
            [10.0, 7.0, 4.0, 1.0],
        ]
    )
    y, variables = conv_module.init_with_output(rng, x)
    self.assertEqual(variables['params']['kernel'].value.shape, (3, 3, 4))
    np.testing.assert_allclose(y, expected)

  def test_single_input_conv_local(self):
    rng = dict(params=jax.random.key(0))
    x = jnp.ones((8, 2))
    conv_module = linear.ConvLocal(
        features=4,
        kernel_size=(3,),
        padding='VALID',
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
    )
    y, variables = conv_module.init_with_output(rng, x)
    self.assertEqual(variables['params']['kernel'].value.shape, (6, 3 * 2, 4))
    np.testing.assert_allclose(y, np.full((6, 4), 7.0))

  def test_group_conv(self):
    rng = dict(params=jax.random.key(0))
    x = jnp.ones((1, 8, 4))
    conv_module = linear.Conv(
        features=4,
        kernel_size=(3,),
        feature_group_count=2,
        padding='VALID',
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
    )
    y, variables = conv_module.init_with_output(rng, x)
    self.assertEqual(variables['params']['kernel'].value.shape, (3, 2, 4))
    np.testing.assert_allclose(y, np.full((1, 6, 4), 7.0))

  @parameterized.product(
      n_batch=(1, 3),
      n_features=(1, 2),
      kernel_size=(1, 2, 3, 9),
      n_input_features=(1, 3),
      input_size=(1, 8, 16),
      module=(linear.Conv, linear.ConvLocal),
  )
  def test_circular_conv_1d_constant(
      self,
      n_batch,
      n_features,
      kernel_size,
      n_input_features,
      input_size,
      module,
  ):  # pylint: disable=g-doc-args
    """Test 1D convolution with circular padding.

    Filter with all elements equal to 1 applied on an input with all elements
    equal to 1. Result should have the same shape as input (except for the
    feature dimension) and have all elements equal to
    `n_input_features * kernel_lin_size`.
    """
    rng = dict(params=jax.random.key(0))
    x = jnp.ones((n_batch, input_size, n_input_features))
    conv_module = module(
        features=n_features,
        kernel_size=(kernel_size,),
        padding=utils.circular_pre_conv_padding,
        kernel_init=initializers.ones,
        bias_init=initializers.zeros,
    )
    y, variables = conv_module.init_with_output(rng, x)

    kernel_shape = self._get_kernel_shape(
        x.shape, (kernel_size,), module, n_features
    )

    self.assertEqual(
        variables['params']['kernel'].value.shape,
        kernel_shape,
    )
    correct_ans = np.full(
        (n_batch, input_size, n_features), kernel_size * n_input_features
    )
    np.testing.assert_allclose(y, correct_ans)

  def _get_kernel_shape(self, input_shape, kernel_size, module, n_features):
    if module == linear.Conv:
      kernel_shape = kernel_size + (input_shape[-1], n_features)
    elif module == linear.ConvLocal:
      kernel_shape = input_shape[1:-1] + (
          input_shape[-1] * np.prod(kernel_size),
          n_features,
      )
    else:
      raise ValueError(module)
    return kernel_shape

  @parameterized.product(
      n_batch=(1, 3),
      n_features=(1, 2, 10),
      kernel_lin_size=(1, 2, 3, 9),
      n_input_features=(1, 5),
      input_x_size=(14,),
      input_y_size=(5, 10),
      module=(linear.Conv, linear.ConvLocal),
  )
  def test_circular_conv_2d_constant(
      self,
      n_batch,
      n_features,
      kernel_lin_size,
      n_input_features,
      input_x_size,
      input_y_size,
      module,
  ):  # pylint: disable=g-doc-args
    """Test 2D convolution with circular padding.

    Square filter with all elements equal to 1 applied on an input with all
    elements equal to 1. Result should have the same shape as input (except for
    the feature dimension) and have all elements equal to
    `n_input_features * kernel_lin_size^2`.
    """
    rng = dict(params=jax.random.key(0))
    x = jnp.ones((n_batch, input_x_size, input_y_size, n_input_features))
    kernel_size = (kernel_lin_size, kernel_lin_size)
    conv_module = module(
        features=n_features,
        kernel_size=kernel_size,
        padding=utils.circular_pre_conv_padding,
        kernel_init=initializers.ones,
        bias_init=initializers.zeros,
    )
    y, variables = conv_module.init_with_output(rng, x)

    kernel_shape = self._get_kernel_shape(
        x.shape, kernel_size, module, n_features
    )

    self.assertEqual(
        variables['params']['kernel'].value.shape,
        kernel_shape,
    )
    correct_ans = np.full(
        (n_batch, input_x_size, input_y_size, n_features),
        kernel_lin_size * kernel_lin_size * n_input_features,
    )
    np.testing.assert_allclose(y, correct_ans)

  def test_circular_conv_1d_custom(self):
    """Test 1d convolution with circular padding and a stride."""
    rng = dict(params=jax.random.key(0))
    x = np.arange(1, 6)
    x = np.expand_dims(x, (0, 2))
    kernel = np.array((1, 2, 1))
    kernel = np.expand_dims(kernel, (1, 2))

    conv_module = linear.Conv(
        features=1,
        kernel_size=(3,),
        strides=(3,),
        padding=utils.circular_pre_conv_padding,
        kernel_init=lambda key, shape, dtype: kernel,
        bias_init=initializers.zeros,
    )
    y, variables = conv_module.init_with_output(rng, x)

    self.assertEqual(variables['params']['kernel'].value.shape, (3, 1, 1))
    # Compare with manually computed convolution
    correct_ans = np.array((5 + 2 * 1 + 2, 3 + 2 * 4 + 5))
    correct_ans = np.expand_dims(correct_ans, (0, 2))
    np.testing.assert_allclose(y, correct_ans)

  def test_circular_conv_local_1d_custom(self):
    """Test 1d local convolution with circular padding and a stride."""
    rng = dict(params=jax.random.key(0))
    x = np.arange(1, 6)
    x = np.expand_dims(x, (0, 2))
    kernel = np.array(((-1, 2, 3), (4, 5, 6)))
    kernel = np.expand_dims(kernel, (2,))
    conv_module = linear.ConvLocal(
        features=1,
        kernel_size=(3,),
        strides=(3,),
        padding=utils.circular_pre_conv_padding,
        kernel_init=lambda key, shape, dtype: kernel,
        bias_init=initializers.zeros,
    )
    y, variables = conv_module.init_with_output(rng, x)

    self.assertEqual(variables['params']['kernel'].value.shape, (2, 3, 1))
    # Compare with manually computed convolution
    correct_ans = np.array((-1 * 5 + 2 * 1 + 3 * 2, 4 * 3 + 5 * 4 + 6 * 5))
    correct_ans = np.expand_dims(correct_ans, (0, 2))
    np.testing.assert_allclose(y, correct_ans)

  def test_circular_conv_1d_dilation(self):
    """Test 1d convolution with circular padding and kernel dilation."""
    rng = dict(params=jax.random.key(0))
    x = np.arange(1, 6)
    x = np.expand_dims(x, (0, 2))
    kernel = np.array((1, 2, 1))
    kernel = np.expand_dims(kernel, (1, 2))

    conv_module = linear.Conv(
        features=1,
        kernel_size=(3,),
        padding=utils.circular_pre_conv_padding,
        kernel_init=lambda key, shape, dtype: kernel,
        bias_init=initializers.zeros,
        kernel_dilation=(3,),
    )
    y, variables = conv_module.init_with_output(rng, x)

    self.assertEqual(variables['params']['kernel'].value.shape, (3, 1, 1))
    # Compare with manually computed convolution
    correct_ans = np.array(
        (
            3 + 2 * 1 + 4,
            4 + 2 * 2 + 5,
            5 + 2 * 3 + 1,
            1 + 2 * 4 + 2,
            2 + 2 * 5 + 3,
        )
    )
    correct_ans = np.expand_dims(correct_ans, (0, 2))
    np.testing.assert_allclose(y, correct_ans)

  def test_circular_conv_local_1d_dilation(self):
    """Test 1d local convolution with circular padding and kernel dilation."""
    rng = dict(params=jax.random.key(0))
    x = np.arange(1, 6)
    x = np.expand_dims(x, (0, 2))
    kernel = np.array(
        ((1, 2, 1), (3, 4, 5), (-1, 1, 2), (2, 3, 4), (-1, -2, -3))
    )
    kernel = np.expand_dims(kernel, (2,))

    conv_module = linear.ConvLocal(
        features=1,
        kernel_size=(3,),
        padding=utils.circular_pre_conv_padding,
        kernel_init=lambda key, shape, dtype: kernel,
        bias_init=initializers.zeros,
        kernel_dilation=(3,),
    )
    y, variables = conv_module.init_with_output(rng, x)

    self.assertEqual(variables['params']['kernel'].value.shape, (5, 3, 1))
    # Compare with manually computed convolution
    correct_ans = np.array(
        (
            1 * 3 + 2 * 1 + 1 * 4,
            3 * 4 + 4 * 2 + 5 * 5,
            -1 * 5 + 1 * 3 + 2 * 1,
            2 * 1 + 3 * 4 + 4 * 2,
            -1 * 2 + -2 * 5 + -3 * 3,
        )
    )
    correct_ans = np.expand_dims(correct_ans, (0, 2))
    np.testing.assert_allclose(y, correct_ans)

  def test_circular_conv_2d_custom(self):
    """Test 2d convolution with circular padding on a 3x3 example."""
    rng = dict(params=jax.random.key(0))
    x = np.array(((1, 2, 3), (4, 5, 6), (7, 8, 9)))
    x = np.expand_dims(x, (0, 3))
    kernel = np.array(((0, 1, 0), (1, 2, 1), (0, 1, 0)))
    kernel = np.expand_dims(kernel, (2, 3))

    conv_module = linear.Conv(
        features=1,
        kernel_size=(3, 3),
        padding=utils.circular_pre_conv_padding,
        kernel_init=lambda key, shape, dtype: kernel,
        bias_init=initializers.zeros,
    )
    y, variables = conv_module.init_with_output(rng, x)

    self.assertEqual(variables['params']['kernel'].value.shape, (3, 3, 1, 1))
    # Compare with manually computed convolution
    correct_ans = np.array((
        (2 * 1 + 7 + 2 + 4 + 3, 2 * 2 + 8 + 3 + 5 + 1, 2 * 3 + 9 + 1 + 6 + 2),
        (2 * 4 + 1 + 5 + 7 + 6, 2 * 5 + 2 + 6 + 8 + 4, 2 * 6 + 3 + 4 + 9 + 5),
        (2 * 7 + 4 + 8 + 1 + 9, 2 * 8 + 5 + 9 + 2 + 7, 2 * 9 + 6 + 7 + 3 + 8),
    ))
    correct_ans = np.expand_dims(correct_ans, (0, 3))
    np.testing.assert_allclose(y, correct_ans)

  def test_circular_conv_local_2d_custom(self):
    """Test 2d local convolution with circular padding on a 3x3 example."""
    rng = dict(params=jax.random.key(0))
    x = np.array(((1, 2, 3), (4, 5, 6), (7, 8, 9)))
    x = np.expand_dims(x, (0, 3))
    kernel = np.array((
        (
            ((0, 1, 0), (1, 2, 1), (0, 1, 0)),
            ((0, 1, 0), (1, 3, 1), (0, 1, 0)),
            ((0, 1, 0), (1, 4, 1), (0, 1, 0)),
        ),
        (
            ((0, 1, 0), (1, 5, 1), (0, 1, 0)),
            ((0, 1, 0), (1, 6, 1), (0, 1, 0)),
            ((0, 1, 0), (1, 7, 1), (0, 1, 0)),
        ),
        (
            ((0, 1, 0), (1, 8, 1), (0, 1, 0)),
            ((0, 1, 0), (1, 9, 1), (0, 1, 0)),
            ((0, 1, 0), (1, 10, 1), (0, 1, 0)),
        ),
    ))
    kernel = np.expand_dims(kernel, (3,))
    kernel = np.reshape(kernel, (3, 3, 9, 1))

    conv_module = linear.ConvLocal(
        features=1,
        kernel_size=(3, 3),
        padding=utils.circular_pre_conv_padding,
        kernel_init=lambda key, shape, dtype: kernel,
        bias_init=initializers.zeros,
    )
    y, variables = conv_module.init_with_output(rng, x)

    self.assertEqual(variables['params']['kernel'].value.shape, (3, 3, 9, 1))
    # Compare with manually computed convolution
    correct_ans = np.array((
        (2 * 1 + 7 + 2 + 4 + 3, 3 * 2 + 8 + 3 + 5 + 1, 4 * 3 + 9 + 1 + 6 + 2),
        (5 * 4 + 1 + 5 + 7 + 6, 6 * 5 + 2 + 6 + 8 + 4, 7 * 6 + 3 + 4 + 9 + 5),
        (8 * 7 + 4 + 8 + 1 + 9, 9 * 8 + 5 + 9 + 2 + 7, 10 * 9 + 6 + 7 + 3 + 8),
    ))
    correct_ans = np.expand_dims(correct_ans, (0, 3))
    np.testing.assert_allclose(y, correct_ans)

  def test_causal_conv1d(self):
    rng = dict(params=jax.random.key(0))
    x = jnp.ones((1, 8, 4))
    conv_module = linear.Conv(
        features=4,
        kernel_size=(3,),
        padding=utils.causal_1d_pre_conv_padding,
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
    )
    y, _ = conv_module.init_with_output(rng, x)
    correct_ans = np.array(
        [
            [
                [5.0, 5.0, 5.0, 5.0],
                [9.0, 9.0, 9.0, 9.0],
                [13.0, 13.0, 13.0, 13.0],
                [13.0, 13.0, 13.0, 13.0],
                [13.0, 13.0, 13.0, 13.0],
                [13.0, 13.0, 13.0, 13.0],
                [13.0, 13.0, 13.0, 13.0],
                [13.0, 13.0, 13.0, 13.0],
            ]
        ]
    )
    np.testing.assert_allclose(y, correct_ans)
    np.testing.assert_array_equal(correct_ans.shape, y.shape)

  @parameterized.product(
      use_bias=(True, False),
  )
  def test_conv_transpose(self, use_bias):
    rng = dict(params=jax.random.key(0))
    x = jnp.ones((1, 8, 3))
    conv_transpose_module = linear.ConvTranspose(
        features=4,
        use_bias=use_bias,
        kernel_size=(3,),
        padding='VALID',
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
    )
    y, variables = conv_transpose_module.init_with_output(rng, x)
    self.assertEqual(variables['params']['kernel'].value.shape, (3, 3, 4))
    correct_ans = np.array(
        [
            [
                [4.0, 4.0, 4.0, 4.0],
                [7.0, 7.0, 7.0, 7.0],
                [10.0, 10.0, 10.0, 10.0],
                [10.0, 10.0, 10.0, 10.0],
                [10.0, 10.0, 10.0, 10.0],
                [10.0, 10.0, 10.0, 10.0],
                [10.0, 10.0, 10.0, 10.0],
                [10.0, 10.0, 10.0, 10.0],
                [7.0, 7.0, 7.0, 7.0],
                [4.0, 4.0, 4.0, 4.0],
            ]
        ]
    )
    if not use_bias:
      correct_ans -= 1.0
    np.testing.assert_allclose(y, correct_ans)

  @parameterized.product(
      use_bias=(True, False),
  )
  def test_multibatch_input_conv_transpose(self, use_bias):
    rng = dict(params=jax.random.key(0))
    x = jnp.ones((2, 5, 8, 3))
    conv_transpose_module = linear.ConvTranspose(
        features=4,
        use_bias=use_bias,
        kernel_size=(3,),
        padding='VALID',
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
    )
    y, variables = conv_transpose_module.init_with_output(rng, x)
    self.assertEqual(variables['params']['kernel'].value.shape, (3, 3, 4))
    correct_ans = np.array(
        [
            [
                [4.0, 4.0, 4.0, 4.0],
                [7.0, 7.0, 7.0, 7.0],
                [10.0, 10.0, 10.0, 10.0],
                [10.0, 10.0, 10.0, 10.0],
                [10.0, 10.0, 10.0, 10.0],
                [10.0, 10.0, 10.0, 10.0],
                [10.0, 10.0, 10.0, 10.0],
                [10.0, 10.0, 10.0, 10.0],
                [7.0, 7.0, 7.0, 7.0],
                [4.0, 4.0, 4.0, 4.0],
            ]
        ]
    )
    correct_ans = np.repeat(correct_ans[None], repeats=2, axis=0)
    correct_ans = np.repeat(correct_ans, repeats=5, axis=1)
    if not use_bias:
      correct_ans -= 1.0
    np.testing.assert_allclose(y, correct_ans)

  def test_single_input_conv_transpose(self):
    rng = dict(params=jax.random.key(0))
    x = jnp.ones((8, 3))
    conv_transpose_module = linear.ConvTranspose(
        features=4,
        kernel_size=(3,),
        padding='VALID',
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
    )
    y, variables = conv_transpose_module.init_with_output(rng, x)
    self.assertEqual(variables['params']['kernel'].value.shape, (3, 3, 4))
    correct_ans = np.array(
        [
            [4.0, 4.0, 4.0, 4.0],
            [7.0, 7.0, 7.0, 7.0],
            [10.0, 10.0, 10.0, 10.0],
            [10.0, 10.0, 10.0, 10.0],
            [10.0, 10.0, 10.0, 10.0],
            [10.0, 10.0, 10.0, 10.0],
            [10.0, 10.0, 10.0, 10.0],
            [10.0, 10.0, 10.0, 10.0],
            [7.0, 7.0, 7.0, 7.0],
            [4.0, 4.0, 4.0, 4.0],
        ]
    )
    np.testing.assert_allclose(y, correct_ans)

  def test_single_input_masked_conv_transpose(self):
    rng = dict(params=jax.random.key(0))
    x = jnp.ones((8, 3))
    m = jnp.tril(jnp.ones((3, 3, 4)))
    conv_transpose_module = linear.ConvTranspose(
        features=4,
        kernel_size=(3,),
        padding='VALID',
        mask=m,
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
    )
    y, variables = conv_transpose_module.init_with_output(rng, x)
    self.assertEqual(variables['params']['kernel'].value.shape, (3, 3, 4))
    correct_ans = np.array(
        [
            [4.0, 3.0, 2.0, 1.0],
            [7.0, 5.0, 3.0, 1.0],
            [10.0, 7.0, 4.0, 1.0],
            [10.0, 7.0, 4.0, 1.0],
            [10.0, 7.0, 4.0, 1.0],
            [10.0, 7.0, 4.0, 1.0],
            [10.0, 7.0, 4.0, 1.0],
            [10.0, 7.0, 4.0, 1.0],
            [7.0, 5.0, 3.0, 1.0],
            [4.0, 3.0, 2.0, 1.0],
        ]
    )
    np.testing.assert_allclose(y, correct_ans)

  @parameterized.product(
      n_batch=(1, 3),
      n_features=(1, 2),
      kernel_size=(1, 2, 3, 9),
      n_input_features=(1, 3),
      input_size=(1, 8, 16),
  )
  def test_circular_conv_transpose_1d_constant(
      self,
      n_batch,
      n_features,
      kernel_size,
      n_input_features,
      input_size
  ):  # pylint: disable=g-doc-args
    """Test 1D transposed convolution with circular padding.

    Filter with all elements equal to 1 applied on an input with all elements
    equal to 1. Result should have the same shape as input (except for the
    feature dimension) and have all elements equal to
    `n_input_features * kernel_lin_size`.
    """
    rng = dict(params=jax.random.key(0))
    x = jnp.ones((n_batch, input_size, n_input_features))
    conv_module = linear.ConvTranspose(
        features=n_features,
        kernel_size=(kernel_size,),
        padding=functools.partial(
            utils.circular_post_conv_padding,
            transpose_kernel=False,
        ),
        kernel_init=initializers.ones,
        bias_init=initializers.zeros,
    )
    y, variables = conv_module.init_with_output(rng, x)

    self.assertEqual(
        variables['params']['kernel'].value.shape,
        (kernel_size, n_input_features, n_features),
    )
    correct_ans = np.full(
        (n_batch, input_size, n_features), kernel_size * n_input_features
    )
    np.testing.assert_allclose(y, correct_ans)

  @parameterized.product(
      n_batch=(1, 3),
      n_features=(1, 2, 10),
      kernel_lin_size=(1, 2, 3, 9),
      n_input_features=(1, 5),
      input_x_size=(14,),
      input_y_size=(5, 10),
  )
  def test_circular_conv_transpose_2d_constant(
      self,
      n_batch,
      n_features,
      kernel_lin_size,
      n_input_features,
      input_x_size,
      input_y_size,
  ):  # pylint: disable=g-doc-args
    """Test 2D transposed convolution with circular padding.

    Square filter with all elements equal to 1 applied on an input with all
    elements equal to 1. Result should have the same shape as input (except for
    the feature dimension) and have all elements equal to
    `n_input_features * kernel_lin_size^2`.
    """
    rng = dict(params=jax.random.key(0))
    x = jnp.ones((n_batch, input_x_size, input_y_size, n_input_features))
    conv_module = linear.ConvTranspose(
        features=n_features,
        kernel_size=(kernel_lin_size, kernel_lin_size),
        padding=functools.partial(
            utils.circular_post_conv_padding,
            transpose_kernel=False,
        ),
        kernel_init=initializers.ones,
        bias_init=initializers.zeros,
    )
    y, variables = conv_module.init_with_output(rng, x)

    self.assertEqual(
        variables['params']['kernel'].value.shape,
        (kernel_lin_size, kernel_lin_size, n_input_features, n_features),
    )
    correct_ans = np.full(
        (n_batch, input_x_size, input_y_size, n_features),
        kernel_lin_size * kernel_lin_size * n_input_features,
    )
    np.testing.assert_allclose(y, correct_ans)

  def test_circular_conv_transpose_2d_with_vmap(self):
    layer = linear.ConvTranspose(
        features=5,
        kernel_size=(3,),
        padding=functools.partial(
            utils.circular_post_conv_padding,
            transpose_kernel=False,
        ),
    )

    # this is ok
    sample_input = jnp.ones((1, 32, 2))
    out, variables = layer.init_with_output(jax.random.key(0), sample_input)
    self.assertEqual(out.shape, (1, 32, 5))

    batch_input = jnp.ones((8, 32, 2))
    batch_apply = jax.vmap(layer.apply, in_axes=(None, 0))

    # this breaks with the error provided
    batch_out = batch_apply(variables, batch_input)
    self.assertEqual(batch_out.shape, (8, 32, 5))

  def test_circular_conv_transpose_1d_custom(self):
    """Test 1d transposed convolution with circular padding and a stride."""
    rng = dict(params=jax.random.key(0))
    x = np.arange(1, 6)
    x = np.expand_dims(x, (0, 2))
    kernel = np.array((1, 2, 1))
    kernel = np.expand_dims(kernel, (1, 2))

    conv_module = linear.ConvTranspose(
        features=1,
        kernel_size=(3,),
        strides=(3,),
        padding=functools.partial(
            utils.circular_post_conv_padding,
            transpose_kernel=False,
        ),
        kernel_init=lambda key, shape, dtype: kernel,
        bias_init=initializers.zeros,
    )
    y, variables = conv_module.init_with_output(rng, x)

    self.assertEqual(variables['params']['kernel'].value.shape, (3, 1, 1))
    # Compare with manually computed convolution
    correct_ans = np.array(
        (  # pyformat: disable
            1 * 1,
            1 * 2,
            1 * 1,
            2 * 1,
            2 * 2,
            2 * 1,
            3 * 1,
            3 * 2,
            3 * 1,
            4 * 1,
            4 * 2,
            4 * 1,
            5 * 1,
            5 * 2,
            5 * 1,
        )
    )
    correct_ans = np.expand_dims(correct_ans, (0, 2))
    np.testing.assert_allclose(y, correct_ans)

  def test_circular_conv_transpose_2d_custom(self):
    """Test 2d transposed convolution with circular padding on a 3x3 example."""
    rng = dict(params=jax.random.key(0))
    x = np.array(
        (
            (1, 2, 3),
            (4, 5, 6),
            (7, 8, 9),
        )
    )
    x = np.expand_dims(x, (0, 3))
    kernel = np.array(((0, 1, 0), (1, 2, 1), (0, 1, 0)))
    kernel = np.expand_dims(kernel, (2, 3))

    conv_module = linear.ConvTranspose(
        features=1,
        kernel_size=(3, 3),
        padding=functools.partial(
            utils.circular_post_conv_padding,
            transpose_kernel=False,
        ),
        kernel_init=lambda key, shape, dtype: kernel,
        bias_init=initializers.zeros,
    )
    y, variables = conv_module.init_with_output(rng, x)

    self.assertEqual(variables['params']['kernel'].value.shape, (3, 3, 1, 1))
    # Compare with manually computed convolution
    correct_ans = np.array(
        (
            (18, 21, 24),
            (27, 30, 33),
            (36, 39, 42),
        )
    )
    correct_ans = np.expand_dims(correct_ans, (0, 3))
    np.testing.assert_allclose(y, correct_ans)

  def test_circular_conv_transpose_2d_custom_bias(self):
    """Test 2d transposed convolution with circular padding on a 2x2 example with bias."""
    rng = dict(params=jax.random.key(0))
    x = np.array(((1, 2), (3, 4)))
    x = np.expand_dims(x, (0, 3))
    kernel = np.array(
        (
            (1, 2),
            (3, 4),
        )
    )
    kernel = np.expand_dims(kernel, (2, 3))

    conv_module = linear.ConvTranspose(
        features=1,
        kernel_size=(2, 2),
        padding=functools.partial(
            utils.circular_post_conv_padding,
            transpose_kernel=False,
        ),
        kernel_init=lambda key, shape, dtype: kernel,
        bias_init=initializers.ones,
    )
    y, variables = conv_module.init_with_output(rng, x)

    self.assertEqual(variables['params']['kernel'].value.shape, (2, 2, 1, 1))
    # Compare with manually computed convolution
    correct_ans = np.array(
        (
            (21, 23),
            (29, 31),
        )
    )
    correct_ans = np.expand_dims(correct_ans, (0, 3))
    np.testing.assert_allclose(y, correct_ans)

  @parameterized.product(use_bias=(True, False))
  def test_transpose_kernel_conv_transpose(self, use_bias):
    rng = dict(params=jax.random.key(0))
    x = jnp.ones((1, 15, 15, 3))
    conv_module = linear.ConvTranspose(
        features=4,
        use_bias=use_bias,
        strides=(2, 2),
        kernel_size=(6, 6),
        padding=functools.partial(
            utils.circular_post_conv_padding,
            transpose_kernel=True,
        ),
        transpose_kernel=True,
    )
    y, variables = conv_module.init_with_output(rng, x)
    self.assertEqual(variables['params']['kernel'].value.shape, (6, 6, 4, 3))
    self.assertEqual(y.shape, (1, 30, 30, 4))

  @parameterized.product(module=(linear.Conv, linear.ConvLocal))
  def test_int_kernel_size(self, module):
    conv = module(features=4, kernel_size=3)
    x = jnp.ones((8, 3))
    with self.assertRaises(TypeError):
      conv.init(jax.random.key(0), x)

if __name__ == '__main__':
  absltest.main()
