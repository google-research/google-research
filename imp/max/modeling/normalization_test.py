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

"""Tests for multimodal modules."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from imp.max.modeling import normalization
from imp.max.utils import sharding


_TOKENS_SHARDINGS = ('data', None, None, 'model')


def _create_global_mesh():
  return jax.sharding.Mesh(
      sharding.create_tpu_device_mesh(ici_mesh_shape=(1, 1),
                                      dcn_mesh_shape=(1, 1)),
      ['data', 'model'],
  )


class NormalizationTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'baseline',
          'reduction_axes': -1,
      }, {
          'testcase_name': 'with_mask',
          'reduction_axes': 3,
          'with_mask': True,
      }, {
          'testcase_name': 'non_last_dim',
          'reduction_axes': 2,
      }, {
          'testcase_name': 'last_dim_with_others',
          'reduction_axes': (2, 3),
      }, {
          'testcase_name': 'others_without_last_dim',
          'reduction_axes': (1, 2),
      }, {
          'testcase_name': 'batch_inclusive',
          'reduction_axes': (0, 1, 2, 3),
      },
      {
          'testcase_name': 'not_fast_variance',
          'reduction_axes': -1,
          'use_fast_variance': False,
      },
  )
  def test_layernorm(self,
                     reduction_axes=-1,
                     use_fast_variance=True,
                     with_mask=False):

    if isinstance(reduction_axes, int):
      scale_bias_shardings = (_TOKENS_SHARDINGS[reduction_axes],)
    else:
      scale_bias_shardings = (_TOKENS_SHARDINGS[reduction_axes[-1]],)

    layernorm = normalization.LayerNorm(
        use_bias=True,
        use_scale=True,
        epsilon=1e-5,
        scale_init=jax.nn.initializers.normal(),
        bias_init=jax.nn.initializers.normal(),
        reduction_axes=reduction_axes,
        use_fast_variance=use_fast_variance,
        shardings=scale_bias_shardings,
    )
    key_inputs, key_params = jax.random.split(jax.random.key(0))
    input_shape = (2, 1, 3, 4)
    if with_mask:
      key_inputs, key_mask = jax.random.split(key_inputs)
      mask = jax.random.choice(
          key_mask, reduction_axes, input_shape).astype(bool)
      mask = mask.at[Ellipsis, :2].set(True)  # guarantee at least 2 elements
    else:
      mask = None

    @jax.jit
    def _run_forward(inputs):
      variables = layernorm.init(rngs={'params': key_params},
                                 inputs=inputs,
                                 mask=mask)
      outputs = layernorm.apply(variables=variables, inputs=inputs, mask=mask)
      return outputs, variables

    with _create_global_mesh():
      inputs = jax.random.normal(key_inputs, shape=input_shape)
      if mask is not None:
        inputs = jnp.where(mask, inputs, jnp.nan)
      inputs = sharding.shard_array(inputs, _TOKENS_SHARDINGS)
      if not use_fast_variance:
        inputs += 1e6  # This blows up fast variance, but should work otherwise.
      outputs, variables = _run_forward(inputs)

    self.assertEqual(inputs.dtype, outputs.dtype)
    self.assertEqual(inputs.shape, outputs.shape)

    expected_mean_centered = (
        inputs - inputs.mean(axis=reduction_axes, keepdims=True, where=mask))
    expected_multiplier = jax.lax.rsqrt(
        inputs.var(axis=reduction_axes, keepdims=True, where=mask) + 1e-5)
    expected_multiplier *= variables['params']['scale'].value
    expected_outputs = (
        expected_mean_centered * expected_multiplier
        + variables['params']['bias'].value)

    np.testing.assert_allclose(outputs, expected_outputs, atol=1e-4)

    # Assert shardings are propagated properly
    self.assertEqual(variables['params']['scale'].names,
                     scale_bias_shardings)
    self.assertEqual(variables['params']['bias'].names,
                     scale_bias_shardings)

  @parameterized.named_parameters(
      {
          'testcase_name': 'baseline',
          'reduction_axes': -1,
      }, {
          'testcase_name': 'with_mask',
          'reduction_axes': 3,
          'with_mask': True,
      }, {
          'testcase_name': 'non_last_dim',
          'reduction_axes': 2,
      }, {
          'testcase_name': 'last_dim_with_others',
          'reduction_axes': (2, 3),
      }, {
          'testcase_name': 'others_without_last_dim',
          'reduction_axes': (1, 2),
      }, {
          'testcase_name': 'batch_inclusive',
          'reduction_axes': (0, 1, 2, 3),
      },
      {
          'testcase_name': 'not_fast_variance',
          'reduction_axes': -1,
          'use_fast_variance': False,
      },
  )
  def test_rmsnorm(self,
                   reduction_axes=-1,
                   use_fast_variance=True,
                   with_mask=False):

    if isinstance(reduction_axes, int):
      scale_bias_shardings = (_TOKENS_SHARDINGS[reduction_axes],)
    else:
      scale_bias_shardings = (_TOKENS_SHARDINGS[reduction_axes[-1]],)

    rmsnorm = normalization.RMSNorm(
        use_scale=True,
        epsilon=1e-5,
        scale_init=jax.nn.initializers.normal(),
        reduction_axes=reduction_axes,
        use_fast_variance=use_fast_variance,
        shardings=scale_bias_shardings,
    )
    key_inputs, key_params = jax.random.split(jax.random.key(0))
    input_shape = (2, 1, 3, 4)
    if with_mask:
      key_inputs, key_mask = jax.random.split(key_inputs)
      mask = jax.random.choice(
          key_mask, reduction_axes, input_shape).astype(bool)
      mask = mask.at[Ellipsis, :2].set(True)  # guarantee at least 2 elements
    else:
      mask = None

    @jax.jit
    def _run_forward(inputs):
      variables = rmsnorm.init(rngs={'params': key_params},
                               inputs=inputs,
                               mask=mask)
      outputs = rmsnorm.apply(variables=variables, inputs=inputs, mask=mask)
      return outputs, variables

    with _create_global_mesh():
      inputs = jax.random.normal(key_inputs, shape=input_shape)
      if mask is not None:
        inputs = jnp.where(mask, inputs, jnp.nan)
      inputs = sharding.shard_array(inputs, _TOKENS_SHARDINGS)
      if not use_fast_variance:
        inputs += 1e6  # This blows up fast variance, but should work otherwise.
      outputs, variables = _run_forward(inputs)

    self.assertEqual(inputs.dtype, outputs.dtype)
    self.assertEqual(inputs.shape, outputs.shape)

    expected_multiplier = jax.lax.rsqrt(
        jnp.mean(jax.lax.square(inputs),
                 axis=reduction_axes,
                 keepdims=True,
                 where=mask) + 1e-5)
    expected_multiplier *= variables['params']['scale'].value
    expected_outputs = inputs * expected_multiplier

    np.testing.assert_allclose(outputs, expected_outputs, atol=1e-4)

    # Assert shardings are propagated properly
    self.assertEqual(variables['params']['scale'].names,
                     scale_bias_shardings)


if __name__ == '__main__':
  absltest.main()
