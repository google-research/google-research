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

"""Tests for transformers."""
import copy
import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import aqt.jax.v2.config as aqt_config
import chex
import jax
from jax import lax
import jax.numpy as jnp

from imp.max.core import utils
from imp.max.modeling import transformers
from imp.max.utils import sharding


AQT_8BITS_CFG = aqt_config.DotGeneral(fwd=aqt_config.DotGeneralRaw.make(8, 8),
                                      dlhs=aqt_config.DotGeneralRaw.make(8, 8),
                                      drhs=aqt_config.DotGeneralRaw.make(8, 8))


def _create_global_mesh():
  return jax.sharding.Mesh(
      sharding.create_tpu_device_mesh(ici_mesh_shape=(1, 1, 1),
                                      dcn_mesh_shape=(1, 1, 1)),
      ['data', 'layers', 'model'],
  )


def _create_global_moe_mesh():
  return jax.sharding.Mesh(
      sharding.create_tpu_device_mesh(ici_mesh_shape=(1, 1, 1, 1),
                                      dcn_mesh_shape=(1, 1, 1, 1)),
      ['expert', 'data', 'layers', 'model'],
  )


@dataclasses.dataclass
class Shardings:
  tokens = ('data', None, None, 'model')  # (b, n, t, d)
  attention_mask = ('data', None, None, None, None)  # (b, n, 1, t, t)
  attention_bias = (None, None, 'model', None, None)  # (1, 1, h, t, t)
  ffn_inner_kernel = (None, 'model')  # (d, d_ff)
  ffn_outer_kernel = ('model', None)  # (d_ff, d)
  ffn_intermediate = ('data', None, None, 'model')  # (b, n, t, d)
  mha_qkv_kernel = (None, 'model', None)  # (d, h, d_h)
  mha_out_kernel = ('model', None, None)  # (h, d_h, d)
  mha_activation = ('data', None, None, 'model', None)  # (b, n, t, h, d_h)
  layernorm = (None,)  # (d,)
  scan_axis = 'layers'


# pylint: disable=line-too-long
@dataclasses.dataclass
class MoeShardings:
  tokens = (('expert', 'data'), None, None, 'model')  # (b, n, t, d)
  attention_mask = (('expert', 'data'), None, None, None, None)  # (b, n, 1, t, t)
  attention_bias = (None, None, 'model', None, None)  # (1, 1, h, t, t)
  ffn_inner_kernel = (None, 'model')  # (d, d_ff)
  ffn_outer_kernel = ('model', None)  # (d_ff, d)
  ffn_intermediate = (('expert', 'data'), None, None, 'model')  # (b, n, t, d)
  routed_ffn_intermediate = ('data', 'model')  # (r, d_ff)
  mha_qkv_kernel = (None, 'model', None)  # (d, h, d_h)
  mha_out_kernel = ('model', None, None)  # (h, d_h, d)
  mha_activation = (('expert', 'data'), None, None, 'model', None)  # (b, n, t, h, d_h)
  layernorm = (None,)  # (d,)
  router_kernel = (None, None)
  scan_axis = 'layers'
  expert_axis = 'expert'
# pylint: enable=line-too-long


class FeedForwardTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'baseline',
          'shape_without_d': (1, 2, 3),
          'd_ff': 1,
          'd_model': 2,
          'dropout_rate': 0.,
          'use_bias': True,
          'dtype': jnp.float32,
      }, {
          'testcase_name': 'with_dropout',
          'shape_without_d': (1, 2, 3),
          'd_ff': 1,
          'd_model': 2,
          'dropout_rate': 0.5,
          'use_bias': False,
          'dtype': jnp.float32,
          'deterministic': False
      }, {
          'testcase_name': 'more_dims',
          'shape_without_d': (1, 2, 3, 4),
          'd_ff': 1,
          'd_model': 2,
          'dropout_rate': 0.5,
          'use_bias': False,
          'dtype': jnp.bfloat16,
          'deterministic': False
      }, {
          'testcase_name': 'quantized_8bits',
          'shape_without_d': (1, 2, 3, 4),
          'd_ff': 1,
          'd_model': 2,
          'dropout_rate': 0.5,
          'use_bias': False,
          'dtype': jnp.bfloat16,
          'deterministic': False,
          'dot_general': utils.make_dot_general(AQT_8BITS_CFG),
      }, {
          'testcase_name': 'lora',
          'shape_without_d': (1, 2, 3, 4),
          'd_ff': 1,
          'd_model': 2,
          'dropout_rate': 0.5,
          'use_bias': False,
          'dtype': jnp.bfloat16,
          'deterministic': False,
          'lora_rank': 2,
          'lora_scale': 1.,
      })
  def test_feed_forward(self,
                        shape_without_d,
                        d_ff,
                        d_model,
                        dropout_rate=0,
                        use_bias=True,
                        dtype=jnp.float32,
                        deterministic=True,
                        dot_general=lax.dot_general,
                        precision=None,
                        lora_rank=2,
                        lora_scale=0.):

    input_shape = shape_without_d + (d_model,)
    shardings = Shardings()
    if len(input_shape) == 5:
      tokens = list(shardings.tokens)
      ffn_intermediate = list(shardings.ffn_intermediate)
      tokens.insert(-1, None)
      ffn_intermediate.insert(-1, None)
      shardings.tokens = tuple(tokens)
      shardings.ffn_intermediate = tuple(ffn_intermediate)
    elif len(input_shape) != 4:
      raise ValueError('Invalid input shape.')

    ffn_layer = transformers.FeedForward(
        d_ff=d_ff,
        d_model=d_model,
        dropout_rate=dropout_rate,
        use_bias=use_bias,
        dtype=dtype,
        approximate_gelu=True,
        dot_general=dot_general,
        precision=precision,
        lora_rank=lora_rank,
        lora_scale=lora_scale,
        inner_kernel_shardings=shardings.ffn_inner_kernel,
        outer_kernel_shardings=shardings.ffn_outer_kernel,
        intermediate_shardings=shardings.ffn_intermediate,
    )

    @jax.jit
    def _run_forward(inputs):
      outputs, variables = ffn_layer.init_with_output(
          rngs={'params': jax.random.key(1),
                'dropout': jax.random.key(2)},
          inputs=inputs,
          deterministic=deterministic)
      return outputs, variables

    with _create_global_mesh():
      inputs = jnp.ones(input_shape)
      inputs = sharding.shard_array(inputs, shardings.tokens)
      outputs, variables = _run_forward(inputs)

    # Assert output shape
    chex.assert_shape(outputs, input_shape)

    # Assert shardings are propagated properly
    self.assertEqual(variables['params']['wi']['kernel'].names,
                     shardings.ffn_inner_kernel)
    self.assertEqual(variables['params']['wo']['kernel'].names,
                     shardings.ffn_outer_kernel)


class TransformerEncoderLayerTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'baseline',
          'batch_size': 1,
          'n_instances': 2,
          'qkv_length': 3,
          'd_model': 5,
          'num_heads': 5,
          'd_ff': 6,
          'use_bias': True,
          'dtype': jnp.float32,
      },
      {
          'testcase_name': 'multiple_heads',
          'batch_size': 2,
          'n_instances': 2,
          'qkv_length': 3,
          'd_model': 16,
          'num_heads': 4,
          'd_ff': 32,
          'use_bias': True,
          'dtype': jnp.bfloat16,
      },
      {
          'testcase_name': 'with_dropout',
          'batch_size': 2,
          'n_instances': 2,
          'qkv_length': 3,
          'd_model': 16,
          'num_heads': 4,
          'd_ff': 32,
          'use_bias': False,
          'dtype': jnp.float32,
          'deterministic': False
      },
      {
          'testcase_name': 'd_model_nondivisible_num_heads',
          'batch_size': 1,
          'n_instances': 2,
          'qkv_length': 3,
          'd_model': 6,
          'num_heads': 5,
          'd_ff': 6,
          'use_bias': True,
          'dtype': jnp.float32,
      }, {
          'testcase_name': 'qk_layernorm',
          'batch_size': 1,
          'n_instances': 2,
          'qkv_length': 3,
          'd_model': 5,
          'num_heads': 5,
          'd_ff': 6,
          'use_bias': True,
          'dtype': jnp.float32,
          'qk_layernorm': True,
      }, {
          'testcase_name': 'quantized_8bits',
          'batch_size': 1,
          'n_instances': 2,
          'qkv_length': 3,
          'd_model': 5,
          'num_heads': 5,
          'd_ff': 6,
          'use_bias': True,
          'dtype': jnp.float32,
          'qk_layernorm': True,
          'mha_qkv_dot_general': utils.make_dot_general(AQT_8BITS_CFG),
          'mha_out_dot_general': utils.make_dot_general(AQT_8BITS_CFG),
          'mha_einsum_dot_general': utils.make_dot_general(AQT_8BITS_CFG),
          'ffn_dot_general': utils.make_dot_general(AQT_8BITS_CFG),
      }, {
          'testcase_name': 'lora',
          'batch_size': 1,
          'n_instances': 2,
          'qkv_length': 3,
          'd_model': 5,
          'num_heads': 5,
          'd_ff': 6,
          'use_bias': True,
          'dtype': jnp.float32,
          'qk_layernorm': True,
          'lora_rank': 2,
          'lora_scale': 1.,
      }
  )
  def test_encoder_layer(self,
                         batch_size,
                         n_instances,
                         qkv_length,
                         d_model,
                         num_heads,
                         d_ff,
                         dropout_rate=0.1,
                         use_bias=True,
                         dtype=jnp.float32,
                         use_attention_mask=False,
                         use_attention_bias=False,
                         deterministic=True,
                         qk_layernorm=False,
                         mha_qkv_dot_general=lax.dot_general,
                         mha_out_dot_general=lax.dot_general,
                         mha_einsum_dot_general=lax.dot_general,
                         ffn_dot_general=lax.dot_general,
                         precision=None,
                         lora_rank=2,
                         lora_scale=0.):

    shardings = Shardings()
    encoder_layer = transformers.TransformerEncoderLayer(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout_rate=dropout_rate,
        use_bias=use_bias,
        dtype=dtype,
        qk_layernorm=qk_layernorm,
        approximate_gelu=True,
        mha_qkv_dot_general=mha_qkv_dot_general,
        mha_out_dot_general=mha_out_dot_general,
        mha_einsum_dot_general=mha_einsum_dot_general,
        ffn_dot_general=ffn_dot_general,
        precision=precision,
        lora_rank=lora_rank,
        lora_scale=lora_scale,
        ffn_inner_kernel_shardings=shardings.ffn_inner_kernel,
        ffn_outer_kernel_shardings=shardings.ffn_outer_kernel,
        ffn_intermediate_shardings=shardings.ffn_intermediate,
        mha_qkv_kernel_shardings=shardings.mha_qkv_kernel,
        mha_out_kernel_shardings=shardings.mha_out_kernel,
        mha_activation_shardings=shardings.mha_activation,
        layernorm_shardings=shardings.layernorm,
    )
    @jax.jit
    def _run_forward(inputs, attention_mask, attention_bias):
      outputs, variables = encoder_layer.init_with_output(
          rngs={'params': jax.random.key(1)},
          inputs=inputs,
          attention_mask=attention_mask,
          attention_bias=attention_bias,
          deterministic=deterministic)
      return outputs, variables

    with _create_global_mesh():
      inputs = jnp.ones((batch_size, n_instances, qkv_length, d_model))
      inputs = sharding.shard_array(inputs, shardings.tokens)
      if use_attention_mask:
        attention_mask = jnp.ones(
            (batch_size, n_instances, 1, qkv_length, qkv_length))
        attention_mask = sharding.shard_array(
            attention_mask, shardings.attention_mask)
      else:
        attention_mask = None
      if use_attention_bias:
        attention_bias = jnp.ones((1, 1, num_heads, qkv_length, qkv_length))
        attention_bias = sharding.shard_array(
            attention_bias, shardings.attention_bias)
      else:
        attention_bias = None
      outputs, variables = _run_forward(inputs, attention_mask, attention_bias)

    # Assert outputs shape
    chex.assert_shape(outputs, (batch_size, n_instances, qkv_length, d_model))

    # Assert shardings are propagated properly
    ffn_params = variables['params']['feed_forward']
    mha_params = variables['params']['self_attention']
    sa_ln_params = variables['params']['layer_norm_sa']
    ffn_ln_params = variables['params']['layer_norm_ffn']
    self.assertEqual(ffn_params['wi']['kernel'].names,
                     shardings.ffn_inner_kernel)
    self.assertEqual(ffn_params['wo']['kernel'].names,
                     shardings.ffn_outer_kernel)
    for l in ('q', 'k', 'v'):
      self.assertEqual(mha_params[l]['kernel'].names, shardings.mha_qkv_kernel)
    self.assertEqual(mha_params['o']['kernel'].names, shardings.mha_out_kernel)
    self.assertEqual(sa_ln_params['scale'].names, shardings.layernorm)
    self.assertEqual(ffn_ln_params['scale'].names, shardings.layernorm)


class TransformerEncoderTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'baseline',
          'batch_size': 1,
          'n_instances': 2,
          'qkv_length': 3,
          'd_model': 5,
          'num_heads': 5,
          'd_ff': 6,
          'use_bias': False,
          'dtype': jnp.float32,
      }, {
          'testcase_name': 'scanned',
          'batch_size': 1,
          'n_instances': 2,
          'qkv_length': 3,
          'd_model': 5,
          'num_heads': 5,
          'd_ff': 6,
          'use_bias': False,
          'scanned_layers': True,
          'dtype': jnp.float32,
      }, {
          'testcase_name': 'scanned_and_remat',
          'batch_size': 1,
          'n_instances': 2,
          'qkv_length': 3,
          'd_model': 5,
          'num_heads': 5,
          'd_ff': 6,
          'use_bias': False,
          'scanned_layers': True,
          'remat': 'full',
          'dtype': jnp.float32,
      },
      {
          'testcase_name': 'multiple_heads',
          'batch_size': 2,
          'n_instances': 2,
          'qkv_length': 3,
          'd_model': 16,
          'num_heads': 4,
          'd_ff': 32,
          'use_bias': True,
          'dtype': jnp.float32,
      },
      {
          'testcase_name': 'with_dropout',
          'batch_size': 2,
          'n_instances': 2,
          'qkv_length': 3,
          'd_model': 16,
          'num_heads': 4,
          'd_ff': 32,
          'use_bias': True,
          'dtype': jnp.float32,
          'deterministic': False,
      },
      {
          'testcase_name': 'qk_layernorm',
          'batch_size': 1,
          'n_instances': 2,
          'qkv_length': 3,
          'd_model': 5,
          'num_heads': 5,
          'd_ff': 6,
          'use_bias': False,
          'dtype': jnp.float32,
          'qk_layernorm': True,
      },
      {
          'testcase_name': 'quantized_8bits',
          'batch_size': 1,
          'n_instances': 2,
          'qkv_length': 3,
          'd_model': 5,
          'num_heads': 5,
          'd_ff': 6,
          'use_bias': True,
          'dtype': jnp.float32,
          'qk_layernorm': True,
          'mha_qkv_dot_general': utils.make_dot_general(AQT_8BITS_CFG),
          'mha_out_dot_general': utils.make_dot_general(AQT_8BITS_CFG),
          'mha_einsum_dot_general': utils.make_dot_general(AQT_8BITS_CFG),
          'ffn_dot_general': utils.make_dot_general(AQT_8BITS_CFG),
      },
      {
          'testcase_name': 'lora',
          'batch_size': 1,
          'n_instances': 2,
          'qkv_length': 3,
          'd_model': 5,
          'num_heads': 5,
          'd_ff': 6,
          'use_bias': False,
          'dtype': jnp.float32,
          'qk_layernorm': True,
          'lora_rank': 2,
          'lora_scale': 1.,
      },
      {
          'testcase_name': 'nonapproximate_gelu',
          'batch_size': 1,
          'n_instances': 2,
          'qkv_length': 3,
          'd_model': 5,
          'num_heads': 5,
          'd_ff': 6,
          'use_bias': False,
          'dtype': jnp.float32,
          'qk_layernorm': False,
          'approximate_gelu': False,
      },
  )
  def test_encoder(
      self,
      batch_size,
      n_instances,
      qkv_length,
      d_model,
      num_heads,
      d_ff,
      num_layers=2,
      dropout_rate=0.1,
      use_bias=True,
      dtype=jnp.float32,
      use_attention_mask=False,
      use_attention_bias=False,
      deterministic=True,
      approximate_gelu=True,
      qk_layernorm=False,
      mha_qkv_dot_general=lax.dot_general,
      mha_out_dot_general=lax.dot_general,
      mha_einsum_dot_general=lax.dot_general,
      ffn_dot_general=lax.dot_general,
      precision=None,
      lora_rank=2,
      lora_scale=0.0,
      scanned_layers=False,
      remat='zero',
  ):
    shardings = Shardings()
    encoder = transformers.TransformerEncoder(
        num_heads=num_heads,
        d_model=d_model,
        d_ff=d_ff,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        use_bias=use_bias,
        remat=remat,
        scanned_layers=scanned_layers,
        scan_axis=0,
        dtype=dtype,
        qk_layernorm=qk_layernorm,
        scan_sharding_axis=shardings.scan_axis,
        ffn_inner_kernel_shardings=shardings.ffn_inner_kernel,
        ffn_outer_kernel_shardings=shardings.ffn_outer_kernel,
        ffn_intermediate_shardings=shardings.ffn_intermediate,
        mha_qkv_kernel_shardings=shardings.mha_qkv_kernel,
        mha_out_kernel_shardings=shardings.mha_out_kernel,
        mha_activation_shardings=shardings.mha_activation,
        layernorm_shardings=shardings.layernorm,
        mha_qkv_dot_general=mha_qkv_dot_general,
        mha_out_dot_general=mha_out_dot_general,
        mha_einsum_dot_general=mha_einsum_dot_general,
        ffn_dot_general=ffn_dot_general,
        precision=precision,
        lora_rank=lora_rank,
        lora_scale=lora_scale,
        approximate_gelu=approximate_gelu,
    )
    @jax.jit
    def _run_forward(inputs, attention_mask, attention_bias):
      outputs, variables = encoder.init_with_output(
          rngs={'params': jax.random.key(1),
                'dropout': jax.random.key(2)},
          inputs=inputs,
          attention_mask=attention_mask,
          attention_bias=attention_bias,
          deterministic=deterministic)
      return outputs, variables

    with _create_global_mesh():
      inputs = jnp.ones((batch_size, n_instances, qkv_length, d_model))
      inputs = sharding.shard_array(inputs, shardings.tokens)

      if use_attention_mask:
        attention_mask = jnp.ones(
            (batch_size, n_instances, 1, qkv_length, qkv_length))
        attention_mask = sharding.shard_array(
            attention_mask, shardings.attention_mask)
      else:
        attention_mask = None
      if use_attention_bias:
        attention_bias = jnp.ones((num_heads, qkv_length, qkv_length))
        attention_bias = sharding.shard_array(
            attention_bias, shardings.attention_bias)
      else:
        attention_bias = None
      outputs, variables = _run_forward(inputs, attention_mask, attention_bias)

      # Assert output shape
      chex.assert_shape(outputs, (batch_size, n_instances, qkv_length, d_model))

      # Assert shardings are propagated properly in scanned layers
      if scanned_layers:
        scanned_params = variables['params']['layer_scan']
        ffn_params = scanned_params['feed_forward']
        mha_params = scanned_params['self_attention']
        sa_ln_params = scanned_params['layer_norm_sa']
        ffn_ln_params = scanned_params['layer_norm_ffn']
        # Scan axis is 0, hence its sharding name leads
        self.assertEqual(ffn_params['wi']['kernel'].names,
                         (shardings.scan_axis, *shardings.ffn_inner_kernel))
        self.assertEqual(ffn_params['wo']['kernel'].names,
                         (shardings.scan_axis, *shardings.ffn_outer_kernel))
        for l in ('q', 'k', 'v'):
          self.assertEqual(mha_params[l]['kernel'].names,
                           (shardings.scan_axis, *shardings.mha_qkv_kernel))
        self.assertEqual(mha_params['o']['kernel'].names,
                         (shardings.scan_axis, *shardings.mha_out_kernel))
        self.assertEqual(sa_ln_params['scale'].names,
                         (shardings.scan_axis, *shardings.layernorm))
        self.assertEqual(ffn_ln_params['scale'].names,
                         (shardings.scan_axis, *shardings.layernorm))

  @parameterized.named_parameters(
      {
          'testcase_name': '2d_mask',
          'attention_mask_shape': (2, 3),
          'attention_bias_shape': (1, 3, 3),
          'error_type': ValueError
      },
      {
          'testcase_name': '3d_mask',
          'attention_mask_shape': (2, 1, 3),
          'attention_bias_shape': (1, 3, 3),
          'error_type': ValueError
      },
      {
          'testcase_name': '4d_mask',
          'attention_mask_shape': (2, 1, 3, 3),
          'attention_bias_shape': (5, 3, 3),
          'error_type': ValueError
      },
      {
          'testcase_name': '4d_bias',
          'attention_mask_shape': (2, 1, 1, 3, 3),
          'attention_bias_shape': (5, 2, 3, 3),
          'error_type': ValueError
      },
      {
          'testcase_name': '2d_bias',
          'attention_mask_shape': (2, 1, 1, 3, 3),
          'attention_bias_shape': (1, 3),
          'error_type': ValueError
      },
      {
          'testcase_name': 'bias_mismatch',
          'attention_mask_shape': (2, 1, 1, 3, 3),
          'attention_bias_shape': (5, 3, 9),
          'error_type': ValueError
      },
      {
          'testcase_name': 'mask_mismatch',
          'attention_mask_shape': (2, 1, 1, 3, 4),
          'attention_bias_shape': (5, 3, 3),
          'error_type': ValueError
      },
  )
  def test_encoder_attention_mismatch(self, attention_mask_shape,
                                      attention_bias_shape, error_type):
    shardings = Shardings()
    encoder = transformers.TransformerEncoder(
        d_model=15,
        num_heads=5,
        d_ff=7,
        num_layers=1,
        use_bias=True,
        dropout_rate=0.,
        remat='zero',
        scanned_layers=False,
        scan_axis=0,
        dtype=jnp.float32,
        qk_layernorm=False,
        approximate_gelu=True,
        scan_sharding_axis=shardings.scan_axis,
        ffn_inner_kernel_shardings=shardings.ffn_inner_kernel,
        ffn_outer_kernel_shardings=shardings.ffn_outer_kernel,
        ffn_intermediate_shardings=shardings.ffn_intermediate,
        mha_qkv_kernel_shardings=shardings.mha_qkv_kernel,
        mha_out_kernel_shardings=shardings.mha_out_kernel,
        mha_activation_shardings=shardings.mha_activation,
        layernorm_shardings=shardings.layernorm,
        mha_qkv_dot_general=lax.dot_general,
        mha_out_dot_general=lax.dot_general,
        mha_einsum_dot_general=lax.dot_general,
        ffn_dot_general=lax.dot_general,
        precision=None,
        lora_rank=2,
        lora_scale=0.,
    )
    @jax.jit
    def _run_forward(inputs, attention_mask, attention_bias):
      outputs, _ = encoder.init_with_output(
          rngs={'params': jax.random.key(1)},
          inputs=inputs,
          attention_mask=attention_mask,
          attention_bias=attention_bias,
      )
      return outputs

    with _create_global_mesh():
      inputs = jnp.ones((2, 1, 3, 15))
      attention_mask = jnp.ones(attention_mask_shape)
      attention_bias = jnp.ones(attention_bias_shape)
      with self.assertRaises(error_type):
        _run_forward(inputs, attention_mask, attention_bias)


class MoeTransformerEncoderTest(parameterized.TestCase):

  @property
  def sparse_moe_config(self):
    shardings = MoeShardings()
    return dict(
        d_model=4,
        num_heads=2,
        d_ff=8,
        num_layers=4,
        use_bias=True,
        dropout_rate=0.,
        remat='zero',
        scan_axis=0,
        dtype=jnp.float32,
        lora_rank=2,
        lora_scale=0.,
        num_experts=2,
        max_group_size=16,
        capacity_factor=2.,
        min_expert_capacity=4,
        jitter_noise=0.1,
        comm_dtype=jnp.float32,
        optimize_parallel_comms=False,
        strict_group_size=False,
        router_type='ExpertsChooseMasked',
        num_selected_experts=0,  # Unused with 'ExpertsChooseMasked'.
        router_bias=False,
        split_params=True,
        batch_prioritized_routing=False,  # Unused with 'ExpertsChooseMasked'.
        ignore_padding_tokens=False,
        qk_layernorm=False,
        model_axis_size=1,
        model_axis_name=shardings.ffn_inner_kernel[-1],
        scan_sharding_axis=shardings.scan_axis,
        tokens_shardings=shardings.tokens,
        router_kernel_shardings=shardings.router_kernel,
        ffn_inner_kernel_shardings=shardings.ffn_inner_kernel,
        ffn_outer_kernel_shardings=shardings.ffn_outer_kernel,
        ffn_intermediate_shardings=shardings.ffn_intermediate,
        routed_ffn_intermediate_shardings=shardings.routed_ffn_intermediate,
        mha_qkv_kernel_shardings=shardings.mha_qkv_kernel,
        mha_out_kernel_shardings=shardings.mha_out_kernel,
        mha_activation_shardings=shardings.mha_activation,
        layernorm_shardings=shardings.layernorm,
        mha_qkv_dot_general=lax.dot_general,
        mha_out_dot_general=lax.dot_general,
        mha_einsum_dot_general=lax.dot_general,
        ffn_dot_general=lax.dot_general,
        precision=None,
        approximate_gelu=True,
        router_kwargs=(('router_name', 'default_router'),),
    )

  @property
  def soft_moe_config(self):
    shardings = MoeShardings()
    return dict(
        d_model=4,
        num_heads=2,
        d_ff=8,
        num_layers=4,
        use_bias=True,
        dropout_rate=0.,
        remat='zero',
        scan_axis=0,
        dtype=jnp.float32,
        lora_rank=2,
        lora_scale=0.,
        num_experts=2,
        jitter_noise=0.1,
        expert_capacity=4,
        comm_dtype=jnp.float32,
        optimize_parallel_comms=False,
        split_params=True,
        ignore_padding_tokens=False,
        qk_layernorm=False,
        model_axis_size=1,
        model_axis_name=shardings.ffn_inner_kernel[-1],
        scan_sharding_axis=shardings.scan_axis,
        tokens_shardings=shardings.tokens,
        router_kernel_shardings=shardings.router_kernel,
        ffn_inner_kernel_shardings=shardings.ffn_inner_kernel,
        ffn_outer_kernel_shardings=shardings.ffn_outer_kernel,
        ffn_intermediate_shardings=shardings.ffn_intermediate,
        routed_ffn_intermediate_shardings=shardings.routed_ffn_intermediate,
        mha_qkv_kernel_shardings=shardings.mha_qkv_kernel,
        mha_out_kernel_shardings=shardings.mha_out_kernel,
        mha_activation_shardings=shardings.mha_activation,
        layernorm_shardings=shardings.layernorm,
        mha_qkv_dot_general=lax.dot_general,
        mha_out_dot_general=lax.dot_general,
        mha_einsum_dot_general=lax.dot_general,
        ffn_dot_general=lax.dot_general,
        precision=None,
        approximate_gelu=True,
        router_kwargs=(('router_name', 'default_router'),),
    )

  @parameterized.named_parameters(
      {
          'testcase_name': 'every_layer',
          'num_moe_layers': 4,
          'moe_layers_distribution': 'uniform',
          'scanned_layers': False,
      },
      {
          'testcase_name': 'every_layer_scan',
          'num_moe_layers': 4,
          'moe_layers_distribution': 'uniform',
          'scanned_layers': True,
      },
      {
          'testcase_name': 'every_layer_scan_and_remat',
          'num_moe_layers': 4,
          'moe_layers_distribution': 'uniform',
          'scanned_layers': True,
          'remat': 'full',
      },
      {
          'testcase_name': 'every_layer_scan_quantized_8bits',
          'num_moe_layers': 4,
          'moe_layers_distribution': 'uniform',
          'scanned_layers': True,
          'mha_qkv_dot_general': utils.make_dot_general(AQT_8BITS_CFG),
          'mha_out_dot_general': utils.make_dot_general(AQT_8BITS_CFG),
          'mha_einsum_dot_general': utils.make_dot_general(AQT_8BITS_CFG),
          'ffn_dot_general': utils.make_dot_general(AQT_8BITS_CFG),
      },
      {
          'testcase_name': 'every_layer_scan_lora',
          'num_moe_layers': 4,
          'moe_layers_distribution': 'uniform',
          'scanned_layers': True,
          'lora_rank': 2,
          'lora_scale': 1.,
      },
      {
          'testcase_name': 'every_2_layers',
          'num_moe_layers': 4,
          'moe_layers_distribution': 'uniform',
          'scanned_layers': False,
      },
      {
          'testcase_name': 'every_2_layers_lora',
          'num_moe_layers': 4,
          'moe_layers_distribution': 'uniform',
          'scanned_layers': False,
          'lora_rank': 2,
          'lora_scale': 1.,
      },
      {
          'testcase_name': 'last_2_layers',
          'num_moe_layers': 4,
          'moe_layers_distribution': 'last',
          'scanned_layers': False,
      },
      {
          'testcase_name': 'last_2_layers_scan',
          'num_moe_layers': 2,
          'moe_layers_distribution': 'last',
          'scanned_layers': True,
      },
      {
          'testcase_name': 'last_2_layers_scan_and_remat',
          'num_moe_layers': 2,
          'moe_layers_distribution': 'last',
          'scanned_layers': True,
          'remat': 'full',
      },
  )
  def test_sparse_moe_transformer_encoder(self, **kwargs):
    shd = MoeShardings()
    config = copy.deepcopy(self.sparse_moe_config)
    config.update(kwargs)
    moe_encoder = transformers.SparseMoeTransformerEncoder(**config)
    @jax.jit
    def _run_forward(inputs):
      outputs, variables = moe_encoder.init_with_output(
          rngs={'params': jax.random.key(1),
                'dropout': jax.random.key(2),
                'jitter': jax.random.key(3)},
          inputs=inputs)
      return outputs, variables

    batch_size, n_instances, qkv_length, d_model = 2, 2, 8, 4
    with _create_global_moe_mesh():
      inputs = jnp.ones((batch_size, n_instances, qkv_length, d_model))
      inputs = sharding.shard_array(inputs, shd.tokens)
      outputs, variables = _run_forward(inputs)

    # Assert output shapes
    chex.assert_shape(outputs, (batch_size, n_instances, qkv_length, d_model))

    # Assert variables are propagated properly
    if (kwargs['moe_layers_distribution'] == 'uniform'
        and kwargs['scanned_layers']):
      scanned_params = variables['params']['layer_scan_moe']
      ffn_params = scanned_params['feed_forward']['experts']
      router_params = scanned_params['feed_forward']['router']
      mha_params = scanned_params['self_attention']
      sa_ln_params = scanned_params['layer_norm_sa']
      ffn_ln_params = scanned_params['layer_norm_ffn']
      # Scan axis is 0, hence its sharding name leads
      self.assertEqual(
          router_params['w']['kernel'].names,
          (shd.scan_axis, *shd.router_kernel))
      # Scan is called over the vmapped expert, hence parameter shardings are
      # prepended by expert_axis first and then by scan_axis
      self.assertEqual(
          ffn_params['wi']['kernel'].names,
          (shd.scan_axis, shd.expert_axis, *shd.ffn_inner_kernel))
      self.assertEqual(
          ffn_params['wo']['kernel'].names,
          (shd.scan_axis, shd.expert_axis, *shd.ffn_outer_kernel))
      for l in ('q', 'k', 'v'):
        self.assertEqual(mha_params[l]['kernel'].names,
                         (shd.scan_axis, *shd.mha_qkv_kernel))
      self.assertEqual(mha_params['o']['kernel'].names,
                       (shd.scan_axis, *shd.mha_out_kernel))
      self.assertEqual(sa_ln_params['scale'].names,
                       (shd.scan_axis, *shd.layernorm))
      self.assertEqual(ffn_ln_params['scale'].names,
                       (shd.scan_axis, *shd.layernorm))

  @parameterized.named_parameters(
      {
          'testcase_name': 'every_layer',
          'num_moe_layers': 4,
          'moe_layers_distribution': 'uniform',
          'scanned_layers': False,
      },
      {
          'testcase_name': 'every_layer_scan',
          'num_moe_layers': 4,
          'moe_layers_distribution': 'uniform',
          'scanned_layers': True,
      },
      {
          'testcase_name': 'every_layer_scan_and_remat',
          'num_moe_layers': 4,
          'moe_layers_distribution': 'uniform',
          'scanned_layers': True,
          'remat': 'full',
      },
      {
          'testcase_name': 'every_layer_scan_quantized_8bits',
          'num_moe_layers': 4,
          'moe_layers_distribution': 'uniform',
          'scanned_layers': True,
          'mha_qkv_dot_general': utils.make_dot_general(AQT_8BITS_CFG),
          'mha_out_dot_general': utils.make_dot_general(AQT_8BITS_CFG),
          'mha_einsum_dot_general': utils.make_dot_general(AQT_8BITS_CFG),
          'ffn_dot_general': utils.make_dot_general(AQT_8BITS_CFG),
      },
      {
          'testcase_name': 'every_layer_scan_lora',
          'num_moe_layers': 4,
          'moe_layers_distribution': 'uniform',
          'scanned_layers': True,
          'lora_rank': 2,
          'lora_scale': 1.,
      },
      {
          'testcase_name': 'every_2_layers',
          'num_moe_layers': 4,
          'moe_layers_distribution': 'uniform',
          'scanned_layers': False,
      },
      {
          'testcase_name': 'every_2_layers_lora',
          'num_moe_layers': 4,
          'moe_layers_distribution': 'uniform',
          'scanned_layers': False,
          'lora_rank': 2,
          'lora_scale': 1.,
      },
      {
          'testcase_name': 'last_2_layers',
          'num_moe_layers': 4,
          'moe_layers_distribution': 'last',
          'scanned_layers': False,
      },
      {
          'testcase_name': 'last_2_layers_scan',
          'num_moe_layers': 2,
          'moe_layers_distribution': 'last',
          'scanned_layers': True,
      },
      {
          'testcase_name': 'last_2_layers_scan_and_remat',
          'num_moe_layers': 2,
          'moe_layers_distribution': 'last',
          'scanned_layers': True,
          'remat': 'full',
      },
  )
  def test_soft_moe_transformer_encoder(self, **kwargs):
    shd = MoeShardings()
    config = copy.deepcopy(self.soft_moe_config)
    config.update(kwargs)
    moe_encoder = transformers.SoftMoeTransformerEncoder(**config)
    @jax.jit
    def _run_forward(inputs):
      outputs, variables = moe_encoder.init_with_output(
          rngs={'params': jax.random.key(1),
                'dropout': jax.random.key(2),
                'jitter': jax.random.key(3)},
          inputs=inputs)
      return outputs, variables

    batch_size, n_instances, qkv_length, d_model = 2, 2, 8, 4
    with _create_global_moe_mesh():
      inputs = jnp.ones((batch_size, n_instances, qkv_length, d_model))
      inputs = sharding.shard_array(inputs, shd.tokens)
      outputs, variables = _run_forward(inputs)

    # Assert output shapes
    chex.assert_shape(outputs, (batch_size, n_instances, qkv_length, d_model))

    # Assert variables are propagated properly
    if (kwargs['moe_layers_distribution'] == 'uniform'
        and kwargs['scanned_layers']):
      scanned_params = variables['params']['layer_scan_moe']
      ffn_params = scanned_params['feed_forward']['experts']
      router_params = scanned_params['feed_forward']['router']
      mha_params = scanned_params['self_attention']
      sa_ln_params = scanned_params['layer_norm_sa']
      ffn_ln_params = scanned_params['layer_norm_ffn']
      # Scan axis is 0, hence its sharding name leads
      self.assertEqual(
          router_params['mu'].names, (shd.scan_axis, *shd.router_kernel))
      # Scan is called over the vmapped expert, hence parameter shardings are
      # prepended by expert_axis first and then by scan_axis
      self.assertEqual(
          ffn_params['wi']['kernel'].names,
          (shd.scan_axis, shd.expert_axis, *shd.ffn_inner_kernel))
      self.assertEqual(
          ffn_params['wo']['kernel'].names,
          (shd.scan_axis, shd.expert_axis, *shd.ffn_outer_kernel))
      for l in ('q', 'k', 'v'):
        self.assertEqual(mha_params[l]['kernel'].names,
                         (shd.scan_axis, *shd.mha_qkv_kernel))
      self.assertEqual(mha_params['o']['kernel'].names,
                       (shd.scan_axis, *shd.mha_out_kernel))
      self.assertEqual(sa_ln_params['scale'].names,
                       (shd.scan_axis, *shd.layernorm))
      self.assertEqual(ffn_ln_params['scale'].names,
                       (shd.scan_axis, *shd.layernorm))

  @parameterized.named_parameters(
      {
          'testcase_name': 'wrong_distribution',
          'num_moe_layers': 1,
          'moe_layers_distribution': 'this-does-not-exist',
          'scanned_layers': False,
          'error_type': ValueError,
          'error_regex': 'Wrong value for moe_layers_distribution',
      },
      {
          'testcase_name': 'wrong_num',
          'num_moe_layers': 0,
          'moe_layers_distribution': 'uniform',
          'scanned_layers': False,
          'error_type': ValueError,
          'error_regex': 'Wrong value for num_moe_layers',
      },
      {
          'testcase_name': 'cannot_scan_alternate_layers',
          'num_moe_layers': 2,
          'moe_layers_distribution': 'uniform',
          'scanned_layers': True,
          'error_type': ValueError,
          'error_regex': 'Scanned layers cannot be used with alternated',
      },
      {
          'testcase_name': 'uniform_not_divisor',
          'num_moe_layers': 3,
          'moe_layers_distribution': 'uniform',
          'scanned_layers': False,
          'error_type': ValueError,
          'error_regex': 'num_moe_layers must be a divisor of num_layers',
      },
      {
          'testcase_name': 'too_large_last_num',
          'num_moe_layers': 5,
          'moe_layers_distribution': 'last',
          'scanned_layers': False,
          'error_type': ValueError,
          'error_regex': 'you cannot have num_moe_layers = 5 > 4',
      },
  )
  def test_moe_transformer_raises(self, error_type, error_regex, **kwargs):

    config = copy.deepcopy(self.sparse_moe_config)
    config.update(kwargs)
    moe_encoder = transformers.SparseMoeTransformerEncoder(**config)
    @jax.jit
    def _run_forward(inputs):
      outputs, variables = moe_encoder.init_with_output(
          rngs={'params': jax.random.key(1)}, inputs=inputs)
      return outputs, variables

    batch_size, n_instances, qkv_length, d_model = 2, 2, 8, 4
    with _create_global_moe_mesh():
      inputs = jnp.ones((batch_size, n_instances, qkv_length, d_model))
      with self.assertRaisesRegex(error_type, error_regex):
        _ = _run_forward(inputs)


if __name__ == '__main__':
  absltest.main()
