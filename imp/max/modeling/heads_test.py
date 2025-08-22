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

"""Tests for heads."""
import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp

from imp.max.core import constants
from imp.max.modeling import heads
from imp.max.utils import sharding


@dataclasses.dataclass
class Shardings:
  tokens = ('data', None, None, None)  # (b, n, t, d)
  mha_activation = ('data', None, None, 'model', None)  # (b, n, t, h, d)
  mha_qkv_kernel = (None, 'model', None)  # (dim, heads, d_h)
  mha_out_kernel = (None, 'model', None)  # (heads, d_h, dim)
  ffn_inner_kernel = (None, 'model')  # (d, d_ff)
  ffn_outer_kernel = ('model', None)  # (d_ff, d)
  ffn_intermediate = ('data', None, None, 'model')  # (b, n, t, d_ff)
  probe_kernel = (None, None)  # (1, d)
  probe_activation = ('data', None, None, None)  # (b, n, t, d)


def _create_global_mesh():
  return jax.sharding.Mesh(
      sharding.create_tpu_device_mesh(ici_mesh_shape=(1, 1),
                                      dcn_mesh_shape=(1, 1)),
      ['data', 'model'],
  )


class HeadsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'baseline',
          'batch_size': 1,
          'n_instances': 2,
          'q_length': 3,
          'd_head': 7,
          'mlp_dim': None,
          'num_heads': 5,
          'buggy': False,
          'use_bias': False,
          'dropout_rate': 0.1,
          'deterministic': False,
      }, {
          'testcase_name': 'explicit_mlp_dim',
          'batch_size': 1,
          'n_instances': 2,
          'q_length': 3,
          'd_head': 7,
          'mlp_dim': 11,
          'num_heads': 5,
          'buggy': False,
          'use_bias': False,
          'dropout_rate': 0.1,
          'deterministic': False,
      }, {
          'testcase_name': 'buggy',
          'batch_size': 1,
          'n_instances': 2,
          'q_length': 3,
          'd_head': 7,
          'mlp_dim': None,
          'num_heads': 5,
          'buggy': True,
          'use_bias': False,
          'dropout_rate': 0.1,
          'deterministic': False,
      }, {
          'testcase_name': 'deterministic',
          'batch_size': 1,
          'n_instances': 2,
          'q_length': 3,
          'd_head': 7,
          'mlp_dim': None,
          'num_heads': 5,
          'buggy': False,
          'use_bias': False,
          'dropout_rate': 0.1,
          'deterministic': True,
      }, {
          'testcase_name': 'lora',
          'batch_size': 1,
          'n_instances': 2,
          'q_length': 3,
          'd_head': 7,
          'mlp_dim': None,
          'num_heads': 5,
          'buggy': False,
          'use_bias': False,
          'dropout_rate': 0.1,
          'deterministic': False,
          'lora_rank': 2,
          'lora_scale': 1.0,
      })
  def test_map_head(self,
                    batch_size,
                    n_instances,
                    q_length,
                    d_head,
                    mlp_dim,
                    num_heads,
                    buggy,
                    use_bias,
                    dropout_rate,
                    deterministic,
                    lora_rank=2,
                    lora_scale=0.):
    shardings = Shardings()
    map_head = heads.MAPHead(
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        buggy=buggy,
        use_bias=use_bias,
        dropout_rate=dropout_rate,
        lora_rank=lora_rank,
        lora_scale=lora_scale,
        mha_qkv_kernel_shardings=shardings.mha_qkv_kernel,
        mha_out_kernel_shardings=shardings.mha_out_kernel,
        mha_activation_shardings=shardings.mha_activation,
        ffn_inner_kernel_shardings=shardings.ffn_inner_kernel,
        ffn_outer_kernel_shardings=shardings.ffn_outer_kernel,
        ffn_intermediate_shardings=shardings.ffn_intermediate,
        probe_kernel_shardings=shardings.probe_kernel,
        probe_activation_shardings=shardings.probe_activation,
    )

    @jax.jit
    def _run_forward(inputs):
      variables = map_head.init(
          rngs={'params': jax.random.key(1)},
          x=inputs)
      outputs = map_head.apply(
          variables=variables,
          rngs={'dropout': jax.random.key(2)},
          x=inputs,
          deterministic=deterministic)
      return outputs, variables

    with _create_global_mesh():
      inputs = jnp.ones(
          (batch_size, n_instances, q_length, d_head * num_heads))
      inputs = sharding.shard_array(inputs, shardings.tokens)
      outputs, variables = _run_forward(inputs)

    chex.assert_shape(outputs, (batch_size, n_instances, d_head * num_heads))

    # Assert shardings are propagated properly
    params = variables['params']
    for layer_name, layer_shardings in zip(
        ['q', 'k', 'v', 'o'],
        [*([shardings.mha_qkv_kernel]*3), shardings.mha_out_kernel]
    ):
      self.assertEqual(
          params['cross_attention'][layer_name]['kernel'].names,
          layer_shardings,
      )
    for layer_name, layer_shardings in zip(
        ['wi', 'wo'],
        [shardings.ffn_inner_kernel, shardings.ffn_outer_kernel]
    ):
      self.assertEqual(
          params['feed_forward'][layer_name]['kernel'].names, layer_shardings)
    self.assertEqual(params['probe'].names, shardings.probe_kernel)

  @parameterized.named_parameters(
      {
          'testcase_name': 'special_token',
          'batch_size': 1,
          'n_instances': 2,
          'seq_length': 3,
          'd_model': 4,
          'aggregation_type': constants.AggregationType.SPECIAL_TOKEN,
      }, {
          'testcase_name': 'global_average_pool',
          'batch_size': 1,
          'n_instances': 2,
          'seq_length': 3,
          'd_model': 4,
          'aggregation_type': constants.AggregationType.GLOBAL_AVERAGE_POOL,
      }, {
          'testcase_name': 'global_sum_pool',
          'batch_size': 1,
          'n_instances': 2,
          'seq_length': 3,
          'd_model': 4,
          'aggregation_type': constants.AggregationType.GLOBAL_SUM_POOL,
      }, {
          'testcase_name': 'global_max_pool',
          'batch_size': 1,
          'n_instances': 2,
          'seq_length': 3,
          'd_model': 4,
          'aggregation_type': constants.AggregationType.GLOBAL_MAX_POOL,
      }, {
          'testcase_name': 'wrong_aggregation_type',
          'batch_size': 1,
          'n_instances': 2,
          'seq_length': 3,
          'd_model': 4,
          'aggregation_type': (
              constants.AggregationType.MULTI_HEAD_ATTENTION_POOL),
      }, {
          'testcase_name': 'input_shape',
          'batch_size': 3,
          'n_instances': 4,
          'seq_length': 2,
          'd_model': 5,
          'aggregation_type': constants.AggregationType.SPECIAL_TOKEN,
      })
  def test_aggregator_head(
      self,
      batch_size,
      n_instances,
      seq_length,
      d_model,
      aggregation_type):
    @jax.jit
    def _run_forward(inputs):
      return heads.NonParametricAggregatorHead(
          aggregation_type=aggregation_type)(inputs=inputs)

    model_outputs = jnp.ones(
        (batch_size, n_instances, seq_length, d_model))

    if aggregation_type == constants.AggregationType.MULTI_HEAD_ATTENTION_POOL:
      with self.assertRaises(NotImplementedError):
        _run_forward(model_outputs)
    else:
      outputs = _run_forward(model_outputs)
      chex.assert_shape(
          outputs[constants.DataFeatureName.FEATURES_AGG],
          (batch_size, n_instances, d_model))

      # Special token reduces the temporal dimension by 1
      if aggregation_type == constants.AggregationType.SPECIAL_TOKEN:
        seq_length -= 1

      chex.assert_shape(
          outputs[constants.DataFeatureName.FEATURES],
          (batch_size, n_instances, seq_length, d_model))

if __name__ == '__main__':
  absltest.main()
