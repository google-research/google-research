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

"""Tests for attention modules."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp

from imp.max.modeling import attention
from imp.max.utils import sharding


_TOKENS_SHARDINGS = ('data', None, None, None)  # (b, n, t, d)
_ATT_MASK_SHARDINGS = ('data', None, None, None, None)  # (b, n, 1, t, t)
_BIAS_SHARDINGS = (None, None, 'model', None, None)  # (1, 1, h, t, t)
_ACTIVATION_SHARDINGS = ('data', None, None, 'model', None)  # (b, n, t, h, d)
_QKV_KERNEL_SHARDINGS = (None, 'model', None)  # (dim, heads, d_h)
_OUT_KERNEL_SHARDINGS = (None, 'model', None)  # (heads, d_h, dim)
_LAYERNORM_SHARDNIGS = (None,)


def _create_global_mesh():
  return jax.sharding.Mesh(
      sharding.create_tpu_device_mesh(ici_mesh_shape=(1, 1),
                                      dcn_mesh_shape=(1, 1)),
      ['data', 'model'],
  )


class MultiHeadAttentionTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'baseline',
          'batch_size': 1,
          'n_instances': 2,
          'q_length': 3,
          'kv_length': None,
          'd_head': 7,
          'num_heads': 4,
      }, {
          'testcase_name': 'baseline_decode',
          'batch_size': 1,
          'n_instances': 2,
          'q_length': 1,  # query length should be 1 in decode mode
          'kv_length': 3,
          'd_head': 7,
          'num_heads': 4,
          'decode': True,
      }, {
          'testcase_name': 'kv_specified',
          'batch_size': 10,
          'n_instances': 20,
          'q_length': 30,
          'kv_length': 50,
          'd_head': 7,
          'num_heads': 8,
      }, {
          'testcase_name': 'non_deterministic',
          'batch_size': 1,
          'n_instances': 2,
          'q_length': 3,
          'kv_length': None,
          'd_head': 7,
          'num_heads': 4,
          'dropout_rate': 0.5,
          'deterministic': False
      }, {
          'testcase_name': 'attention_mask',
          'batch_size': 10,
          'n_instances': 2,
          'q_length': 3,
          'kv_length': None,
          'd_head': 7,
          'num_heads': 4,
          'use_attention_mask': True
      }, {
          'testcase_name': 'attention_bias',
          'batch_size': 10,
          'n_instances': 20,
          'q_length': 30,
          'kv_length': 40,
          'd_head': 7,
          'num_heads': 4,
          'use_attention_bias': True
      }, {
          'testcase_name': 'attention_bias_and_mask',
          'batch_size': 10,
          'n_instances': 20,
          'q_length': 30,
          'kv_length': 40,
          'd_head': 7,
          'num_heads': 4,
          'use_attention_mask': True,
          'use_attention_bias': True
      }, {
          'testcase_name': 'd_model_nondivisible_num_heads',
          'batch_size': 1,
          'n_instances': 2,
          'q_length': 3,
          'kv_length': None,
          'd_head': 7,
          'num_heads': 4,
          'd_model': 40,  # d_model != d_head * num_heads
      }, {
          'testcase_name': 'qk_layernorm',
          'batch_size': 1,
          'n_instances': 2,
          'q_length': 3,
          'kv_length': None,
          'd_head': 7,
          'num_heads': 4,
          'qk_layernorm': True
      }, {
          'testcase_name': 'lora',
          'batch_size': 1,
          'n_instances': 2,
          'q_length': 3,
          'kv_length': None,
          'd_head': 7,
          'num_heads': 4,
          'qk_layernorm': True,
          'lora_rank': 2,
          'lora_scale': 1.0
      })
  def test_multihead_attention(self,
                               batch_size,
                               n_instances,
                               q_length,
                               kv_length,
                               d_head,
                               num_heads,
                               d_model=None,
                               dropout_rate=0.1,
                               use_attention_mask=False,
                               use_attention_bias=False,
                               deterministic=True,
                               qk_layernorm=False,
                               lora_rank=2,
                               lora_scale=0.,
                               decode=False):
    if d_model is None:
      d_model = d_head * num_heads

    mha = attention.MultiHeadAttention(
        d_head=d_head,
        num_heads=num_heads,
        d_model=d_model,
        dropout_rate=dropout_rate,
        qk_layernorm=qk_layernorm,
        lora_rank=lora_rank,
        lora_scale=lora_scale,
        qkv_kernel_shardings=_QKV_KERNEL_SHARDINGS,
        out_kernel_shardings=_OUT_KERNEL_SHARDINGS,
        activation_shardings=_ACTIVATION_SHARDINGS,
        layernorm_shardings=_LAYERNORM_SHARDNIGS,
    )

    @jax.jit
    def _run_forward(queries, kv, attention_mask, attention_bias):
      mutable = ['cache'] if decode else []
      variables = mha.init(
          rngs={'params': jax.random.key(1)},
          mutable=mutable + ['params'],
          query=queries,
          key=kv,
          value=kv,
          decode=decode,
      )
      outputs = mha.apply(
          variables=variables,
          rngs={'dropout': jax.random.key(2)},
          mutable=mutable,
          query=queries,
          key=kv,
          value=kv,
          attention_mask=attention_mask,
          attention_bias=attention_bias,
          deterministic=deterministic,
          decode=decode,
      )
      return outputs, variables

    with _create_global_mesh():
      # Prepare inputs and masks/biases
      queries = jnp.ones((batch_size, n_instances, q_length, d_head))
      queries = sharding.shard_array(queries, _TOKENS_SHARDINGS)
      if kv_length is None:
        kv_length = q_length
      kv = jnp.ones((batch_size, n_instances, kv_length, d_head))
      kv = sharding.shard_array(kv, _TOKENS_SHARDINGS)
      if use_attention_mask:
        attention_mask = jnp.ones(
            (batch_size, n_instances, 1, q_length, kv_length))
        attention_mask = sharding.shard_array(
            attention_mask, _ATT_MASK_SHARDINGS)
      else:
        attention_mask = None
      if use_attention_bias:
        attention_bias = jnp.ones((1, 1, num_heads, q_length, kv_length))
        attention_bias = sharding.shard_array(attention_bias, _BIAS_SHARDINGS)
      else:
        attention_bias = None

      queries = sharding.shard_array(queries, _TOKENS_SHARDINGS)
      kv = sharding.shard_array(kv, _TOKENS_SHARDINGS)
      outputs, variables = _run_forward(
          queries, kv, attention_mask, attention_bias)
      if decode:
        outputs, mutables = outputs
      else:
        outputs, mutables = outputs[0], {}

    expected_output_dim = (
        d_model if d_model is not None else d_head * num_heads)
    chex.assert_shape(
        outputs, (batch_size, n_instances, q_length, expected_output_dim))

    if decode:
      self.assertEqual(mutables['cache']['cache_index'], 1)
      chex.assert_shape(
          mutables['cache']['cached_key'],
          (batch_size, n_instances, num_heads, d_head, kv_length),
      )
      chex.assert_shape(
          mutables['cache']['cached_value'],
          (batch_size, n_instances, num_heads, d_head, kv_length),
      )

    # Assert shardings are propagated properly
    for l in ('q', 'k', 'v'):
      self.assertEqual(
          variables['params'][l]['kernel'].names, _QKV_KERNEL_SHARDINGS)
    self.assertEqual(
        variables['params']['o']['kernel'].names, _OUT_KERNEL_SHARDINGS)

  @parameterized.named_parameters(
      {
          'testcase_name': 'efficient_attention',
          'batch_size': 1,
          'n_instances': 2,
          'q_length': 3,
          'kv_length': None,
          'd_head': 7,
          'num_heads': 5,
      },
      )
  def test_efficient_multihead_attention(
      self,
      batch_size,
      n_instances,
      q_length,
      kv_length,
      d_head,
      num_heads,
      d_model=None,
      dropout_rate=0.1,
      deterministic=True,
      qk_layernorm=False,
  ):
    if d_model is None:
      d_model = d_head * num_heads

    mha = attention.MultiHeadAttention(
        d_head=d_head,
        num_heads=num_heads,
        d_model=d_model,
        dropout_rate=dropout_rate,
        qk_layernorm=qk_layernorm,
        efficient_attention=True,
        qkv_kernel_shardings=_QKV_KERNEL_SHARDINGS,
        out_kernel_shardings=_OUT_KERNEL_SHARDINGS,
        activation_shardings=_ACTIVATION_SHARDINGS,
        layernorm_shardings=_LAYERNORM_SHARDNIGS,
    )

    @jax.jit
    def _run_forward(queries, kv):
      variables = mha.init(
          rngs={'params': jax.random.key(1)},
          query=queries,
          key=kv,
          value=kv,
      )
      return mha.apply(
          variables=variables,
          rngs={'dropout': jax.random.key(2)},
          query=queries,
          key=kv,
          value=kv,
          deterministic=deterministic,
      )

    with _create_global_mesh():
      queries = jnp.ones((batch_size, n_instances, q_length, d_head))
      queries = sharding.shard_array(queries, _TOKENS_SHARDINGS)
      if kv_length is None:
        kv_length = q_length
      kv = jnp.ones((batch_size, n_instances, kv_length, d_head))
      kv = sharding.shard_array(kv, _TOKENS_SHARDINGS)
      outputs = _run_forward(queries, kv)

    expected_output_dim = d_model if d_model is not None else d_head * num_heads
    chex.assert_shape(
        outputs, (batch_size, n_instances, q_length, expected_output_dim)
    )

  def test_multihead_attention_fail(self):

    @jax.jit
    def _run_forward(inputs):
      mha = attention.MultiHeadAttention(d_head=10, num_heads=11, d_model=110)
      variables = mha.init(
          rngs={'params': jax.random.key(1)},
          query=inputs,
          key=inputs,
          value=inputs)
      return mha.apply(
          variables=variables, rngs={}, query=inputs, key=inputs, value=inputs)

    inputs = jnp.ones((2, 3, 4))
    with self.assertRaises(ValueError):
      _run_forward(inputs)

  def test_multihead_attention_sharding_fail(self):

    @jax.jit
    def _run_forward(inputs):
      mha = attention.MultiHeadAttention(d_head=7,
                                         num_heads=2,
                                         d_model=14,
                                         qkv_kernel_shardings=(None, 'model'))
      variables = mha.init(
          rngs={'params': jax.random.key(1)},
          query=inputs,
          key=inputs,
          value=inputs)
      return mha.apply(
          variables=variables, rngs={}, query=inputs, key=inputs, value=inputs)

    with _create_global_mesh():
      inputs = jnp.ones((2, 3, 4))
      with self.assertRaises(ValueError):
        _run_forward(inputs)


if __name__ == '__main__':
  absltest.main()
