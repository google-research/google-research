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

"""Tests for embeddings."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np

from imp.max.modeling import embeddings
from imp.max.utils import sharding

_EXPECTED_FLATTENED_RANK = 4
_EMBEDDING_SHARDINGS = ('model', None)  # (vocab, embed)
_LAYERNORM_SHARDINGS = (None,)  # (embed,)


def _create_global_mesh():
  return jax.sharding.Mesh(
      sharding.create_tpu_device_mesh(ici_mesh_shape=(1, 1),
                                      dcn_mesh_shape=(1, 1)),
      ['data', 'model'],
  )


class PosBiasTest(absltest.TestCase):

  def test_get_relative_position_bucket(self):
    relative_position = np.array([[-10, -9, -8, -7, -6], [-5, -4, -3, -2, -1],
                                  [0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    unidirectional_buckets = embeddings.get_relative_position_bucket(
        relative_position, bidirectional=False, num_buckets=4, max_distance=3)
    np.testing.assert_array_equal(
        unidirectional_buckets,
        np.array([[3, 3, 3, 3, 3], [3, 3, 3, 2, 1], [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]]))
    unidirectional_many_buckets = embeddings.get_relative_position_bucket(
        relative_position, bidirectional=False, num_buckets=8, max_distance=10)
    np.testing.assert_array_equal(
        unidirectional_many_buckets,
        np.array([[7, 7, 7, 6, 5], [4, 4, 3, 2, 1], [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]]))
    unidirectional_large_distance = embeddings.get_relative_position_bucket(
        relative_position, bidirectional=False, num_buckets=16, max_distance=9)
    np.testing.assert_array_equal(
        unidirectional_large_distance,
        np.array([[15, 15, 8, 7, 6], [5, 4, 3, 2, 1], [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]]))
    bidirectional_buckets = embeddings.get_relative_position_bucket(
        relative_position, bidirectional=True, num_buckets=4, max_distance=4)
    np.testing.assert_array_equal(
        bidirectional_buckets,
        np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 3, 3, 3, 3],
                  [3, 3, 3, 3, 3]]))
    bidirectional_many_buckets = embeddings.get_relative_position_bucket(
        relative_position, bidirectional=True, num_buckets=15, max_distance=4)
    np.testing.assert_array_equal(
        bidirectional_many_buckets,
        np.array([[6, 6, 6, 6, 6], [6, 6, 3, 2, 1], [0, 8, 9, 10, 13],
                  [13, 13, 13, 13, 13]]))
    bidirectional_large_distance = embeddings.get_relative_position_bucket(
        relative_position, bidirectional=True, num_buckets=16, max_distance=8)
    np.testing.assert_array_equal(
        bidirectional_large_distance,
        np.array([[7, 7, 7, 7, 6], [5, 4, 3, 2, 1], [0, 9, 10, 11, 12],
                  [13, 14, 15, 15, 15]]))

  def test_get_realtive_position_bucket_misconfigured(self):
    relative_position = np.array([0, -1, 1])
    with self.assertRaises(ValueError):
      embeddings.get_relative_position_bucket(
          relative_position, bidirectional=False, num_buckets=8, max_distance=3)
    with self.assertRaises(ValueError):
      embeddings.get_relative_position_bucket(
          relative_position, bidirectional=False, num_buckets=8, max_distance=4)
    with self.assertRaises(ValueError):
      embeddings.get_relative_position_bucket(
          relative_position, bidirectional=True, num_buckets=16, max_distance=4)
    with self.assertRaises(ValueError):
      embeddings.get_relative_position_bucket(
          relative_position, bidirectional=True, num_buckets=16, max_distance=3)


def _get_output_shape(input_shape):
  if len(input_shape) < _EXPECTED_FLATTENED_RANK:
    return (1,) * (_EXPECTED_FLATTENED_RANK - len(input_shape)) + input_shape
  return input_shape


class TemporalPosEncodeTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'temp_pos_embds',
          'input_shape': (27, 2, 5, 4),
          'pos_buckets': 5,
          'dropout_rate': 0.2,
          'deterministic': True,
      }, {
          'testcase_name': 'no_batch_temp_pos_embds',
          'input_shape': (2, 3, 4),
          'pos_buckets': 3,
          'dropout_rate': 0.1,
          'deterministic': True,
      }, {
          'testcase_name': 'no_batch_no_instance_temp_pos_embds',
          'input_shape': (2, 3),
          'pos_buckets': 2,
          'dropout_rate': 0.9,
          'deterministic': True,
      }, {
          'testcase_name': 'no_batch_no_instance_temp_pos_embds_nondet',
          'input_shape': (3, 2),
          'pos_buckets': 3,
          'dropout_rate': 0.5,
          'deterministic': False,
      })
  def test_temp_pos_encode(self,
                           input_shape,
                           pos_buckets,
                           dropout_rate,
                           deterministic):

    embedding = embeddings.TemporalPosEncode(
        hidden_size=input_shape[-1],
        pos_buckets=pos_buckets,
        dropout_rate=dropout_rate,
        embedding_shardings=_EMBEDDING_SHARDINGS,
        layernorm_shardings=_LAYERNORM_SHARDINGS,
    )

    @jax.jit
    def _run_forward(inputs):
      variables = embedding.init(
          rngs={'params': jax.random.key(1)}, inputs=inputs)
      outputs = embedding.apply(
          variables=variables,
          rngs={'dropout': jax.random.key(2)},
          inputs=inputs,
          deterministic=deterministic,
      )
      return outputs, variables

    inputs = jnp.ones(input_shape)
    with _create_global_mesh():
      outputs, variables = _run_forward(inputs)

    chex.assert_shape(outputs, _get_output_shape(input_shape))

    # Assert shardings are propagated properly
    params = variables['params']
    self.assertEqual(
        params['temporal_postition_embeddings']['embedding'].names,
        _EMBEDDING_SHARDINGS)
    self.assertEqual(params['layer_norm']['scale'].names, _LAYERNORM_SHARDINGS)


class SpectroTemporalPosEncodeTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'spectro_temp_pos_embds',
          'batch_size': None,
          'pos_buckets': (2, 3),
          'hidden_size': 20,
          'dropout_rate': 0.1,
          'deterministic': True,
      }, {
          'testcase_name': 'large_spectro_temp_pos_embds',
          'batch_size': 200,
          'pos_buckets': (50, 20),
          'hidden_size': 100,
          'dropout_rate': 0.9,
          'deterministic': True,
      })
  def test_spectro_temp_pos_encode(self,
                                   pos_buckets,
                                   batch_size,
                                   hidden_size,
                                   dropout_rate,
                                   deterministic):
    embedding = embeddings.SpectroTemporalPosEncode(
        hidden_size=hidden_size,
        pos_buckets=pos_buckets,
        dropout_rate=dropout_rate,
        embedding_shardings=_EMBEDDING_SHARDINGS,
        layernorm_shardings=_LAYERNORM_SHARDINGS,
    )

    @jax.jit
    def _run_forward(inputs):
      variables = embedding.init(
          rngs={'params': jax.random.key(1)}, inputs=inputs)
      outputs = embedding.apply(
          variables=variables,
          rngs={'dropout': jax.random.key(2)},
          inputs=inputs, deterministic=deterministic)
      return outputs, variables

    token_size = np.prod(pos_buckets)
    input_shape = (3, token_size, hidden_size)
    if batch_size:
      input_shape = (batch_size,) + input_shape
    inputs = jnp.ones(input_shape)
    with _create_global_mesh():
      outputs, variables = _run_forward(inputs)
    chex.assert_shape(outputs, _get_output_shape(input_shape))

    # Assert shardings are propagated properly
    params = variables['params']
    self.assertEqual(
        params['temporal_postition_embeddings']['embedding'].names,
        _EMBEDDING_SHARDINGS)
    self.assertEqual(
        params['spectoral_postition_embeddings']['embedding'].names,
        _EMBEDDING_SHARDINGS)
    self.assertEqual(params['layer_norm']['scale'].names, _LAYERNORM_SHARDINGS)

  @parameterized.named_parameters(('too_few', (2,), 10, IndexError),
                                  ('too_many', (2, 4, 5), 6, TypeError))
  def test_wrong_buckets(self, pos_buckets, hidden_size, error_type):

    @jax.jit
    def _run_forward(inputs):
      embedding = embeddings.SpectroTemporalPosEncode(
          hidden_size=hidden_size, pos_buckets=pos_buckets, dropout_rate=0.1)
      variables = embedding.init(
          rngs={'params': jax.random.key(1)}, inputs=inputs)
      return embedding.apply(
          variables=variables,
          rngs={'dropout': jax.random.key(2)},
          inputs=inputs)

    token_size = np.prod(pos_buckets)
    input_shape = (5, 3, token_size, hidden_size)
    inputs = jnp.ones(input_shape)
    with self.assertRaises(error_type):
      _run_forward(inputs)


class SpatioTemporalPosEncodeTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'spatio_temp_pos_embds',
          'batch_size': None,
          'pos_buckets': (2, 3, 5),
          'hidden_size': 20,
          'dropout_rate': 0.1,
          'deterministic': True,
      }, {
          'testcase_name': 'large_spatio_temp_pos_embds',
          'batch_size': 200,
          'pos_buckets': (50, 20, 15),
          'hidden_size': 100,
          'dropout_rate': 0.9,
          'deterministic': True,
      })
  def test_spatio_temp_pos_encode(self,
                                  pos_buckets,
                                  batch_size,
                                  hidden_size,
                                  dropout_rate,
                                  deterministic):

    embedding = embeddings.SpatioTemporalPosEncode(
        hidden_size=hidden_size,
        pos_buckets=pos_buckets,
        dropout_rate=dropout_rate,
        embedding_shardings=_EMBEDDING_SHARDINGS,
        layernorm_shardings=_LAYERNORM_SHARDINGS,
    )
    @jax.jit
    def _run_forward(inputs):
      variables = embedding.init(
          rngs={'params': jax.random.key(1)}, inputs=inputs)
      outputs = embedding.apply(
          variables=variables,
          rngs={'dropout': jax.random.key(2)},
          inputs=inputs, deterministic=deterministic)
      return outputs, variables

    token_size = np.prod(pos_buckets)
    input_shape = (3, token_size, hidden_size)
    if batch_size:
      input_shape = (batch_size,) + input_shape
    inputs = jnp.ones(input_shape)
    with _create_global_mesh():
      outputs, variables = _run_forward(inputs)
    chex.assert_shape(outputs, _get_output_shape(input_shape))

    # Assert shardings are propagated properly
    params = variables['params']
    for layer_name in ['temporal_postition_embeddings',
                       'vertical_postition_embeddings',
                       'horizontal_postition_embeddings']:
      self.assertEqual(
          params[layer_name]['embedding'].names, _EMBEDDING_SHARDINGS)
    self.assertEqual(params['layer_norm']['scale'].names, _LAYERNORM_SHARDINGS)

  @parameterized.named_parameters(('too_few', (2, 3), 10, IndexError),
                                  ('too_many', (2, 3, 4, 5), 6, TypeError))
  def test_wrong_buckets(self, pos_buckets, hidden_size, error_type):

    @jax.jit
    def _run_forward(inputs):
      embedding = embeddings.SpatioTemporalPosEncode(
          hidden_size=hidden_size, pos_buckets=pos_buckets, dropout_rate=0.1)
      variables = embedding.init(
          rngs={'params': jax.random.key(1)}, inputs=inputs)
      return embedding.apply(
          variables=variables,
          rngs={'dropout': jax.random.key(2)},
          inputs=inputs)

    token_size = np.prod(pos_buckets)
    input_shape = (5, 3, token_size, hidden_size)
    inputs = jnp.ones(input_shape)
    with self.assertRaises(error_type):
      _run_forward(inputs)


class PositionBias1DTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('pos_bias_1d', 5, 8, 10, True, 10, 7),
      ('pos_bias_1d_no_klen', 50, 100, 100, True, 2, None),
      ('pos_bias_1d_no_klen_uni', 2, 5, 20, False, 10, None),
      ('pos_bias_1d_uni', 50, 100, 100, False, 13, 11))
  def test_position_bias_1d(self, num_heads, num_relative_buckets,
                            max_relative_distnace, bidirectional, qlen, klen):

    position_bias = embeddings.PositionBias1D(
        num_heads=num_heads,
        num_relative_buckets=num_relative_buckets,
        max_relative_distance=max_relative_distnace,
        bidirectional=bidirectional,
        embedding_shardings=_EMBEDDING_SHARDINGS,
    )

    @jax.jit
    def _run_forward():
      variables = position_bias.init(
          rngs={'params': jax.random.key(1)}, qlen=qlen, klen=klen)
      outputs = position_bias.apply(variables=variables, qlen=qlen, klen=klen)
      return outputs, variables

    with _create_global_mesh():
      outputs, variables = _run_forward()
    chex.assert_shape(outputs, (num_heads, qlen, klen or qlen))

    # Assert shardings are propagated properly
    params = variables['params']
    self.assertEqual(
        params['relative_temporal_attention_bias']['embedding'].names,
        _EMBEDDING_SHARDINGS)


class PositionBias3DTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('pos_bias_3d', 5, (4, 6, 8), (10, 20, 30), True, 10, 2, 20),
      ('pos_bias_3d_uni', 2, (5, 7, 9), (20, 15, 10), False, 10, 7, 5),
  )
  def test_position_bias_3d(self, num_heads, num_relative_buckets,
                            max_relative_distnace, bidirectional, tlen, vlen,
                            hlen):

    position_bias = embeddings.PositionBias3D(
        num_heads=num_heads,
        num_relative_buckets=num_relative_buckets,
        max_relative_distance=max_relative_distnace,
        bidirectional=bidirectional,
        embedding_shardings=_EMBEDDING_SHARDINGS,
    )

    @jax.jit
    def _run_forward():
      variables = position_bias.init(
          rngs={'params': jax.random.key(1)},
          tlen=tlen,
          vlen=vlen,
          hlen=hlen)
      outputs = position_bias.apply(variables=variables,
                                    tlen=tlen, vlen=vlen, hlen=hlen)
      return outputs, variables

    with _create_global_mesh():
      outputs, variables = _run_forward()
    attention_size = tlen * vlen * hlen
    chex.assert_shape(outputs, (num_heads, attention_size, attention_size))

    # Assert shardings are propagated properly
    params = variables['params']
    for layer_name in ['relative_temporal_attention_bias',
                       'relative_vertical_attention_bias',
                       'relative_horizontal_attention_bias']:
      self.assertEqual(
          params[layer_name]['embedding'].names, _EMBEDDING_SHARDINGS)

  @parameterized.named_parameters(
      ('pos_bias_3d_too_few_buckets', (4, 6), (10, 20, 30)),
      ('pos_bias_3d_too_few_distances', (4, 6, 8), (10, 20)))
  def test_position_bias_3d_wrong_input(self, num_relative_buckets,
                                        max_relative_distnace):

    tlen = vlen = hlen = num_heads = 10

    @jax.jit
    def _run_forward():
      bias = embeddings.PositionBias3D(
          num_heads=num_heads,
          num_relative_buckets=num_relative_buckets,
          max_relative_distance=max_relative_distnace,
          bidirectional=True)
      variables = bias.init(
          rngs={'params': jax.random.key(1)},
          tlen=tlen,
          vlen=vlen,
          hlen=hlen)
      return bias.apply(variables=variables, tlen=tlen, vlen=vlen, hlen=hlen)

    with self.assertRaises(IndexError):
      _run_forward()


if __name__ == '__main__':
  absltest.main()
