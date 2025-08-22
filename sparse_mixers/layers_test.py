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

"""Tests for layers."""

import functools
from typing import Any, Dict, Mapping
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import flax
import jax
from jax import numpy as jnp
import ml_collections
import numpy as np

from sparse_mixers import layers
from sparse_mixers import routing

jax.config.update("jax_threefry_partitionable", False)

# Type Stubs
FrozenDict = flax.core.frozen_dict.FrozenDict
Layer = Any
PRNGKey = layers.PRNGKey


NUM_CLASSES = 2


def init_layer_variables(
    key, module,
    init_batch):
  """Initialize layer parameters."""
  params_key, dropout_key, jitter_key = jax.random.split(key, num=3)

  return module.init(
      {
          "params": params_key,
          "dropout": dropout_key,
          "jitter": jitter_key
      }, **init_batch)


class FeedForwardLayerTest(absltest.TestCase):

  def test_feed_forward_layer(self):
    batch_size = 3
    max_seq_length = 16
    hidden_dim = 12
    rng = jax.random.PRNGKey(0)

    feed_forward_layer = layers.FeedForwardLayer(d_ff=8, dropout_rate=0.1)
    init_batch = {
        "input_emb": jnp.ones((1, max_seq_length, hidden_dim), jnp.float32)
    }
    params = init_layer_variables(rng, feed_forward_layer, init_batch)["params"]

    expected_keys = {"intermediate", "output"}
    self.assertEqual(params.keys(), expected_keys)

    rng, init_rng = jax.random.split(rng)
    input_emb = jax.random.randint(
        init_rng, (batch_size, max_seq_length, hidden_dim), minval=0, maxval=10)
    outputs = feed_forward_layer.apply({"params": params},
                                       rngs={"dropout": rng},
                                       input_emb=input_emb)

    self.assertEqual(outputs.shape, (batch_size, max_seq_length, hidden_dim))


class AttentionLayerTest(parameterized.TestCase):

  def test_attention(self):
    batch_size = 2
    input_seq_length = 2
    hidden_dim = 4
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    attention_layer = layers.AttentionLayer(
        num_heads=2,
        d_model=hidden_dim,
        dropout_rate=0.1,
        pad_id=3)
    inputs = {
        "input_emb":
            jax.random.uniform(
                init_rng, (batch_size, input_seq_length, hidden_dim),
                minval=0,
                maxval=1),
        "input_ids":
            jax.random.randint(
                init_rng, (batch_size, input_seq_length), minval=0, maxval=10),
    }
    init_layer_variables(rng, attention_layer, inputs)

    params = init_layer_variables(rng, attention_layer, inputs)["params"]

    expected_keys = {"self"}
    self.assertEqual(params.keys(), expected_keys)

    outputs = attention_layer.apply({"params": params},
                                    rngs={"dropout": rng},
                                    **inputs)

    expected_output = [
        [
            [2.0673181e-04, 1.0010562e-03, -4.1518142e-04, -5.1020604e-04],
            [1.8721429e-04, 1.5502818e-03, -6.3518737e-04, -8.5086958e-04],
        ],
        [
            [2.5694639e-05, 7.4733491e-04, -3.4021208e-04, -4.5509319e-04],
            [4.7545444e-05, 1.1858117e-03, -4.7121668e-04, -5.4524373e-04],
        ],
    ]
    np.testing.assert_allclose(outputs, expected_output, rtol=1e-6, atol=1e-6)


class MixingLayersTest(parameterized.TestCase):

  def test_init_fourier(self):
    max_seq_length = 8
    d_model = 4
    init = functools.partial(
        layers._init_fourier_transform,
        input_seq_length=max_seq_length,
        d_model=d_model,
        precision=jax.lax.Precision.DEFAULT)

    fft_fourier = init(use_fft=True)
    matmul_fourier = init(use_fft=False)
    inputs = jax.random.uniform(
        jax.random.PRNGKey(42), (max_seq_length, d_model), minval=-2, maxval=2)
    # Expected output for use_fft=True and use_fft=False should be the same.
    np.testing.assert_allclose(
        fft_fourier(inputs), matmul_fourier(inputs), rtol=1e-6, atol=1e-6)

  def test_init_fourier_bad_long_seq(self):
    with self.assertRaisesRegex(
        ValueError,
        "must be a power of 2 to take advantage of FFT optimizations"):
      _ = layers._init_fourier_transform(
          use_fft=True,
          input_seq_length=8194,
          d_model=2,
          precision=jax.lax.Precision.DEFAULT)

  def test_fourier_transform(self):
    batch_size = 2
    input_seq_length = 4
    hidden_dim = 2
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    fourier_layer = layers.FourierTransform(use_fft=False)
    input_emb = jax.random.uniform(
        init_rng, (batch_size, input_seq_length, hidden_dim),
        minval=0,
        maxval=10)
    init_layer_variables(rng, fourier_layer, {
        "input_emb": input_emb,
    })

    # FourierTransform layer has no learnable params.
    outputs = fourier_layer.apply({"params": {}}, input_emb=input_emb)

    expected_output = [
        [
            [35.555874, -1.1852341],
            [-3.5240781, 3.302263],
            [0.59148693, 12.0786085],
            [-3.5240781, 3.302263],
        ],
        [
            [35.518036, 15.53099],
            [-9.507425, -3.950809],
            [7.7935295, -8.306814],
            [-9.507425, -3.950809],
        ],
    ]
    np.testing.assert_allclose(outputs, expected_output, rtol=1e-6, atol=1e-6)

  def test_hartley_transform(self):
    batch_size = 2
    input_seq_length = 4
    hidden_dim = 2
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    hartley_layer = layers.HartleyTransform(use_fft=False)
    input_emb = jax.random.uniform(
        init_rng, (batch_size, input_seq_length, hidden_dim),
        minval=0,
        maxval=10)
    init_layer_variables(rng, hartley_layer, {
        "input_emb": input_emb,
    })

    # HartleyTransform layer has no learnable params.
    outputs = hartley_layer.apply({"params": {}}, input_emb=input_emb)

    expected_output = [
        [
            [35.555874, -1.1852341],
            [-6.725869, 11.0328245],
            [0.59148693, 12.0786085],
            [-0.32228732, -4.428298],
        ],
        [
            [35.518036, 15.53099],
            [-9.823376, -5.477189],
            [7.7935295, -8.306814],
            [-9.191475, -2.4244287],
        ],
    ]
    np.testing.assert_allclose(outputs, expected_output, rtol=1e-6, atol=1e-6)

  def test_linear_transform(self):
    batch_size = 1
    input_seq_length = 2
    hidden_dim = 2
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    linear_layer = layers.LinearTransform()
    input_emb = jax.random.uniform(
        init_rng, (batch_size, input_seq_length, hidden_dim),
        minval=0,
        maxval=13)
    params = init_layer_variables(rng, linear_layer, {
        "input_emb": input_emb,
    })["params"]

    expected_keys = {"hidden_kernel", "input_kernel"}
    self.assertEqual(params.keys(), expected_keys)

    outputs = linear_layer.apply({"params": params}, input_emb=input_emb)

    expected_output = [
        [
            [0.00458921, 0.00194825],
            [-0.00955103, 0.0006038],
        ],
    ]
    np.testing.assert_allclose(outputs, expected_output, rtol=1e-6, atol=1e-6)

  def test_construct_circulant_matrix(self):
    circ_inputs = jnp.arange(4.)
    expected_circulant_matrix = jnp.array(
        [
            [0, 3, 2, 1],  #
            [1, 0, 3, 2],  #
            [2, 1, 0, 3],  #
            [3, 2, 1, 0]
        ],
        dtype=jnp.float32)
    np.testing.assert_allclose(
        layers._circulant_matrix(circ_inputs), expected_circulant_matrix)

  def test_fft_circulant_matmul(self):
    circ_vector_dim_zero = jnp.arange(4.)
    circ_vector_dim_one = jnp.arange(6.)
    inputs = jnp.arange(24.).reshape((4, 6))

    expected_output = jnp.einsum("ni,kj,ij->nk",
                                 layers._circulant_matrix(circ_vector_dim_zero),
                                 layers._circulant_matrix(circ_vector_dim_one),
                                 inputs)

    np.testing.assert_allclose(
        layers._apply_2d_fft_circulant(inputs, circ_vector_dim_zero,
                                       circ_vector_dim_one), expected_output)

  @parameterized.parameters(dict(use_fft=False), dict(use_fft=True))
  def test_circulant_transform(self, use_fft):
    batch_size = 1
    input_seq_length = 2
    hidden_dim = 4
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    circulant_layer = layers.CirculantTransform(use_fft=use_fft)
    input_emb = jax.random.uniform(
        init_rng, (batch_size, input_seq_length, hidden_dim),
        minval=0,
        maxval=5)
    params = init_layer_variables(rng, circulant_layer, {
        "input_emb": input_emb,
    })["params"]

    expected_keys = {"hidden_kernel", "input_kernel"}
    self.assertEqual(params.keys(), expected_keys)

    outputs = circulant_layer.apply({"params": params}, input_emb=input_emb)

    # Expected output for use_fft=True and use_fft=False are the same.
    expected_output = [
        [
            [0.00340233, 0.00365852, 0.00036887, 0.00267642],
            [0.00367175, 0.00376533, 0.00063781, 0.00220538],
        ],
    ]
    np.testing.assert_allclose(outputs, expected_output, rtol=1e-6, atol=1e-6)

  def test_construct_toeplitz_matrix(self):
    toe_inputs = jnp.arange(7.)
    expected_toeplitz_matrix = jnp.array([
        [3, 4, 5, 6],
        [2, 3, 4, 5],
        [1, 2, 3, 4],
        [0, 1, 2, 3],
    ],
                                         dtype=jnp.float32)
    np.testing.assert_allclose(
        layers._toeplitz_matrix(toe_inputs), expected_toeplitz_matrix)

  def test_embed_in_circulant(self):
    inputs = jnp.arange(7)
    np.testing.assert_allclose(
        layers._embed_in_circulant(inputs), jnp.array([3, 2, 1, 0, 3, 6, 5, 4]))

  def test_fft_toeplitz_matmul(self):
    toe_vector_dim_zero = jnp.arange(7.)
    toe_vector_dim_one = jnp.arange(5.)
    inputs = jnp.arange(12.).reshape((4, 3))

    expected_output = jnp.einsum("ni,kj,ij->nk",
                                 layers._toeplitz_matrix(toe_vector_dim_zero),
                                 layers._toeplitz_matrix(toe_vector_dim_one),
                                 inputs)

    np.testing.assert_allclose(
        layers._apply_2d_fft_toeplitz(inputs, toe_vector_dim_zero,
                                      toe_vector_dim_one),
        expected_output,
        rtol=1e-6,
        atol=1e-6)

  @parameterized.parameters(dict(use_fft=False), dict(use_fft=True))
  def test_toeplitz_transform(self, use_fft):
    batch_size = 1
    input_seq_length = 2
    hidden_dim = 4
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    toeplitz_layer = layers.ToeplitzTransform(use_fft=use_fft)
    input_emb = jax.random.uniform(
        init_rng, (batch_size, input_seq_length, hidden_dim),
        minval=0,
        maxval=5)

    params = init_layer_variables(rng, toeplitz_layer, {
        "input_emb": input_emb,
    })["params"]

    expected_keys = {"hidden_kernel", "input_kernel"}
    self.assertEqual(params.keys(), expected_keys)

    outputs = toeplitz_layer.apply({"params": params}, input_emb=input_emb)

    # Expected output for use_fft=True and use_fft=False are the same.
    expected_output = [
        [
            [-0.00077546, -0.00070286, -0.00084842, -0.00011822],
            [0.00092456, 0.00152007, 0.00167474, -0.00176105],
        ],
    ]
    np.testing.assert_allclose(outputs, expected_output, rtol=1e-6, atol=1e-6)


class EncoderBlockTest(parameterized.TestCase):

  def test_construct_encoder_block_correctly(self):
    max_seq_length = 14
    hidden_dim = 8
    rng = jax.random.PRNGKey(0)

    feed_forward_layer = layers.FeedForwardLayer(d_ff=8)
    mixing_layer = layers.LinearTransform()
    attention_layer = layers.AttentionLayer(num_heads=1, d_model=2)

    init_batch = {
        "input_emb": jnp.ones((1, max_seq_length, hidden_dim), jnp.float32),
        "input_ids": jnp.ones((1, max_seq_length), jnp.int32),
    }

    # Success case.
    encoder_block = layers.EncoderBlock(
        feed_forward_sublayer=feed_forward_layer,
        mixing_sublayer=mixing_layer,
        attention_sublayer=None)
    _ = init_layer_variables(rng, encoder_block, init_batch)

    # Failure case.
    with self.assertRaisesRegex(
        ValueError, "One, and only one, of {self.mixing_sublayer, "
        "self.attention_sublayer} must be nonempty"):
      encoder_block = layers.EncoderBlock(
          feed_forward_sublayer=feed_forward_layer,
          mixing_sublayer=mixing_layer,
          attention_sublayer=attention_layer)
      _ = init_layer_variables(rng, encoder_block, init_batch)

    # Failure case.
    with self.assertRaisesRegex(
        ValueError, "One, and only one, of {self.mixing_sublayer, "
        "self.attention_sublayer} must be nonempty"):
      encoder_block = layers.EncoderBlock(
          feed_forward_sublayer=feed_forward_layer,
          mixing_sublayer=None,
          attention_sublayer=None)
      _ = init_layer_variables(rng, encoder_block, init_batch)

  def test_encoder_block_feed_forward(self):
    batch_size = 2
    max_seq_length = 14
    hidden_dim = 8
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    feed_forward_layer = layers.FeedForwardLayer(d_ff=8, dropout_rate=0.0)
    mixing_layer = layers.LinearTransform()
    encoder_block = layers.EncoderBlock(
        feed_forward_sublayer=feed_forward_layer,
        mixing_sublayer=mixing_layer,
        attention_sublayer=None)
    input_emb = jax.random.uniform(
        init_rng, (batch_size, max_seq_length, hidden_dim), minval=0, maxval=10)
    input_ids = jax.random.randint(
        init_rng, (batch_size, max_seq_length, hidden_dim), minval=0, maxval=20)
    params = init_layer_variables(rng, encoder_block, {
        "input_emb": input_emb,
        "input_ids": input_ids
    })["params"]

    expected_keys = {
        "mixing_sublayer", "mixing_layer_norm", "output_layer_norm",
        "feed_forward_sublayer"
    }
    self.assertEqual(params.keys(), expected_keys)

    outputs = encoder_block.apply({"params": params},
                                  rngs={"dropout": rng},
                                  input_emb=input_emb,
                                  input_ids=input_ids)
    self.assertEqual(outputs.shape, (batch_size, max_seq_length, hidden_dim))

  def test_encoder_block_switch(self):
    batch_size = 2
    max_seq_length = 14
    num_tokens = batch_size * max_seq_length
    hidden_dim = 8
    rng = jax.random.PRNGKey(0)
    init_rng, dropout_key, jitter_key = jax.random.split(rng, num=3)

    expert = layers.FeedForwardLayer(d_ff=4, dropout_rate=0.0, name="mlp")
    router = routing.TokensChooseMaskedRouter(
        router_weights=routing.RouterWeights(name="router_weights"),
        jitter_noise=0.01,
        num_selected_experts=1,
        batch_prioritized_routing=True,
        dtype=jnp.float32)
    moe_layer = layers.MoeLayer(
        num_experts=2,
        router=router,
        max_group_size=num_tokens,
        train_capacity_factor=1.0,
        eval_capacity_factor=1.0,
        expert=expert)

    mixing_layer = layers.LinearTransform()
    encoder_block = layers.EncoderBlock(
        feed_forward_sublayer=moe_layer,
        mixing_sublayer=mixing_layer,
        attention_sublayer=None)
    input_emb = jax.random.uniform(
        init_rng, (batch_size, max_seq_length, hidden_dim), minval=0, maxval=10)
    input_ids = jax.random.randint(
        init_rng, (batch_size, max_seq_length, hidden_dim), minval=0, maxval=20)
    params = init_layer_variables(rng, encoder_block, {
        "input_emb": input_emb,
        "input_ids": input_ids
    })["params"]

    expected_keys = {
        "mixing_sublayer", "mixing_layer_norm", "output_layer_norm",
        "feed_forward_sublayer"
    }
    self.assertEqual(params.keys(), expected_keys)

    outputs, state = encoder_block.apply({"params": params},
                                         rngs={
                                             "dropout": dropout_key,
                                             "jitter": jitter_key
                                         },
                                         mutable=["intermediates"],
                                         input_emb=input_emb,
                                         input_ids=input_ids)
    self.assertEqual(outputs.shape, (batch_size, max_seq_length, hidden_dim))

    self.assertIn("intermediates", state)
    jax.tree_util.tree_map(
        functools.partial(np.testing.assert_allclose, rtol=1e-5),
        flax.core.freeze(state["intermediates"]),
        FrozenDict({
            "feed_forward_sublayer": {
                "diversity_metrics":
                    layers.DiversityMetrics(
                        auxiliary_loss=0.9997709,
                        router_z_loss=0.48709542,
                        fraction_tokens_left_behind=0.03571427,
                        expert_usage=0.96428573,
                        router_confidence=0.51779515)
            }
        }))


class ProjectionLayerTest(parameterized.TestCase):

  def test_classification_output_projection(self):
    batch_size = 2
    max_seq_length = 14
    hidden_dim = 8
    rng = jax.random.PRNGKey(0)

    classification_projection = layers.OutputProjection(
        n_out=NUM_CLASSES, name="classification")
    init_batch = {
        "input_emb": jnp.ones((1, max_seq_length, hidden_dim), jnp.float32)
    }
    params = init_layer_variables(rng, classification_projection,
                                  init_batch)["params"]

    expected_keys = {"output_bias", "output_kernel"}
    self.assertEqual(params.keys(), expected_keys)

    rng, init_rng = jax.random.split(rng)
    input_emb = jax.random.randint(
        init_rng, (batch_size, max_seq_length, hidden_dim), minval=0, maxval=10)
    outputs = classification_projection.apply({"params": params},
                                              input_emb=input_emb)
    self.assertEqual(outputs.shape, (batch_size, max_seq_length, NUM_CLASSES))

  def test_kernel_output_projection(self):
    batch_size = 2
    max_seq_length = 14
    hidden_dim = 8
    rng = jax.random.PRNGKey(0)

    rng, kernel_rng = jax.random.split(rng)
    kernel = jax.random.uniform(
        kernel_rng, (NUM_CLASSES, hidden_dim), minval=-1.0, maxval=1.0)
    kernel_projection = layers.OutputProjection(
        kernel=jnp.asarray(kernel), name="predictions_output")
    init_batch = {
        "input_emb": jnp.ones((1, max_seq_length, hidden_dim), jnp.float32)
    }
    params = init_layer_variables(rng, kernel_projection, init_batch)["params"]

    expected_keys = {"output_bias"}
    self.assertEqual(params.keys(), expected_keys)

    rng, init_rng = jax.random.split(rng)
    input_emb = jax.random.randint(
        init_rng, (batch_size, max_seq_length, hidden_dim), minval=0, maxval=10)
    outputs = kernel_projection.apply({"params": params}, input_emb=input_emb)
    self.assertEqual(outputs.shape, (batch_size, max_seq_length, NUM_CLASSES))


class EmbeddingTest(parameterized.TestCase):

  def test_embedding_layer(self):
    config = ml_collections.ConfigDict({
        "batch_size": 3,
        "vocab_size": 1000,
        "d_emb": 32,
        "max_seq_length": 64,
        "type_vocab_size": 2,
        "d_model": 4,
        "dropout_rate": 0.1,
        "dtype": jnp.float32
    })
    frozen_config = ml_collections.FrozenConfigDict(config)
    rng = jax.random.PRNGKey(100)

    embedding_layer = layers.EmbeddingLayer(config=frozen_config)
    init_batch = {
        "input_ids": jnp.ones((1, frozen_config.max_seq_length), jnp.int32),
        "type_ids": jnp.ones((1, frozen_config.max_seq_length), jnp.int32)
    }
    params = init_layer_variables(rng, embedding_layer, init_batch)["params"]

    expected_keys = {
        "word", "position", "type", "layer_norm", "hidden_mapping_in"
    }
    self.assertEqual(params.keys(), expected_keys)

    rng, init_rng = jax.random.split(rng)
    inputs = {
        "input_ids":
            jax.random.randint(
                init_rng,
                (frozen_config.batch_size, frozen_config.max_seq_length),
                minval=0,
                maxval=13),
        "type_ids":
            jax.random.randint(
                init_rng,
                (frozen_config.batch_size, frozen_config.max_seq_length),
                minval=0,
                maxval=2)
    }
    outputs = embedding_layer.apply({"params": params},
                                    rngs={"dropout": rng},
                                    **inputs)
    self.assertEqual(outputs.shape,
                     (frozen_config.batch_size, frozen_config.max_seq_length,
                      frozen_config.d_model))

  def test_positional_encoding(self):
    batch_size = 3
    max_seq_length = 14
    hidden_dim = 6
    rng = jax.random.PRNGKey(10)

    positional_encoding_layer = layers.PositionalEncoding(
        seq_length=max_seq_length)
    init_batch = {
        "word_embeddings":
            jnp.ones((1, max_seq_length, hidden_dim), jnp.float32),
    }
    params = init_layer_variables(rng, positional_encoding_layer,
                                  init_batch)["params"]

    expected_keys = {"embedding"}
    self.assertEqual(params.keys(), expected_keys)

    rng, init_rng = jax.random.split(rng)
    word_embeddings = jax.random.randint(
        init_rng, (batch_size, max_seq_length, hidden_dim), minval=0, maxval=13)
    outputs = positional_encoding_layer.apply({"params": params},
                                              word_embeddings=word_embeddings)
    self.assertEqual(outputs.shape, (1, max_seq_length, hidden_dim))


class GatherTest(parameterized.TestCase):

  def test_gather(self):
    example = jnp.arange(12.).reshape(4, 3)
    # Shape [BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM] = [2,4,3].
    batch = jnp.array([example, -example])

    # Shape [BATCH_SIZE, MAX_PREDICTIONS_PER_SEQ] = [2,2].
    indices = jnp.array([[0, 3], [1, 2]])

    outputs = layers.gather(sequence=batch, indices=indices)

    # Shape [BATCH_SIZE * MAX_PREDICTIONS_PER_SEQ, HIDDEN_DIM] = [4,3]
    self.assertEqual(outputs.shape, (4, 3))

    expected = jnp.array([[0, 1, 2], [9, 10, 11], [-3, -4, -5], [-6, -7, -8]],
                         dtype=jnp.float32)
    np.testing.assert_allclose(outputs, expected, atol=1e-12)

  def test_gather_incorrect_batch_sizes(self):
    example = jnp.arange(12.).reshape(4, 3)
    # Shape [BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM] = [2,4,3].
    batch = jnp.array([example, -example])

    with self.assertRaisesRegex(
        ValueError, "Input sequence and indices must have the same batch size"):
      # Shape [BATCH_SIZE, MAX_PREDICTIONS_PER_SEQ] = [1,2].
      indices = jnp.array([[1, 2]])
      _ = layers.gather(sequence=batch, indices=indices)

  def test_gather_bad_indices(self):
    example = jnp.arange(12.).reshape(4, 3)
    # Shape [BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM] = [2,4,3].
    batch = jnp.array([example, -example])

    with self.assertRaisesRegex(
        ValueError,
        "predictions per sequence cannot be greater than the maximum sequence"):
      # Shape [BATCH_SIZE, MAX_PREDICTIONS_PER_SEQ] = [2,5].
      indices = jnp.array([jnp.arange(5), jnp.arange(5)])
      _ = layers.gather(sequence=batch, indices=indices)


class SwitchLayerTest(parameterized.TestCase):

  @parameterized.parameters(dict(dispatch="scatter"), dict(dispatch="mask"))
  def test_moe_layer_runs(self, dispatch):
    batch_size = 3
    max_seq_length = 4
    num_tokens = batch_size * max_seq_length
    hidden_dim = 2
    num_experts = 4
    rng = jax.random.PRNGKey(0)

    if dispatch == "mask":
      router = routing.TokensChooseMaskedRouter(
          router_weights=routing.RouterWeights(name="router_weights"),
          jitter_noise=0.,
          num_selected_experts=2,
          batch_prioritized_routing=True,
          dtype=jnp.float32)
    else:
      router = routing.TokensChooseScatterRouter(
          router_weights=routing.RouterWeights(name="router_weights"),
          jitter_noise=0.,
          num_selected_experts=2,
          batch_prioritized_routing=True,
          dtype=jnp.float32)

    expert = layers.FeedForwardLayer(d_ff=2, dropout_rate=0.1, name="mlp")
    moe_layer = layers.MoeLayer(
        num_experts=num_experts,
        max_group_size=num_tokens,
        router=router,
        train_capacity_factor=1.5,
        eval_capacity_factor=1.5,
        expert=expert,
        axis_name="batch")
    init_batch = {
        "input_emb": jnp.ones((1, max_seq_length, hidden_dim), jnp.float32)
    }
    params = init_layer_variables(rng, moe_layer, init_batch)["params"]

    expected_keys = {"router", "expert"}
    self.assertEqual(params.keys(), expected_keys)

    dropout_rng, jitter_rng, init_rng = jax.random.split(rng, num=3)
    input_emb = jax.random.uniform(
        init_rng, (batch_size, max_seq_length, hidden_dim),
        minval=-10,
        maxval=10)
    actual_outputs, state = moe_layer.apply({"params": params},
                                            rngs={
                                                "dropout": dropout_rng,
                                                "jitter": jitter_rng
                                            },
                                            mutable=["intermediates"],
                                            input_emb=input_emb)

    self.assertEqual(actual_outputs.shape,
                     (batch_size, max_seq_length, hidden_dim))

    self.assertIn("diversity_metrics", state["intermediates"])

  def test_scatter_mask_dispatch_equal(self):
    batch_size = 4
    max_seq_length = 4
    hidden_dim = 2
    num_experts = 2
    tokens_per_group = 8
    num_groups = batch_size * max_seq_length // tokens_per_group

    rng = jax.random.PRNGKey(0)

    expert = layers.FeedForwardLayer(d_ff=2, dropout_rate=0.1, name="mlp")
    moe_layer_factory = functools.partial(
        layers.MoeLayer,
        num_experts=num_experts,
        max_group_size=tokens_per_group,
        train_capacity_factor=1.,
        eval_capacity_factor=1.,
        expert=expert,
        split_params=False)  # Ensures all experts start with same params

    router_weights = routing.RouterWeights(name="router_weights")
    masked_router = routing.TokensChooseMaskedRouter(
        router_weights=router_weights,
        jitter_noise=0.,
        num_selected_experts=2,
        batch_prioritized_routing=True,
        dtype=jnp.float32)
    masked_moe_layer = moe_layer_factory(router=masked_router)
    scatter_router = routing.TokensChooseScatterRouter(
        router_weights=router_weights,
        jitter_noise=0.,
        num_selected_experts=2,
        batch_prioritized_routing=True,
        dtype=jnp.float32)
    scatter_moe_layer = moe_layer_factory(router=scatter_router)

    input_emb = jax.random.uniform(
        rng, (batch_size, max_seq_length, hidden_dim), minval=-10, maxval=10)

    # Mock the router weights to ensure both layers compute with the same
    # logits.
    mock_router_logits = jax.random.uniform(
        rng, (num_groups, tokens_per_group, num_experts), minval=-1, maxval=1)
    with mock.patch.object(
        masked_router, "router_weights", return_value=mock_router_logits):
      masked_outputs, _ = masked_moe_layer.init_with_output(
          rng, input_emb, deterministic=True)
    with mock.patch.object(
        scatter_router, "router_weights", return_value=mock_router_logits):
      scatter_outputs, _ = scatter_moe_layer.init_with_output(
          rng, input_emb, deterministic=True)

    expected_outputs = jnp.array([
        [
            [-8.16194050e-04, -3.92473085e-05],
            [-8.87976727e-04, 6.41788647e-05],
            [1.51725704e-04, 5.44631148e-05],
            [0.00000000e+00, 0.00000000e+00],
        ],
        [
            [-1.63517136e-03, 7.32473345e-05],
            [6.99331111e-04, -4.98824847e-05],
            [-7.68527039e-04, -1.00117592e-04],
            [3.73630854e-03, 1.74387533e-04],
        ],
        [
            [1.09393802e-03, 5.09395104e-05],
            [-4.27273808e-05, 1.12514383e-04],
            [3.19827022e-03, 1.41921133e-04],
            [2.31421960e-04, -2.57078882e-05],
        ],
        [
            [0.00000000e+00, 0.00000000e+00],
            [1.65408337e-03, 1.62946199e-05],
            [2.29193736e-03, 1.07774074e-04],
            [-9.18464328e-04, -4.17242954e-05],
        ],
    ],
                                 dtype=jnp.float32)

    np.testing.assert_allclose(
        masked_outputs, expected_outputs, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        scatter_outputs, expected_outputs, rtol=1e-6, atol=1e-6)

  # pmap works with a local group size and number of tokens.
  @parameterized.parameters(
      dict(
          max_group_size=8,
          num_tokens=32,
          num_experts=2,
          expected_num_groups=4),
      dict(
          max_group_size=16,
          num_tokens=32,
          num_experts=2,
          expected_num_groups=2),
      dict(
          max_group_size=3,
          num_tokens=32,
          num_experts=4,
          expected_num_groups=16),
      dict(
          max_group_size=32,
          num_tokens=32,
          num_experts=2,
          expected_num_groups=1),
      dict(
          max_group_size=64,
          num_tokens=32,
          num_experts=2,
          expected_num_groups=1))
  def test_num_groups(self, max_group_size, num_tokens,
                      num_experts, expected_num_groups):
    expert = layers.FeedForwardLayer(d_ff=2)
    router = routing.ExpertsChooseMaskedRouter(
        router_weights=routing.RouterWeights(name="router_weights"),
        jitter_noise=0.,
        dtype=jnp.float32)
    moe_layer = layers.MoeLayer(
        num_experts=num_experts,
        router=router,
        max_group_size=max_group_size,
        train_capacity_factor=1.,
        eval_capacity_factor=1.,
        expert=expert)

    num_groups = moe_layer._num_groups(num_tokens, max_group_size)
    self.assertEqual(num_groups, expected_num_groups)


if __name__ == "__main__":
  absltest.main()
