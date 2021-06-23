# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Tests f_net.layers."""

import functools
from typing import Any, Dict, Mapping

from absl.testing import absltest
import jax
from jax import numpy as jnp
import ml_collections
import numpy as np

from f_net import layers

# Type Stubs
PRNGKey = Any
Layer = Any

NUM_CLASSES = 2


def init_layer_variables(
    key, module,
    init_batch):
  """Initialize layer parameters."""
  key, dropout_key = jax.random.split(key)

  return module.init({"params": key, "dropout": dropout_key}, **init_batch)


class LayersTest(absltest.TestCase):

  def test_feed_forward_layer(self):
    batch_size = 3
    max_seq_length = 16
    hidden_dim = 12
    rng = jax.random.PRNGKey(0)

    feed_forward_layer = layers.FeedForwardLayer(d_ff=8, dropout_rate=0.1)
    init_batch = {
        "inputs": jnp.ones((1, max_seq_length, hidden_dim), jnp.float32)
    }
    params = init_layer_variables(rng, feed_forward_layer, init_batch)["params"]

    expected_keys = {"intermediate", "output"}
    self.assertEqual(params.keys(), expected_keys)

    rng, init_rng = jax.random.split(rng)
    inputs = jax.random.randint(
        init_rng, (batch_size, max_seq_length, hidden_dim), minval=0, maxval=10)
    outputs = feed_forward_layer.apply({"params": params},
                                       rngs={"dropout": rng},
                                       inputs=inputs)
    self.assertEqual(outputs.shape, (batch_size, max_seq_length, hidden_dim))

  def test_fourier_transform(self):
    batch_size = 4
    max_seq_length = 16
    hidden_dim = 8
    rng = jax.random.PRNGKey(0)

    identity = functools.partial(jnp.matmul, b=jnp.identity(hidden_dim))
    fourier_layer = layers.FourierTransform(fourier_transform=identity)
    init_batch = {
        "inputs": jnp.ones((1, max_seq_length, hidden_dim), jnp.float32)
    }
    init_layer_variables(rng, fourier_layer, init_batch)

    rng, init_rng = jax.random.split(rng)
    inputs = jax.random.randint(
        init_rng, (batch_size, max_seq_length, hidden_dim), minval=0, maxval=10)
    # FourierTransform layer has no learnable params.
    outputs = fourier_layer.apply({"params": {}}, inputs=inputs)
    self.assertEqual(outputs.shape, (batch_size, max_seq_length, hidden_dim))

  def test_identity_transform(self):
    batch_size = 8
    max_seq_length = 8
    hidden_dim = 8
    rng = jax.random.PRNGKey(0)

    identity_layer = layers.IdentityTransform()
    init_batch = {
        "inputs": jnp.ones((1, max_seq_length, hidden_dim), jnp.float32)
    }
    init_layer_variables(rng, identity_layer, init_batch)

    rng, init_rng = jax.random.split(rng)
    inputs = jax.random.randint(
        init_rng, (batch_size, max_seq_length, hidden_dim), minval=0, maxval=10)
    # IdentityTransform layer has no learnable params.
    outputs = identity_layer.apply({"params": {}}, inputs=inputs)
    self.assertEqual(outputs.shape, (batch_size, max_seq_length, hidden_dim))

  def test_linear_transform(self):
    batch_size = 8
    max_seq_length = 16
    hidden_dim = 32
    rng = jax.random.PRNGKey(0)

    linear_layer = layers.LinearTransform()
    init_batch = {
        "inputs": jnp.ones((1, max_seq_length, hidden_dim), jnp.float32)
    }
    params = init_layer_variables(rng, linear_layer, init_batch)["params"]

    expected_keys = {"hidden_kernel", "seq_kernel"}
    self.assertEqual(params.keys(), expected_keys)

    rng, init_rng = jax.random.split(rng)
    inputs = jax.random.randint(
        init_rng, (batch_size, max_seq_length, hidden_dim), minval=0, maxval=13)
    outputs = linear_layer.apply({"params": params}, inputs=inputs)
    self.assertEqual(outputs.shape, (batch_size, max_seq_length, hidden_dim))

  def test_random_transform(self):
    batch_size = 4
    max_seq_length = 16
    hidden_dim = 8
    rng = jax.random.PRNGKey(0)
    rng, weight_key = jax.random.split(rng)

    random_layer = layers.RandomTransform(
        max_seq_length=max_seq_length, d_model=hidden_dim, key=weight_key)
    init_batch = {
        "inputs": jnp.ones((1, max_seq_length, hidden_dim), jnp.float32)
    }
    init_layer_variables(rng, random_layer, init_batch)

    rng, init_rng = jax.random.split(rng)
    inputs = jax.random.randint(
        init_rng, (batch_size, max_seq_length, hidden_dim), minval=0, maxval=10)
    # RandomTransform layer has no learnable params.
    outputs = random_layer.apply({"params": {}}, inputs=inputs)
    self.assertEqual(outputs.shape, (batch_size, max_seq_length, hidden_dim))

  def test_encoder_block(self):
    batch_size = 2
    max_seq_length = 14
    hidden_dim = 8
    rng = jax.random.PRNGKey(0)

    feed_forward_layer = layers.FeedForwardLayer(d_ff=8, dropout_rate=0.0)
    mixing_layer = layers.IdentityTransform()
    encoder_block = layers.EncoderBlock(
        feed_forward_sublayer=feed_forward_layer, mixing_sublayer=mixing_layer)
    init_batch = {
        "inputs": jnp.ones((1, max_seq_length, hidden_dim), jnp.float32),
        "padding_mask": jnp.ones((1, max_seq_length), jnp.int32)
    }
    params = init_layer_variables(rng, encoder_block, init_batch)["params"]

    expected_keys = {
        "mixing_layer_norm", "output_layer_norm", "feed_forward_sublayer"
    }
    self.assertEqual(params.keys(), expected_keys)

    rng, init_rng = jax.random.split(rng)
    inputs = {
        "inputs":
            jax.random.randint(
                init_rng, (batch_size, max_seq_length, hidden_dim),
                minval=0,
                maxval=10),
        "padding_mask":
            jax.random.randint(
                init_rng, (batch_size, max_seq_length), minval=0, maxval=1)
    }

    outputs = encoder_block.apply({"params": params},
                                  rngs={"dropout": rng},
                                  **inputs)
    self.assertEqual(outputs.shape, (batch_size, max_seq_length, hidden_dim))

  def test_classification_output_projection(self):
    batch_size = 2
    max_seq_length = 14
    hidden_dim = 8
    rng = jax.random.PRNGKey(0)

    classification_projection = layers.OutputProjection(
        n_out=NUM_CLASSES, name="classification")
    init_batch = {
        "inputs": jnp.ones((1, max_seq_length, hidden_dim), jnp.float32)
    }
    params = init_layer_variables(rng, classification_projection,
                                  init_batch)["params"]

    expected_keys = {"output_bias", "output_kernel"}
    self.assertEqual(params.keys(), expected_keys)

    rng, init_rng = jax.random.split(rng)
    inputs = jax.random.randint(
        init_rng, (batch_size, max_seq_length, hidden_dim), minval=0, maxval=10)
    outputs = classification_projection.apply({"params": params}, inputs=inputs)
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
        "inputs": jnp.ones((1, max_seq_length, hidden_dim), jnp.float32)
    }
    params = init_layer_variables(rng, kernel_projection, init_batch)["params"]

    expected_keys = {"output_bias"}
    self.assertEqual(params.keys(), expected_keys)

    rng, init_rng = jax.random.split(rng)
    inputs = jax.random.randint(
        init_rng, (batch_size, max_seq_length, hidden_dim), minval=0, maxval=10)
    outputs = kernel_projection.apply({"params": params}, inputs=inputs)
    self.assertEqual(outputs.shape, (batch_size, max_seq_length, NUM_CLASSES))

  def test_embedding_layer(self):
    config = ml_collections.ConfigDict({
        "batch_size": 3,
        "vocab_size": 1000,
        "d_emb": 32,
        "max_seq_length": 64,
        "type_vocab_size": 2,
        "d_model": 4,
        "dropout_rate": 0.1
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
        max_seq_length=max_seq_length)
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

  def test_gather(self):
    example = jnp.arange(12.).reshape(4, 3)
    # Shape [BATCH_SIZE, MAX_SEQ_LENGTH, HIDDEN_DIM] = [2,4,3].
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
    # Shape [BATCH_SIZE, MAX_SEQ_LENGTH, HIDDEN_DIM] = [2,4,3].
    batch = jnp.array([example, -example])

    with self.assertRaisesRegex(
        ValueError, "Input sequence and indices must have the same batch size"):
      # Shape [BATCH_SIZE, MAX_PREDICTIONS_PER_SEQ] = [1,2].
      indices = jnp.array([[1, 2]])
      _ = layers.gather(sequence=batch, indices=indices)

  def test_gather_bad_indices(self):
    example = jnp.arange(12.).reshape(4, 3)
    # Shape [BATCH_SIZE, MAX_SEQ_LENGTH, HIDDEN_DIM] = [2,4,3].
    batch = jnp.array([example, -example])

    with self.assertRaisesRegex(
        ValueError,
        "predictions per sequence cannot be greater than the maximum sequence"):
      # Shape [BATCH_SIZE, MAX_PREDICTIONS_PER_SEQ] = [2,5].
      indices = jnp.array([jnp.arange(5), jnp.arange(5)])
      _ = layers.gather(sequence=batch, indices=indices)


if __name__ == "__main__":
  absltest.main()
