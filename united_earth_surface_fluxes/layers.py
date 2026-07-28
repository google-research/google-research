# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

# pylint: disable=logging-fstring-interpolation
"""Module for layers.py."""

import tensorflow as tf


class LayerReweighting(tf.keras.layers.Layer):

  def build(self, input_shape):
    self.w = self.add_weight(
        'kernel', shape=(input_shape[-1],), initializer='ones'
    )

  def call(self, x, training=None):
    return x * self.w


class SelfModulatedLayerNorm(tf.keras.layers.Layer):
  """A layer that applies layer normalization and self-modulates it."""

  def __init__(self, gate_init=0.3, **kwargs):
    super().__init__(**kwargs)
    self.gate_init = gate_init

    self.norm = tf.keras.layers.LayerNormalization()

  def build(self, input_shape):

    a_init = tf.constant(self.gate_init)
    bias_init_val = tf.math.log(a_init / (1 - a_init))

    self.gate_network = tf.keras.layers.Dense(
        input_shape[-1],
        activation='sigmoid',
        bias_initializer=tf.keras.initializers.Constant(bias_init_val),
    )

  def call(self, x, training=None):
    normalized_x = self.norm(x)
    a = self.gate_network(x)
    return normalized_x * a + x * (1 - a)


class BlockLayer(tf.keras.layers.Layer):
  """Standard functional generic base block layer."""

  def __init__(self, hid_dim, **kwargs):
    super().__init__(**kwargs)
    dense_layer = tf.keras.layers.Dense(hid_dim, use_bias=False)
    self.all_layers = tf.keras.Sequential([
        dense_layer,
        SelfModulatedLayerNorm(),
        tf.keras.layers.ELU(),
    ])

  def call(self, x, training=None):
    x = self.all_layers(x, training=training)
    return x


class FiLMLookupTable(tf.keras.layers.Layer):
  """FiLM layer implemented as an embedding lookup table per expert."""

  def __init__(self, num_experts, modulate_dim, **kwargs):
    super().__init__(**kwargs)
    self.num_experts = num_experts
    self.modulate_dim = modulate_dim
    self.gamma_embeddings = tf.keras.layers.Embedding(
        num_experts,
        modulate_dim,
        embeddings_initializer='ones',
        name='film_gamma_lookup',
    )
    self.beta_embeddings = tf.keras.layers.Embedding(
        num_experts,
        modulate_dim,
        embeddings_initializer='zeros',
        name='film_beta_lookup',
    )

  def call(self, inputs):
    """Calculates modulated representations using FiLM.

    Args:
      inputs: A tuple containing the main features x and the expert_idx.

    Returns:
      The modulated representations.
    """
    x, expert_idx = inputs
    gamma = self.gamma_embeddings(expert_idx)
    beta = self.beta_embeddings(expert_idx)
    return x * gamma + beta


class CustomTransformerEncoder(tf.keras.layers.Layer):
  """A Transformer Encoder block using SelfModulatedLayerNorm."""

  def __init__(
      self,
      embed_dim,
      num_heads,
      ff_dim,
      activation='elu',
      gate_init=0.3,
      **kwargs,
  ):
    super().__init__(**kwargs)
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.ff_dim = ff_dim

    # 1. Multi-Head Attention
    self.mha = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=embed_dim
    )

    self.ffn = tf.keras.Sequential([
        tf.keras.layers.Dense(ff_dim, activation=activation, use_bias=False),
        tf.keras.layers.Dense(embed_dim, use_bias=False),
    ])

    self.norm1 = SelfModulatedLayerNorm(gate_init=gate_init)
    self.norm2 = SelfModulatedLayerNorm(gate_init=gate_init)

  def call(self, x, training=None):
    attn_output = self.mha(x, x, x, training=training)
    x_norm1 = self.norm1(x + attn_output, training=training)
    ffn_output = self.ffn(x_norm1, training=training)
    # Add & Norm (Residual + Your custom norm)
    x_norm2 = self.norm2(x_norm1 + ffn_output, training=training)

    return x_norm2


class ExpertModel(tf.keras.layers.Layer):
  """A specialized expert model for a single land type."""

  def __init__(
      self,
      hid_dim,
      num_layers,
      num_outputs,
      **kwargs,
  ):
    super().__init__(**kwargs)
    self.block_layers = tf.keras.Sequential(
        [LayerReweighting()]
        + [
            BlockLayer(
                hid_dim,
            )
            for _ in range(num_layers)
        ]
    )
    self.output_layer = tf.keras.layers.Dense(num_outputs)

  def call(self, x, training=None):
    x = self.block_layers(x, training=training)
    x = self.output_layer(x, training=training)
    return x


class SoftGatingNetwork(tf.keras.layers.Layer):
  """Learned Gating Network (Router) for Soft-Routing MoE.

  It produces weights for each expert based on the input context.
  """

  def __init__(self, num_experts, **kwargs):
    super().__init__(**kwargs)
    self.dense = tf.keras.layers.Dense(
        num_experts,
        name='gating_logits',
        kernel_initializer='he_normal',
        bias_initializer='zeros',
    )
    # Softmax ensures the weights sum to 1, acting like probabilities
    self.softmax = tf.keras.layers.Softmax(axis=-1, name='gating_softmax')

  def call(self, x):
    # x is the shared context vector (aware_representation from FiLM)
    logits = self.dense(x)
    # Output shape: (B, num_experts) - Learned mixture weights
    return self.softmax(logits)
