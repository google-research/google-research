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
"""Mixture of Experts model architectures."""

import tensorflow as tf
from united_earth_surface_fluxes import layers


class BaseMoE(tf.keras.Model):
  """Base class for Mixture of Experts models."""

  def __init__(
      self,
      hid_dim,
      num_experts,
      num_outputs,
      expert_num_layers,
      **kwargs,
  ):
    super().__init__(**kwargs)
    self.num_experts = num_experts
    self.num_outputs = num_outputs
    self.expert_models = [
        layers.ExpertModel(
            hid_dim=hid_dim,
            num_layers=expert_num_layers,
            num_outputs=num_outputs,
        )
        for _ in range(self.num_experts)
    ]

  def run_experts(self, shared_representation, training=None):
    """Runs all experts on the given input."""
    expert_input = shared_representation
    expert_outputs = [
        expert_model(expert_input, training=training)
        for expert_model in self.expert_models
    ]
    # Shape: (B, num_outputs, num_experts)
    final_output = tf.stack(expert_outputs, axis=-1)
    return final_output

  def call(self, inputs, training=None, mask=None):
    raise NotImplementedError('Subclasses should implement this method.')


class UniTeD(BaseMoE):
  """UniTeD with shared expert and FiLM lookup table."""

  def __init__(
      self,
      hid_dim,
      trunk_num_layers,
      num_experts,
      num_outputs,
      expert_num_layers,
      transformer_embed_dim,
      transformer_ff_dim,
      transformer_num_heads,
      **kwargs,
  ):
    super().__init__(
        hid_dim=hid_dim,
        num_experts=num_experts,
        num_outputs=num_outputs,
        expert_num_layers=expert_num_layers,
        **kwargs,
    )
    del self.expert_models
    self.film_layer = layers.FiLMLookupTable(num_experts, transformer_embed_dim)
    self.shared_expert = layers.ExpertModel(
        hid_dim=hid_dim,
        num_layers=expert_num_layers,
        num_outputs=num_outputs,
    )
    self.feature_projector = tf.keras.layers.LocallyConnected1D(
        filters=transformer_embed_dim,
        kernel_size=1,
        activation='silu',
        use_bias=False,
        name='feature_projector',
    )
    self.trunk_layers = [
        layers.CustomTransformerEncoder(
            embed_dim=transformer_embed_dim,
            num_heads=transformer_num_heads,
            ff_dim=transformer_ff_dim,
            activation='elu',
            name=f'custom_transformer_encoder_{i}',
        )
        for i in range(trunk_num_layers)
    ]
    self.pool_layer = tf.keras.layers.GlobalAveragePooling1D(
        name='context_vector'
    )

  def run_trunk(self, x, training=None):
    """Applies transformer trunk."""
    x = tf.expand_dims(x, axis=-1)
    x = self.feature_projector(x)
    for layer in self.trunk_layers:
      x = layer(x, training=training)
    return self.pool_layer(x)

  def call(self, inputs, training=None, mask=None):
    x, _ = inputs  # land_fractions are not used internally.
    shared_representation = self.run_trunk(x, training=training)
    expert_outputs = []
    for i in range(self.num_experts):
      aware_representation = self.film_layer([shared_representation, i])
      expert_input = aware_representation
      expert_input = tf.nn.silu(expert_input)
      expert_outputs.append(self.shared_expert(expert_input, training=training))
    return tf.stack(expert_outputs, axis=-1)


class FracMoE(BaseMoE):
  """Mixture of Experts model with a Transformer Trunk."""

  def __init__(
      self,
      hid_dim,
      trunk_num_layers,
      num_experts,
      num_outputs,
      expert_num_layers,
      transformer_embed_dim,
      transformer_ff_dim,
      transformer_num_heads,
      **kwargs,
  ):
    super().__init__(
        hid_dim=hid_dim,
        num_experts=num_experts,
        num_outputs=num_outputs,
        expert_num_layers=expert_num_layers,
        **kwargs,
    )

    # Input Projection
    # Shape:(B, features, 1) -> (B, features,transformer_embed_dim)
    self.feature_projector = tf.keras.layers.LocallyConnected1D(
        filters=transformer_embed_dim,
        kernel_size=1,
        activation='silu',
        use_bias=False,
        name='feature_projector',
    )

    self.trunk_layers = [
        layers.CustomTransformerEncoder(
            embed_dim=transformer_embed_dim,
            num_heads=transformer_num_heads,
            ff_dim=transformer_ff_dim,
            activation='elu',
            name=f'custom_transformer_encoder_{i}',
        )
        for i in range(trunk_num_layers)
    ]

    # Pooling (B, features, transformer_embed_dim) -> (B, transformer_embed_dim)
    self.pool_layer = tf.keras.layers.GlobalAveragePooling1D(
        name='context_vector'
    )

  def run_trunk(self, x, land_fractions, training=None):
    """Applies the transformer trunk to the input."""
    del land_fractions
    x = tf.expand_dims(x, axis=-1)
    x = self.feature_projector(x)

    for layer in self.trunk_layers:
      x = layer(x, training=training)
    shared_representation = self.pool_layer(x)
    return shared_representation

  def call(self, inputs, training=None, mask=None):
    x, land_fractions = inputs
    shared_representation = self.run_trunk(x, land_fractions, training=training)
    expert_preds = self.run_experts(shared_representation, training=training)
    return expert_preds


class SoftRoutingMoE(FracMoE):
  """Soft-Routing MoE Baseline using learned gates."""

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    # Instantiate the Gating Network
    self.gating_network = layers.SoftGatingNetwork(
        num_experts=self.num_experts, name='soft_gating_network'
    )

  def run_trunk(self, x, land_fractions, training=None):
    """Applies the transformer trunk to the input."""
    x = tf.concat([x, land_fractions], axis=-1)
    x = tf.expand_dims(x, axis=-1)
    x = self.feature_projector(x)

    for layer in self.trunk_layers:
      x = layer(x, training=training)
    shared_representation = self.pool_layer(x)
    return shared_representation

  def call(self, inputs, training=None, mask=None):
    x, land_fractions = inputs
    shared_representation = self.run_trunk(x, land_fractions, training=training)
    learned_weights = self.gating_network(shared_representation)
    final_output = self.run_experts(shared_representation, training=training)

    return final_output, learned_weights
