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
"""Module for model_baselines.py."""

import tensorflow as tf
from united_earth_surface_fluxes import layers


class FTTransformer(tf.keras.Model):
  """FTTransformer baseline model.

  Reference: https://arxiv.org/abs/2106.01342.
  """

  def __init__(
      self,
      hid_dim,
      trunk_num_layers,
      num_outputs,
      expert_num_layers,
      transformer_embed_dim,
      transformer_ff_dim,
      transformer_num_heads,
      **kwargs,
  ):
    kwargs.pop('num_experts', None)
    super().__init__(**kwargs)

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

    self.expert = layers.ExpertModel(
        hid_dim=hid_dim,
        num_layers=expert_num_layers,
        num_outputs=num_outputs,
    )

  def call(self, inputs, training=None, mask=None):
    x, land_fractions = inputs
    features = tf.concat([x, land_fractions], axis=-1)

    proc_features = tf.expand_dims(features, axis=-1)
    proc_features = self.feature_projector(proc_features)

    for layer in self.trunk_layers:
      proc_features = layer(proc_features, training=training)

    output = self.pool_layer(proc_features)

    output = self.expert(output, training=training)
    return output


class MLP(tf.keras.Model):
  """MLP baseline model inspired by FTTransformer but with MLP trunk."""

  def __init__(
      self,
      hid_dim,
      trunk_num_layers,
      num_outputs,
      expert_num_layers,
      **kwargs,
  ):
    kwargs.pop('num_experts', None)
    kwargs.pop('transformer_embed_dim', None)
    kwargs.pop('transformer_ff_dim', None)
    kwargs.pop('transformer_num_heads', None)
    super().__init__(**kwargs)

    trunk_dim = int(hid_dim * 1.5)
    self.init_rew = layers.LayerReweighting()
    self.init_linear_layer = tf.keras.layers.Dense(trunk_dim)
    self.trunk_module = tf.keras.Sequential(
        [
            layers.BlockLayer(
                trunk_dim,
            )
            for _ in range(trunk_num_layers)
        ]
    )
    self.expert = layers.ExpertModel(
        hid_dim=hid_dim,
        num_layers=expert_num_layers,
        num_outputs=num_outputs,
    )

  def call(self, inputs, training=None, mask=None):
    x, land_fractions = inputs
    features = tf.concat([x, land_fractions], axis=-1)
    proc_features = self.init_rew(features, training=training)
    proc_features = self.init_linear_layer(proc_features)
    trunk_output = self.trunk_module(proc_features, training=training)
    output = self.expert(trunk_output, training=training)
    return output
