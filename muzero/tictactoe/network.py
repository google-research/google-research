# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

# Lint as: python3
# pylint: disable=g-complex-comprehension
# pylint: disable=missing-docstring
"""MLP+LSTM network for use with MuZero."""

import tensorflow as tf

from muzero import network


class MLPandLSTM(network.AbstractEncoderandLSTM):

  def __init__(self, mlp_sizes, *args, **kwargs):
    super().__init__(*args, **kwargs)
    mlp_layers = [
        tf.keras.Sequential([
            tf.keras.layers.Dense(size, activation='relu', use_bias=False),
            tf.keras.layers.LayerNormalization(),
        ],
                            name='intermediate_{}'.format(idx))
        for idx, size in enumerate(mlp_sizes)
    ]
    self._observation_encoder = tf.keras.Sequential(
        mlp_layers, name='observation_encoder')

  def _encode_observation(self, observation, training=True):
    return self._observation_encoder(observation, training=training)
