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

# pylint: disable=unused-argument
"""Build UVFA with image observation for state input."""

from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers

from hal.agent.common import tensor_concat
from hal.agent.UVFA_repr_tf2.uvfa_state import StateUVFA2


class ImageUVFA2(StateUVFA2):
  """UVFA that uses the image observation.

  Attributes:
    layers_variables: weight variables of all layers
  """

  def __init__(self, cfg):
    self.layers_variables = {}
    StateUVFA2.__init__(self, cfg)

  def _process_image(self, input_shape, cfg, name, embedding_length):
    inputs = layers.Input(shape=input_shape)
    goal_embedding = layers.Input(shape=(embedding_length))
    expand_dims = tf.keras.layers.Lambda(
        lambda inputs: tf.expand_dims(inputs[0], axis=inputs[1]))
    out = inputs
    for cfg in cfg.conv_layer_config:
      if cfg[0] < 0:
        conv = layers.Conv2D(
            filters=-cfg[0],
            kernel_size=cfg[1],
            strides=cfg[2],
            activation=tf.nn.relu,
            padding='SAME')
        out = conv(out)
      else:
        # Film
        conv = layers.Conv2D(
            filters=cfg[0],
            kernel_size=cfg[1],
            strides=cfg[2],
            activation=None,
            padding='SAME')
        out = conv(out)
        out = layers.BatchNormalization(center=False, scale=False)(out)
        gamma = layers.Dense(cfg[0])(goal_embedding)
        beta = layers.Dense(cfg[0])(goal_embedding)
        gamma = expand_dims((expand_dims((gamma, 1)), 1))
        beta = expand_dims((expand_dims((beta, 1)), 1))
        out = layers.Multiply()([out, gamma])
        out = layers.Add()([out, beta])
        out = layers.ReLU()(out)
    all_inputs = {'state_input': inputs, 'goal_embedding': goal_embedding}
    overall_layer = tf.keras.Model(
        name='vl_embedding', inputs=all_inputs, outputs=out)
    return overall_layer

  def build_q_discrete(self, cfg, name, embedding_length):
    """"Build the q value network.

    Args:
      cfg: configuration object
      name: name of the model
      embedding_length: length of the embedding of the instruction

    Returns:
      the q value network
    """
    input_shape = (cfg.img_resolution, cfg.img_resolution, 3)
    inputs = tf.keras.layers.Input(shape=input_shape)
    goal_embedding = tf.keras.layers.Input(shape=(embedding_length))
    all_inputs = {'state_input': inputs, 'goal_embedding': goal_embedding}
    factors = [8, 10, 10]

    process_layer = self._process_image(input_shape, cfg, name,
                                        embedding_length)
    process_layer.build(input_shape)
    out_shape = process_layer.output_shape
    final_layer = DiscreteFinalLayer(factors, out_shape)

    processed_input = process_layer(all_inputs)
    processed_all_inputs = {
        'state_input': processed_input,
        'goal_embedding': goal_embedding
    }
    q_out = final_layer(processed_all_inputs)
    model = tf.keras.Model(name=name, inputs=all_inputs, outputs=q_out)
    return model


class DiscreteFinalLayer(layers.Layer):
  """Keras layer for projection.

  Attributes:
      factors: size of each action axis
      out_shape: shape of the action
  """

  def __init__(self, factors, out_shape):
    super(DiscreteFinalLayer, self).__init__()
    self.factors = factors
    self.out_shape = out_shape
    self._initializer = tf.initializers.glorot_uniform()
    self._projection_mat = tf.Variable(
        self._initializer(shape=(1, sum(factors), np.prod(out_shape[1:-1]))),
        trainable=True,
        name='projection_matrix')
    self._dense_layer = layers.Dense(out_shape[-1])
    self._conv_layer_1 = layers.Conv2D(100, 1, 1)
    self._conv_layer_2 = layers.Conv2D(32, 1, 1)
    self._conv_layer_3 = layers.Conv2D(1, 1, 1)

  def call(self, inputs):
    goal_embedding = inputs['goal_embedding']
    state_inputs = inputs['state_input']
    projection_mat = tf.tile(self._projection_mat,
                             [tf.shape(state_inputs)[0], 1, 1])
    out = tf.reshape(state_inputs,
                     (-1, np.prod(self.out_shape[1:-1]), self.out_shape[-1]))
    out = tf.matmul(projection_mat, out)
    # [B, factor[0], s3] [B, factor[1], s3] [B, factor[2], s3]
    fac1, fac2, fac3 = tf.split(out, self.factors, axis=1)
    out = tensor_concat(fac1, fac2, fac3)  # [B, f1, f2, f3, s3]
    # [B, 800, s3*3]
    out = tf.reshape(out, [-1, np.prod(self.factors), self.out_shape[-1] * 3])
    goal_tile = tf.expand_dims(self._dense_layer(goal_embedding), 1)
    goal_tile = tf.tile(goal_tile, multiples=[1, np.prod(self.factors), 1])
    out = tf.concat([out, goal_tile], axis=-1)
    out = tf.expand_dims(out, axis=1)
    out = tf.nn.relu(self._conv_layer_1(out))
    out = tf.nn.relu(self._conv_layer_2(out))
    out = self._conv_layer_3(out)
    return tf.squeeze(out, axis=[1, 3])
