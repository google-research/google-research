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
"""Build UVFA with state observation for image input."""

from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf

from hal.agent.common import encoder
from hal.agent.common import film_params
from hal.agent.common import stack_dense_layer
from hal.agent.common import tensor_concat
from hal.agent.UVFA_repr.uvfa_state import StateUVFA


class ImageUVFA(StateUVFA):
  """UVFA agent that uses image observation."""

  def create_models(self, cfg):
    """Builds the computation graph for the agent."""

    def make_policy():
      """Returns one copy of the model."""
      artifact = {}
      if cfg.intruction_repr == 'language':
        trainable_encoder = cfg.trainable_encoder
        print('The encoder is trainable: {}'.format(trainable_encoder))
        embedding = tf.get_variable(
            name='word_embedding',
            shape=(cfg.vocab_size, cfg.embedding_size),
            dtype=tf.float32,
            trainable=trainable_encoder)
        _, goal_embedding = encoder(
            self.word_inputs,
            embedding,
            cfg.encoder_n_unit,
            trainable=trainable_encoder)
        artifact['embedding'] = embedding
      elif cfg.intruction_repr == 'one_hot':
        print('Goal input for one-hot max len {}'.format(
            cfg.max_sequence_length))
        one_hot_goal = tf.one_hot(self.word_inputs, cfg.max_sequence_length)
        one_hot_goal.set_shape([None, cfg.max_sequence_length])
        layer_cfg = [cfg.max_sequence_length // 8, cfg.encoder_n_unit]
        goal_embedding = stack_dense_layer(one_hot_goal, layer_cfg)
      else:
        raise ValueError('Unrecognized instruction type: {}'.format(
            cfg.instruction_repr))
      artifact['goal_embedding'] = goal_embedding
      all_q = self.build_q_factor_discrete(cfg, goal_embedding)

      predict_action = tf.argmax(all_q, axis=-1)
      action = tf.placeholder(shape=None, dtype=tf.int32)
      action_onehot = tf.one_hot(
          action, cfg.ac_dim[0], dtype=tf.float32)
      q = tf.reduce_sum(
          tf.multiply(all_q, action_onehot), axis=1)
      artifact.update(
          {
              'all_q': all_q,
              'predict_action': predict_action,
              'action_ph': action,
              'action_onehot': action_onehot,
              'q': q
          }
      )
      return artifact

    ############################# Build ###############################
    with tf.variable_scope('online'):
      self.online_artifact = make_policy()
    with tf.variable_scope('target'):
      self.target_artifact = make_policy()

    self.action_ph = self.online_artifact['action_ph']
    self.predict_action = self.online_artifact['predict_action']
    self.all_q = self.online_artifact['all_q']
    self.q = self.online_artifact['q']

    self.action_ph_target = self.target_artifact['action_ph']
    self.all_q_target = self.target_artifact['all_q']

  def build_q_factor_discrete(self, cfg, goal_embedding):
    """"Build the q value network.

    Args:
      cfg: configuration object
      goal_embedding: tensor that contains the goal embedding

    Returns:
      the q value tensor
    """
    n_layer_channel = []
    for layer_config in cfg.conv_layer_config:
      if layer_config[0] > 0:
        n_layer_channel.append(layer_config[0])
    layer_film_params = film_params(goal_embedding, n_layer_channel)
    out = self.inputs
    for cfg, param in zip(cfg.conv_layer_config, layer_film_params):
      if cfg[0] < 0:
        out = tf.layers.conv2d(out, -cfg[0], cfg[1], cfg[2], padding='SAME')
        out = tf.nn.relu(out)
      else:
        out = tf.layers.conv2d(out, cfg[0], cfg[1], cfg[2], padding='SAME')
        out = tf.layers.batch_normalization(
            out, center=False, scale=False, training=self.is_training)
        gamma, beta = tf.split(param, 2, axis=1)
        out *= tf.expand_dims(tf.expand_dims(gamma, 1), 1)
        out += tf.expand_dims(tf.expand_dims(beta, 1), 1)
        out = tf.nn.relu(out)

    out_shape = out.get_shape()
    factors = [8, 10, 10]

    # [B, s1*s2, s3]
    out = tf.reshape(out, (-1, np.prod(out_shape[1:-1]), out_shape[-1]))
    projection_mat = tf.get_variable(
        name='projection_matrix',
        shape=(sum(factors), np.prod(out_shape[1:-1])),
        dtype=tf.float32, trainable=True)
    projection_mat = tf.expand_dims(projection_mat, axis=0)
    projection_mat = tf.tile(projection_mat, [tf.shape(out)[0], 1, 1])
    out = tf.matmul(projection_mat, out)  # [B, sum(fac), s3]
    # [B, factor[0], s3] [B, factor[1], s3] [B, factor[2], s3]
    fac1, fac2, fac3 = tf.split(out, factors, axis=1)
    out = tensor_concat(fac1, fac2, fac3)  # [B, f1, f2, f3, s3]
    # [B, 800, s3*3]
    out = tf.reshape(out, [-1, np.prod(factors), out_shape[-1]*3])
    print('tensor concat: {}'.format(out))
    goal_tile = tf.expand_dims(
        tf.layers.dense(goal_embedding, out_shape[-1]), 1)  # [B, 1, s3]
    print('goal: {}'.format(goal_tile))
    goal_tile = tf.tile(
        goal_tile, multiples=[1, np.prod(factors), 1])
    # TODO(ydjiang): include context vector too?
    out = tf.concat([out, goal_tile], axis=-1)
    out = tf.expand_dims(out, axis=1)
    # TODO(ydjiang): wider network here?
    out = tf.nn.relu(tf.layers.conv2d(out, 100, 1, 1))
    out = tf.nn.relu(tf.layers.conv2d(out, 32, 1, 1))
    out = tf.layers.conv2d(out, 1, 1, 1)
    return tf.squeeze(out, axis=[1, 3])
