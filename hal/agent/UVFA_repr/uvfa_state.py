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

# Lint as: python3
# pylint: disable=unused-argument
# pylint: disable=g-missing-from-attributes
"""Build UVFA with state observation for state input."""

from __future__ import absolute_import
from __future__ import division

import os
import random

import numpy as np
import tensorflow as tf

from hal.agent.common import encoder
from hal.agent.common import get_vars
from hal.agent.common import stack_conv_layer
from hal.agent.common import stack_dense_layer
from hal.agent.common import vector_tensor_product


class StateUVFA:
  """UVFA that uses the ground truth states.

  Attributes:
    cfg: configuration object
    sess: tf session
    saver: Tensorflow Saver
    global_step: integer tensor that counts how many step the agent has taken
    increment_global_step_op: operation to increase the global step
    word_inputs: input placeholder for the instruction
    inputs: input placeholder of the observation
    is_training: placeholder that indicates if the agent is in training mode
  """

  def __init__(self, cfg, sess=None):
    """Initialize the State UVFA.

    Args:
      cfg: configuration object for the agent
      sess: optional tf sesseion
    """
    # making session
    self.sess = sess
    if not sess:
      self.sess = tf.Session()
    self.cfg = cfg
    self._build()
    self.saver = tf.train.Saver()

  def _build(self):
    self.global_step = tf.get_variable(
        'global_step',
        shape=(),
        dtype=tf.int32,
        initializer=tf.zeros_initializer())
    self.increment_global_step_op = tf.assign_add(self.global_step, 1)
    self.create_inputs(self.cfg)
    self.create_models(self.cfg)
    self.create_ops(self.cfg)

  def step(self, obs, g, env, explore_prob=0.0, debug=False):
    """Take a step in the environment and predicts an action.

    Args:
      obs: observation from which action will be taken
      g: the instruction / goal
      env: the RL environment
      explore_prob: probability to take random action
      debug: if the mode is debug and extra information should be printed

    Returns:
      an action
    """
    if random.uniform(0, 1) < explore_prob:
      return env.sample_random_action()
    else:
      feed_dict = {
          self.inputs: [obs], self.word_inputs: [g], self.is_training: False
      }
      action = self.sess.run(self.predict_action, feed_dict)
      return np.squeeze(action)

  def train(self, batch):
    """Takes a single gradient step on the batch."""
    feed_dict = {
        self.inputs: batch['obs_next'],
        self.word_inputs: batch['g'],
        self.is_training: True
    }
    p_action, q_next_target = self.sess.run(
        [self.predict_action, self.all_q_target,], feed_dict)
    if self.cfg.action_type == 'perfect':
      indices = np.arange(p_action.shape[0])
      double_q = q_next_target[indices, p_action[:, 0], p_action[:, 1]]
      batch['action'] = np.stack(batch['action'])
    elif self.cfg.action_type == 'discrete':
      double_q = q_next_target[:, p_action]
    else:
      raise ValueError(
          'Unrecognized action type: {}'.format(self.cfg.action_type))

    r = batch['reward']
    if self.cfg.masking_q:
      double_q *= (1. - (r >= self.cfg.reward_scale).astype(np.float32))

    gamma = self.cfg.discount
    regression_target = np.clip(r + gamma * double_q, -1. / (1 - gamma), 10.0)

    _, loss = self.sess.run(
        [self.train_op, self.loss],
        {
            self.inputs: batch['obs'],
            self.q_next: regression_target,
            self.action_ph: batch['action'],
            self.word_inputs: batch['g'],
            self.is_training: True
        }
    )
    return {'loss': loss}

  def save_model(self, model_dir):
    """Saves the current weights to model_dir decorated by global step."""
    self.saver.save(
        self.sess,
        os.path.join(model_dir, 'model.ckpt'),
        global_step=self.sess.run(self.global_step)
    )

  def load_model(self, model_dir):
    """Loads model from model_dir."""
    self.saver.restore(
        self.sess,
        tf.train.latest_checkpoint(model_dir),
    )

  def increase_global_step(self):
    """Increments global step."""
    return self.sess.run(self.increment_global_step_op)

  def get_global_step(self):
    """Returns the current global step."""
    return self.sess.run(self.global_step)

  def create_inputs(self, cfg):
    """Builds input placeholders."""
    if cfg.intruction_repr == 'language':
      self.word_inputs = tf.placeholder(
          shape=(None, None), dtype=tf.int32, name='text_ph')
    elif cfg.intruction_repr == 'one_hot':
      self.word_inputs = tf.placeholder(
          shape=(None), dtype=tf.int32, name='goal_ph')
    else:
      raise ValueError('Unrecognized instruction type: {}'.format(
          cfg.instruction_repr))

    # variable number of inputs ([B, ?, di])
    self.inputs = tf.placeholder(
        shape=[None] + cfg.obs_dim, dtype=tf.float32, name='input_ph')
    print('Input placeholder: {}'.format(self.inputs))
    self.is_training = tf.placeholder(
        shape=(), dtype=tf.bool, name='training_indicator_ph')

  def create_models(self, cfg):
    """Builds the computation graph for the agent."""
    def make_policy():
      """Build one copy of the model."""
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

      if cfg.action_type == 'perfect':
        print('using perfect action Q function...')
        all_q, predict_object, predict_object_action = self.build_q_perfect(
            cfg, goal_embedding)
        predict_action = tf.stack(
            [predict_object, predict_object_action], axis=1)
        action = tf.placeholder(shape=(None, 2), dtype=tf.int32)
        stacked_indices = tf.concat(
            [tf.expand_dims(tf.range(0, tf.shape(action)[0]), axis=1), action],
            axis=1
        )
        q = tf.gather_nd(all_q, stacked_indices)
        artifact.update(
            {
                'all_q': all_q,
                'predict_object': predict_object,
                'predict_object_action': predict_object_action,
                'predict_action': predict_action,
                'action_ph': action,
                'q': q,
            }
        )
      elif cfg.action_type == 'discrete':
        print('using discrete action Q function...')
        ac_dim = cfg.per_input_ac_dim[0]
        all_q = self.build_q_discrete(goal_embedding, ac_dim)
        predict_action = tf.argmax(all_q, axis=-1)
        action = tf.placeholder(shape=None, dtype=tf.int32)
        action_onehot = tf.one_hot(
            action, ac_dim, dtype=tf.float32)
        q = tf.reduce_sum(
            tf.multiply(all_q, action_onehot), axis=1)
        artifact.update(
            {
                'all_q': all_q,
                'predict_action': predict_action,
                'action_ph': action,
                'action_onehot': action_onehot,
                'q': q,
            }
        )
      else:
        raise ValueError('Unrecognized action type: {}'.format(
            cfg.action_type))
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

  def create_ops(self, cfg):
    """Create training and updating ops.

    Args:
      cfg: configuration of the experiments
    """
    self.q_next = tf.placeholder(shape=None, dtype=tf.float32)
    self.loss = tf.losses.huber_loss(
        self.q_next, self.q, reduction=tf.losses.Reduction.MEAN)
    self.optimizer = tf.train.AdamOptimizer(learning_rate=cfg.learning_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='online')
    with tf.control_dependencies(update_ops):
      self.train_op = self.optimizer.minimize(self.loss)

    self.target_update_op = tf.group([
        v_t.assign(cfg.polyak_rate*v_t + (1-cfg.polyak_rate)*v)
        for v, v_t in zip(get_vars('online'), get_vars('target'))
    ])
    self.init_op = tf.global_variables_initializer()
    self.target_init_op = tf.group([
        v_t.assign(v)
        for v, v_t in zip(get_vars('online'), get_vars('target'))
    ])

  def init_networks(self):
    """Initializes the weights."""
    self.sess.run(self.init_op)

  def update_target_network(self):
    """Run the moving average update operations."""
    self.sess.run(self.target_update_op)

  def build_q_perfect(self, cfg, goal_embedding):
    """Build the Q-function for perfect action space.

    Args:
      cfg: configuration of the experiments
      goal_embedding: embedding tensor of the instructions

    Returns:
      q_out: output q values
      input_select: object with maximum q value
      action_select: actions with maximum q value
    """
    per_input_ac_dim = cfg.per_input_ac_dim
    des_len, inner_len = cfg.descriptor_length, cfg.inner_product_length

    num_object = tf.shape(self.inputs)[1]
    tp_concat = vector_tensor_product(self.inputs, self.inputs)
    conv_layer_cfg = [[des_len * 8, 1, 1], [des_len * 4, 1, 1], [des_len, 1, 1]]
    # [B, ?, ?, des_len]
    tp_concat = stack_conv_layer(tp_concat, conv_layer_cfg)

    # similarity with goal
    goal_key = stack_dense_layer(
        goal_embedding, [inner_len*2, inner_len])  # [B, d_inner]
    goal_key = tf.expand_dims(goal_key, 1)  # [B, 1, d_inner]
    # [B, ?, ?, d_inner]
    obs_query = tf.layers.conv2d(tp_concat, inner_len, 1, padding='same')
    # [B, ?*?, d_inner]
    obs_query = tf.reshape(obs_query, [-1, num_object**2, inner_len])
    obs_query_t = tf.transpose(obs_query, perm=(0, 2, 1))  # [B, d_inner, ?*?]
    inner = tf.matmul(goal_key, obs_query_t)  # [B, 1, ?*?]
    weight = tf.nn.softmax(inner, axis=-1)  # [B, 1, ?*?]
    prod = tf.matmul(
        weight,
        tf.reshape(tp_concat, [-1, num_object**2, des_len]))  # [B, 1, des_len]

    goal_embedding_ = tf.expand_dims(goal_embedding, 1)  # [B, 1, dg]
    # [B, ?, dg]
    goal_embedding_ = tf.tile(goal_embedding_, multiples=[1, num_object, 1])
    # [B, ?, des_len]
    pair_wise_summary = tf.tile(prod, multiples=[1, num_object, 1])
    # [B, ?, des_len+di+dg]
    augemented_inputs = tf.concat(
        [self.inputs, pair_wise_summary, goal_embedding_], axis=-1)
    # [B, ?, 1, des_len+di+dg]
    augemented_inputs = tf.expand_dims(augemented_inputs, axis=2)
    conv_layer_cfg = [
        [per_input_ac_dim*64, 1, 1],
        [per_input_ac_dim*64, 1, 1],
        [per_input_ac_dim, 1, 1]
    ]
    # [B, ?, per_input_ac_dim]
    q_out = tf.squeeze(
        stack_conv_layer(augemented_inputs, conv_layer_cfg), axis=2)
    input_max_q = tf.reduce_max(q_out, axis=2)
    input_select = tf.argmax(input_max_q, axis=1)
    action_max_q = tf.reduce_max(q_out, axis=1)
    action_select = tf.argmax(action_max_q, axis=1)

    return q_out, input_select, action_select

  def build_q_discrete(self, cfg, goal_embedding):
    """Returns the q function for discrete action space.

    Args:
      cfg: configuration of the experiments
      goal_embedding: embedding tensor of the instructions

    Returns:
      output q values of all actions
    """
    ac_dim = cfg.ac_dim
    des_len, inner_len = cfg.descriptor_length, cfg.inner_product_length

    num_object = tf.shape(self.inputs)[1]
    tp_concat = vector_tensor_product(self.inputs, self.inputs)
    conv_layer_cfg = [[des_len*8, 1, 1], [des_len*4, 1, 1], [des_len, 1, 1]]
    # [B, ?, ?, des_len]
    tp_concat = stack_conv_layer(tp_concat, conv_layer_cfg)

    # similarity with goal
    goal_key = stack_dense_layer(
        goal_embedding, [inner_len*2, inner_len])  # [B, d_inner]
    goal_key = tf.expand_dims(goal_key, 1)  # [B, 1, d_inner]
    # [B, ?, ?, d_inner]
    obs_query = tf.layers.conv2d(tp_concat, inner_len, 1, padding='same')
    # [B, ?*?, d_inner]
    obs_query = tf.reshape(obs_query, [-1, num_object**2, inner_len])
    obs_query_t = tf.transpose(obs_query, perm=(0, 2, 1))  # [B, d_inner, ?*?]
    inner = tf.matmul(goal_key, obs_query_t)  # [B, 1, ?*?]
    weight = tf.nn.softmax(inner, axis=-1)  # [B, 1, ?*?]
    prod = tf.matmul(
        weight,
        tf.reshape(tp_concat, [-1, num_object**2, des_len])
    )  # [B, 1, des_len]
    goal_embedding_ = tf.expand_dims(goal_embedding, 1)  # [B, 1, dg]
    # [B, ?, dg]
    goal_embedding_ = tf.tile(goal_embedding_, multiples=[1, num_object, 1])
    # [B, ?, des_len]
    pair_wise_summary = tf.tile(prod, multiples=[1, num_object, 1])
    # [B, ?, des_len+di+dg]
    augemented_inputs = tf.concat(
        [self.inputs, pair_wise_summary, goal_embedding_], axis=-1)
    # [B, ?, 1, des_len+di+dg]
    augemented_inputs = tf.expand_dims(augemented_inputs, axis=2)
    cfg = [[ac_dim//8, 1, 1], [ac_dim//8, 1, 1]]
    heads = []
    for _ in range(8):
      # [B, ?, 1, ac_dim//8]
      head_out = stack_conv_layer(augemented_inputs, cfg)
      weights = tf.layers.conv2d(head_out, 1, 1)  # [B, ?, 1, 1]
      softmax_weights = tf.nn.softmax(weights, axis=1)  # [B, ?, 1, 1]
      heads.append(
          tf.reduce_sum(softmax_weights*head_out, axis=(1, 2))
      )
    # heads = 8 X [B, ac_dim//8]
    out = tf.concat(heads, axis=1)  # [B, ac_dim]
    return tf.layers.dense(out, ac_dim)
