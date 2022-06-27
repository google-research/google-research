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
# pylint: disable=line-too-long
"""Build DQN agent with state observation for state input."""

from __future__ import absolute_import
from __future__ import division

import random

import numpy as np
import tensorflow as tf

from hal.agent.tf2_utils import soft_variables_update
from hal.agent.tf2_utils import stack_conv_layer
# from hal.agent.tf2_utils import stack_dense_layer
from hal.agent.tf2_utils import vector_tensor_product


class StateDQN:
  """DQN that uses the ground truth states.

  Attributes:
    cfg: Configuration object
    saver: tensorflow weight saver
    manager: tensorflow checkpoint manager
    global_step: how many simulation steps have been taken
    online_models: model that contains the most recent weight
    target_models: model that contains the weight of the moving average
    optimizer: the optimizer used for training
  """

  def __init__(self, cfg):
    self.cfg = cfg
    self._build()
    if 'model_dir' in cfg.as_dict():
      self.saver = tf.train.Checkpoint(
          optimizer=self.optimizer,
          online_model=self.online_models,
          target_model=self.target_models,
          step=self.global_step)
      self.manager = tf.train.CheckpointManager(
          self.saver, cfg.model_dir, max_to_keep=5, checkpoint_name='model')

  def _build(self):
    """Builds the model and creates the weight tensors."""
    with tf.name_scope('StateDQN'):
      self.global_step = tf.Variable(0, name='global_step')
      self._create_models(self.cfg)

  def _create_models(self, cfg):
    """Build the computation graph for the agent."""
    self.online_models = self._build_q_naive(cfg, 'online_network')
    self.target_models = self._build_q_naive(cfg, 'target_network')
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)

  def step(self, obs, env, explore_prob=0.0, debug=False):
    """Returns an action based on obs."""
    obs = tf.convert_to_tensor(np.expand_dims(obs, axis=0), dtype=tf.float32)
    if random.uniform(0, 1) < explore_prob:
      return np.int32(env.sample_random_action())
    else:
      predict_action = np.int32(np.squeeze(self._step(obs).numpy()))
      return predict_action

  @tf.function
  def _step(self, obs):
    """Take step using the tf models."""
    online_values = self.online_models(obs, training=False)
    if self.cfg.action_type == 'discrete':
      predict_action = tf.argmax(online_values, axis=1)
      return predict_action
    elif self.cfg.action_type == 'perfect':
      input_max_q = tf.reduce_max(online_values, axis=2)
      input_select = tf.argmax(input_max_q, axis=1)
      action_max_q = tf.reduce_max(online_values, axis=1)
      action_select = tf.argmax(action_max_q, axis=1)
      predict_action = tf.stack([input_select, action_select], axis=1)
      return predict_action

  def train(self, batch):
    """Take a single gradient step on the batch."""

    if self.cfg.action_type == 'perfect':
      batch['action'] = np.stack(batch['action'])

    # TODO(ydjiang): figure out why reward cannot be converted to tensor as is
    batch['reward'] = list(batch['reward'])

    # make everything tensor except for g
    for k in batch:
      batch[k] = tf.convert_to_tensor(batch[k])

    loss = self._train(batch).numpy()

    return {'loss': loss}

  @tf.function
  def _train(self, batch):
    with tf.GradientTape() as tape:
      loss = self._loss(batch)
    variables = self.online_models.trainable_variables
    gradients = tape.gradient(loss, variables)
    self.optimizer.apply_gradients(zip(gradients, variables))
    return loss

  def _loss(self, batch):
    """Compute td loss on a batch."""
    obs, action, obs_next = batch['obs'], batch['action'], batch['obs_next']
    r = batch['reward']

    q = self._compute_q(obs, action)
    q_next = self._compute_q_next(obs_next)
    if self.cfg.masking_q:
      mask = tf.cast(r == self.cfg.max_reward, dtype=tf.float32)
      q_next *= mask

    gamma = self.cfg.discount
    td_target = tf.clip_by_value(r + gamma * q_next, -30.0, 10.0)
    loss = tf.compat.v1.losses.huber_loss(
        tf.stop_gradient(td_target), q, reduction=tf.losses.Reduction.MEAN)
    return loss

  def _compute_q(self, obs, action):
    """Compute q values of current observation."""
    online_q_values = self.online_models(obs, training=True)
    if self.cfg.action_type == 'perfect':
      stacked_indices = tf.concat(
          [tf.expand_dims(tf.range(tf.shape(action)[0]), axis=1), action],
          axis=1)
      predicted_q = tf.gather_nd(online_q_values, stacked_indices)
    elif self.cfg.action_type == 'discrete':
      ac_dim = self.cfg.per_input_ac_dim[0]
      action_onehot = tf.one_hot(action, ac_dim, dtype=tf.float32)
      predicted_q = tf.reduce_sum(
          tf.multiply(online_q_values, action_onehot), axis=1)
    else:
      raise ValueError('Unrecognized action type: {}'.format(
          self.cfg.action_type))
    return predicted_q

  def _compute_q_next(self, obs_next):
    """Compute q values of next observation."""
    online_values = self.online_models(obs_next, training=True)
    target_values = self.target_models(obs_next, training=True)
    if self.cfg.action_type == 'discrete':
      predict_action = tf.argmax(online_values, axis=1)
      double_q = target_values[:, predict_action]
    elif self.cfg.action_type == 'perfect':
      input_max_q = tf.reduce_max(online_values, axis=2)
      input_select = tf.argmax(input_max_q, axis=1, output_type=tf.int32)
      action_max_q = tf.reduce_max(online_values, axis=1)
      action_select = tf.argmax(action_max_q, axis=1, output_type=tf.int32)
      indices = tf.range(input_max_q.shape[0], dtype=tf.int32)
      stacked_indices = tf.stack([indices, input_select, action_select], axis=1)
      double_q = tf.gather_nd(target_values, stacked_indices)
    else:
      raise ValueError('Unrecognized action type: {}'.format(
          self.cfg.action_type))
    return double_q

  def _build_q_naive(self, cfg, name):
    """Build the naive q function."""
    per_input_ac_dim = cfg.per_input_ac_dim
    inputs = tf.keras.layers.Input(shape=(5, cfg.obs_dim[1]))
    flat_input = tf.reshape(inputs, (-1, int(5 * cfg.obs_dim[1])))
    out = tf.keras.layers.Dense(256, activation='relu')(flat_input)
    out = tf.keras.layers.Dense(256, activation='relu')(out)
    out = tf.keras.layers.Dense(5 * per_input_ac_dim)(out)
    out = tf.keras.layers.Reshape((5, per_input_ac_dim))(out)
    return tf.keras.Model(name=name, inputs=inputs, outputs=out)

  def _build_q_perfect(self, cfg, name):
    """Build the q function for perfect action space."""
    per_input_ac_dim = cfg.per_input_ac_dim
    des_len = cfg.descriptor_length

    inputs = tf.keras.layers.Input(shape=(cfg.obs_dim[0], cfg.obs_dim[1]))

    shape_layer = tf.keras.layers.Lambda(tf.shape)
    num_object = shape_layer(inputs)[1]
    tp_concat_orig = vector_tensor_product(inputs, inputs)
    conv_layer_cfg = [[des_len * 8, 1, 1], [des_len * 4, 1, 1], [des_len, 1, 1]]
    # [B, ?, ?, des_len]
    tp_concat = stack_conv_layer(conv_layer_cfg)(tp_concat_orig)

    expand_dims_layer = tf.keras.layers.Lambda(
        lambda inputs: tf.expand_dims(inputs[0], axis=inputs[1]))

    # [B, ?, ?, 1]
    conv_layer_cfg = [[32, 3, 1], [16, 3, 1], [1, 1, 1]]
    obs_query = stack_conv_layer(conv_layer_cfg)(tp_concat)
    # [B, 1, ?*?]
    obs_query = tf.keras.layers.Reshape((1, -1))(obs_query)
    matmul_layer = tf.keras.layers.Lambda(
        lambda inputs: tf.matmul(inputs[0], inputs[1]))
    weight = tf.keras.layers.Softmax()(obs_query)  # [B, 1, ?*?]
    prod = matmul_layer((weight, tf.keras.layers.Reshape(
        (-1, des_len))(tp_concat)))  # [B, 1, des_len]

    tile_layer = tf.keras.layers.Lambda(
        lambda inputs: tf.tile(inputs[0], multiples=inputs[1]))
    # [B, ?, des_len]
    pair_wise_summary = tile_layer((prod, [1, num_object, 1]))
    # [B, ?, des_len+di]
    augemented_inputs = tf.keras.layers.Concatenate(axis=-1)(
        [inputs, pair_wise_summary])
    # [B, ?, 1, des_len+di+dg]
    augemented_inputs = expand_dims_layer((augemented_inputs, 2))
    conv_layer_cfg = [[per_input_ac_dim * 64, 1, 1],
                      [per_input_ac_dim * 64, 1, 1], [per_input_ac_dim, 1, 1]]
    # [B, ?, per_input_ac_dim]
    q_out = tf.keras.layers.Reshape((-1, per_input_ac_dim))(
        stack_conv_layer(conv_layer_cfg)(augemented_inputs))
    q_out_layer = tf.keras.Model(name=name, inputs=inputs, outputs=q_out)
    return q_out_layer

  def _build_q_discrete(self, cfg, name):
    """Build the q function for discrete action space."""
    ac_dim = cfg.ac_dim

    inputs = tf.keras.layers.Input(shape=(cfg.obs_dim[0], cfg.obs_dim[1]))

    # [B, ?, 1, des_len+di+dg]
    augemented_inputs = tf.expand_dims(inputs, axis=2)
    cfg = [[ac_dim // 8, 1, 1], [ac_dim // 8, 1, 1]]
    heads = []
    for _ in range(8):
      # [B, ?, 1, ac_dim//8]
      head_out = stack_conv_layer(cfg)(augemented_inputs)
      weights = tf.keras.layers.Conv2D(1, 1, 1)(head_out)  # [B, ?, 1, 1]
      softmax_weights = tf.nn.softmax(weights, axis=1)  # [B, ?, 1, 1]
      heads.append(tf.reduce_sum(softmax_weights * head_out, axis=(1, 2)))
    # heads = 8 X [B, ac_dim//8]
    out = tf.concat(heads, axis=1)  # [B, ac_dim]
    out = tf.keras.layers.Dense(ac_dim)(out)
    q_out_layer = tf.keras.Model(name=name, inputs=inputs, outputs=out)
    return q_out_layer

  def update_target_network(self, polyak_rate=None):
    """"Update the target network with moving average."""
    online_var = self.online_models.trainable_variables
    target_var = self.target_models.trainable_variables
    target_update_op = soft_variables_update(online_var, target_var,
                                             self.cfg.polyak_rate)
    tf.group(*target_update_op)

  def init_networks(self):
    """Initialize the weights."""
    print('No need to initialize in eager mode.')

  def save_model(self, model_dir=None):
    """Save current model weights with manager or to model_dir."""
    assert self.manager or model_dir, 'No manager and no model dir!'
    save_path = self.manager.save()
    print('Save model: step {} to {}'.format(int(self.global_step), save_path))

  def load_model(self, model_dir=None):
    """Load current model weights with manager or from model_dir."""
    assert self.manager or model_dir, 'No manager and no model dir!'
    if not model_dir:
      model_dir = self.manager.latest_checkpoint
    else:
      model_dir = tf.train.latest_checkpoint(model_dir)
    self.saver.restore(model_dir)

  def increase_global_step(self):
    """Increment gloabl step by 1."""
    return self.global_step.assign_add(1).numpy()

  def get_global_step(self):
    """Get the current gloabl step as python integer."""
    return int(self.global_step)
