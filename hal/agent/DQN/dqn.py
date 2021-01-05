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
"""Simple Double DQN agent."""
# pylint: disable=line-too-long

import numpy as np
import tensorflow as tf


def mlp_policy(input_shape, action_shape):
  """Returns a keras model of fully connected layers."""
  return tf.keras.Sequential([
      tf.keras.layers.Dense(64, input_shape=(input_shape,), activation='relu'),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(action_shape),
  ])


class DoubleDQN:
  """A basic double dqn agent.

  Attributes:
    learning_rate: learning rate of the optimizer
    gamma: future discount of the agent
    online_model: model that contains the most recent weight
    target_model: model that contains the weight of the moving average
    optimizer: the optimizer used for training
    loss: the loss function
    global_step: how many simulation steps have been taken
    learn_step: how many optimization steps have been taken
    batch_size: size of the batch
    state_size: size of the observation
    action_size: size of the action
    saver: tensorflow weight saver
    manager: tensorflow checkpoint manager
  """

  def __init__(self,
               learning_rate=0.001,
               discount=0.99,
               batch_size=64,
               state_size=4,
               action_size=2,
               use_huber_loss=False,
               state_process_fn=None,
               action_process_fn=None,
               action_post_process_fn=None,
               model_dir=None):
    """Initialize the double dqn agent.

    Args:
      learning_rate: learning rate of the optimizer
      discount: future discount of the agent
      batch_size: size of the batch
      state_size: size of the observation
      action_size: size of the action
      use_huber_loss: whether to use huber loss or l2 loss
      state_process_fn: function that process state before compute
      action_process_fn: function that process action before compute
      action_post_process_fn:  function that process state after compute
      model_dir: optional directory for saving weights
    """

    # hyper parameters
    self.learning_rate = learning_rate
    self.gamma = discount

    self.online_model = mlp_policy(state_size, action_size)
    self.target_model = mlp_policy(state_size, action_size)
    self.online_model.build()
    self.target_model.build()
    self.optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate,)
    self.loss = tf.keras.losses.Huber()

    self.global_step = tf.Variable(1, name='global_step')
    self.learn_step = tf.Variable(1, name='learn_step')

    self.batch_size = batch_size
    self.state_size = state_size
    self.action_size = action_size
    self._use_huber_loss = use_huber_loss

    self._state_process_fn = state_process_fn
    self._action_process_fn = action_process_fn
    self._action_post_process_fn = action_post_process_fn

    if model_dir:
      self.saver = tf.train.Checkpoint(
          optimizer=self.optimizer,
          online_model=self.online_model,
          target_model=self.target_model,
          step=self.global_step)
      self.manager = tf.train.CheckpointManager(
          self.saver, model_dir, max_to_keep=5, checkpoint_name='model')

  def update_target_network(self):
    """"Update the target network with moving average."""
    self.target_model.set_weights(self.online_model.get_weights())

  def step(self, state, env, epsilon):
    """"Produce action on a state."""
    del env
    state = state[None, :]
    if self._state_process_fn:
      state = self._state_process_fn(state)
    if np.random.rand() <= epsilon:
      action = np.random.choice(self.action_size)
    else:
      state = tf.convert_to_tensor(state, dtype=tf.float32)
      q_value = self.online_model(state, training=False)[0]
      action = np.argmax(q_value)
    if self._action_post_process_fn:
      action = self._action_post_process_fn(action)
    return action

  def train(self, batch):
    """Train the agent on a batch of transitions."""
    states = batch['obs']
    actions = batch['action']
    rewards = batch['reward']
    next_states = batch['obs_next']
    dones = batch['done']

    if self._state_process_fn:
      states = self._state_process_fn(states)
      next_states = self._state_process_fn(next_states)

    if self._action_process_fn:
      actions = self._action_process_fn(actions)

    batch = {
        'states':
            tf.convert_to_tensor(np.vstack(states), dtype=tf.float32),
        'actions':
            tf.convert_to_tensor(actions, dtype=tf.int32),
        'rewards':
            tf.convert_to_tensor(rewards, dtype=tf.float32),
        'next_states':
            tf.convert_to_tensor(np.vstack(next_states), dtype=tf.float32),
        'dones':
            tf.convert_to_tensor(dones, dtype=tf.float32)
    }
    loss = self._train(batch).numpy()
    return {'loss': loss}

  @tf.function
  def _train(self, batch):
    """Update models with data in batch."""
    dqn_variable = self.online_model.trainable_variables
    with tf.GradientTape() as tape:
      tape.watch(dqn_variable)
      error = self._loss(batch)
    dqn_grads = tape.gradient(error, dqn_variable)
    self.optimizer.apply_gradients(zip(dqn_grads, dqn_variable))
    return error

  def _loss(self, batch):
    """Compute the td loss."""
    states, next_states = batch['states'], batch['next_states']
    rewards, dones, actions = batch['rewards'], batch['dones'], batch['actions']
    target_q = self.target_model(next_states, training=True)
    online_q = self.online_model(next_states, training=True)
    next_action = tf.argmax(online_q, axis=1)
    target_value = tf.reduce_sum(
        tf.one_hot(next_action, self.action_size) * target_q, axis=1)

    target_value = (1 - dones) * self.gamma * target_value + rewards
    main_q = self.online_model(states)
    main_value = tf.reduce_sum(
        tf.one_hot(actions, self.action_size) * main_q, axis=1)
    if self._use_huber_loss:
      error = self.loss(target_value, main_value)
    else:
      error = tf.square(main_value - tf.stop_gradient(target_value)) * 0.5
      error = tf.reduce_mean(error)
    return error

  def init_networks(self):
    """Initialize the weights."""
    print('No need to initialize in eager mode.')

  def save_model(self, model_dir=None):
    """Save current model weights with manager or to model_dir.

    Args:
      model_dir: optional directory for saving the model weights
    """
    assert self.manager or model_dir, 'No manager and no model dir!'
    save_path = self.manager.save()
    print('Save model: step {} to {}'.format(int(self.global_step), save_path))

  def load_model(self, model_dir=None):
    """Load current model weights with manager or from model_dir.

    Args:
      model_dir: optional directory for saving the model weights
    """
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
    """Get the current value of global step in python integer."""
    return int(self.global_step)
