# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Implementation of DualDICE https://openreview.net/pdf?id=SygrvzwniE ."""

import numpy as np
import tensorflow.compat.v2 as tf
from value_dice import keras_utils
from value_dice import twin_sac


EPS = np.finfo(np.float32).eps


def weighted_softmax(x, weights, axis=0):
  x = x - tf.reduce_max(x, axis=axis)
  return weights * tf.exp(x) / tf.reduce_sum(
      weights * tf.exp(x), axis=axis, keepdims=True)


class ValueDICE(object):
  """Class that implements DualDICE training."""

  def __init__(self, state_dim, action_dim, log_interval, nu_lr, actor_lr,
               alpha_init, hidden_size):
    """Creates a DualDICE object.

    Args:
      state_dim: State size.
      action_dim: Action size.
      log_interval: Log losses every N steps.
      nu_lr: nu network learning rate.
      actor_lr: actor learning rate.
      alpha_init: Initial temperature value.
      hidden_size: A number of hidden units.
    """

    self.nu_net = tf.keras.Sequential([
        tf.keras.layers.Dense(
            hidden_size,
            input_shape=(state_dim + action_dim,),
            activation=tf.nn.relu,
            kernel_initializer='orthogonal'),
        tf.keras.layers.Dense(
            hidden_size, activation=tf.nn.relu,
            kernel_initializer='orthogonal'),
        tf.keras.layers.Dense(
            1, kernel_initializer='orthogonal', use_bias=False)
    ])

    self.log_interval = log_interval

    self.avg_loss = tf.keras.metrics.Mean('dual dice loss', dtype=tf.float32)
    self.avg_ratio = tf.keras.metrics.Mean('dual dice ratio', dtype=tf.float32)

    self.avg_nu_expert = tf.keras.metrics.Mean(
        'dual dice expert', dtype=tf.float32)
    self.avg_nu_rb = tf.keras.metrics.Mean('dual dice nu rb', dtype=tf.float32)

    self.nu_reg_metric = tf.keras.metrics.Mean('nu reg', dtype=tf.float32)

    self.nu_optimizer = tf.keras.optimizers.Adam(learning_rate=nu_lr)

    self.actor = twin_sac.Actor(state_dim, action_dim)
    self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
    self.avg_actor_loss = tf.keras.metrics.Mean('actor_loss', dtype=tf.float32)
    self.avg_alpha_loss = tf.keras.metrics.Mean('alpha_loss', dtype=tf.float32)
    self.avg_actor_entropy = tf.keras.metrics.Mean(
        'actor_entropy', dtype=tf.float32)
    self.avg_alpha = tf.keras.metrics.Mean('alpha', dtype=tf.float32)

    self.log_alpha = tf.Variable(tf.math.log(alpha_init), trainable=True)
    self.alpha_optimizer = tf.keras.optimizers.Adam()

  @property
  def alpha(self):
    return tf.exp(self.log_alpha)

  @tf.function
  def update(self,
             expert_dataset_iter,
             policy_dataset_iter,
             discount,
             replay_regularization=0.05,
             nu_reg=10.0):
    """A function that updates nu network.

    When replay regularization is non-zero, it learns
    (d_pi * (1 - replay_regularization) + d_rb * replay_regulazation) /
    (d_expert * (1 - replay_regularization) + d_rb * replay_regulazation)
    instead.

    Args:
      expert_dataset_iter: An tensorflow graph iteratable over expert data.
      policy_dataset_iter: An tensorflow graph iteratable over training policy
        data, used for regularization.
      discount: An MDP discount.
      replay_regularization: A fraction of samples to add from a replay buffer.
      nu_reg: A grad penalty regularization coefficient.
    """

    (expert_states, expert_actions,
     expert_next_states) = expert_dataset_iter.get_next()

    expert_initial_states = expert_states

    rb_states, rb_actions, rb_next_states, _, _ = policy_dataset_iter.get_next(
    )[0]

    with tf.GradientTape(
        watch_accessed_variables=False, persistent=True) as tape:
      tape.watch(self.actor.variables)
      tape.watch(self.nu_net.variables)

      _, policy_next_actions, _ = self.actor(expert_next_states)
      _, rb_next_actions, rb_log_prob = self.actor(rb_next_states)

      _, policy_initial_actions, _ = self.actor(expert_initial_states)

      # Inputs for the linear part of DualDICE loss.
      expert_init_inputs = tf.concat(
          [expert_initial_states, policy_initial_actions], 1)

      expert_inputs = tf.concat([expert_states, expert_actions], 1)
      expert_next_inputs = tf.concat([expert_next_states, policy_next_actions],
                                     1)

      rb_inputs = tf.concat([rb_states, rb_actions], 1)
      rb_next_inputs = tf.concat([rb_next_states, rb_next_actions], 1)

      expert_nu_0 = self.nu_net(expert_init_inputs)
      expert_nu = self.nu_net(expert_inputs)
      expert_nu_next = self.nu_net(expert_next_inputs)

      rb_nu = self.nu_net(rb_inputs)
      rb_nu_next = self.nu_net(rb_next_inputs)

      expert_diff = expert_nu - discount * expert_nu_next
      rb_diff = rb_nu - discount * rb_nu_next

      linear_loss_expert = tf.reduce_mean(expert_nu_0 * (1 - discount))

      linear_loss_rb = tf.reduce_mean(rb_diff)

      rb_expert_diff = tf.concat([expert_diff, rb_diff], 0)
      rb_expert_weights = tf.concat([
          tf.ones(expert_diff.shape) * (1 - replay_regularization),
          tf.ones(rb_diff.shape) * replay_regularization
      ], 0)

      rb_expert_weights /= tf.reduce_sum(rb_expert_weights)
      non_linear_loss = tf.reduce_sum(
          tf.stop_gradient(
              weighted_softmax(rb_expert_diff, rb_expert_weights, axis=0)) *
          rb_expert_diff)

      linear_loss = (
          linear_loss_expert * (1 - replay_regularization) +
          linear_loss_rb * replay_regularization)

      loss = (non_linear_loss - linear_loss)

      alpha = tf.random.uniform(shape=(expert_inputs.shape[0], 1))

      nu_inter = alpha * expert_inputs + (1 - alpha) * rb_inputs
      nu_next_inter = alpha * expert_next_inputs + (1 - alpha) * rb_next_inputs

      nu_inter = tf.concat([nu_inter, nu_next_inter], 0)

      with tf.GradientTape(watch_accessed_variables=False) as tape2:
        tape2.watch(nu_inter)
        nu_output = self.nu_net(nu_inter)
      nu_grad = tape2.gradient(nu_output, [nu_inter])[0] + EPS
      nu_grad_penalty = tf.reduce_mean(
          tf.square(tf.norm(nu_grad, axis=-1, keepdims=True) - 1))

      nu_loss = loss + nu_grad_penalty * nu_reg
      pi_loss = -loss + keras_utils.orthogonal_regularization(self.actor.trunk)

    nu_grads = tape.gradient(nu_loss, self.nu_net.variables)
    pi_grads = tape.gradient(pi_loss, self.actor.variables)

    self.nu_optimizer.apply_gradients(zip(nu_grads, self.nu_net.variables))
    self.actor_optimizer.apply_gradients(zip(pi_grads, self.actor.variables))

    del tape

    self.avg_nu_expert(expert_nu)
    self.avg_nu_rb(rb_nu)

    self.nu_reg_metric(nu_grad_penalty)
    self.avg_loss(loss)

    self.avg_actor_loss(pi_loss)
    self.avg_actor_entropy(-rb_log_prob)

    if tf.equal(self.nu_optimizer.iterations % self.log_interval, 0):
      tf.summary.scalar(
          'train dual dice/loss',
          self.avg_loss.result(),
          step=self.nu_optimizer.iterations)
      keras_utils.my_reset_states(self.avg_loss)

      tf.summary.scalar(
          'train dual dice/nu expert',
          self.avg_nu_expert.result(),
          step=self.nu_optimizer.iterations)
      keras_utils.my_reset_states(self.avg_nu_expert)

      tf.summary.scalar(
          'train dual dice/nu rb',
          self.avg_nu_rb.result(),
          step=self.nu_optimizer.iterations)
      keras_utils.my_reset_states(self.avg_nu_rb)

      tf.summary.scalar(
          'train dual dice/nu reg',
          self.nu_reg_metric.result(),
          step=self.nu_optimizer.iterations)
      keras_utils.my_reset_states(self.nu_reg_metric)

    if tf.equal(self.actor_optimizer.iterations % self.log_interval, 0):
      tf.summary.scalar(
          'train sac/actor_loss',
          self.avg_actor_loss.result(),
          step=self.actor_optimizer.iterations)
      keras_utils.my_reset_states(self.avg_actor_loss)

      tf.summary.scalar(
          'train sac/actor entropy',
          self.avg_actor_entropy.result(),
          step=self.actor_optimizer.iterations)
      keras_utils.my_reset_states(self.avg_actor_entropy)
