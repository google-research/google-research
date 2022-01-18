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

"""Implementation of DualDICE."""
import typing
import tensorflow as tf
from tensorflow_addons import optimizers as tfa_optimizers
import tqdm
from policy_eval.q_fitter import CriticNet


class DualDICE(object):
  """Implementation of DualDICE."""

  def __init__(self, state_dim, action_dim, weight_decay):
    self.nu = CriticNet(state_dim, action_dim)
    self.zeta = CriticNet(state_dim, action_dim)

    self.nu_optimizer = tfa_optimizers.AdamW(
        learning_rate=1e-4, beta_1=0.0, beta_2=0.99, weight_decay=weight_decay)
    self.zeta_optimizer = tfa_optimizers.AdamW(
        learning_rate=1e-3, beta_1=0.0, beta_2=0.99, weight_decay=weight_decay)

  @tf.function
  def update(self, initial_states, initial_actions,
             initial_weights, states, actions,
             next_states, next_actions, masks,
             weights, discount):
    """Updates parameters.

    Args:
      initial_states: A batch of states.
      initial_actions: A batch of actions sampled from target policy.
      initial_weights: A batch of weights for the initial states.
      states: A batch of states.
      actions: A batch of actions sampled from behavior policy.
      next_states: A batch of next states.
      next_actions: A batch of next actions sampled from target policy.
      masks: A batch of masks indicating the end of the episodes.
      weights: A batch of weights.
      discount: An MDP discount factor.

    Returns:
      Critic loss.
    """
    with tf.GradientTape(
        watch_accessed_variables=False, persistent=True) as tape:
      tape.watch(self.nu.trainable_variables)
      tape.watch(self.zeta.trainable_variables)

      nu = self.nu(states, actions)
      nu_next = self.nu(next_states, next_actions)
      nu_0 = self.nu(initial_states, initial_actions)

      zeta = self.zeta(states, actions)

      nu_loss = (
          tf.reduce_sum(weights * (
              (nu - discount * masks * nu_next) * zeta - tf.square(zeta) / 2)) /
          tf.reduce_sum(weights) -
          tf.reduce_sum(initial_weights *
                        (1 - discount) * nu_0) / tf.reduce_sum(initial_weights))
      zeta_loss = -nu_loss

    nu_grads = tape.gradient(nu_loss, self.nu.trainable_variables)
    zeta_grads = tape.gradient(zeta_loss, self.zeta.trainable_variables)

    self.nu_optimizer.apply_gradients(
        zip(nu_grads, self.nu.trainable_variables))
    self.zeta_optimizer.apply_gradients(
        zip(zeta_grads, self.zeta.trainable_variables))

    del tape

    tf.summary.scalar(
        'train/nu loss', nu_loss, step=self.nu_optimizer.iterations)
    tf.summary.scalar(
        'train/zeta loss', zeta_loss, step=self.zeta_optimizer.iterations)

    return nu_loss

  @tf.function
  def estimate_returns(
      self,
      tf_dataset_iter,
      num_samples = 100):
    """Estimated returns for a target policy.

    Args:
      tf_dataset_iter: Iterator over the dataset.
      num_samples: Number of samples used to estimate the returns.

    Returns:
      Estimated returns.
    """
    pred_returns = 0.0
    pred_ratio = 0.0
    for _ in tqdm.tqdm(range(num_samples), desc='Estimating Returns'):
      states, actions, _, rewards, _, weights, _ = next(tf_dataset_iter)
      zeta = self.zeta(states, actions)
      pred_ratio += tf.reduce_sum(weights * zeta) / tf.reduce_sum(weights)
      pred_returns += tf.reduce_sum(
          weights * zeta * rewards) / tf.reduce_sum(weights)
    return pred_returns / num_samples, pred_ratio / num_samples
