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

"""Behavioral cloning module."""

import typing

from dm_env import specs as dm_env_specs
import tensorflow as tf

from rl_repr.batch_rl import policies


class BehavioralCloning(object):
  """Training class for behavioral clonning."""

  def __init__(self,
               state_dim,
               action_spec,
               mixture = False,
               hidden_dims = (256, 256),
               embed_model=None,
               finetune=False):
    """Creates networks.

    Args:
      state_dim: State size.
      action_spec: Action spec.
      mixture: Whether policy is a mixture of Gaussian.
      hidden_dims: List of hidden dimensions.
      embed_model: Pretrained embedder.
      finetune: Whether to finetune the pretrained embedder.
    """
    self.action_spec = action_spec
    self.embed_model = embed_model
    self.finetune = finetune
    input_state_dim = (
        self.embed_model.get_input_state_dim()
        if self.embed_model else state_dim)

    if mixture:
      self.policy = policies.MixtureGuassianPolicy(
          input_state_dim, action_spec, hidden_dims=hidden_dims)
    else:
      self.policy = policies.DiagGuassianPolicy(
          input_state_dim, action_spec, hidden_dims=hidden_dims)

    boundaries = [180_000, 190_000]
    values = [1e-3, 1e-4, 1e-5]
    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries, values)

    self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)

    self.log_alpha = tf.Variable(tf.math.log(1.0), trainable=True)
    self.alpha_optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate_fn)

    self.target_entropy = -self.action_spec.shape[0]

  @tf.function
  def update_step(self, dataset_iter):
    """Performs a single training step.

    Args:
      dataset_iter: Iterator over dataset samples.

    Returns:
      Dictionary with losses to track.
    """
    states, actions, rewards, _, _ = next(dataset_iter)

    trainable_variables = self.policy.trainable_variables
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      if self.embed_model:
        states = self.embed_model(
            states, actions, rewards, stop_gradient=(not self.finetune))
        if hasattr(self.embed_model,
                   'ctx_length') and self.embed_model.ctx_length:
          assert (len(actions.shape) == 3)
          actions = actions[:, self.embed_model.ctx_length - 1, :]
        trainable_variables += self.embed_model.trainable_variables

      tape.watch(trainable_variables)
      data_log_probs = self.policy.log_probs(states, actions)
      _, log_probs = self.policy(states, sample=True, with_log_probs=True)
      alpha = tf.math.exp(self.log_alpha)
      loss = tf.reduce_mean(alpha * log_probs - data_log_probs)

    grads = tape.gradient(loss, trainable_variables)

    self.optimizer.apply_gradients(zip(grads, trainable_variables))

    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch([self.log_alpha])
      alpha = tf.math.exp(self.log_alpha)
      alpha_loss = tf.reduce_mean(alpha *
                                  (-log_probs - self.target_entropy))

    alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])
    self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.log_alpha]))

    return {'bc_actor_loss': loss, 'entropy': -tf.reduce_mean(log_probs),
            'alpha': alpha, 'alpha_loss': alpha_loss}

  @tf.function
  def act(self, states, actions=None, rewards=None):
    if self.embed_model:
      states = self.embed_model(states, actions, rewards)
    return self.policy(states, sample=False)
