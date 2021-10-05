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

"""Behavioral Clonning training."""
import typing

from dm_env import specs as dm_env_specs
import tensorflow as tf
from tf_agents.specs.tensor_spec import TensorSpec

from representation_batch_rl.batch_rl import policies


class BehavioralCloning(object):
  """Training class for behavioral clonning."""

  def __init__(self,
               observation_spec,
               action_spec,
               mixture = False,
               env_name = ''):
    """BC class init.

    Args:
      observation_spec: observation space
      action_spec: action space
      mixture: use a mixture model?
      env_name: name of env
    Returns:
      None
    """
    del env_name
    assert len(observation_spec.shape) == 1
    state_dim = observation_spec.shape[0]

    self.action_spec = action_spec
    if mixture:
      self.policy = policies.MixtureGuassianPolicy(state_dim, action_spec)
    else:
      self.policy = policies.DiagGuassianPolicy(state_dim, action_spec)

    boundaries = [800_000, 900_000]
    values = [1e-3, 1e-4, 1e-5]
    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries, values)

    self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)

    self.log_alpha = tf.Variable(tf.math.log(1.0), trainable=True)
    self.alpha_optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate_fn)

    self.target_entropy = -action_spec.shape[0]

  @property
  def alpha(self):
    return tf.exp(self.log_alpha)

  @tf.function
  def update_step(self, dataset_iter):
    """Performs a single training step.

    Args:
      dataset_iter: Iterator over dataset samples.

    Returns:
      Dictionary with losses to track.
    """

    transition = next(dataset_iter)
    states = transition.observation[:, 0]
    actions = transition.action[:, 0]

    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self.policy.trainable_variables)
      log_probs, entropy = self.policy.log_probs(
          states, actions, with_entropy=True)

      loss = -tf.reduce_mean(self.alpha * entropy + log_probs)

    grads = tape.gradient(loss, self.policy.trainable_variables)

    self.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))

    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch([self.log_alpha])
      alpha_loss = tf.reduce_mean(self.alpha * (entropy - self.target_entropy))

    alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])
    self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.log_alpha]))

    return {
        'bc_actor_loss': loss,
        'bc_alpha': self.alpha,
        'bc_alpha_loss': alpha_loss,
        'bc_log_probs': tf.reduce_mean(log_probs),
        'bc_entropy': tf.reduce_mean(entropy)
    }

  @tf.function
  def act(self, states):
    return self.policy(states, sample=False)
