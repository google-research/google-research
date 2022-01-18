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

"""An implementation of model based policy evaluation."""
import tensorflow as tf
from tensorflow_addons import optimizers as tfa_optimizers
import tensorflow_probability as tfp


class ForwardModel(tf.keras.Model):
  """A class that implement a forward model."""

  def __init__(self, state_dim, action_dim):
    super(ForwardModel, self).__init__()
    self.model = tf.keras.Sequential([
        tf.keras.layers.Dense(
            256,
            input_dim=state_dim + action_dim,
            activation=tf.nn.relu,
            kernel_initializer='orthogonal'),
        tf.keras.layers.Dense(
            256, activation=tf.nn.relu, kernel_initializer='orthogonal'),
        tf.keras.layers.Dense(
            256, activation=tf.nn.relu, kernel_initializer='orthogonal'),
        tf.keras.layers.Dense(
            256, activation=tf.nn.relu, kernel_initializer='orthogonal'),
        tf.keras.layers.Dense(state_dim, kernel_initializer='orthogonal')
    ])

  @tf.function
  def call(self, x, a):
    x_a = tf.concat([x, a], -1)
    return self.model(x_a) + x


class RewardModel(tf.keras.Model):
  """A class that implement a reward model."""

  def __init__(self, state_dim, action_dim):
    super(RewardModel, self).__init__()
    self.model = tf.keras.Sequential([
        tf.keras.layers.Dense(
            256,
            input_dim=state_dim + action_dim,
            activation=tf.nn.relu,
            kernel_initializer='orthogonal'),
        tf.keras.layers.Dense(
            256, activation=tf.nn.relu, kernel_initializer='orthogonal'),
        tf.keras.layers.Dense(
            256, activation=tf.nn.relu, kernel_initializer='orthogonal'),
        tf.keras.layers.Dense(
            256, activation=tf.nn.relu, kernel_initializer='orthogonal'),
        tf.keras.layers.Dense(1, kernel_initializer='orthogonal')
    ])

  @tf.function
  def call(self, x, a):
    x_a = tf.concat([x, a], -1)
    return tf.squeeze(self.model(x_a), 1)


class ModelBased(object):
  """A class that learns models and estimated returns via rollouts."""

  def __init__(self, state_dim, action_dim, learning_rate, weight_decay):
    """Creates networks and optimizers for model based policy evaluation.

    Args:
      state_dim: State size.
      action_dim: Action size.
      learning_rate: Critic learning rate.
      weight_decay: Weight decay.
    """
    self.dynamics_net = ForwardModel(state_dim, action_dim)
    self.rewards_net = RewardModel(state_dim, action_dim)
    self.done_net = RewardModel(state_dim, action_dim)

    self.dyn_optimizer = tfa_optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay)
    self.reward_optimizer = tfa_optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay)
    self.done_optimizer = tfa_optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay)

  @tf.function
  def update(self, states, actions,
             next_states, rewards, masks,
             weights):
    """Updates critic parameters.

    Args:
      states: A batch of states.
      actions: A batch of actions.
      next_states: A batch of next states.
      rewards: A batch of rewards.
      masks: A batch of masks.
      weights: A batch of weights.

    Returns:
      Critic loss.
    """
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self.dynamics_net.trainable_variables)
      pred_state = self.dynamics_net(states, actions)
      dyn_loss = tf.losses.mean_squared_error(next_states, pred_state)
      dyn_loss = tf.reduce_mean(dyn_loss * weights)

    grads = tape.gradient(dyn_loss, self.dynamics_net.trainable_variables)

    self.dyn_optimizer.apply_gradients(
        zip(grads, self.dynamics_net.trainable_variables))

    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self.rewards_net.trainable_variables)
      pred_rewards = self.rewards_net(states, actions)
      reward_loss = tf.losses.mean_squared_error(rewards, pred_rewards)
      reward_loss = tf.reduce_mean(reward_loss * weights)

    grads = tape.gradient(reward_loss, self.rewards_net.trainable_variables)

    self.reward_optimizer.apply_gradients(
        zip(grads, self.rewards_net.trainable_variables))

    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self.done_net.trainable_variables)
      pred_dones = self.done_net(states, actions)
      done_loss = tf.keras.losses.binary_crossentropy(
          masks, pred_dones, from_logits=True)
      done_loss = tf.reduce_mean(done_loss * weights)

    grads = tape.gradient(done_loss, self.done_net.trainable_variables)

    self.done_optimizer.apply_gradients(
        zip(grads, self.done_net.trainable_variables))

    tf.summary.scalar('train/dyn_loss', dyn_loss,
                      step=self.dyn_optimizer.iterations)
    tf.summary.scalar('train/rew_loss', reward_loss,
                      step=self.reward_optimizer.iterations)
    tf.summary.scalar('train/done_loss', done_loss,
                      step=self.done_optimizer.iterations)

  def estimate_returns(self,
                       initial_states,
                       weights,
                       get_action,
                       discount,
                       min_reward,
                       max_reward,
                       min_state,
                       max_state,
                       clip=True,
                       horizon=1000):
    """Compute returns via rollouts.

    Args:
      initial_states: A batch of initial states.
      weights: Weights.
      get_action: An policy.
      discount: MDP discount.
      min_reward: Min reward in the dataset.
      max_reward: Max reward in the dataset.
      min_state: Min state in the dataset.
      max_state: Max state in the dataset.
      clip: Whether to clip obs and rewards.
      horizon: Rollout horizon.

    Returns:
      Estimated returns via rollouts.
    """
    returns = 0
    states = initial_states

    masks = tf.ones((initial_states.shape[0],), dtype=tf.float32)

    for i in range(horizon):
      actions = get_action(states)

      pred_rewards = self.rewards_net(states, actions)
      if clip:
        pred_rewards = tf.clip_by_value(pred_rewards, min_reward,
                                        max_reward)
      logits = self.done_net(states, actions)
      mask_dist = tfp.distributions.Bernoulli(logits=logits)
      masks *= tf.cast(mask_dist.sample(), tf.float32)

      returns += (discount**i) * masks * pred_rewards

      states = self.dynamics_net(states, actions)
      if clip:
        states = tf.clip_by_value(states, min_state, max_state)
    return tf.reduce_sum(
        weights * returns) / tf.reduce_sum(weights) * (1 - discount)
