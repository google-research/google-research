# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Tensorflow evaluation tools for StochasticDecisionTransformers."""
import tensorflow as tf


def evaluate_stochastic_decision_transformer_episode(
    env,
    state_dim,
    act_dim,
    model,
    max_ep_len,
    scale,
    state_mean,
    state_std,
    target_return,
    reward_noise_fn=None,
):
  """Evaluate a DecisionTransformer model for one episode in env.

  Args:
    env: A gym environment.
    state_dim: An int dimension of observations.
    act_dim: An int dimension of actions.
    model: A DecisionTransformer model.
    max_ep_len: An integer max episode length.
    scale: A float for scaling rewards.
    state_mean: A np.ndarray for scaling states.
    state_std: A np.ndarray for scaling states.
    target_return: A float representing the desired returns for the episode.

  Returns:
    episode_return: A float returns for the episode.
    episode_length: An int episode length.
  """

  state_mean = tf.convert_to_tensor(state_mean)
  state_std = tf.convert_to_tensor(state_std)

  state = env.reset()

  # note that the latest action and reward will be "padding"
  states = tf.convert_to_tensor(state, dtype=tf.float32)
  states = tf.reshape(states, [1, state_dim])
  actions = tf.zeros((0, act_dim), dtype=tf.float32)

  target_return = tf.zeros((0, 1), dtype=tf.float32)
  timesteps = tf.convert_to_tensor(0, dtype=tf.int64)
  timesteps = tf.reshape(timesteps, (1, 1))

  episode_return, episode_length = 0, 0
  for t in range(max_ep_len):
    # add padding
    padded_actions = tf.concat([actions, tf.zeros((1, act_dim))], axis=0)
    padded_target_return = tf.concat([target_return, tf.zeros((1, 1))], axis=0)

    action = model.get_action(
        (states - state_mean) / state_std,
        padded_actions,
        padded_target_return,
        timesteps,
    )

    actions = tf.concat([actions, tf.reshape(action, (1, act_dim))], axis=0)
    action = action.numpy()

    state, reward, done, _ = env.step(action)

    if reward_noise_fn:
      reward = reward_noise_fn(env, reward)

    target_return = tf.concat(
        [target_return,
         tf.reshape(tf.cast(reward, tf.float32), (1, 1))],
        axis=0)

    cur_state = tf.convert_to_tensor(state, dtype=tf.float32)
    cur_state = tf.reshape(cur_state, [1, state_dim])
    states = tf.concat([states, cur_state], axis=0)
    timesteps = tf.concat(
        [timesteps, tf.ones((1, 1), dtype=tf.int64) * (t + 1)], axis=1)

    episode_return += reward
    episode_length += 1

    if done:
      break

  return episode_return, episode_length
