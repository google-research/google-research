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

"""Policy evaluation."""
import typing
import tensorflow.compat.v2 as tf


def evaluate(
    env,
    policy,
    num_episodes = 10,
    ctx_length = None,
    state_mask_fn = None
):
  """Evaluates the policy.

  Args:
    env: Environment to evaluate the policy on.
    policy: Policy to evaluate.
    num_episodes: A number of episodes to average the policy on.
    ctx_length: number of previous steps to compute context from.
    state_mask_fn: state masking function for partially obs envs.

  Returns:
    Averaged reward and a total number of steps.
  """
  total_timesteps = 0
  total_returns = 0.0

  def apply_mask(observation):
    if state_mask_fn:
      return tf.convert_to_tensor(state_mask_fn(observation.numpy()))
    return observation

  for _ in range(num_episodes):
    timestep = env.reset()
    if ctx_length:
      states = [apply_mask(timestep.observation) for _ in range(ctx_length)]
      actions = [
          tf.zeros(policy.action_spec.shape)[None, :] for _ in range(ctx_length)
      ]
      rewards = [[0.] for _ in range(ctx_length)]
    while not timestep.is_last():
      if ctx_length:
        states.append(apply_mask(timestep.observation))
        if len(states) > ctx_length:
          states.pop(0)
          actions.pop(0)
          rewards.pop(0)
        action = policy.act(
            tf.stack(states, axis=1),
            actions=tf.stack(actions, axis=1),
            rewards=tf.stack(rewards, axis=1))
        actions.append(action)
      else:
        action = policy.act(apply_mask(timestep.observation))

      timestep = env.step(action)
      if ctx_length:
        rewards.append(timestep.reward)

      total_returns += timestep.reward[0]
      total_timesteps += 1

  return total_returns / num_episodes, total_timesteps / num_episodes
