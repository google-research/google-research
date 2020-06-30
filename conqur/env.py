# coding=utf-8
# Copyright 2020 The Google Research Authors.
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
"""Wrapper class for environment."""

import numpy as np


class Environment:
  """Base class for environment object."""

  def __init__(self):
    self.action_vals = np.zeros((self.num_actions,), dtype='f8')

  def sample_batch(self,
                   eps_explore,
                   max_batch_size,
                   random_state,
                   parent_layer,
                   target_layer,
                   max_episode_length=27000):
    """Samples a batch of (state, action, reward, next_state) transitions.

    These transitions are sampled from episodes, which are generated from
    an epsilon greedy exploration strategy.

    Args:
      eps_explore: float, the probability of exploring a random action.
      max_batch_size: int, the maximum transitions per batch.
      random_state: np.random.RandomState, for maintaining the random seed.
      parent_layer: keras.layers, A input the agent network to be rollout.
      target_layer: keras weights, input target network used to compute Q-values
        of the target net.
      max_episode_length: int, maximum number of transitions per episode.

    Returns:
      A batch consisting of a list of transitions (s, a, r, next_s) where
        s is a vector state representation, a is a discrete action choice,
        r is a scalar reward and next_s is the next state vector.
    """
    batch_size = 0
    batch = []
    while True:
      episode = self.sample_episode(
          eps_explore,
          random_state,
          parent_layer,
          target_layer,
          max_episode_length=max_episode_length)
      if batch_size + len(episode) >= max_batch_size:
        batch.extend(episode[:max_batch_size - batch_size])
        break
      else:
        batch.extend(episode)
        batch_size += len(episode)
    return batch

  def evaluate_policy(self,
                      random_state,
                      online_layer,
                      max_episode_length=27000,
                      epsilon_eval=0.001):
    """Evaluates a given policy using Monte Carlo rollouts.

    Args:
      random_state: np.random.RandomState, random number state.
      online_layer: keras weights, as a copy of the agent.
      max_episode_length: int, maximum length of the rollouts.
      epsilon_eval: float, exploration probability used during policy
        evaluation.

    Returns:
      float, the total return of rolling out the agent interacting with env.
    """
    _, sum_returns, num_episodes = self.runner.dopamine_monte_carlo_rollout(
        epsilon_eval,
        random_state,
        online_layer,
        max_episode_length=max_episode_length)
    return sum_returns * 1.0 / num_episodes, 0.0, 0.0
