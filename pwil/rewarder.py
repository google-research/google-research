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

"""Rewarder class implementation."""

import copy
import random

import dm_env
import numpy as np
import ot
from sklearn import preprocessing


class PWILRewarder(object):
  """Rewarder class to compute PWIL rewards."""

  def __init__(self,
               demonstrations,
               subsampling,
               env_specs,
               num_demonstrations=1,
               time_horizon=1000.,
               alpha=5.,
               beta=5.,
               observation_only=False):
    """Initialize the rewarder.

    Args:
      demonstrations: list of expert episodes, comes under the following format:
        [
          [{observation: o_1, action: a_1}, ...], # episode 1
          [{observation: o'_1, action: a'_1}, ...], # episode 2
          ...
        ]
      subsampling: int describing the demonstrations subsamplig frequency.
      env_specs: description of the actions, observations, etc.
      num_demonstrations: int describing the number of demonstration episodes
                          to select at random.
      time_horizon: int time length of the task.
      alpha: float scaling the reward function.
      beta: float controling the kernel size of the reward function.
      observation_only: boolean whether or not to use action to compute reward.
    """
    self.num_demonstrations = num_demonstrations
    self.time_horizon = time_horizon
    self.subsampling = subsampling

    # Observations and actions are flat.
    dim_act = env_specs.actions.shape[0]
    dim_obs = env_specs.observations.shape[0]
    self.reward_sigma = beta * time_horizon / np.sqrt(dim_act + dim_obs)
    self.reward_scale = alpha

    self.observation_only = observation_only
    self.demonstrations = self.filter_demonstrations(demonstrations)
    self.vectorized_demonstrations = self.vectorize(self.demonstrations)
    self.scaler = self.get_scaler()

  def filter_demonstrations(self, demonstrations):
    """Select a subset of expert demonstrations.

    Select n episodes at random.
    Subsample transitions in these episodes.
    Offset the start transition before subsampling at random.

    Args:
      demonstrations: list of expert demonstrations

    Returns:
      filtered_demonstrations: list of filtered expert demonstrations
    """
    filtered_demonstrations = []
    random.shuffle(demonstrations)
    for episode in demonstrations[:self.num_demonstrations]:
      # Random episode start.
      random_offset = random.randint(0, self.subsampling-1)
      # Subsampling.
      subsampled_episode = episode[random_offset::self.subsampling]
      # Specify step types of demonstrations.
      for transition in subsampled_episode:
        transition['step_type'] = dm_env.StepType.MID
      subsampled_episode[0]['step_type'] = dm_env.StepType.FIRST
      subsampled_episode[-1]['step_type'] = dm_env.StepType.LAST
      filtered_demonstrations += subsampled_episode
    return filtered_demonstrations

  def vectorize(self, demonstrations):
    """Convert filtered expert demonstrations to numpy array.

    Args:
      demonstrations: list of expert demonstrations

    Returns:
      numpy array with dimension:
      [num_expert_transitions, dim_observation] if observation_only
      [num_expert_transitions, (dim_observation + dim_action)] otherwise
    """
    if self.observation_only:
      demonstrations = [t['observation'] for t in demonstrations]
    else:
      demonstrations = [np.concatenate([t['observation'], t['action']])
                        for t in demonstrations]
    return np.array(demonstrations)

  def get_scaler(self):
    """Defines a scaler to derive the standardized Euclidean distance."""
    scaler = preprocessing.StandardScaler()
    scaler.fit(self.vectorized_demonstrations)
    return scaler

  def reset(self):
    """Makes all expert transitions available and initialize weights."""
    self.expert_atoms = copy.deepcopy(
        self.scaler.transform(self.vectorized_demonstrations)
    )
    num_expert_atoms = len(self.expert_atoms)
    self.expert_weights = np.ones(num_expert_atoms) / (num_expert_atoms)

  def compute_reward(self, obs_act):
    """Computes reward as presented in Algorithm 1."""
    # Scale observation and action.
    if self.observation_only:
      agent_atom = obs_act['observation']
    else:
      agent_atom = np.concatenate([obs_act['observation'], obs_act['action']])
    agent_atom = np.expand_dims(agent_atom, axis=0)  # add dim for scaler
    agent_atom = self.scaler.transform(agent_atom)[0]

    cost = 0.
    # As we match the expert's weights with the agent's weights, we might
    # raise an error due to float precision, we substract a small epsilon from
    # the agent's weights to prevent that.
    weight = 1. / self.time_horizon - 1e-6
    norms = np.linalg.norm(self.expert_atoms - agent_atom, axis=1)
    while weight > 0:
      # Get closest expert state action to agent's state action.
      argmin = norms.argmin()
      expert_weight = self.expert_weights[argmin]

      # Update cost and weights.
      if weight >= expert_weight:
        weight -= expert_weight
        cost += expert_weight * norms[argmin]
        self.expert_weights = np.delete(self.expert_weights, argmin, 0)
        self.expert_atoms = np.delete(self.expert_atoms, argmin, 0)
        norms = np.delete(norms, argmin, 0)
      else:
        cost += weight * norms[argmin]
        self.expert_weights[argmin] -= weight
        weight = 0

    reward = self.reward_scale * np.exp(-self.reward_sigma * cost)
    return reward.astype('float32')

  def compute_w2_dist_to_expert(self, trajectory):
    """Computes Wasserstein 2 distance to expert demonstrations."""
    self.reset()
    if self.observation_only:
      trajectory = [t['observation'] for t in trajectory]
    else:
      trajectory = [np.concatenate([t['observation'], t['action']])
                    for t in trajectory]

    trajectory = self.scaler.transform(trajectory)
    trajectory_weights = 1./len(trajectory) * np.ones(len(trajectory))
    cost_matrix = ot.dist(trajectory, self.expert_atoms, metric='euclidean')
    w2_dist = ot.emd2(trajectory_weights, self.expert_weights, cost_matrix)
    return w2_dist
