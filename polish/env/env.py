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

"""Wrapper class for a gym-like environment."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
import os
from absl import logging
import gin
import gym
import numpy as np
from scipy.stats import multivariate_normal
from polish.mcts import mcts_player
from polish.utils import tf_utils
import polish.utils.running_mean_std as running_mean_std


@gin.configurable
class MCTSEnv(object):
  """A gym-like environment for building trajectories.

  Attributes:
    env: game environment (should support gym environment APIs, `step, reset`).
    estimator: rollout policy (value and policy network) TF estimator.
    serving_input_fn: Input function for model predictions.
    clip_ob: Clip value for obesrvations (states).
    clip_rew: Clip value for rewards.
    epsilon: Epsilon value used to avoid zero-division.
    obs_normalized: Whether to normalize environment observations.
    reward_normalized: Whether to normalize environment rewards.
    env_states: Array for MuJoCo environment internal states.
    trajectory_states: Array for trajectory states.
    trajectory_actions: Array for trajectory actions.
    trajectory_values: Array for trajectory state-values.
    trajectory_returns: Array for trajectory returns.
    trajectory_means: Array for current policy mean values.
    trajectory_dones: Array for indicating whether a state is a terminal state.
    trajectory_logstds: Array for current policy logstd values.
    trajectory_neg_logprobs: Array for trajectory negative log probabilities.
    trajectory_per_episode_rewards: Array for trajectory `episode` rewards. Each
      trajectoy may contain multiple episodes.
    trajectory_per_episode_lengths: Array for trajectory `episode` lengths. Each
      trajectory may contain multiple episodes.
    trajectory_per_step_rewards: Array for trajectory rewards (per step).
    mcts_player: An MCTSPlayer instance for generating MCTS rollouts.
    mcts_sampling: If True, the current iteration uses MCTS to generate
      demonstration data.
  """

  def __init__(self,
               env,
               estimator,
               serving_input_fn,
               gamma=0.99,
               lam=0.95,
               tanh_action_clipping=False,
               obs_normalized=True,
               reward_normalized=True,
               clip_ob=10.,
               clip_rew=10.,
               epsilon=1e-8,
               mcts_enable=False,
               num_envs=1,
               mcts_start_step_frac=0.1,
               mcts_end_step_frac=0.9,
               num_iterations=156160,
               mcts_sim_decay_factor=0.8,
               mcts_sampling_frac=0.1,
               mcts_collect_data_freq=1,
               random_action_sampling_freq=0.0,
               checkpoint_dir=None):
    """Creates a gym-like environment with some added functionalities.

    Args:
      env: an instance of gym environment.
      estimator: a TF estimator instance used to call `prediction` on.
      serving_input_fn: the serving input function specifies what the caller of
        the estimator `predict` method must provide. the `serving_input_fn`
        tells the model what data it has to get from the user.
      gamma: the discount factor multiplied by future rewards from the
        environment. gamma value is generally used to dampen the effect of
        future reward on the agent's choice. That is, gamma value makes future
        rewards are worth less than immediate rewards.
      lam: Generalized Advantage Estimator (GAE) parameter.
      tanh_action_clipping: If set, performs tanh action clipping. Enabling tanh
        action clipping bound the actions to [-1, 1]. See
        https://arxiv.org/pdf/1801.01290.pdf for details.
      obs_normalized: if True, the observations from environment are normalized.
      reward_normalized: if True, the rewards from environment are normalized.
      clip_ob: the range for clipping the observations. The observations are
        clipped into the range [-clip_ob, clip_ob] after normalization.
      clip_rew: the range for clipping the rewards. The rewards are clipped into
        the range [-clip_rew, clip_rew] after normalization.
      epsilon: an infinitesimal value used to prevent divide-by-zero in
        normalizing the data.
      mcts_enable: if True, the samples are taken from MCTS simulations.
      num_envs: indicates the number of parallel environments in MCTS player.
      mcts_start_step_frac: The sampling step at which MCTS sampling starts.
      mcts_end_step_frac: The sampling step at which MCTS sampling stops.
      num_iterations: total number of training iterations (including epochs).
      mcts_sim_decay_factor: decay number of MCTS simulations with this value.
      mcts_sampling_frac: across all the data samples this fraction of MCTS
        sampling occurs.
      mcts_collect_data_freq: As MCTS is costly, we do not want to collect MCTS
        data for each training iteration. Instead, every
        `mcts_collect_data_freq`, we perform MCTS sampling.
      random_action_sampling_freq: the percentage of the children's move that
        are exploratory (completely random).
      checkpoint_dir: use checkpoint dir and create 'mcts_data' this directory
        for holding MCTS data.
    """
    if random_action_sampling_freq < 0.0 or random_action_sampling_freq > 1.0:
      raise ValueError('ranom_action_sampling_freq should be '
                       'between [0.0, 1.0]!')

    self.env = env
    self.estimator = estimator
    self.serving_input_fn = serving_input_fn

    # Private attributes.
    self._policy = None
    self._last_value = None
    self._last_done = None
    self._gym_state = None
    self._episode_length = 0
    self._episode_reward = 0.
    self._env_done = False
    self._tanh_action_clipping = tanh_action_clipping
    self._gamma = gamma
    self._lam = lam
    self._mcts_enable = mcts_enable
    self._first_time_call = True
    self._num_envs = num_envs

    self._sampling_step = 0
    self._num_mcts_samples = 0
    self._random_gen = np.random
    self._random_action_sampling_freq = random_action_sampling_freq
    self._mcts_data_dir = os.path.join(checkpoint_dir, 'mcts_data')

    if (mcts_start_step_frac < 0.0 or mcts_start_step_frac > 1.5):
      raise ValueError(
          'MCTS start step should be a value between 0.0 and 1.0'
          ' indicating after what fraction of data sampling we should switch'
          ' to MCTS sampling')

    if (mcts_end_step_frac < 0.0 or mcts_end_step_frac > 1.5):
      raise ValueError(
          'MCTS end step should be a value between 0.0 and 1.0'
          ' indicating after what fraction of data sampling we should stop'
          ' MCTS sampling')

    if mcts_end_step_frac <= mcts_start_step_frac:
      raise ValueError('MCTS end step should be greater than MCTS start step')

    if mcts_sampling_frac > 1.0:
      raise ValueError(
          'Among all the sampling iterations this fraction of'
          ' MCTS sampling occurs. Negative value indicates no MCTS sampling.')

    if mcts_collect_data_freq < 1.0:
      raise ValueError(
          'mcts_collect_data_freq must be greater than one. That is, '
          ' how many times to perform MCTS sampling.')
    self._num_iterations = num_iterations
    self._mcts_start_step_frac = int(mcts_start_step_frac *
                                     self._num_iterations)
    self._mcts_end_step_frac = int(mcts_end_step_frac * self._num_iterations)
    self._mcts_sampling_frac = mcts_sampling_frac
    self._mcts_collect_data_freq = mcts_collect_data_freq
    self._mcts_sim_decay_factor = mcts_sim_decay_factor

    # Observation and return normalizers.
    self._ob_rms = running_mean_std.RunningMeanStd(
        shape=(1, self.env.observation_space.shape[0]))
    self._ret_rms = running_mean_std.RunningMeanStd(shape=())
    # Return placeholder.
    self._ret = np.zeros(1)

    # Property getter/setter.
    self._clip_ob = clip_ob
    self._clip_rew = clip_rew
    self._epsilon = epsilon
    self._obs_normalized = obs_normalized
    self._reward_normalized = reward_normalized
    self.mcts_sampling = False

    if self._mcts_enable:
      self.prepare_mcts_player()

    self.reset()

    self.initialize_trajectory_data()

  @property
  def clip_ob(self):
    return self._clip_ob

  @clip_ob.setter
  def clip_ob(self, clip_ob):
    self._clip_ob = clip_ob

  @property
  def clip_rew(self):
    return self._clip_rew

  @clip_rew.setter
  def clip_rew(self, clip_rew):
    self._clip_rew = clip_rew

  @property
  def epsilon(self):
    return self._epsilon

  @epsilon.setter
  def epsilon(self, epsilon):
    self._epsilon = epsilon

  @property
  def obs_normalized(self):
    return self._obs_normalized

  @obs_normalized.setter
  def obs_normalized(self, obs_normalized):
    self._obs_normalized = obs_normalized

  @property
  def reward_normalized(self):
    return self._reward_normalized

  @reward_normalized.setter
  def reward_normalized(self, reward_normalized):
    self._reward_normalized = reward_normalized

  def initialize_trajectory_data(self):
    """Initialize trajectory data to empty lists."""

    # Variables for one trajectory.
    # Each trajectory may consist of multiple episodes.
    self.trajectory_states = []
    self.trajectory_actions = []
    self.trajectory_values = []
    self.trajectory_returns = []
    self.trajectory_means = []
    self.trajectory_logstds = []
    self.trajectory_neg_logprobs = []
    self.trajectory_per_episode_rewards = []
    self.trajectory_per_episode_lengths = []
    self.trajectory_per_step_rewards = []
    self.trajectory_dones = []
    self.env_states = []

  def step(self, action):
    """Take one action in the environment.

    Args:
      action: action to be taken on the environment.

    Returns:
      state: next state.
      reward: reward.
      done: final state.
      info: information about the state.
    """
    state, reward, done, _ = self.env.step(action)
    return state, reward, done, _

  def reset(self):
    """Reset the environment to an initial state.

    Returns:
      The initial state of the environment.
    """
    # Reset the environment
    self._gym_state = self.env.reset()
    # Reset return normalizer to zero
    self._ret = np.zeros(1)
    # Normalize observation
    self._gym_state = self._norm_clip_ob(np.asarray([self._gym_state]))[0]

    return self._gym_state

  def prepare_mcts_player(self):
    """Initializes variables for MCTS sampling.

    This function is called during initialization of the `Env` class, only if
      `mcts_enable` is true.
    """
    # Retrieve the environment name.
    env_name = self.env.unwrapped.spec.id
    env_constructor = functools.partial(gym.make, env_name)
    # Parallel environments used in MCTS simulation during
    # expand and evaluate phase.
    temp_env = env_constructor()
    env_action_space = temp_env.action_space.shape[0]

    envs = [
        env_constructor() for _ in range(self._num_envs)
    ]
    self._tree_env = envs

    self.mcts_player = mcts_player.MCTSPlayer(
        tree_env=self._tree_env,
        call_policy=self.call_policy,
        max_episode_steps=1000,
        env_action_space=env_action_space,
        num_envs=self._num_envs)
    self._current_mcts_player = self.mcts_player

  def call_policy(self, state, only_normalized=False):
    """Run policy on a state.

    Args:
      state: state tensor [Batch, *].
      only_normalized: if true, the input data are normalized and clipped.

    Returns:
      action, value, neg_logprob, mu, var in tensor [Batch, *].
    """
    estimator_prediction = {}
    # This check is for MCTSPlayer during simulation phase.
    # We do not want to update observation runnign mean and std (`_ob_rms`)
    # for the environment observations during MCTS simulations step.
    if only_normalized:
      state = self._norm_clip_ob(state, update_rms=False)
    # Call TF estimator predictor and retrieve the predictions:
    #  `action`: sampled action from the policy distribution.
    #  `value`: state-value for the given state.
    #  `neg_logprob`: negative log of probability distribution function (pdf)
    #    of the sampled action.
    #  `mean`: mean value of the policy distibution.
    #  `logstd`: log of standard deviation of the policy distribution.
    estimator_out = self._policy({'mcts_features': state,
                                  'policy_features': state})
    estimator_prediction['action'] = estimator_out['action']
    estimator_prediction['value'] = estimator_out['value']
    estimator_prediction['neg_logprob'] = estimator_out['neg_logprob']
    estimator_prediction['mean'] = estimator_out['mean']
    estimator_prediction['logstd'] = estimator_out['logstd']
    return estimator_prediction

  def update_action(self, input_action):
    """Update the action value.

    Args:
      input_action: input action array.

    Returns:
      updated action value after tanh clipping (if applicable).
    """
    # tanh action clipping is a technique to map infinite action space
    # from gaussian distribution to [-1,1]
    # https://arxiv.org/pdf/1801.01290.pdf
    if self._tanh_action_clipping:
      return np.tanh(input_action)
    else:
      return input_action

  def _norm_clip_ob(self, obs, update_rms=True):
    """Observation normalization and clipping.

    Args:
      obs: observation tensor from environment [*, state_size].
      update_rms: if true, observation running mean gets updated.

    Returns:
      normalized and clipped observation.
    """
    assert isinstance(obs, np.ndarray), ('The observation array MUST be a '
                                         'numpy array.')
    if update_rms:
      self._ob_rms.update(obs)

    if self.obs_normalized:
      obs = np.clip(
          (obs - self._ob_rms.mean) / np.sqrt(self._ob_rms.var + self.epsilon),
          0. - self.clip_ob, self.clip_ob)
      return obs
    return obs

  def mcts_initialization(self, init_state=None, init_action=None):
    """Initialize MCTS player and expand/evaluate root node."""

    self._current_mcts_player.initialize_game(init_state)
    # At the beginning, we expand the root (first node of the tree).
    # This step is necessary as we do not have any other basis to select
    # a child from root.
    first_node = self._current_mcts_player.root.select_leaf()

    # Retrieve the root observation.
    self._gym_state = first_node.observ

    # Normalize and clip the initial observation (root observation).
    self._gym_state = self._norm_clip_ob(np.asarray([self._gym_state]))[0]

    # Call the policy/state-value network (using tf.estimator).
    policy_out = self.call_policy([self._gym_state])

    # Update first node state-value.
    first_node.network_value = policy_out['value'][0]

    # Create a Multivariate Normal Distribution from the given `mean` and
    # `logstd`. This distribution is a replica of the policy distribution that
    # exists in the tf.estimator. We need this distribtion to sample a set
    # of actions.
    mcts_dist = multivariate_normal(
        mean=policy_out['mean'][0],
        cov=np.diag(np.power(np.exp(policy_out['logstd'][0]), 2)))

    sampled_actions = self._current_mcts_player.sample_actions(
        mcts_dist=mcts_dist)
    if init_action is not None:
      # always include the action taken by the policy as a choice.
      sampled_actions[-1] = init_action
    # Calcualte probabilities for the sampled actions.
    child_probs = mcts_dist.pdf(sampled_actions)
    # update `move_to_action` for the root node.
    for i, a in enumerate(sampled_actions):
      first_node.move_to_action[i] = a

    # Expand each action one by one and populate child node statistics.
    first_iteration = True
    child_reward = np.zeros(0)
    child_observ = np.zeros(0)
    child_state_qpos = np.zeros(0)
    child_state_qvel = np.zeros(0)
    child_done = np.zeros(0)

    for mcts_env, mcts_action in zip(self._current_mcts_player.tree_env,
                                     sampled_actions):
      mcts_env.reset()
      mcts_env.set_state(first_node.state.qpos, first_node.state.qvel)
      observ, reward, done, _ = mcts_env.step(mcts_action)
      state = mcts_env.sim.get_state()

      if first_iteration:
        child_reward = np.array([reward])
        child_observ = np.array([observ])
        child_state_qpos = np.array([state.qpos])
        child_state_qvel = np.array([state.qvel])
        child_done = np.array([done])
        first_iteration = False
      else:
        child_reward = np.concatenate((child_reward, np.array([reward])))
        child_observ = np.concatenate((child_observ, [observ]))
        child_state_qpos = np.concatenate((child_state_qpos, [state.qpos]))
        child_state_qvel = np.concatenate((child_state_qvel, [state.qvel]))
        child_done = np.concatenate((child_done, np.array([done])))
    # Update the reward value for the selected leaf's children and perform
    # backup step.
    max_num_actions = self._current_mcts_player.max_num_actions
    first_node.child_reward = child_reward[:max_num_actions]
    first_node.move_to_observ = child_observ[:max_num_actions]
    first_node.move_to_state = [(qpos, qvel) for qpos, qvel in zip(
        child_state_qpos[:max_num_actions],
        child_state_qvel[:max_num_actions])]
    first_node.move_to_done = child_done[:max_num_actions]
    # Update the values for all the children by calling the value network.
    network_children = self.call_policy(
        first_node.move_to_observ, only_normalized=True)
    first_node.child_w = network_children['value']
    # Incorporate the results up to root (`backup` step in MCTS).
    first_node.incorporate_results(
        child_probs=child_probs,
        node_value=policy_out['value'][0],
        up_to=first_node)

  def run_mcts_trajectory(self, max_horizon):
    """Run a trajectory with length max_horizon using MCTS.

    Args:
      max_horizon: maximum number of steps for the trajectory.

    Returns:
      Update trajectory_* (`states`, `actions`, `values`, `neg_logprobs`,
        `rewards`, `dones`) with the new trajectory.
    """

    self.initialize_trajectory_data()

    # Take steps in the environment for `max_horizon` number of steps.
    for _ in range(max_horizon):
      self.env_states.append(
          self._current_mcts_player.tree_env[0].sim.get_state())

      self._current_mcts_player.root.inject_noise()
      move = self._current_mcts_player.suggest_move()

      # Append the current observation to the game trajectory.
      self.trajectory_states.append(self._gym_state)

      self._current_mcts_player.play_move(move)

      mcts_action = self._current_mcts_player.game_actions[-1]
      mcts_value = self._current_mcts_player.game_values[-1]
      mcts_done = self._current_mcts_player.game_dones[-1]
      mcts_prob = self._current_mcts_player.game_probs[-1]
      mcts_reward = self._current_mcts_player.game_rewards[-1]
      mcts_observ = self._current_mcts_player.game_observs[-1]
      mcts_mean = self._current_mcts_player.game_means[-1]
      mcts_logstd = self._current_mcts_player.game_logstd[-1]
      reward = mcts_reward

      # The probabilities are already normalized.
      mcts_neg_logprob = mcts_prob

      self.trajectory_actions.append(mcts_action)
      self.trajectory_values.append(mcts_value)
      self.trajectory_dones.append(mcts_done)
      self.trajectory_neg_logprobs.append(mcts_neg_logprob)

      self.trajectory_means.append(mcts_mean)
      self.trajectory_logstds.append(mcts_logstd)
      # Take the sampled action in the environment and get the reward.
      self._gym_state = mcts_observ
      self._env_done = mcts_done
      # Normalize and clip the next obeservation.
      self._gym_state = self._norm_clip_ob(np.asarray([self._gym_state]))[0]

      # Update current episode reward and length.
      self._episode_reward += reward
      self._episode_length += 1

      # Update return value.
      self._ret = self._ret * self._gamma + reward

      # Update running mean/std for reward.
      self._ret_rms.update(np.asarray(self._ret))
      if self.reward_normalized:
        reward = np.clip(reward / np.sqrt(self._ret_rms.var + self.epsilon),
                         0. - self.clip_rew, self.clip_rew)
      self.trajectory_per_step_rewards.append(reward)

      if self._env_done:
        self.trajectory_per_episode_rewards.append(self._episode_reward)
        self.trajectory_per_episode_lengths.append(self._episode_length)
        # Reset return normalizer to zero.
        self._ret = np.zeros(1)
        # Initialize MCTS player and expand root node.
        self.mcts_initialization()
        self._ret[0] = 0.
        self._episode_reward = 0.
        self._episode_length = 0

      self._last_done = self._env_done

    # Get the last state-value.
    policy_out = self.call_policy([self._gym_state])
    self._last_value = policy_out['value'][0]

    # If the max_horizon is not enough for one episode, record the reward
    # and length here.
    if not self.trajectory_per_episode_rewards:
      self.trajectory_per_episode_rewards.append(self._episode_reward)
      self.trajectory_per_episode_lengths.append(self._episode_length)

    # Calculate return.
    self.calc_returns()

  def run_trajectory(self, max_horizon):
    """Run a trajectory with length max_horizon using a policy network.

    Args:
      max_horizon: maximum number of steps for the trajectory.

    Returns:
      Update trajectory_* (`states`, `actions`, `values`, `neg_logprobs`,
        `rewards`, `dones`) with the new trajectory.
    """

    self.initialize_trajectory_data()

    # Take steps in the environment for `max_horizon` number of steps.
    for _ in range(max_horizon):
      # Call policy network.
      policy_out = self.call_policy([self._gym_state])
      self.trajectory_states.append(self._gym_state)

      self.env_states.append(self.env.sim.get_state())

      # Sample an action from the policy network (`Gaussian Distribution`).
      orig_action = policy_out['action'][0]
      # Perform action clipping if it is enabled.
      action = self.update_action(orig_action)
      self.trajectory_actions.append(orig_action)
      # Get state-value for the current state.
      self.trajectory_values.append(policy_out['value'][0])
      # Calculate negative log probability (if `tanh` clipping is enabled
      # we need to add a correction to log probability).
      # Check: https://arxiv.org/pdf/1801.01290.pdf
      if self._tanh_action_clipping:
        neg_logprobs = policy_out['neg_logprob'][0]
        new_logprobs = -neg_logprobs - np.sum(
            np.log(1.0 - (np.tanh(orig_action)**2.0) + self.epsilon))
        self.trajectory_neg_logprobs.append(-new_logprobs)
      else:
        self.trajectory_neg_logprobs.append(policy_out['neg_logprob'][0])
      # Append the status of curren state (done/not done).
      self.trajectory_dones.append(self._env_done)

      self.trajectory_means.append(policy_out['mean'][0])
      self.trajectory_logstds.append(policy_out['logstd'][0])

      # Take the sampled action in the environment and get the reward.
      self._gym_state, reward, self._env_done, _ = self.step(action)
      # Update current episode reward and length.
      self._episode_reward += reward
      self._episode_length += 1

      # Update return value.
      self._ret = self._ret * self._gamma + reward

      # Update running mean/std for reward.
      self._ret_rms.update(np.asarray(self._ret))
      if self.reward_normalized:
        reward = np.clip(reward / np.sqrt(self._ret_rms.var + self.epsilon),
                         0. - self.clip_rew, self.clip_rew)
      self.trajectory_per_step_rewards.append(reward)

      # Normalize and clip the next obeservation.
      self._gym_state = self._norm_clip_ob(np.asarray([self._gym_state]))[0]
      if self._env_done:
        self.trajectory_per_episode_rewards.append(self._episode_reward)
        self.trajectory_per_episode_lengths.append(self._episode_length)
        self._gym_state = self.reset()
        self._ret[0] = 0.
        self._episode_reward = 0.
        self._episode_length = 0

      self._last_done = self._env_done

    # Get the last state-value.
    policy_out = self.call_policy([self._gym_state])
    self._last_value = policy_out['value'][0]

    # If the max_horizon is not enough for one episode, record the reward
    # and length here.
    if not self.trajectory_per_episode_rewards:
      self.trajectory_per_episode_rewards.append(self._episode_reward)
      self.trajectory_per_episode_lengths.append(self._episode_length)

    # Calculate return.
    self.calc_returns()

  def calc_returns(self):
    """Calculate return.

      Update `_epi_returns` array for trajectory returns.

    """
    # Convert all the arrays to numpy arrays.
    self.trajectory_states = np.asarray(
        self.trajectory_states, dtype=self.env.observation_space.dtype)
    self.trajectory_actions = np.asarray(
        self.trajectory_actions, dtype=np.float32)
    self.trajectory_values = np.asarray(
        self.trajectory_values, dtype=np.float32)
    self.trajectory_neg_logprobs = np.asarray(
        self.trajectory_neg_logprobs, dtype=np.float32)
    self.trajectory_means = np.asarray(self.trajectory_means, dtype=np.float32)
    self.trajectory_logstds = np.asarray(
        self.trajectory_logstds, dtype=np.float32)
    self.trajectory_per_episode_rewards = np.asarray(
        self.trajectory_per_episode_rewards, dtype=np.float32)
    self.trajectory_per_episode_lengths = np.asarray(
        self.trajectory_per_episode_lengths, dtype=np.float32)
    self.trajectory_dones = np.asarray(self.trajectory_dones, dtype=np.bool)
    self.trajectory_per_step_rewards = np.asarray(
        self.trajectory_per_step_rewards, dtype=np.float32)

    # Perform calculation.
    mb_returns = np.zeros_like(self.trajectory_per_step_rewards)
    mb_advs = np.zeros_like(self.trajectory_per_step_rewards)
    lastgaelam = 0
    for t in reversed(range(len(self.trajectory_per_step_rewards))):
      if t == len(self.trajectory_per_step_rewards) - 1:
        nextnonterminal = 1.0 - self._last_done
        nextvalues = self._last_value
      else:
        nextnonterminal = 1.0 - self.trajectory_dones[t + 1]
        nextvalues = self.trajectory_values[t + 1]
      delta = self.trajectory_per_step_rewards[t] + (
          self._gamma * nextvalues * nextnonterminal) - (
              self.trajectory_values[t])
      mb_advs[t] = lastgaelam = delta + (
          self._gamma * self._lam * nextnonterminal * lastgaelam)
    mb_returns = mb_advs + self.trajectory_values
    self.trajectory_returns = mb_returns

  def update_estimator(self, test_mode=False):
    """Update the estimator from the most recent checkpoint.

    Args:
      test_mode: If set, it does not call tf_utils.
    """
    if not test_mode:
      self._policy = tf_utils.create_predictor(self.estimator,
                                               self.serving_input_fn)

  def mcts_sample_enable(self):
    """Indicates whether we should switch to MCTS data sampling.

    Returns:
      a boolean indicating whether MCTS sampling should start.
    """

    if not self._mcts_enable:
      return False

    if self._random_gen.uniform() <= self._mcts_sampling_frac:
      return True

    if ((self._sampling_step >= self._mcts_start_step_frac) and
        (self._sampling_step < self._mcts_end_step_frac)):
      return True

    return False

  def initialize_episode_data(self):
    # If we switch from PPO sampling to MCTS sampling or vice versa,
    # reset `episode_length` and `episode_reward` to zero.
    # That is, throwing away the data from PPO sampling.
    # Also, restart the return normalization. We treat this like starting
    # a new episode.
    self._episode_length = 0
    self._episode_reward = 0.
    self._ret = np.zeros(1)
    self._ret[0] = 0.

  def play(self, max_steps, test_mode=False):
    """Runs max_steps in the environment.

    Args:
      max_steps: Maximum number of steps to run.
      test_mode: If set, it does not update the policy.

    Returns:
      An array of states, actions, values, neg_logprobs, rewards, and returns.
    """
    # Update the estimator with the most recent checkpoint.
    self.update_estimator(test_mode)

    if self._first_time_call and self._mcts_enable:
      self.mcts_initialization()
      self._first_time_call = False

    if self.mcts_sample_enable():
      logging.info('MCTS Sampling...')
      # Update number of MCTS simulation with a decay factor.
      num_mcts_sim = max(
          self._current_mcts_player.num_mcts_sim * self._mcts_sim_decay_factor,
          4.0)
      mcts_temperature = max(
          self._current_mcts_player.temp_threshold *
          self._mcts_sim_decay_factor, 1.0)
      # Update MCTS hyperparameters
      self._current_mcts_player.num_mcts_sim = num_mcts_sim
      self._current_mcts_player.temp_threshold = mcts_temperature

      # If this is the first time performing MCTS sampling,
      # we need to perform a hard reset.
      if not self.mcts_sampling:
        self.initialize_episode_data()

      # Perform MCTS sampling only if the frequency of MCTS sampling
      # limit is met.
      if self._num_mcts_samples % self._mcts_collect_data_freq == 0:
        # Run trajectory for the specified number of steps using MCTS.
        self.run_mcts_trajectory(max_steps)

      self.mcts_sampling = True
      self._num_mcts_samples += 1
    else:
      # Run trajectory for the specified number of steps using policy network.
      logging.info('Policy Sampling...')

      # If the last sampling was MCTS, we need to perform a hard reset.
      if self.mcts_sampling:
        self.initialize_episode_data()

      self.run_trajectory(max_steps)
      self.mcts_sampling = False

    self._sampling_step += 1
