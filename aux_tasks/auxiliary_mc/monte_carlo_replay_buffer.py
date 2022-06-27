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

"""Monte carlo replay with support for auxiliary MC rewards."""

from dopamine.replay_memory import circular_replay_buffer as crb
import gin
import numpy as np


ReplayElement = crb.ReplayElement


@gin.configurable
class OutOfGraphReplayBufferWithMC(crb.OutOfGraphReplayBuffer):
  """This is an extension of the circular RB that handles Monte Carlo rollouts.

  Specifically, it supports two kinds of sampling:
  - Regular n-step sampling.
  - MonteCarlo rollout sampling (for a different value of n).
  """

  def __init__(self,
               observation_shape,
               stack_size,
               replay_capacity=1000000,
               batch_size=32,
               update_horizon=1,
               gamma=0.99,
               max_sample_attempts=1000,
               extra_storage_types=None,
               observation_dtype=np.uint8,
               monte_carlo_rollout_length=10,
               extra_monte_carlo_storage_types=None,
               reverse_fill=True):
    """Initializes OutOfGraphReplayBufferWithMC.

    Note that not all constructor parameters are replicated here. The rest can
    be set via a gin config file.

    Args:
      observation_shape: tuple of ints.
      stack_size: int, number of frames to use in state stack.
      replay_capacity: int, number of transitions to keep in memory.
      batch_size: int.
      update_horizon: int, length of update ('n' in n-step update).
      gamma: int, the discount factor.
      max_sample_attempts: int, the maximum number of attempts allowed to
        get a sample.
      extra_storage_types: list of ReplayElements defining the type of the extra
        contents that will be stored and returned by sample_transition_batch.
      observation_dtype: np.dtype, type of the observations. Defaults to
        np.uint8 for Atari 2600.
      monte_carlo_rollout_length: int, number of transitions to sample for the
        Monte Carlo rollout.
      extra_monte_carlo_storage_types: list of ReplayElements defining the type
        of the extra Monte carlo contents that will be stored and returned by
        sample_transition_batch.
      reverse_fill: bool, specifies whether we reverse-fill the returns upon
        finishing an episode, or whether we do n-step rollouts (where
        n=monte_carlo_rollout_length) for computing the MonteCarlo returns.
    """
    self._monte_carlo_rollout_length = monte_carlo_rollout_length
    self._reverse_fill = reverse_fill
    if extra_storage_types is None:
      extra_storage_types = []
    extra_storage_types += [
        ReplayElement('monte_carlo_reward', (), np.float32)
    ]
    self._keys_with_monte_carlo = ['reward']
    for element in extra_monte_carlo_storage_types:
      extra_storage_types += [
          ReplayElement(
              f'monte_carlo_{element.name}', element.shape, element.type)
      ]
      self._keys_with_monte_carlo.append(element.name)

    super().__init__(
        observation_shape,
        stack_size,
        replay_capacity,
        batch_size,
        update_horizon=update_horizon,
        gamma=gamma,
        max_sample_attempts=max_sample_attempts,
        extra_storage_types=extra_storage_types,
        observation_dtype=observation_dtype)

  def sample_index_batch(self, batch_size):
    """Returns a batch of valid indices sampled uniformly.

    Args:
      batch_size: int, number of indices returned.

    Returns:
      list of ints, a batch of valid indices sampled uniformly.

    Raises:
      RuntimeError: If the batch was not constructed after maximum number of
        tries.
    """
    if self._reverse_fill:
      horizon = self._update_horizon
    else:
      horizon = max(self._update_horizon, self._monte_carlo_rollout_length)
    if self.is_full():
      # add_count >= self._replay_capacity > self._stack_size
      min_id = self.cursor() - self._replay_capacity + self._stack_size - 1
      max_id = self.cursor() - horizon
    else:
      # add_count < self._replay_capacity
      min_id = self._stack_size - 1
      max_id = self.cursor() - horizon
      if max_id <= min_id:
        raise RuntimeError('Cannot sample a batch with fewer than stack size '
                           '({}) + horizon ({}) transitions.'.
                           format(self._stack_size, horizon))

    indices = []
    attempt_count = 0
    while (len(indices) < batch_size and
           attempt_count < self._max_sample_attempts):
      attempt_count += 1
      index = np.random.randint(min_id, max_id) % self._replay_capacity
      if self.is_valid_transition(index):
        indices.append(index)
    if len(indices) != batch_size:
      raise RuntimeError(
          'Max sample attempts: Tried {} times but only sampled {}'
          ' valid indices. Batch size is {}'.
          format(self._max_sample_attempts, len(indices), batch_size))

    return indices

  def is_valid_transition(self, index):
    """Checks if the index contains a valid transition."""
    is_valid = super().is_valid_transition(index)
    if is_valid and self._reverse_fill:
      # Checks whether monte carlo reward is already filled.
      for key in self._keys_with_monte_carlo:
        if np.any(np.isnan(self._store[f'monte_carlo_{key}'][index])):
          return False
    return is_valid

  def sample_transition_batch(self, batch_size=None, indices=None):
    """Returns a batch of transitions (including any extra contents).

    There are two different horizons being considered here, one for the regular
    transitions, and one for doing Monte Carlo rollouts for estimating returns.

    Args:
      batch_size: int, number of transitions returned. If None, the default
        batch_size will be used.
      indices: None or list of ints, the indices of every transition in the
        batch. If None, sample the indices uniformly.

    Returns:
      transition_batch: tuple of np.arrays with the shape and type as in
        get_transition_elements().

    Raises:
      ValueError: If an element to be sampled is missing from the replay buffer.
    """
    if batch_size is None:
      batch_size = self._batch_size
    if indices is None:
      indices = self.sample_index_batch(batch_size)
    assert len(indices) == batch_size

    transition_elements = self.get_transition_elements(batch_size)
    batch_arrays = self._create_batch_arrays(batch_size)
    for batch_element, state_index in enumerate(indices):
      # Get transitions for regular updates.
      trajectory_indices = [(state_index + j) % self._replay_capacity
                            for j in range(self._update_horizon)]
      trajectory_terminals = self._store['terminal'][trajectory_indices]
      is_terminal_transition = trajectory_terminals.any()
      if not is_terminal_transition:
        trajectory_length = self._update_horizon
      else:
        # np.argmax of a bool array returns the index of the first True.
        trajectory_length = np.argmax(trajectory_terminals.astype(bool),
                                      0) + 1
      next_state_index = state_index + trajectory_length
      trajectory_discount_vector = (
          self._cumulative_discount_vector[:trajectory_length])
      trajectory_rewards = self.get_range(self._store['reward'], state_index,
                                          next_state_index)

      if not self._reverse_fill:
        # Get transitions for Monte Carlo rollouts.
        monte_carlo_indices = [(state_index + j) % self._replay_capacity
                               for j in range(self._monte_carlo_rollout_length)]
        monte_carlo_terminals = self._store['terminal'][monte_carlo_indices]
        is_monte_carlo_terminal_transition = monte_carlo_terminals.any()
        if not is_monte_carlo_terminal_transition:
          monte_carlo_length = self._monte_carlo_rollout_length
        else:
          # np.argmax of a bool array returns the index of the first True.
          monte_carlo_length = np.argmax(monte_carlo_terminals.astype(bool),
                                         0) + 1

        # TODO(charlinel): Use `monte_carlo_length`, not `trajectory_length`.
        # Unfortunately it's not a drop-in replacement.
        next_state_monte_carlo_index = state_index + trajectory_length
        monte_carlo_discount_vector = (
            self._cumulative_discount_vector[:monte_carlo_length])
        monte_carlo_rewards = {}
        for key in self._keys_with_monte_carlo:
          monte_carlo_rewards[f'monte_carlo_{key}'] = self.get_range(
              self._store[key], state_index, next_state_monte_carlo_index)

      # Fill the contents of each array in the sampled batch.
      assert len(transition_elements) == len(batch_arrays)
      for element_array, element in zip(batch_arrays, transition_elements):
        if element.name == 'state':
          element_array[batch_element] = self.get_observation_stack(state_index)
        elif element.name == 'reward':
          # compute the discounted sum of rewards in the trajectory.
          element_array[batch_element] = trajectory_discount_vector.dot(
              trajectory_rewards)
        elif 'monte_carlo' in element.name and not self._reverse_fill:
          # compute the discounted sum of rewards in the trajectory.
          element_array[batch_element] = monte_carlo_discount_vector.dot(
              monte_carlo_rewards[element.name])
        elif element.name == 'next_state':
          element_array[batch_element] = self.get_observation_stack(
              (next_state_index) % self._replay_capacity)
        elif element.name == 'terminal':
          element_array[batch_element] = is_terminal_transition
        elif element.name == 'indices':
          element_array[batch_element] = state_index
        elif element.name in list(self._store.keys()):
          element_array[batch_element] = (
              self._store[element.name][state_index])
        # We assume the other elements are filled in by the subclass.

    return batch_arrays

  def get_add_args_signature(self):
    add_args = [
        ReplayElement('observation', self._observation_shape,
                      self._observation_dtype),
        ReplayElement('action', self._action_shape, self._action_dtype),
        ReplayElement('reward', self._reward_shape, self._reward_dtype),
        ReplayElement('terminal', (), self._terminal_dtype)
    ]

    for extra_replay_element in self._extra_storage_types:
      if 'monte_carlo' not in extra_replay_element.name:
        add_args.append(extra_replay_element)
    return add_args

  def _create_storage(self):
    super()._create_storage()
    for key in self._keys_with_monte_carlo:
      self._store[f'monte_carlo_{key}'][:] = np.nan

  def _record_monte_carlo_returns(self):
    cursor = (self.cursor() - 1) % self._replay_capacity
    for key in self._keys_with_monte_carlo:
      accumulated_returns = self._store[key][cursor]
      monte_carlo_key = f'monte_carlo_{key}'
      self._store[monte_carlo_key][cursor] = accumulated_returns
      cursor = (cursor - 1) % self._replay_capacity
      # Iterate until you reach a previous terminal state or episode boundary.
      while not (self._store['terminal'][cursor] or
                 (cursor in self.episode_end_indices)):
        accumulated_returns *= self._gamma
        accumulated_returns += self._store[key][cursor]
        self._store[monte_carlo_key][cursor] = accumulated_returns
        cursor = (cursor - 1) % self._replay_capacity

  def add(self,
          observation,
          action,
          reward,
          terminal,
          *args,
          priority=None,
          episode_end=False):
    """Adds transition to the replay and fills MC returns at terminal states."""
    super().add(observation, action, reward, terminal, *args,
                priority=priority, episode_end=episode_end)
    if (episode_end or terminal) and self._reverse_fill:
      self._record_monte_carlo_returns()
