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

from typing import Sequence

from dopamine.replay_memory import circular_replay_buffer as crb
import gin
import numpy as np


ReplayElement = crb.ReplayElement


@gin.configurable
class MultiMCReplayBuffer(crb.OutOfGraphReplayBuffer):
  """This is an extension of the circular RB that handles Monte Carlo rollouts.

  Specifically, it supports two kinds of sampling:
  - Regular n-step sampling.
  - MonteCarlo rollout sampling (for a different value of n).
  """

  def __init__(self,
               observation_shape,
               stack_size,
               replay_capacity = 1000000,
               batch_size = 32,
               update_horizon = 10,
               gamma = 0.99,
               max_sample_attempts = 1000,
               extra_storage_types = (),
               extra_reward_storage_types = (),
               observation_dtype = np.uint8,
               num_additional_discount_factors = 32):
    """Initializes OutOfGraphReplayBufferWithMC.

    Note that not all constructor parameters are replicated here. The rest can
    be set via a gin config file.

    Args:
      observation_shape: tuple of ints.
      stack_size: int, number of frames to use in state stack.
      replay_capacity: int, number of transitions to keep in memory.
      batch_size: int.
      update_horizon: int, the length of the MC horizon.
      gamma: int, the discount factor.
      max_sample_attempts: int, the maximum number of attempts allowed to
        get a sample.
      extra_storage_types: list of ReplayElements defining the type of the extra
        contents that will be stored and returned by sample_transition_batch.
      extra_reward_storage_types: list of ReplayElements defining the type of
        the extra contents that will be stored and returned by
        sample_transition_batch. These elements will also be subject to
        the MC return calculation.
      observation_dtype: np.dtype, type of the observations. Defaults to
        np.uint8 for Atari 2600.
      num_additional_discount_factors: The number of additional discount
        factors to train with. These additional discount factors must
        be supplied to sample_transition_batch.
    """
    self._num_discount_factors = num_additional_discount_factors

    # TODO(joshgreaves): Implement.
    if extra_reward_storage_types:
      raise NotImplementedError(
          'extra_reward_storage_types hasn\'t been implemented yet.')

    extra_storage_types = list(extra_storage_types)
    extra_storage_types.extend([
        ReplayElement('additional_discount_factor_returns',
                      (self._num_discount_factors,),
                      np.float32),
    ])

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

  def sample_transition_batch(self,
                              batch_size=None,
                              indices=None,
                              *,
                              extra_discounts):
    """Returns a batch of transitions (including any extra contents).

    There are two different horizons being considered here, one for the regular
    transitions, and one for doing Monte Carlo rollouts for estimating returns.

    Args:
      batch_size: int, number of transitions returned. If None, the default
        batch_size will be used.
      indices: None or list of ints, the indices of every transition in the
        batch. If None, sample the indices uniformly.
      extra_discounts: If supplied, will compute the return for each discount
        in the sequence.

    Returns:
      transition_batch: tuple of np.arrays with the shape and type as in
        get_transition_elements().

    Raises:
      ValueError: If an element to be sampled is missing from the replay buffer.
    """
    if len(extra_discounts) != self._num_discount_factors:
      raise ValueError(
          f'The number of supplied discount factors ({len(extra_discounts)}) ',
          'must be equal to num_discount_factors '
          f'({self._num_discount_factors})')

    if batch_size is None:
      batch_size = self._batch_size
    if indices is None:
      indices = self.sample_index_batch(batch_size)
    assert len(indices) == batch_size

    extra_discounts = np.asarray(extra_discounts).reshape(-1, 1)
    extra_discounts = np.tile(
        extra_discounts, (1, self._update_horizon))
    extra_discounts[:, 0] = 1.0  # Required to make first step undiscounted.
    extra_discounts = np.cumprod(extra_discounts, axis=1)

    transition_elements = self.get_transition_elements(batch_size)
    batch_arrays = self._create_batch_arrays(batch_size)

    for batch_index, state_index in enumerate(indices):
      # Get transitions for regular updates.
      trajectory_indices = [(state_index + j) % self._replay_capacity
                            for j in range(self._update_horizon)]
      trajectory_terminals = self._store['terminal'][trajectory_indices]
      is_terminal_transition = trajectory_terminals.any()

      if not is_terminal_transition:
        trajectory_length = self._update_horizon
      else:
        # np.argmax of a bool array returns the index of the first True.
        trajectory_length = (
            np.argmax(trajectory_terminals.astype(bool), 0) + 1)

      next_state_index = state_index + trajectory_length
      trajectory_discount_vector = (
          self._cumulative_discount_vector[:trajectory_length])
      extra_discount_vector = extra_discounts[:, :trajectory_length]
      trajectory_rewards = self.get_range(self._store['reward'],
                                          state_index,
                                          next_state_index)

      # Fill the contents of each array in the sampled batch.
      assert len(transition_elements) == len(batch_arrays)
      for element_array, element in zip(batch_arrays, transition_elements):
        if element.name == 'state':
          element_array[batch_index] = self.get_observation_stack(state_index)
        elif element.name == 'reward':
          # compute the discounted sum of rewards in the trajectory.
          element_array[batch_index] = trajectory_discount_vector.dot(
              trajectory_rewards)
        elif element.name == 'additional_discount_factor_returns':
          element_array[batch_index] = (
              extra_discount_vector.dot(trajectory_rewards))
        elif element.name == 'next_state':
          element_array[batch_index] = self.get_observation_stack(
              (next_state_index) % self._replay_capacity)
        elif element.name == 'terminal':
          element_array[batch_index] = is_terminal_transition
        elif element.name == 'indices':
          element_array[batch_index] = state_index
        elif element.name in list(self._store.keys()):
          element_array[batch_index] = (
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
      if extra_replay_element.name == 'additional_discount_factor_returns':
        continue
      add_args.append(extra_replay_element)
    return add_args
