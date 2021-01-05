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

"""The standard DQN replay memory.

This implementation is an out-of-graph replay memory + in-graph wrapper. It
supports vanilla n-step updates of the form typically found in the literature,
i.e. where rewards are accumulated for n steps and the intermediate trajectory
is not exposed to the agent. This does not allow, for example, performing
off-policy corrections.
"""

import collections
import gzip
import math
import os
import pickle

import gin
import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow.contrib import staging as contrib_staging


# Defines a type describing part of the tuple returned by the replay
# memory. Each element of the tuple is a tensor of shape [batch, ...] where
# ... is defined the 'shape' field of ReplayElement. The tensor type is
# given by the 'type' field. The 'name' field is for convenience and ease of
# debugging.
ReplayElement = (
    collections.namedtuple('shape_type', ['name', 'shape', 'type']))

# A prefix that can not collide with variable names for checkpoint files.
STORE_FILENAME_PREFIX = '$store$_'

# This constant determines how many iterations a checkpoint is kept for.
CHECKPOINT_DURATION = 4


def invalid_range(cursor, replay_capacity, stack_size, update_horizon):
  """Returns a array with the indices of cursor-related invalid transitions.

  There are update_horizon + stack_size invalid indices:
    - The update_horizon indices before the cursor, because we do not have a
      valid N-step transition (including the next state).
    - The stack_size indices on or immediately after the cursor.
  If N = update_horizon, K = stack_size, and the cursor is at c, invalid
  indices are:
    c - N, c - N + 1, ..., c, c + 1, ..., c + K - 1.

  It handles special cases in a circular buffer in the beginning and the end.

  Args:
    cursor: int, the position of the cursor.
    replay_capacity: int, the size of the replay memory.
    stack_size: int, the size of the stacks returned by the replay memory.
    update_horizon: int, the agent's update horizon.

  Returns:
    np.array of size stack_size with the invalid indices.
  """
  assert cursor < replay_capacity, 'cursor: %d' % cursor
  return np.array([(cursor - update_horizon + i) % replay_capacity
                   for i in range(stack_size + update_horizon)])


class OutOfGraphReplayBuffer(object):
  """A simple out-of-graph Replay Buffer.

  Stores transitions, state, action, reward, next_state, terminal (and any
  extra contents specified) in a circular buffer and provides a uniform
  transition sampling function.

  When the states consist of stacks of observations storing the states is
  inefficient. This class writes observations and constructs the stacked states
  at sample time.

  Attributes:
    add_count: int, counter of how many transitions have been added (including
      the blank ones at the beginning of an episode).
    invalid_range: np.array, an array with the indices of cursor-related invalid
      transitions
  """

  def __init__(self,
               observation_shape,
               stack_size,
               replay_capacity,
               batch_size,
               update_horizon=1,
               gamma=0.99,
               max_sample_attempts=1000,
               extra_storage_types=None,
               observation_dtype=np.uint8,
               terminal_dtype=np.uint8,
               action_shape=(),
               action_dtype=np.int32,
               reward_shape=(),
               reward_dtype=np.float32,
               trajectory_value='return',
               replay_forgetting='default'):
    """Initializes OutOfGraphReplayBuffer.

    Args:
      observation_shape: tuple of ints.
      stack_size: int, number of frames to use in state stack.
      replay_capacity: int, number of transitions to keep in memory.
      batch_size: int.
      update_horizon: int, length of update ('n' in n-step update).
      gamma: int, the discount factor.
      max_sample_attempts: int, the maximum number of attempts allowed to get a
        sample.
      extra_storage_types: list of ReplayElements defining the type of the extra
        contents that will be stored and returned by sample_transition_batch.
      observation_dtype: np.dtype, type of the observations. Defaults to
        np.uint8 for Atari 2600.
      terminal_dtype: np.dtype, type of the terminals. Defaults to np.uint8 for
        Atari 2600.
      action_shape: tuple of ints, the shape for the action vector. Empty tuple
        means the action is a scalar.
      action_dtype: np.dtype, type of elements in the action.
      reward_shape: tuple of ints, the shape of the reward vector. Empty tuple
        means the reward is a scalar.
      reward_dtype: np.dtype, type of elements in the reward.
      trajectory_value:  str, Metric for evaluating quality of trajectories for
        forgetting purposes.  One of ['return'].
      replay_forgetting:  str, What strategy to employ for forgetting old
        trajectories.  One of ['default', 'elephant'].

    Raises:
      ValueError: If replay_capacity is too small to hold at least one
        transition.
    """
    assert isinstance(observation_shape, tuple)
    if replay_capacity < update_horizon + stack_size:
      raise ValueError('There is not enough capacity to cover '
                       'update_horizon and stack_size.')

    tf.logging.info(
        'Creating a %s replay memory with the following parameters:',
        self.__class__.__name__)
    tf.logging.info('\t observation_shape: %s', str(observation_shape))
    tf.logging.info('\t observation_dtype: %s', str(observation_dtype))
    tf.logging.info('\t terminal_dtype: %s', str(terminal_dtype))
    tf.logging.info('\t stack_size: %d', stack_size)
    tf.logging.info('\t replay_capacity: %d', replay_capacity)
    tf.logging.info('\t batch_size: %d', batch_size)
    tf.logging.info('\t update_horizon: %d', update_horizon)
    tf.logging.info('\t gamma: %f', gamma)
    tf.logging.info('\t trajectory_value: %s', trajectory_value)
    tf.logging.info('\t replay_forgetting: %s', replay_forgetting)

    self._action_shape = action_shape
    self._action_dtype = action_dtype
    self._reward_shape = reward_shape
    self._reward_dtype = reward_dtype
    self._observation_shape = observation_shape
    self._stack_size = stack_size
    self._state_shape = self._observation_shape + (self._stack_size,)
    self._replay_capacity = replay_capacity
    self._batch_size = batch_size
    self._update_horizon = update_horizon
    self._gamma = gamma
    self._observation_dtype = observation_dtype
    self._terminal_dtype = terminal_dtype
    self._max_sample_attempts = max_sample_attempts
    if extra_storage_types:
      self._extra_storage_types = extra_storage_types
    else:
      self._extra_storage_types = []
    self._create_storage()
    self.add_count = np.array(0)
    self.invalid_range = np.zeros((self._stack_size))
    # When the horizon is > 1, we compute the sum of discounted rewards as a dot
    # product using the precomputed vector <gamma^0, gamma^1, ..., gamma^{n-1}>.
    self._cumulative_discount_vector = np.array(
        [math.pow(self._gamma, n) for n in range(update_horizon)],
        dtype=np.float32)
    self._trajectory_value = trajectory_value
    self._replay_forgetting = replay_forgetting
    assert trajectory_value == 'return'
    # The cursor for writing into a sorted replay buffer.  When the replay
    # buffer is not yet full, it tracks the cursor.  However, once the replay
    # buffer is full, it will always begin writing at the beginning of the
    # buffer - overwriting the least valuable trajectories.
    self._sorted_cursor = 0

  def _create_storage(self):
    """Creates the numpy arrays used to store transitions."""
    self._store = {}
    for storage_element in self.get_storage_signature():
      array_shape = [self._replay_capacity] + list(storage_element.shape)
      self._store[storage_element.name] = np.zeros(
          array_shape, dtype=storage_element.type)
    self._train_counts = np.empty([self._replay_capacity], dtype=np.int32)

  def _get_trajectory_spans(self):
    """Compute the span of non-looped trajectories."""
    assert (not self.is_empty()), 'Method should not be called on empty buffer.'

    # Record the terminal indices which mark episode boundaries.
    terminals = np.nonzero(self._store['terminal'])[0].tolist()
    num_terminals = len(terminals)
    spans = [(terminals[i] + 1, terminals[i + 1] + 1)
             for i in xrange(num_terminals - 1)]
    return spans

  def _compute_trajectory_value(self, spans):
    """Computes for each trajectory the value."""
    values = []

    for (start_idx, end_idx) in spans:
      if self._trajectory_value == 'return':
        # We currently assign value as max return over the span.
        max_return = max(self._store['return'][start_idx:end_idx])
        values.append(max_return)

    # We need to handle the looped-case specially.
    final_span_end = spans[-1][1]
    first_span_beg = spans[0][0]

    # Check whether the spans begin or end at the replay buffer boundaries.
    if final_span_end == self._replay_capacity:
      final_span_max = 0
    else:
      final_span_max = max(self._store['return'][final_span_end:])
    if first_span_beg == 0:
      first_span_max = 0
    else:
      first_span_max = max(self._store['return'][:first_span_beg])

    looped_span_value = max(final_span_max, first_span_max)
    values.append(looped_span_value)
    return values

  def sort_replay_buffer_trajectories(self):
    """Sort the trajectories within the replay buffer."""
    # We only need to sort the replay buffer once it's full.
    if not self.is_full():
      return

    spans = self._get_trajectory_spans()  # [...,(start_idx, end_id), ...]
    trajectory_values = self._compute_trajectory_value(spans)

    # Sort trajectories in increasing order.
    sorted_trajectory_indices = np.argsort(trajectory_values)

    # Create an empty tmp_store for sorting.
    tmp_store = {}
    for storage_element in self.get_storage_signature():
      array_shape = [self._replay_capacity] + list(storage_element.shape)
      tmp_store[storage_element.name] = np.zeros(
          array_shape, dtype=storage_element.type)

    # Add the sorted trajectories from self._store.
    tmp_cursor = 0
    for index in sorted_trajectory_indices:
      # Handle the looped span.
      if index == len(spans):
        final_span_end = spans[-1][1]
        first_span_beg = spans[0][0]
        traj_len = self._replay_capacity - final_span_end + first_span_beg
      else:
        (traj_start, traj_end) = spans[index]
        traj_len = traj_end - traj_start

      # Add all elements to the self._tmp_store.
      element_signatures = self.get_storage_signature()
      for element in element_signatures:
        element_name = element.name

        # Handle the looped span.
        if index == len(spans):
          element_trajectory_end = self._store[element_name][final_span_end:]
          element_trajectory_beg = self._store[element_name][:first_span_beg]
          element_trajectory = np.concatenate(
              [element_trajectory_end, element_trajectory_beg], 0)
        else:
          element_trajectory = self._store[element_name][traj_start:traj_end]

        tmp_store[element_name][tmp_cursor:tmp_cursor +
                                traj_len] = element_trajectory

      # Move the cursor after each trajectory.
      tmp_cursor += traj_len

    # After all these writes, the tmp_cursor should match the replay buffer
    # length.
    assert tmp_cursor == self._replay_capacity, 'Mismatched'

    # Move the tmp_store to the self._store.
    self._store = tmp_store

    # Return to writing to front.
    assert self.is_full()
    self._sorted_cursor = 0

    del tmp_store, tmp_cursor
    # TODO(liamfedus): reindex the sum_tree in priortized replay

  def get_add_args_signature(self):
    """The signature of the add function.

    Note - Derived classes may return a different signature.

    Returns:
      list of ReplayElements defining the type of the argument signature needed
        by the add function.
    """
    return self.get_storage_signature()

  def get_storage_signature(self):
    """Returns a default list of elements to be stored in this replay memory.

    Note - Derived classes may return a different signature.

    Returns:
      list of ReplayElements defining the type of the contents stored.
    """
    storage_elements = [
        ReplayElement('observation', self._observation_shape,
                      self._observation_dtype),
        ReplayElement('action', self._action_shape, self._action_dtype),
        ReplayElement('reward', self._reward_shape, self._reward_dtype),
        ReplayElement('terminal', (), self._terminal_dtype),
        ReplayElement('step_added', (), np.int32),
    ]

    for extra_replay_element in self._extra_storage_types:
      storage_elements.append(extra_replay_element)
    return storage_elements

  def _add_zero_transition(self):
    """Adds a padding transition filled with zeros (Used in episode beginnings)."""
    zero_transition = []
    for element_type in self.get_add_args_signature():
      zero_transition.append(
          np.zeros(element_type.shape, dtype=element_type.type))
    self._add(*zero_transition)

  def add(self, observation, action, reward, terminal, *args):
    """Adds a transition to the replay memory.

    This function checks the types and handles the padding at the beginning of
    an episode. Then it calls the _add function.

    Since the next_observation in the transition will be the observation added
    next there is no need to pass it.

    If the replay memory is at capacity the oldest transition will be discarded.

    Args:
      observation: np.array with shape observation_shape.
      action: int, the action in the transition.
      reward: float, the reward received in the transition.
      terminal: np.dtype, acts as a boolean indicating whether the transition
        was terminal (1) or not (0).
      *args: extra contents with shapes and dtypes according to
        extra_storage_types.
    """
    step_added = self.add_count
    self._check_add_types(observation, action, reward, terminal, step_added,
                          *args)
    if self.is_empty() or self._store['terminal'][self.cursor() - 1] == 1:
      for _ in range(self._stack_size - 1):
        # Child classes can rely on the padding transitions being filled with
        # zeros. This is useful when there is a priority argument.
        self._add_zero_transition()
    self._add(observation, action, reward, terminal, step_added, *args)

  def _add(self, *args):
    """Internal add method to add to the storage arrays.

    Args:
      *args: All the elements in a transition.
    """
    self._check_args_length(*args)
    transition = {
        e.name: args[idx] for idx, e in enumerate(self.get_add_args_signature())
    }
    self._add_transition(transition)

  def _add_transition(self, transition):
    """Internal add method to add transition dictionary to storage arrays.

    Args:
      transition: The dictionary of names and values of the transition to add to
        the storage.
    """
    cursor = self.cursor()

    for arg_name in transition:
      self._store[arg_name][cursor] = transition[arg_name]
    self._train_counts[cursor] = 0

    self.add_count += 1

    if self._replay_forgetting == 'elephant':
      self._sorted_cursor += 1

    self.invalid_range = invalid_range(self.cursor(), self._replay_capacity,
                                       self._stack_size, self._update_horizon)

  def _check_args_length(self, *args):
    """Check if args passed to the add method have the same length as storage.

    Args:
      *args: Args for elements used in storage.

    Raises:
      ValueError: If args have wrong length.
    """
    if len(args) != len(self.get_add_args_signature()):
      raise ValueError('Add expects {} elements, received {}'.format(
          len(self.get_add_args_signature()), len(args)))

  def _check_add_types(self, *args):
    """Checks if args passed to the add method match those of the storage.

    Args:
      *args: Args whose types need to be validated.

    Raises:
      ValueError: If args have wrong shape or dtype.
    """
    self._check_args_length(*args)
    for arg_element, store_element in zip(args, self.get_add_args_signature()):
      if isinstance(arg_element, np.ndarray):
        arg_shape = arg_element.shape
      elif isinstance(arg_element, tuple) or isinstance(arg_element, list):
        # TODO(b/80536437). This is not efficient when arg_element is a list.
        arg_shape = np.array(arg_element).shape
      else:
        # Assume it is scalar.
        arg_shape = tuple()
      store_element_shape = tuple(store_element.shape)
      if arg_shape != store_element_shape:
        raise ValueError('arg has shape {}, expected {}'.format(
            arg_shape, store_element_shape))

  def is_empty(self):
    """Is the Replay Buffer empty?"""
    return self.add_count == 0

  def is_full(self):
    """Is the Replay Buffer full?"""
    return self.add_count >= self._replay_capacity

  def cursor(self):
    """Index to the location where the next transition will be written."""
    if self._replay_forgetting == 'default':
      return self.add_count % self._replay_capacity
    elif self._replay_forgetting == 'elephant':
      # TODO(liamfedus): Check on sorted_cursor
      return self._sorted_cursor % self._replay_capacity

  def get_range(self, array, start_index, end_index):
    """Returns the range of array at the index handling wraparound if necessary.

    Args:
      array: np.array, the array to get the stack from.
      start_index: int, index to the start of the range to be returned. Range
        will wraparound if start_index is smaller than 0.
      end_index: int, exclusive end index. Range will wraparound if end_index
        exceeds replay_capacity.

    Returns:
      np.array, with shape [end_index - start_index, array.shape[1:]].
    """
    assert end_index > start_index, 'end_index must be larger than start_index'
    assert end_index >= 0
    assert start_index < self._replay_capacity
    if not self.is_full():
      assert end_index <= self.cursor(), (
          'Index {} has not been added.'.format(start_index))

    # Fast slice read when there is no wraparound.
    if start_index % self._replay_capacity < end_index % self._replay_capacity:
      return_array = array[start_index:end_index, Ellipsis]
    # Slow list read.
    else:
      indices = [(start_index + i) % self._replay_capacity
                 for i in range(end_index - start_index)]
      return_array = array[indices, Ellipsis]
    return return_array

  def get_observation_stack(self, index):
    return self._get_element_stack(index, 'observation')

  def _get_element_stack(self, index, element_name):
    state = self.get_range(self._store[element_name],
                           index - self._stack_size + 1, index + 1)
    # The stacking axis is 0 but the agent expects as the last axis.
    return np.moveaxis(state, 0, -1)

  def get_terminal_stack(self, index):
    return self.get_range(self._store['terminal'], index - self._stack_size + 1,
                          index + 1)

  def is_valid_transition(self, index):
    """Checks if the index contains a valid transition.

    Checks for collisions with the end of episodes and the current position
    of the cursor.

    Args:
      index: int, the index to the state in the transition.

    Returns:
      Is the index valid: Boolean.

    """
    # Check the index is in the valid range
    if index < 0 or index >= self._replay_capacity:
      return False
    if not self.is_full():
      # The indices and next_indices must be smaller than the cursor.
      if index >= self.cursor() - self._update_horizon:
        return False
      # The first few indices contain the padding states of the first episode.
      if index < self._stack_size - 1:
        return False

    # Skip transitions that straddle the cursor.
    if index in set(self.invalid_range):
      return False

    # If there are terminal flags in any other frame other than the last one
    # the stack is not valid, so don't sample it.
    if self.get_terminal_stack(index)[:-1].any():
      return False

    return True

  def _create_batch_arrays(self, batch_size):
    """Create a tuple of arrays with the type of get_transition_elements.

    When using the WrappedReplayBuffer with staging enabled it is important to
    create new arrays every sample because StaginArea keeps a pointer to the
    returned arrays.

    Args:
      batch_size: (int) number of transitions returned. If None the default
        batch_size will be used.

    Returns:
      Tuple of np.arrays with the shape and type of get_transition_elements.
    """
    transition_elements = self.get_transition_elements(batch_size)
    batch_arrays = []
    for element in transition_elements:
      batch_arrays.append(np.empty(element.shape, dtype=element.type))
    return tuple(batch_arrays)

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
    if self.is_full():
      # add_count >= self._replay_capacity > self._stack_size
      min_id = self.cursor() - self._replay_capacity + self._stack_size - 1
      max_id = self.cursor() - self._update_horizon
    else:
      # add_count < self._replay_capacity
      min_id = self._stack_size - 1
      max_id = self.cursor() - self._update_horizon
      if max_id <= min_id:
        raise RuntimeError('Cannot sample a batch with fewer than stack size '
                           '({}) + update_horizon ({}) transitions.'.format(
                               self._stack_size, self._update_horizon))

    indices = []
    attempt_count = 0
    while (len(indices) < batch_size and
           attempt_count < self._max_sample_attempts):
      index = np.random.randint(min_id, max_id) % self._replay_capacity
      if self.is_valid_transition(index):
        indices.append(index)
      else:
        attempt_count += 1
    if len(indices) != batch_size:
      raise RuntimeError(
          'Max sample attempts: Tried {} times but only sampled {}'
          ' valid indices. Batch size is {}'.format(self._max_sample_attempts,
                                                    len(indices), batch_size))

    return indices

  def sample_transition_batch(self, batch_size=None, indices=None):
    """Returns a batch of transitions (including any extra contents).

    If get_transition_elements has been overridden and defines elements not
    stored in self._store, an empty array will be returned and it will be
    left to the child class to fill it. For example, for the child class
    OutOfGraphPrioritizedReplayBuffer, the contents of the
    sampling_probabilities are stored separately in a sum tree.

    When the transition is terminal next_state_batch has undefined contents.

    NOTE: This transition contains the indices of the sampled elements. These
    are only valid during the call to sample_transition_batch, i.e. they may
    be used by subclasses of this replay buffer but may point to different data
    as soon as sampling is done.

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
      trajectory_indices = [(state_index + j) % self._replay_capacity
                            for j in range(self._update_horizon)]
      trajectory_terminals = self._store['terminal'][trajectory_indices]
      is_terminal_transition = trajectory_terminals.any()
      if not is_terminal_transition:
        trajectory_length = self._update_horizon
      else:
        # np.argmax of a bool array returns the index of the first True.
        trajectory_length = np.argmax(trajectory_terminals.astype(np.bool),
                                      0) + 1
      next_state_index = state_index + trajectory_length
      trajectory_discount_vector = (
          self._cumulative_discount_vector[:trajectory_length])
      trajectory_rewards = self.get_range(self._store['reward'], state_index,
                                          next_state_index)

      # Fill the contents of each array in the sampled batch.
      assert len(transition_elements) == len(batch_arrays)
      for element_array, element in zip(batch_arrays, transition_elements):
        if element.name == 'state':
          element_array[batch_element] = self.get_observation_stack(state_index)
        elif element.name == 'reward':
          # compute the discounted sum of rewards in the trajectory.
          element_array[batch_element] = np.sum(
              trajectory_discount_vector * trajectory_rewards, axis=0)
        elif element.name == 'next_state':
          element_array[batch_element] = self.get_observation_stack(
              (next_state_index) % self._replay_capacity)
        elif element.name in ('next_action', 'next_reward'):
          element_array[batch_element] = (
              self._store[element.name.lstrip('next_')][(next_state_index) %
                                                        self._replay_capacity])
        elif element.name == 'terminal':
          element_array[batch_element] = is_terminal_transition
        elif element.name == 'indices':
          element_array[batch_element] = state_index
        elif element.name == 'train_counts':
          element_array[batch_element] = self._train_counts[state_index]
        elif element.name == 'steps_until_first_train':
          step_added = self._store['step_added'][state_index]
          steps_since_add = self.add_count - step_added
          already_trained = self._train_counts[state_index] > 0
          steps_until_first_train = -1 if already_trained else steps_since_add
          element_array[batch_element] = steps_until_first_train
        elif element.name == 'age':
          step_added = self._store['step_added'][state_index]
          steps_since_add = self.add_count - step_added
          element_array[batch_element] = steps_since_add
        elif element.name in self._store.keys():
          element_array[batch_element] = (
              self._store[element.name][state_index])
        # We assume the other elements are filled in by the subclass.

    return batch_arrays

  def get_transition_elements(self, batch_size=None):
    """Returns a 'type signature' for sample_transition_batch.

    Args:
      batch_size: int, number of transitions returned. If None, the default
        batch_size will be used.

    Returns:
      signature: A namedtuple describing the method's return type signature.
    """
    batch_size = self._batch_size if batch_size is None else batch_size

    transition_elements = [
        ReplayElement('state', (batch_size,) + self._state_shape,
                      self._observation_dtype),
        ReplayElement('action', (batch_size,) + self._action_shape,
                      self._action_dtype),
        ReplayElement('reward', (batch_size,) + self._reward_shape,
                      self._reward_dtype),
        ReplayElement('next_state', (batch_size,) + self._state_shape,
                      self._observation_dtype),
        ReplayElement('next_action', (batch_size,) + self._action_shape,
                      self._action_dtype),
        ReplayElement('next_reward', (batch_size,) + self._reward_shape,
                      self._reward_dtype),
        ReplayElement('terminal', (batch_size,), self._terminal_dtype),
        ReplayElement('indices', (batch_size,), np.int32),
        ReplayElement('train_counts', (batch_size,), np.int32),
        ReplayElement('steps_until_first_train', (batch_size,), np.int32),
        ReplayElement('age', (batch_size,), np.int32),
    ]
    for element in self._extra_storage_types:
      transition_elements.append(
          ReplayElement(element.name, (batch_size,) + tuple(element.shape),
                        element.type))
    return transition_elements

  def update_train_counts(self, indices):
    """Increments the train count for all transitions that were sampled."""
    for memory_index in indices:
      self._train_counts[memory_index] += 1

  def _generate_filename(self, checkpoint_dir, name, suffix):
    return os.path.join(checkpoint_dir, '{}_ckpt.{}.gz'.format(name, suffix))

  def _return_checkpointable_elements(self):
    """Return the dict of elements of the class for checkpointing.

    Returns:
      checkpointable_elements: dict containing all non private (starting with
      _) members + all the arrays inside self._store.
    """
    checkpointable_elements = {}
    for member_name, member in self.__dict__.items():
      if member_name == '_store':
        for array_name, array in self._store.items():
          checkpointable_elements[STORE_FILENAME_PREFIX + array_name] = array
      elif not member_name.startswith('_'):
        checkpointable_elements[member_name] = member
      elif member_name in ['_train_counts']:  # Exceptions to the above rule.
        checkpointable_elements[member_name] = member
    return checkpointable_elements

  def save(self, checkpoint_dir, iteration_number):
    """Save the OutOfGraphReplayBuffer attributes into a file.

    This method will save all the replay buffer's state in a single file.

    Args:
      checkpoint_dir: str, the directory where numpy checkpoint files should be
        saved.
      iteration_number: int, iteration_number to use as a suffix in naming numpy
        checkpoint files.
    """
    if not tf.gfile.Exists(checkpoint_dir):
      return

    checkpointable_elements = self._return_checkpointable_elements()

    for attr in checkpointable_elements:
      filename = self._generate_filename(checkpoint_dir, attr, iteration_number)
      with tf.gfile.Open(filename, 'wb') as f:
        with gzip.GzipFile(fileobj=f) as outfile:
          # Checkpoint the np arrays in self._store with np.save instead of
          # pickling the dictionary is critical for file size and performance.
          # STORE_FILENAME_PREFIX indicates that the variable is contained in
          # self._store.
          if attr.startswith(STORE_FILENAME_PREFIX):
            array_name = attr[len(STORE_FILENAME_PREFIX):]
            np.save(outfile, self._store[array_name], allow_pickle=False)
          # Some numpy arrays might not be part of storage
          elif isinstance(self.__dict__[attr], np.ndarray):
            np.save(outfile, self.__dict__[attr], allow_pickle=False)
          else:
            pickle.dump(self.__dict__[attr], outfile)

      # After writing a checkpoint file, we garbage collect the checkpoint file
      # that is four versions old.
      stale_iteration_number = iteration_number - CHECKPOINT_DURATION
      if stale_iteration_number >= 0:
        stale_filename = self._generate_filename(checkpoint_dir, attr,
                                                 stale_iteration_number)
        try:
          tf.gfile.Remove(stale_filename)
        except tf.errors.NotFoundError:
          pass

  def load(self, checkpoint_dir, suffix):
    """Restores the object from bundle_dictionary and numpy checkpoints.

    Args:
      checkpoint_dir: str, the directory where to read the numpy checkpointed
        files from.
      suffix: str, the suffix to use in numpy checkpoint files.

    Raises:
      NotFoundError: If not all expected files are found in directory.
    """
    save_elements = self._return_checkpointable_elements()
    # We will first make sure we have all the necessary files available to avoid
    # loading a partially-specified (i.e. corrupted) replay buffer.
    for attr in save_elements:
      filename = self._generate_filename(checkpoint_dir, attr, suffix)
      if not tf.gfile.Exists(filename):
        raise tf.errors.NotFoundError(None, None,
                                      'Missing file: {}'.format(filename))
    # If we've reached this point then we have verified that all expected files
    # are available.
    for attr in save_elements:
      filename = self._generate_filename(checkpoint_dir, attr, suffix)
      with tf.gfile.Open(filename, 'rb') as f:
        with gzip.GzipFile(fileobj=f) as infile:
          if attr.startswith(STORE_FILENAME_PREFIX):
            array_name = attr[len(STORE_FILENAME_PREFIX):]
            self._store[array_name] = np.load(infile, allow_pickle=False)
          elif isinstance(self.__dict__[attr], np.ndarray):
            self.__dict__[attr] = np.load(infile, allow_pickle=False)
          else:
            self.__dict__[attr] = pickle.load(infile)


@gin.configurable(
    blacklist=['observation_shape', 'stack_size', 'update_horizon', 'gamma'])
class WrappedReplayBuffer(object):
  """Wrapper of OutOfGraphReplayBuffer with an in graph sampling mechanism.

  Usage:
    To add a transition:  call the add function.

    To sample a batch:    Construct operations that depend on any of the
                          tensors is the transition dictionary. Every sess.run
                          that requires any of these tensors will sample a new
                          transition.
  """

  def __init__(self,
               observation_shape,
               stack_size,
               use_staging=True,
               replay_capacity=1000000,
               batch_size=32,
               update_horizon=1,
               gamma=0.99,
               wrapped_memory=None,
               max_sample_attempts=1000,
               extra_storage_types=None,
               observation_dtype=np.uint8,
               terminal_dtype=np.uint8,
               action_shape=(),
               action_dtype=np.int32,
               reward_shape=(),
               reward_dtype=np.float32,
               replay_forgetting='default'):
    """Initializes WrappedReplayBuffer.

    Args:
      observation_shape: tuple of ints.
      stack_size: int, number of frames to use in state stack.
      use_staging: bool, when True it would use a staging area to prefetch the
        next sampling batch.
      replay_capacity: int, number of transitions to keep in memory.
      batch_size: int.
      update_horizon: int, length of update ('n' in n-step update).
      gamma: int, the discount factor.
      wrapped_memory: The 'inner' memory data structure. If None, it creates the
        standard DQN replay memory.
      max_sample_attempts: int, the maximum number of attempts allowed to get a
        sample.
      extra_storage_types: list of ReplayElements defining the type of the extra
        contents that will be stored and returned by sample_transition_batch.
      observation_dtype: np.dtype, type of the observations. Defaults to
        np.uint8 for Atari 2600.
      terminal_dtype: np.dtype, type of the terminals. Defaults to np.uint8 for
        Atari 2600.
      action_shape: tuple of ints, the shape for the action vector. Empty tuple
        means the action is a scalar.
      action_dtype: np.dtype, type of elements in the action.
      reward_shape: tuple of ints, the shape of the reward vector. Empty tuple
        means the reward is a scalar.
      reward_dtype: np.dtype, type of elements in the reward.
      replay_forgetting:  str, What strategy to employ for forgetting old
        trajectories.  One of ['default', 'elephant'].

    Raises:
      ValueError: If update_horizon is not positive.
      ValueError: If discount factor is not in [0, 1].
    """
    if replay_capacity < update_horizon + 1:
      raise ValueError('Update horizon ({}) should be significantly smaller '
                       'than replay capacity ({}).'.format(
                           update_horizon, replay_capacity))
    if not update_horizon >= 1:
      raise ValueError('Update horizon must be positive.')
    if not 0.0 <= gamma <= 1.0:
      raise ValueError('Discount factor (gamma) must be in [0, 1].')

    self.batch_size = batch_size

    # Mainly used to allow subclasses to pass self.memory.
    if wrapped_memory is not None:
      self.memory = wrapped_memory
    else:
      self.memory = OutOfGraphReplayBuffer(
          observation_shape,
          stack_size,
          replay_capacity,
          batch_size,
          update_horizon,
          gamma,
          max_sample_attempts,
          observation_dtype=observation_dtype,
          terminal_dtype=terminal_dtype,
          extra_storage_types=extra_storage_types,
          action_shape=action_shape,
          action_dtype=action_dtype,
          reward_shape=reward_shape,
          reward_dtype=reward_dtype,
          replay_forgetting=replay_forgetting)

    self.create_sampling_ops(use_staging)
    tf.logging.info('\t replay_forgetting: %s', replay_forgetting)

  def add(self, observation, action, reward, terminal, *args):
    """Adds a transition to the replay memory.

    Since the next_observation in the transition will be the observation added
    next there is no need to pass it.

    If the replay memory is at capacity the oldest transition will be discarded.

    Args:
      observation: np.array with shape observation_shape.
      action: int, the action in the transition.
      reward: float, the reward received in the transition.
      terminal: np.dtype, acts as a boolean indicating whether the transition
        was terminal (1) or not (0).
      *args: extra contents with shapes and dtypes according to
        extra_storage_types.
    """
    self.memory.add(observation, action, reward, terminal, *args)

  def create_sampling_ops(self, use_staging):
    """Creates the ops necessary to sample from the replay buffer.

    Creates the transition dictionary containing the sampling tensors.

    Args:
      use_staging: bool, when True it would use a staging area to prefetch the
        next sampling batch.
    """
    with tf.name_scope('sample_replay'):
      with tf.device('/cpu:*'):
        transition_type = self.memory.get_transition_elements()
        transition_tensors = tf.py_func(
            self.memory.sample_transition_batch, [],
            [return_entry.type for return_entry in transition_type],
            name='replay_sample_py_func')
        self._set_transition_shape(transition_tensors, transition_type)
        if use_staging:
          transition_tensors = self._set_up_staging(transition_tensors)
          self._set_transition_shape(transition_tensors, transition_type)

        # Unpack sample transition into member variables.
        self.unpack_transition(transition_tensors, transition_type)

  def _set_transition_shape(self, transition, transition_type):
    """Set shape for each element in the transition.

    Args:
      transition: tuple of tf.Tensors.
      transition_type: tuple of ReplayElements descriving the shapes of the
        respective tensors.
    """
    for element, element_type in zip(transition, transition_type):
      element.set_shape(element_type.shape)

  def _set_up_staging(self, transition):
    """Sets up staging ops for prefetching the next transition.

    This allows us to hide the py_func latency. To do so we use a staging area
    to pre-fetch the next batch of transitions.

    Args:
      transition: tuple of tf.Tensors with shape
        memory.get_transition_elements().

    Returns:
      prefetched_transition: tuple of tf.Tensors with shape
        memory.get_transition_elements() that have been previously prefetched.
    """
    transition_type = self.memory.get_transition_elements()

    # Create the staging area in CPU.
    prefetch_area = contrib_staging.StagingArea(
        [shape_with_type.type for shape_with_type in transition_type])

    # Store prefetch op for tests, but keep it private -- users should not be
    # calling _prefetch_batch.
    self._prefetch_batch = prefetch_area.put(transition)
    initial_prefetch = tf.cond(
        tf.equal(prefetch_area.size(), 0),
        lambda: prefetch_area.put(transition), tf.no_op)

    # Every time a transition is sampled self.prefetch_batch will be
    # called. If the staging area is empty, two put ops will be called.
    with tf.control_dependencies([self._prefetch_batch, initial_prefetch]):
      prefetched_transition = prefetch_area.get()

    return prefetched_transition

  def unpack_transition(self, transition_tensors, transition_type):
    """Unpacks the given transition into member variables.

    Args:
      transition_tensors: tuple of tf.Tensors.
      transition_type: tuple of ReplayElements matching transition_tensors.
    """
    self.transition = collections.OrderedDict()
    for element, element_type in zip(transition_tensors, transition_type):
      self.transition[element_type.name] = element

    # TODO(bellemare): These are legacy and should probably be removed in
    # future versions.
    self.states = self.transition['state']
    self.actions = self.transition['action']
    self.rewards = self.transition['reward']
    self.next_states = self.transition['next_state']
    self.next_actions = self.transition['next_action']
    self.next_rewards = self.transition['next_reward']
    self.terminals = self.transition['terminal']
    self.indices = self.transition['indices']

  def save(self, checkpoint_dir, iteration_number):
    """Save the underlying replay buffer's contents in a file.

    Args:
      checkpoint_dir: str, the directory where to read the numpy checkpointed
        files from.
      iteration_number: int, the iteration_number to use as a suffix in naming
        numpy checkpoint files.
    """
    self.memory.save(checkpoint_dir, iteration_number)

  def load(self, checkpoint_dir, suffix):
    """Loads the replay buffer's state from a saved file.

    Args:
      checkpoint_dir: str, the directory where to read the numpy checkpointed
        files from.
      suffix: str, the suffix to use in numpy checkpoint files.
    """
    self.memory.load(checkpoint_dir, suffix)

  def tf_update_train_counts(self, indices):
    """Updates the train counts for the given indices.

    Args:
      indices: tf.Tensor with dtype int32 and shape [n].

    Returns:
       A tf op updating the train count.
    """
    return tf.py_func(
        self.memory.update_train_counts, [indices], [],
        name='replay_update_train_counts_py_func')
