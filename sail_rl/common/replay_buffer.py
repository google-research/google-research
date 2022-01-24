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

"""DQN replay memory with support for SAIL (stores returns)."""
import collections
import itertools

from dopamine.replay_memory import circular_replay_buffer as crb
from dopamine.replay_memory import sum_tree
import gin
import numpy as np
import tensorflow.compat.v2 as tf

PLACEHOLDER_RETURN_VALUE = np.finfo(np.float32).min


class SAILOutOfGraphReplayBuffer(crb.OutOfGraphReplayBuffer):
  """A simple out-of-graph Replay Buffer with support for SAIL.

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

  def __init__(self, observation_shape, stack_size,
               replay_capacity, batch_size, **kwargs):
    self.episode_num_last_completed = -1
    self.curr_episode_start = stack_size - 1
    self.curr_episode_end = stack_size - 2
    super(SAILOutOfGraphReplayBuffer, self).__init__(
        observation_shape, stack_size, replay_capacity, batch_size, **kwargs)

  def get_storage_signature(self):
    """Returns a default list of elements to be stored in this replay memory.

    Note - Derived classes may return a different signature.

    Returns:
      list of ReplayElements defining the type of the contents stored.
    """
    storage_elements = [
        crb.ReplayElement('observation', self._observation_shape,
                          self._observation_dtype),
        crb.ReplayElement('action', self._action_shape, self._action_dtype),
        crb.ReplayElement('reward', self._reward_shape, self._reward_dtype),
        crb.ReplayElement('terminal', (), self._terminal_dtype),
        crb.ReplayElement('return', (), np.float32),
        crb.ReplayElement('episode_num', (), np.int32),
    ]

    for extra_replay_element in self._extra_storage_types:
      storage_elements.append(extra_replay_element)
    return storage_elements

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
    self._check_add_types(observation, action, reward, terminal, *args)
    if self.is_empty() or self._store['terminal'][self.cursor() - 1] == 1:
      for _ in range(self._stack_size - 1):
        # Child classes can rely on the padding transitions being filled with
        # zeros. This is useful when there is a priority argument.
        self._add_zero_transition()
    self._add(observation, action, reward, terminal, *args)
    self.curr_episode_end = (
        self.curr_episode_end + 1) % self._replay_capacity

  def is_valid_transition(self, index, sample_from_complete_episodes=False):
    """Checks if the index contains a valid transition.

    Checks for collisions with the end of episodes and the current position
    of the cursor.

    Args:
      index: int, the index to the state in the transition.
      sample_from_complete_episodes: bool, whether to sample only transitions
        from completed episodes.

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

    # If the episode the transition is from was not inserted entirely,
    # optionally don't sample it.
    if sample_from_complete_episodes:
      episode_num = self._store['episode_num'][index]
      if episode_num > self.episode_num_last_completed:
        return False

    return True

  def sample_index_batch(self, batch_size, sample_from_complete_episodes=False):
    """Returns a batch of valid indices sampled uniformly.

    If sample_complete_episodes=True, only returns indices of transitions where
    the full episode was inserted in the replay buffer.

    Args:
      batch_size: int, number of indices returned.
      sample_from_complete_episodes: bool, whether to sample only transitions
        from completed episodes.

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
                           '({}) + update_horizon ({}) transitions.'.
                           format(self._stack_size, self._update_horizon))

    indices = []
    attempt_count = 0
    while (len(indices) < batch_size and
           attempt_count < self._max_sample_attempts):
      index = np.random.randint(min_id, max_id) % self._replay_capacity
      if self.is_valid_transition(index, sample_from_complete_episodes):
        indices.append(index)
      else:
        attempt_count += 1
    if len(indices) != batch_size:
      raise RuntimeError(
          'Max sample attempts: Tried {} times but only sampled {}'
          ' valid indices. Batch size is {}'.
          format(self._max_sample_attempts, len(indices), batch_size))

    return indices

  def sample_transition_batch(self, batch_size=None, indices=None,
                              sample_from_complete_episodes=True):
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
      sample_from_complete_episodes: bool, whether to sample only transitions
        from completed episodes.

    Returns:
      transition_batch: tuple of np.arrays with the shape and type as in
        get_transition_elements().

    Raises:
      ValueError: If an element to be sampled is missing from the replay buffer.
    """
    if batch_size is None:
      batch_size = self._batch_size
    if indices is None:
      indices = self.sample_index_batch(
          batch_size, sample_from_complete_episodes)
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
        trajectory_length = np.argmax(trajectory_terminals.astype(bool),
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
        crb.ReplayElement('state', (batch_size,) + self._state_shape,
                          self._observation_dtype),
        crb.ReplayElement('action', (batch_size,) + self._action_shape,
                          self._action_dtype),
        crb.ReplayElement('reward', (batch_size,) + self._reward_shape,
                          self._reward_dtype),
        crb.ReplayElement('next_state', (batch_size,) + self._state_shape,
                          self._observation_dtype),
        crb.ReplayElement('next_action', (batch_size,) + self._action_shape,
                          self._action_dtype),
        crb.ReplayElement('next_reward', (batch_size,) + self._reward_shape,
                          self._reward_dtype),
        crb.ReplayElement('terminal', (batch_size,), self._terminal_dtype),
        crb.ReplayElement('indices', (batch_size,), np.int32),
        crb.ReplayElement('return', (batch_size,), np.float32),
        crb.ReplayElement('episode_num', (batch_size,), np.int32),
    ]
    for element in self._extra_storage_types:
      transition_elements.append(
          crb.ReplayElement(element.name, (batch_size,) + tuple(element.shape),
                            element.type))
    return transition_elements

  def _calculate_discounted_returns(self, rewards):
    returns_reversed = itertools.accumulate(rewards[::-1],
                                            lambda x, y: x * self._gamma + y)
    return np.array(list(returns_reversed))[::-1]

  def _get_circular_slice(self, array, start, end):
    assert array.ndim == 1
    if end >= start:
      return array[start: end + 1]
    else:
      return np.concatenate([array[start:], array[:end + 1]])

  def _set_circular_slice(self, array, start, end, values):
    assert array.ndim == 1
    if end >= start:
      assert len(values) == end - start + 1
      array[start: end + 1] = values
    else:
      length_left = len(array) - start
      assert len(values) == end + length_left + 1
      array[start:] = values[:length_left]
      array[:end + 1] = values[length_left:]

  def calculate_and_store_return(self, episode_num):
    """Calculates and updates the return of a given episode based on stored rewards.

    Args:
      episode_num: int, identifier of the episode.
    Raises:
      RuntimeError: if the episode queried does not exist or is unfinished.
    """
    if episode_num != self.episode_num_last_completed + 1:
      raise RuntimeError(
          'The next completed episode should have number {}. '
          'Found `episode_num`={}.'.format(
              self.episode_num_last_completed + 1, episode_num))
    if self._store['terminal'][self.curr_episode_end] != 1:
      raise RuntimeError(
          'Trying to calculate the return of an unfinished episode.')
    rewards = self._get_circular_slice(
        self._store['reward'],
        self.curr_episode_start,
        self.curr_episode_end)
    returns = self._calculate_discounted_returns(rewards)
    self._set_circular_slice(
        self._store['return'],
        self.curr_episode_start,
        self.curr_episode_end,
        returns)
    self.episode_num_last_completed = episode_num
    self.curr_episode_start = (
        self.curr_episode_end + self._stack_size) % self._replay_capacity
    self.curr_episode_end = (
        self.curr_episode_end + self._stack_size - 1) % self._replay_capacity


@gin.configurable(denylist=['observation_shape', 'stack_size',
                            'update_horizon', 'gamma'])
class SAILWrappedReplayBuffer(crb.WrappedReplayBuffer):
  """Wrapper of SAILOutOfGraphReplayBuffer with an in graph sampling mechanism.

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
               reward_dtype=np.float32):
    """Initializes SILWrappedReplayBuffer.

    Args:
      observation_shape: tuple of ints.
      stack_size: int, number of frames to use in state stack.
      use_staging: bool, when True it would use a staging area to prefetch
        the next sampling batch.
      replay_capacity: int, number of transitions to keep in memory.
      batch_size: int.
      update_horizon: int, length of update ('n' in n-step update).
      gamma: int, the discount factor.
      wrapped_memory: The 'inner' memory data structure. If None,
        it creates the standard DQN replay memory.
      max_sample_attempts: int, the maximum number of attempts allowed to
        get a sample.
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

    Raises:
      ValueError: If update_horizon is not positive.
      ValueError: If discount factor is not in [0, 1].
    """
    if replay_capacity < update_horizon + 1:
      raise ValueError(
          'Update horizon ({}) should be significantly smaller '
          'than replay capacity ({}).'.format(update_horizon, replay_capacity))
    if not update_horizon >= 1:
      raise ValueError('Update horizon must be positive.')
    if not 0.0 <= gamma <= 1.0:
      raise ValueError('Discount factor (gamma) must be in [0, 1].')

    self.batch_size = batch_size

    # Mainly used to allow subclasses to pass self.memory.
    if wrapped_memory is not None:
      self.memory = wrapped_memory
    else:
      self.memory = SAILOutOfGraphReplayBuffer(
          observation_shape,
          stack_size,
          replay_capacity,
          batch_size,
          update_horizon=update_horizon,
          gamma=gamma,
          max_sample_attempts=max_sample_attempts,
          observation_dtype=observation_dtype,
          terminal_dtype=terminal_dtype,
          extra_storage_types=extra_storage_types,
          action_shape=action_shape,
          action_dtype=action_dtype,
          reward_shape=reward_shape,
          reward_dtype=reward_dtype)

    self.create_sampling_ops(use_staging)

  @property
  def episode_num_last_completed(self):
    return self.memory.episode_num_last_completed

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
    self.returns = self.transition['return']
    self.episode_nums = self.transition['episode_num']

  def calculate_and_store_return(self, episode_num):
    self.memory.calculate_and_store_return(episode_num)


@gin.configurable
class SAILOutOfGraphPrioritizedReplayBuffer(SAILOutOfGraphReplayBuffer):
  """An out-of-graph Replay Buffer for PER with self-imitation support.
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
               reward_dtype=np.float32):
    """Initializes SAILOutOfGraphPrioritizedReplayBuffer.

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
      terminal_dtype: np.dtype, type of the terminals. Defaults to np.uint8 for
        Atari 2600.
      action_shape: tuple of ints, the shape for the action vector. Empty tuple
        means the action is a scalar.
      action_dtype: np.dtype, type of elements in the action.
      reward_shape: tuple of ints, the shape of the reward vector. Empty tuple
        means the reward is a scalar.
      reward_dtype: np.dtype, type of elements in the reward.
    """
    super(SAILOutOfGraphPrioritizedReplayBuffer, self).__init__(
        observation_shape=observation_shape,
        stack_size=stack_size,
        replay_capacity=replay_capacity,
        batch_size=batch_size,
        update_horizon=update_horizon,
        gamma=gamma,
        max_sample_attempts=max_sample_attempts,
        extra_storage_types=extra_storage_types,
        observation_dtype=observation_dtype,
        terminal_dtype=terminal_dtype,
        action_shape=action_shape,
        action_dtype=action_dtype,
        reward_shape=reward_shape,
        reward_dtype=reward_dtype)

    self.sum_tree = sum_tree.SumTree(replay_capacity)

  def get_add_args_signature(self):
    """The signature of the add function.

    The signature is the same as the one for OutOfGraphReplayBuffer, with an
    added priority.

    Returns:
      list of ReplayElements defining the type of the argument signature needed
        by the add function.
    """
    parent_add_signature = super(SAILOutOfGraphPrioritizedReplayBuffer,
                                 self).get_add_args_signature()
    add_signature = parent_add_signature + [
        crb.ReplayElement('priority', (), np.float32)
    ]
    return add_signature

  def _add(self, *args):
    """Internal add method to add to the underlying memory arrays.

    The arguments need to match add_arg_signature.

    If priority is none, it is set to the maximum priority ever seen.

    Args:
      *args: All the elements in a transition.
    """
    self._check_args_length(*args)

    # Use Schaul et al.'s (2015) scheme of setting the priority of new elements
    # to the maximum priority so far.
    # Picks out 'priority' from arguments and adds it to the sum_tree.
    transition = {}
    for i, element in enumerate(self.get_add_args_signature()):
      if element.name == 'priority':
        priority = args[i]
      else:
        transition[element.name] = args[i]

    self.sum_tree.set(self.cursor(), priority)
    super(SAILOutOfGraphPrioritizedReplayBuffer, self)._add_transition(
        transition)

  def sample_index_batch(self, batch_size, sample_from_complete_episodes=False):
    """Returns a batch of valid indices sampled as in Schaul et al. (2015).

    Args:
      batch_size: int, number of indices returned.
      sample_from_complete_episodes: bool, whether to sample only transitions
        from completed episodes.

    Returns:
      list of ints, a batch of valid indices sampled uniformly.

    Raises:
      Exception: If the batch was not constructed after maximum number of tries.
    """
    # Sample stratified indices. Some of them might be invalid.
    indices = self.sum_tree.stratified_sample(batch_size)
    allowed_attempts = self._max_sample_attempts
    for i in range(len(indices)):
      if not self.is_valid_transition(
          indices[i], sample_from_complete_episodes):
        if allowed_attempts == 0:
          raise RuntimeError(
              'Max sample attempts: Tried {} times but only sampled {}'
              ' valid indices. Batch size is {}'.
              format(self._max_sample_attempts, i, batch_size))
        index = indices[i]
        while not self.is_valid_transition(
            index, sample_from_complete_episodes) and allowed_attempts > 0:
          # If index i is not valid keep sampling others. Note that this
          # is not stratified.
          index = self.sum_tree.sample()
          allowed_attempts -= 1
        indices[i] = index
    return indices

  def sample_transition_batch(self, batch_size=None, indices=None,
                              sample_from_complete_episodes=True):
    """Returns a batch of transitions with extra storage and the priorities.

    The extra storage are defined through the extra_storage_types constructor
    argument.

    When the transition is terminal next_state_batch has undefined contents.

    Args:
      batch_size: int, number of transitions returned. If None, the default
        batch_size will be used.
      indices: None or list of ints, the indices of every transition in the
        batch. If None, sample the indices uniformly.
      sample_from_complete_episodes: bool, whether to sample only transitions
        from completed episodes.

    Returns:
      transition_batch: tuple of np.arrays with the shape and type as in
        get_transition_elements().
    """
    transition = (super(SAILOutOfGraphPrioritizedReplayBuffer, self).
                  sample_transition_batch(batch_size, indices,
                                          sample_from_complete_episodes))
    transition_elements = self.get_transition_elements(batch_size)
    transition_names = [e.name for e in transition_elements]
    probabilities_index = transition_names.index('sampling_probabilities')
    indices_index = transition_names.index('indices')
    indices = transition[indices_index]
    # The parent returned an empty array for the probabilities. Fill it with the
    # contents of the sum tree.
    transition[probabilities_index][:] = self.get_priority(indices)
    return transition

  def set_priority(self, indices, priorities):
    """Sets the priority of the given elements according to Schaul et al.

    Args:
      indices: np.array with dtype int32, of indices in range
        [0, replay_capacity).
      priorities: float, the corresponding priorities.
    """
    assert indices.dtype == np.int32, ('Indices must be integers, '
                                       'given: {}'.format(indices.dtype))
    for index, priority in zip(indices, priorities):
      self.sum_tree.set(index, priority)

  def get_priority(self, indices):
    """Fetches the priorities correspond to a batch of memory indices.

    For any memory location not yet used, the corresponding priority is 0.

    Args:
      indices: np.array with dtype int32, of indices in range
        [0, replay_capacity).

    Returns:
      priorities: float, the corresponding priorities.
    """
    assert indices.shape, 'Indices must be an array.'
    assert indices.dtype == np.int32, ('Indices must be int32s, '
                                       'given: {}'.format(indices.dtype))
    batch_size = len(indices)
    priority_batch = np.empty((batch_size), dtype=np.float32)
    for i, memory_index in enumerate(indices):
      priority_batch[i] = self.sum_tree.get(memory_index)
    return priority_batch

  def get_transition_elements(self, batch_size=None):
    """Returns a 'type signature' for sample_transition_batch.

    Args:
      batch_size: int, number of transitions returned. If None, the default
        batch_size will be used.
    Returns:
      signature: A namedtuple describing the method's return type signature.
    """
    parent_transition_type = (
        super(SAILOutOfGraphPrioritizedReplayBuffer,
              self).get_transition_elements(batch_size))
    probablilities_type = [
        crb.ReplayElement('sampling_probabilities', (batch_size,), np.float32)
    ]
    return parent_transition_type + probablilities_type


@gin.configurable(denylist=['observation_shape', 'stack_size',
                            'update_horizon', 'gamma'])
class SAILWrappedPrioritizedReplayBuffer(
    SAILWrappedReplayBuffer):
  """Wrapper of SAILOutOfGraphPrioritizedReplayBuffer with in-graph sampling.

  Usage:

    * To add a transition:  Call the add function.

    * To sample a batch:  Query any of the tensors in the transition dictionary.
                          Every sess.run that requires any of these tensors will
                          sample a new transition.
  """

  def __init__(self,
               observation_shape,
               stack_size,
               use_staging=False,
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
               reward_dtype=np.float32):
    """Initializes SAILWrappedPrioritizedReplayBuffer.

    Args:
      observation_shape: tuple of ints.
      stack_size: int, number of frames to use in state stack.
      use_staging: bool, when True it would use a staging area to prefetch
        the next sampling batch.
      replay_capacity: int, number of transitions to keep in memory.
      batch_size: int.
      update_horizon: int, length of update ('n' in n-step update).
      gamma: int, the discount factor.
      wrapped_memory: The 'inner' memory data structure. If None, use the
        default prioritized replay.
      max_sample_attempts: int, the maximum number of attempts allowed to
        get a sample.
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

    Raises:
      ValueError: If update_horizon is not positive.
      ValueError: If discount factor is not in [0, 1].
    """
    if wrapped_memory is None:
      wrapped_memory = SAILOutOfGraphPrioritizedReplayBuffer(
          observation_shape, stack_size, replay_capacity, batch_size,
          update_horizon, gamma, max_sample_attempts,
          extra_storage_types=extra_storage_types,
          observation_dtype=observation_dtype)

    super(SAILWrappedPrioritizedReplayBuffer, self).__init__(
        observation_shape,
        stack_size,
        use_staging,
        replay_capacity,
        batch_size,
        update_horizon,
        gamma,
        wrapped_memory=wrapped_memory,
        extra_storage_types=extra_storage_types,
        observation_dtype=observation_dtype,
        terminal_dtype=terminal_dtype,
        action_shape=action_shape,
        action_dtype=action_dtype,
        reward_shape=reward_shape,
        reward_dtype=reward_dtype)

  def tf_set_priority(self, indices, priorities):
    """Sets the priorities for the given indices.

    Args:
      indices: tf.Tensor with dtype int32 and shape [n].
      priorities: tf.Tensor with dtype float and shape [n].

    Returns:
       A tf op setting the priorities for prioritized sampling.
    """
    return tf.numpy_function(
        self.memory.set_priority, [indices, priorities], [],
        name='prioritized_replay_set_priority_py_func')

  def tf_get_priority(self, indices):
    """Gets the priorities for the given indices.

    Args:
      indices: tf.Tensor with dtype int32 and shape [n].

    Returns:
      priorities: tf.Tensor with dtype float and shape [n], the priorities at
        the indices.
    """
    return tf.numpy_function(
        self.memory.get_priority, [indices],
        tf.float32,
        name='prioritized_replay_get_priority_py_func')
