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

"""An implementation of Prioritized Experience Replay (PER).

This implementation is based on the paper "Prioritized Experience Replay"
by Tom Schaul et al. (2015). Many thanks to Tom Schaul, John Quan, and Matteo
Hessel for providing useful pointers on the algorithm and its implementation.
"""

from dopamine.replay_memory import sum_tree
import gin
import numpy as np
import tensorflow.compat.v1 as tf

from experience_replay.replay_memory import circular_replay_buffer
from experience_replay.replay_memory.circular_replay_buffer import ReplayElement


class OutOfGraphPrioritizedReplayBuffer(
    circular_replay_buffer.OutOfGraphReplayBuffer):
  """An out-of-graph Replay Buffer for Prioritized Experience Replay.

  See circular_replay_buffer.py for details.
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
               replay_forgetting='default',
               sample_newest_immediately=False):
    """Initializes OutOfGraphPrioritizedReplayBuffer.

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
      replay_forgetting:  str, What strategy to employ for forgetting old
        trajectories.  One of ['default', 'elephant'].
      sample_newest_immediately: bool, when True, immediately trains on the
        newest transition instead of using the max_priority hack.
    """
    super(OutOfGraphPrioritizedReplayBuffer, self).__init__(
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
        reward_dtype=reward_dtype,
        replay_forgetting=replay_forgetting)

    tf.logging.info('\t replay_forgetting: %s', replay_forgetting)
    self.sum_tree = sum_tree.SumTree(replay_capacity)
    self._sample_newest_immediately = sample_newest_immediately

  def get_add_args_signature(self):
    """The signature of the add function.

    The signature is the same as the one for OutOfGraphReplayBuffer, with an
    added priority.

    Returns:
      list of ReplayElements defining the type of the argument signature needed
        by the add function.
    """
    parent_add_signature = super(OutOfGraphPrioritizedReplayBuffer,
                                 self).get_add_args_signature()
    add_signature = parent_add_signature + [
        ReplayElement('priority', (), np.float32)
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
        if (self._sample_newest_immediately and
            self.add_count > self._stack_size):
          # add_count needs to be above stack_size because otherwise the first
          # transition added to the buffer has priority 0, which means the
          # sum_tree has a total_priority of 0, and thus it cannot sample.
          priority = 0.0
        else:
          priority = args[i]

      else:
        transition[element.name] = args[i]

    self.sum_tree.set(self.cursor(), priority)
    super(OutOfGraphPrioritizedReplayBuffer, self)._add_transition(transition)

  def sample_index_batch(self, batch_size):
    """Returns a batch of valid indices sampled as in Schaul et al. (2015).

    Args:
      batch_size: int, number of indices returned.

    Returns:
      list of ints, a batch of valid indices sampled uniformly.

    Raises:
      Exception: If the batch was not constructed after maximum number of tries.
    """
    manually_sample_newest = False
    if self._sample_newest_immediately:
      # self.cursor() points to the next hole to fill, so need to back up to the
      # the transition that has now accumulated enough data in terms of n-step
      # horizon for the purpose of training.
      newest_transition_index = self.cursor() - (self._update_horizon + 1)
      # Cannot sample newest if either there are not enough transitions in the
      # buffer or these are empty frames used for context at the start of the
      # episode.
      if self.is_valid_transition(newest_transition_index):
        # Sample N - 1 indices and append the newest transition.
        manually_sample_newest = True
        batch_size -= 1

    # Sample stratified indices. Some of them might be invalid.
    indices = self.sum_tree.stratified_sample(batch_size)
    allowed_attempts = self._max_sample_attempts
    for i in range(len(indices)):
      if not self.is_valid_transition(indices[i]):
        if allowed_attempts == 0:
          raise RuntimeError(
              'Max sample attempts: Tried {} times but only sampled {}'
              ' valid indices. Batch size is {}'.
              format(self._max_sample_attempts, i, batch_size))
        index = indices[i]
        while not self.is_valid_transition(index) and allowed_attempts > 0:
          # If index i is not valid keep sampling others. Note that this
          # is not stratified.
          index = self.sum_tree.sample()
          allowed_attempts -= 1
        indices[i] = index

    if manually_sample_newest:
      indices.append(newest_transition_index)

    return indices

  def sample_transition_batch(self, batch_size=None, indices=None):
    """Returns a batch of transitions with extra storage and the priorities.

    The extra storage are defined through the extra_storage_types constructor
    argument.

    When the transition is terminal next_state_batch has undefined contents.

    Args:
      batch_size: int, number of transitions returned. If None, the default
        batch_size will be used.
      indices: None or list of ints, the indices of every transition in the
        batch. If None, sample the indices uniformly.

    Returns:
      transition_batch: tuple of np.arrays with the shape and type as in
        get_transition_elements().
    """
    transition = (super(OutOfGraphPrioritizedReplayBuffer, self).
                  sample_transition_batch(batch_size, indices))
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
        super(OutOfGraphPrioritizedReplayBuffer,
              self).get_transition_elements(batch_size))
    probablilities_type = [
        ReplayElement('sampling_probabilities', (batch_size,), np.float32)
    ]
    return parent_transition_type + probablilities_type


@gin.configurable(
    denylist=['observation_shape', 'stack_size', 'update_horizon', 'gamma'])
class WrappedPrioritizedReplayBuffer(
    circular_replay_buffer.WrappedReplayBuffer):
  """Wrapper of OutOfGraphPrioritizedReplayBuffer with in-graph sampling.

  Usage:

    * To add a transition:  Call the add function.

    * To sample a batch:  Query any of the tensors in the transition dictionary.
                          Every sess.run that requires any of these tensors will
                          sample a new transition.
  """

  def __init__(self,
               observation_shape,
               stack_size,
               use_staging=True,
               replay_capacity=1000000,
               batch_size=32,
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
               replay_forgetting='default',
               sample_newest_immediately=False):
    """Initializes WrappedPrioritizedReplayBuffer.

    Args:
      observation_shape: tuple of ints.
      stack_size: int, number of frames to use in state stack.
      use_staging: bool, when True it would use a staging area to prefetch
        the next sampling batch.
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
      replay_forgetting:  str, What strategy to employ for forgetting old
        trajectories.  One of ['default', 'elephant'].
      sample_newest_immediately: bool, whether to sample a new transition
        immediately for training.

    Raises:
      ValueError: If update_horizon is not positive.
      ValueError: If discount factor is not in [0, 1].
    """
    memory = OutOfGraphPrioritizedReplayBuffer(
        observation_shape, stack_size, replay_capacity, batch_size,
        update_horizon, gamma, max_sample_attempts,
        extra_storage_types=extra_storage_types,
        observation_dtype=observation_dtype,
        replay_forgetting=replay_forgetting,
        sample_newest_immediately=sample_newest_immediately)
    super(WrappedPrioritizedReplayBuffer, self).__init__(
        observation_shape,
        stack_size,
        use_staging,
        replay_capacity,
        batch_size,
        update_horizon,
        gamma,
        wrapped_memory=memory,
        extra_storage_types=extra_storage_types,
        observation_dtype=observation_dtype,
        terminal_dtype=terminal_dtype,
        action_shape=action_shape,
        action_dtype=action_dtype,
        reward_shape=reward_shape,
        reward_dtype=reward_dtype,
        replay_forgetting=replay_forgetting)
    tf.logging.info('\t replay_forgetting: %s', replay_forgetting)

  def tf_set_priority(self, indices, priorities):
    """Sets the priorities for the given indices.

    Args:
      indices: tf.Tensor with dtype int32 and shape [n].
      priorities: tf.Tensor with dtype float and shape [n].

    Returns:
       A tf op setting the priorities for prioritized sampling.
    """
    return tf.py_func(
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
    return tf.py_func(
        self.memory.get_priority, [indices],
        tf.float32,
        name='prioritized_replay_get_priority_py_func')
