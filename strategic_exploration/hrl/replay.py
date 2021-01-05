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

# Adapted from OpenAI Gym Baselines
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
import numpy as np
import random
from collections import defaultdict, deque
from strategic_exploration.hrl.schedule import LinearSchedule
from strategic_exploration.hrl.segment_tree import MinSegmentTree, SumSegmentTree


class ReplayBuffer(object):

  @classmethod
  def from_config(cls, config):
    if config.type == "vanilla":
      return cls(config.max_buffer_size)
    elif config.type == "prioritized":
      return PrioritizedReplayBuffer.from_config(config)
    else:
      raise ValueError("{} not a supported buffer type".format(config.type))

  def __init__(self, size):
    """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
    self._storage = []
    self._maxsize = size
    self._next_idx = 0

  def __len__(self):
    return len(self._storage)

  def add(self, experience):
    if self._next_idx >= len(self._storage):
      self._storage.append(experience)
    else:
      self._storage[self._next_idx] = experience
    self._next_idx = (self._next_idx + 1) % self._maxsize

  def sample(self, batch_size):
    """Sample a batch of experiences.

        Args:
            batch_size (int): How many transitions to sample.

        Returns:
            list[Experience]: sampled experiences, not necessarily unique
        """
    indices = np.random.randint(len(self._storage), size=batch_size)
    return [self._storage[i] for i in indices]


class GroupedReplayBuffer(ReplayBuffer):
  """ReplayBuffer that allocates a maximum amount of elements to

    client-defined groups.
  """

  def __init__(self, max_size):
    """
        Args:
            max_size (int): maximum total number of elements in the buffer
            max_group_size (int): maximum number of elements per group
    """
    self._storage = []
    self._next_index = 0
    self._max_size = max_size

    # Maps group key to list of indices in storage
    self._group_to_indices = defaultdict(lambda: deque())
    # Maps index in storage to group key
    self._index_to_group = {}

  def __len__(self):
    return len(self._storage)

  def add(self, obj):
    """
        Args:
            obj (tuple(Hashable, int, Object)): tuple of the group, max size for
              that group, and the element to store (to be sampled later) Between
              calls, the max size for each group should be consistent.  If
              adding this element exceeds the max group size, it'll evict
              another element from the same group.
    """
    group, max_group_size, elem = obj
    group_indices = self._group_to_indices[group]
    assert len(group_indices) <= max_group_size
    if len(group_indices) == max_group_size:
      # Randomly put somewhere new, otherwise the whole group may be
      # contiguous, resulting in it being easily overwritten
      insertion_index = np.random.randint(len(self._storage))
      swap_index = group_indices.popleft()
      group_indices.append(insertion_index)

      # Add the new element to insertion_index. Give up the slot at
      # swap_index to the group at insertion_inde
      self._storage[swap_index] = self._storage[insertion_index]
      swap_group = self._index_to_group[swap_index] = \
              self._index_to_group[insertion_index]
      self._index_to_group[insertion_index] = group

      # Update other group's group2indices
      self._group_to_indices[swap_group].remove(insertion_index)
      self._group_to_indices[swap_group].append(swap_index)
    else:
      insertion_index = self._next_index
      self._next_index = (self._next_index + 1) % self._max_size
      if insertion_index in self._index_to_group:
        old_group = self._index_to_group[insertion_index]
        self._group_to_indices[old_group].remove(insertion_index)
      self._index_to_group[insertion_index] = group
      self._group_to_indices[group].append(insertion_index)

    if insertion_index < len(self._storage):
      self._storage[insertion_index] = elem
    else:
      self._storage.append(elem)

  def sample(self, batch_size):
    indices = np.random.randint(len(self._storage), size=batch_size)
    return [self._storage[i] for i in indices]


class PrioritizedReplayBuffer(ReplayBuffer):

  @classmethod
  def from_config(cls, config):
    beta_schedule = LinearSchedule.from_config(config.beta_schedule)
    return cls(config.max_buffer_size, config.alpha, beta_schedule)

  def __init__(self, size, alpha, beta_schedule):
    """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        See Also
        --------
        ReplayBuffer.__init__
        """
    super(PrioritizedReplayBuffer, self).__init__(size)
    assert alpha >= 0
    self._alpha = alpha

    it_capacity = 1
    while it_capacity < size:
      it_capacity *= 2

    self._it_sum = SumSegmentTree(it_capacity)
    self._it_min = MinSegmentTree(it_capacity)
    self._max_priority = 1.0
    self._beta_schedule = beta_schedule

  def add(self, *args, **kwargs):
    """See ReplayBuffer.store_effect"""
    idx = self._next_idx
    super(PrioritizedReplayBuffer, self).add(*args, **kwargs)
    self._it_sum[idx] = self._max_priority**self._alpha
    self._it_min[idx] = self._max_priority**self._alpha

  def _sample_proportional(self, batch_size):
    res = []
    for _ in range(batch_size):
      # TODO(szymon): should we ensure no repeats?
      mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
      idx = self._it_sum.find_prefixsum_idx(mass)
      res.append(idx)
    return res

  def sample(self, batch_size):
    """Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        experiences: list[Experience]
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
    beta = self._beta_schedule.step()
    assert beta > 0

    idxes = self._sample_proportional(batch_size)

    weights = []
    p_min = self._it_min.min() / self._it_sum.sum()
    max_weight = (p_min * len(self._storage))**(-beta)

    experiences = []
    for idx in idxes:
      p_sample = self._it_sum[idx] / self._it_sum.sum()
      weight = (p_sample * len(self._storage))**(-beta)
      weights.append(weight / max_weight)
      experiences.append(self._storage[idx])
    weights = np.array(weights)
    return experiences, weights, idxes

  def update_priorities(self, idxes, priorities):
    """Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
    assert len(idxes) == len(priorities)
    for idx, priority in zip(idxes, priorities):
      assert priority > 0
      assert 0 <= idx < len(self._storage)
      self._it_sum[idx] = priority**self._alpha
      self._it_min[idx] = priority**self._alpha

      self._max_priority = max(self._max_priority, priority)
