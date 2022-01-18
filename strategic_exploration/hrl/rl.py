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

"""Define all the core classes for an RL problem."""
import collections
import numpy as np


class Experience(
    collections.namedtuple('Experience',
                           ['state', 'action', 'reward', 'next_state', 'done'])
):
  pass


class Episode(collections.MutableSequence):
  """A list of Experiences."""
  __slots__ = ['_experiences']
  discount_negative_reward = False

  def __init__(self, iterable=None):
    self._experiences = []
    if iterable:
      for item in iterable:
        self.append(item)

  def __getitem__(self, item):
    return self._experiences[item]

  def __setitem__(self, key, value):
    assert isinstance(value, Experience)
    self._experiences[key] = value

  def __delitem__(self, key):
    del self._experiences[key]

  def __len__(self):
    return len(self._experiences)

  def insert(self, index, value):
    assert isinstance(value, Experience)
    return self._experiences.insert(index, value)

  def append(self, experience):
    assert isinstance(experience, Experience)
    self._experiences.append(experience)

  def discounted_return(self, t, gamma):
    """Returns G_t, the discounted return.

        Args:
            t (int): index of the episode (supports negative indexing from back)
            gamma (float): the discount factor

        Returns:
            float
        """

    def discounted_reward(undiscounted, index):
      return undiscounted * np.power(gamma, index)

    if t < -len(self._experiences) or t > len(self._experiences):
      raise ValueError('Index t = {} is out of bounds'.format(t))

    return sum(
        discounted_reward(experience.reward, i)
        for i, experience in enumerate(self._experiences[t:]))

  def __str__(self):
    experiences = '{}'.format(self._experiences)[:50]
    return 'Episode({}..., undiscounted return: {})'.format(
        experiences, self.discounted_return(0, 1.0))

  __repr__ = __str__
