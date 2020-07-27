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
"""A Policy is a sequence of group selectors, that can be static or not."""

import gin

from grouptesting.group_selectors import informative_dorfman
from grouptesting.group_selectors import mutual_information
from grouptesting.group_selectors import origami
from grouptesting.group_selectors import random
from grouptesting.group_selectors import split


@gin.configurable
class Policy(object):
  """A policy is defined as a sequence of Group Selectors."""

  def __init__(self, selectors=None):
    self.index = 0
    self.selectors = [] if selectors is None else selectors

  def reset(self):
    """Restarts index from 0 upon reset."""
    self.index = 0

  def get_selector(self, index=None):
    index = index if index is not None else self.index
    index = min(len(self.selectors) - 1, index)
    return self.selectors[index]

  @property
  def next_selector(self):
    return self.get_selector(self.index + 1)

  def act(self, rng, state):
    """Changes the state by applying the next selector."""
    selector = self.get_selector()
    self.index += 1
    return selector(rng, state)


@gin.configurable
class MimaxPolicy(Policy):
  """A policy that starts with Mimax."""

  def __init__(self, subsequent_selectors=None, **kwargs):
    selectors = [mutual_information.MaxMutualInformation(**kwargs)]
    if subsequent_selectors is not None:
      selectors.extend(
          subsequent_selectors if isinstance(subsequent_selectors, list
                                            ) else [subsequent_selectors])
    super().__init__(selectors=selectors)


@gin.configurable
class OrigamiPolicy(Policy):
  """A policy that starts with Origami."""

  def __init__(self, subsequent_selectors=None):
    selectors = [origami.Origami()]
    if subsequent_selectors is not None:
      selectors.extend(
          subsequent_selectors if isinstance(subsequent_selectors, list
                                            ) else [subsequent_selectors])
    super().__init__(selectors=selectors)


@gin.configurable
class MezardPolicy(Policy):
  """A policy that starts with Mezard (random) selector."""

  def __init__(self, subsequent_selectors=None, **kwargs):
    selectors = [random.Mezard(**kwargs)]
    if subsequent_selectors is not None:
      selectors.extend(
          subsequent_selectors if isinstance(subsequent_selectors, list
                                            ) else [subsequent_selectors])
    super().__init__(selectors=selectors)


@gin.configurable
class Dorfman(Policy):
  """A Dorfman policy that re-tests members in a positive group.

  Attributes:
    second_split : either None (all members of a positive groups are re-tested
      individually) or p-ary (e.g. binary if second_split=2).
  """

  def __init__(self, second_split=None, **kwargs):
    super().__init__(selectors=[
        split.SplitSelector(**kwargs),
        split.SplitPositive(split_factor=second_split)
    ])


@gin.configurable
class InformativeDorfmanPolicy(Policy):
  """Informative Dorfman policy."""

  def __init__(self, **kwargs):
    super().__init__(selectors=[
        informative_dorfman.InformativeDorfman(**kwargs)
    ])
