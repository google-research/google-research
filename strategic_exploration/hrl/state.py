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

import numpy as np
import torch
from gtd.ml.torch.utils import GPUVariable


class State(object):
  """Wraps Atari RAM and pixel states.

  All the getters return *copies* of the
    underlying numpy arrays, so shallow copies are in general safe.
    """

  def __init__(self, ram_state=None, pixel_state=None, step_num=None):
    """Constructs around at least one of RAM state or pixel state.

        Args:
            ram_state (np.array | None): the RAM state
            pixel_state (np.array | None): the pixel state
            step_num (int | None): the number of actions taken previously in
              this episode
    """
    assert ram_state is not None or pixel_state is not None, \
            "RAM and pixel state are both None"

    self._ram_state = ram_state
    self._pixel_state = pixel_state
    self._step_num = step_num
    self._teleport = None
    self._goal = None
    self._object_changes = 0

  @property
  def ram_state(self):
    if self._ram_state is not None:
      return np.array(self._ram_state)
    raise ValueError("RAM state is not set")

  @property
  def pixel_state(self):
    if self._pixel_state is not None:
      return np.array(self._pixel_state)
    raise ValueError("Pixel state is not set")

  @property
  def step_num(self):
    return self._step_num

  @property
  def teleport(self):
    if self._teleport is not None:
      return self._teleport
    raise ValueError("No teleport set: {}".format(self))

  def set_teleport(self, teleport):
    if self._teleport is not None:
      raise ValueError("Teleport already set to: {}".format(self._teleport))
    self._teleport = teleport

  def drop_teleport(self):
    self._teleport = None

  @property
  def goal(self):
    # NOT a copy!
    if self._goal is not None:
      return self._goal
    raise ValueError("No goal set")

  def set_goal(self, goal):
    if self._goal is not None:
      raise ValueError("Goal is already set to: {}".format(self._goal))
    self._goal = goal

  @property
  def unmodified_pixels(self):
    if self._unmodified_pixels is not None:
      return np.array(self._unmodified_pixels)
    raise ValueError("No unmodified pixels set")

  def set_unmodified_pixels(self, unmodified_pixels):
    self._unmodified_pixels = unmodified_pixels

  def drop_unmodified_pixels(self):
    """Drops the reference to unmodified pixels for memory

        optimization.

        Returns:
            np.array: the unmodified pixels whose reference is being dropped
        """
    unmodified_pixels = self._unmodified_pixels
    self._unmodified_pixels = None
    return unmodified_pixels

  def set_object_changes(self, changes):
    self._object_changes = changes

  @property
  def object_changes(self):
    return self._object_changes

  def __hash__(self):
    raise NotImplementedError()

  def __eq__(self, other):
    raise NotImplementedError()

  def __str__(self):
    s = "State("
    if self._ram_state is not None:
      s += "RAM: [{}], ".format(np.sum(self._ram_state))
    if self._pixel_state is not None:
      s += "Pixels: [{}], ".format(np.sum(self._pixel_state))
    s = s[:-2]  # cut trailing comma
    s += ")"
    return s

  __repr__ = __str__
