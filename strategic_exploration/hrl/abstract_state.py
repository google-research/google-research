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

import abc
import numpy as np


def not_configured(_):
  raise ValueError("Call configure_abstract_state before calling AbstractState")


AbstractState = not_configured


def configure_abstract_state(domain):
  global AbstractState
  if "MontezumaRevenge" in domain:
    AbstractState = MontezumaAbstractState
  elif "Pitfall" in domain:
    AbstractState = PitfallAbstractState
  elif "PrivateEye" in domain:
    AbstractState = PrivateEyeAbstractState
  else:
    raise ValueError("{} not a supported domain")


class AbstractStateBase(object):
  __metaclass__ = abc.ABCMeta
  # Subclasses require the following class attributes:
  #  - MAX_PIXEL_X (maximum x-coordinate pixel value)
  #  - MAX_RAM_X (maximum x-coordinate RAM value)
  #  - MAX_PIXEL_Y (maximum y-coordinate pixel value)
  #  - MAX_RAM_Y (maximum y-coordinate RAM value)
  #  - MIN_PIXEL_X (minimum x-coordinate pixel value)
  #  - MIN_RAM_X (minimum x-coordinate RAM value)
  #  - MIN_PIXEL_Y (minimum y-coordinate pixel value)
  #  - MIN_RAM_Y (minimum y-coordinate RAM value)
  #  - X_BUCKET_SIZE (bucket size along x-coordinate in RAM coordinates)
  #  - Y_BUCKET_SIZE (bucket size along x-coordinate in RAM coordinates)

  @classmethod
  def ram_x_range(cls):
    return float(cls.MAX_RAM_X - cls.MIN_RAM_X + 1)

  @classmethod
  def ram_y_range(cls):
    return float(cls.MAX_RAM_Y - cls.MIN_RAM_Y + 1)

  @classmethod
  def ram_to_pixel_x(cls, ram_x):
    """Takes a RAM x coordinate and converts to pixel x coordinate.

        Args: ram_x (int)

        Returns:
            pixel_x (float)
        """
    pixel_x_range = cls.MAX_PIXEL_X - cls.MIN_PIXEL_X + 1
    ram_to_pixel_conversion = pixel_x_range / cls.ram_x_range()
    return ram_x * ram_to_pixel_conversion + cls.MIN_PIXEL_X

  @classmethod
  def ram_to_pixel_y(cls, ram_y):
    """Takes a RAM y coordinate and converts to pixel y coordinate.

        Args: ram_y (int)

        Returns:
            pixel_y (float)
        """
    pixel_y_range = cls.MAX_PIXEL_Y - cls.MIN_PIXEL_Y + 1
    ram_to_pixel_conversion = pixel_y_range / cls.ram_y_range()
    pixel_y = cls.MAX_PIXEL_Y - (ram_y * ram_to_pixel_conversion)
    return pixel_y

  @classmethod
  def bucket_lines(cls):
    """Returns a list of x coordinates and y coordinates which mark the end

        of the buckets. The grid defined by the lines:

            x = x_lines[i]
            y = y_lines[j]

        marks all the buckets.

        Returns:
            x_lines, y_lines (list[float], list[float])
        """
    x_lines = [
        cls.ram_to_pixel_x(x)
        for x in range(0, cls.MAX_RAM_X - cls.MIN_RAM_X + 1, cls.X_BUCKET_SIZE)
    ]
    y_lines = [
        cls.ram_to_pixel_y(y)
        for y in range(0, cls.MAX_RAM_Y - cls.MIN_RAM_Y + 1, cls.Y_BUCKET_SIZE)
    ]
    return x_lines, y_lines

  @abc.abstractproperty
  def numpy(self):
    """Returns center of bucket: np.array(np.float64)"""
    raise NotImplementedError()

  @abc.abstractmethod
  def size():
    """Dimensionality of numpy property."""
    raise NotImplementedError()

  @abc.abstractproperty
  def room_number(self):
    """Returns the room number (float)."""
    raise NotImplementedError()

  @abc.abstractproperty
  def ram_x(self):
    """Returns the RAM (bucketed) x-coordinate."""
    raise NotImplementedError()

  @abc.abstractproperty
  def ram_y(self):
    """Returns the RAM (bucketed) y-coordinate."""
    raise NotImplementedError()

  @abc.abstractproperty
  def match_attributes(self):
    """Returns np.array of attributes.

    If these attributes and the
        room_number match on another abstract state, they can be rendered in
        the same image.
        """
    raise NotImplementedError()

  @property
  def pixel_x(self):
    """Returns the x-coordinate of the center of the bucket that this state

        falls in, scaled to the Atari pixel coordinates.

        Returns:
            float
        """
    return self.ram_to_pixel_x(self.ram_x)

  @property
  def pixel_y(self):
    """Returns the y-coordinate of the center of the bucket that this state

        falls in, scaled to the Atari pixel coordinates.

        Returns:
            float
        """
    return self.ram_to_pixel_y(self.ram_y)

  def __hash__(self):
    return hash(tuple(self.numpy))

  def __eq__(self, other):
    if self.__class__ != other.__class__:
      return False
    else:
      return np.allclose(self.numpy, other.numpy)

  def __ne__(self, other):
    return not self.__eq__(other)


class MontezumaAbstractState(AbstractStateBase):
  """Contains the following information:

    (agent x coordinate, agent y coordinate, room number, inventory)
    extracted from the RAM state. Assumes Montezuma's Revenge.
    """
  X_BUCKET_SIZE = 20  # Size of x buckets in RAM coordinates
  Y_BUCKET_SIZE = 20  # Size of y buckets in RAM coordinates
  MAX_PIXEL_X = 160  # Max x pixel value for agent state
  MIN_PIXEL_X = 0  # Min x pixel value for agent state
  MAX_PIXEL_Y = 180  # Max y pixel value for agent state
  MIN_PIXEL_Y = 60  # Min y pixel value for agent state
  MIN_RAM_X = 1  # Inclusive range of RAM x coordinate
  MAX_RAM_X = 152
  MIN_RAM_Y = 134  # Inclusive range of RAM y coordinate
  MAX_RAM_Y = 254
  NUM_ROOMS = 24.

  def __init__(self, state):
    # When updating this, the size property must also be updated

    bucketed_x = \
        (state.ram_state[42] - self.MIN_RAM_X) / \
        self.X_BUCKET_SIZE * self.X_BUCKET_SIZE + self.X_BUCKET_SIZE / 2
    bucketed_y = \
        (state.ram_state[43] - self.MIN_RAM_Y) / self.Y_BUCKET_SIZE * \
        self.Y_BUCKET_SIZE + self.Y_BUCKET_SIZE / 2

    # Divide and multiply to do bucketing via int div
    self._abstract_state = np.array([
        bucketed_x / self.ram_x_range(),  # bucketed agent x-coordinate
        bucketed_y / self.ram_y_range(),  # bucketed agent y-coordinate
        state.ram_state[3] / self.NUM_ROOMS,  # room number
        state.ram_state[65] / 255.,  # normalized bit-masked inventory
        state.ram_state[66] / 32.,  # normalized bit-masked objects
        state.object_changes / 100.,
    ]).astype(np.float64)

    self._unbucketed = None
    self._x = state.ram_state[42]
    self._y = state.ram_state[43]

  @staticmethod
  def size():
    return 6

  @property
  def numpy(self):
    return self._abstract_state

  @property
  def unbucketed(self):
    # Lazily calculate as optimization
    if self._unbucketed is None:
      self._unbucketed = np.array([
          (self._x - self.MIN_RAM_X) / self.ram_x_range(),
          (self._y - self.MIN_RAM_Y) / self.ram_y_range(),
          self.numpy[2],
          self.numpy[3],
          self.numpy[4],
          self.numpy[5],
      ]).astype(np.float64)
    return self._unbucketed

  @property
  def ram_x(self):
    return int(self.numpy[0] * self.ram_x_range())

  @property
  def ram_y(self):
    return int(self.numpy[1] * self.ram_y_range())

  @property
  def room_number(self):
    return int(self.numpy[2] * self.NUM_ROOMS)

  @property
  def inventory(self):
    return int(self.numpy[3] * 255.)

  @property
  def room_objects(self):
    return int(self.numpy[4] * 32)

  @property
  def match_attributes(self):
    return self.numpy[[3, 4, 5]]

  def __str__(self):
    inv = self.inventory
    num_keys = 0
    if inv & 2:
      num_keys += 1
    if inv & 4:
      num_keys += 1
    if inv & 8:
      num_keys += 1

    human_interp_inv = []
    human_interp_inv.append("{} keys".format(num_keys))
    if inv & 1:
      human_interp_inv.append("cross")
    if inv & 32:
      human_interp_inv.append("sword")
    if inv & 128:
      human_interp_inv.append("torch")
    inv = str(tuple(human_interp_inv))

    return "MZ(({}, {}), room={}, inv={}, objs={}, changes={})".format(
        self.ram_x, self.ram_y, self.room_number, inv, self.room_objects,
        self.numpy[-1])

  __repr__ = __str__


class PitfallAbstractState(AbstractStateBase):
  """Contains the following information:

    (agent x coordinate, agent y coordinate, room number, inventory) extracted
    from the RAM state. Assumes Pitfall.
    """

  X_BUCKET_SIZE = 15  # Size of x buckets in RAM coordinates
  Y_BUCKET_SIZE = 15  # Size of y buckets in RAM coordinates
  MIN_PIXEL_X = 8  # Min x pixel value for agent state
  MAX_PIXEL_X = 148  # Max x pixel value for agent state
  MIN_PIXEL_Y = 81  # Min y pixel value for agent state
  MAX_PIXEL_Y = 171  # Max y pixel value for agent state
  MIN_RAM_X = 8  # Inclusive range of RAM x coordinate
  MAX_RAM_X = 148
  MIN_RAM_Y = 1
  MAX_RAM_Y = 91
  NUM_ROOMS = 256.

  def __init__(self, state):
    # When updating this, the size property must also be updated

    bucketed_x = \
        (state.ram_state[97] - self.MIN_RAM_X) / self.X_BUCKET_SIZE * \
        self.X_BUCKET_SIZE + self.X_BUCKET_SIZE / 2
    bucketed_y = \
        (state.ram_state[105] - self.MIN_RAM_Y) / self.Y_BUCKET_SIZE * \
        self.Y_BUCKET_SIZE + self.Y_BUCKET_SIZE / 2

    self._abstract_state = np.array([
        bucketed_x / self.ram_x_range(),  # bucketed agent x-coordinate
        bucketed_y / self.ram_y_range(),  # bucketed agent y-coordinate
        state.ram_state[1] / self.NUM_ROOMS,  # room identifier
        state.ram_state[113] / 32.,  # remaining treasures bitmask
    ]).astype(np.float64)

    self._unbucketed = None
    self._x = state.ram_state[97]
    self._y = state.ram_state[105]

  @staticmethod
  def size():
    return 4

  @classmethod
  def ram_to_pixel_y(cls, ram_y):
    """Takes a RAM y coordinate and converts to pixel y coordinate.

        Args: ram_y (int)

        Returns:
            pixel_y (float)
        """
    pixel_y = 80 + ram_y
    return pixel_y

  @property
  def numpy(self):
    return self._abstract_state

  @property
  def unbucketed(self):
    # Lazily calculate as optimization
    if self._unbucketed is None:
      self._unbucketed = np.array([
          (self._x - self.MIN_RAM_X) / self.ram_x_range(),
          (self._y - self.MIN_RAM_Y) / self.ram_y_range(),
          self.numpy[2],
          self.numpy[3],
      ]).astype(np.float64)
    return self._unbucketed

  @property
  def ram_x(self):
    return int(self.numpy[0] * self.ram_x_range())

  @property
  def ram_y(self):
    return int(self.numpy[1] * self.ram_y_range())

  @property
  def room_number(self):
    return int(self.numpy[2] * self.NUM_ROOMS)

  @property
  def treasure_count(self):
    return int(self.numpy[3])

  @property
  def match_attributes(self):
    return self.numpy[[3]]

  def __str__(self):
    return "PitfallAbstractState({})".format(self.numpy)

  __repr__ = __str__


class PrivateEyeAbstractState(AbstractStateBase):
  """Contains the following information:

    (agent x coordinate, agent y coordinate, room number, inventory)
    extracted from the RAM state.
    """
  X_BUCKET_SIZE = 40  # Size of x buckets in RAM coordinates
  Y_BUCKET_SIZE = 20  # Size of y buckets in RAM coordinates
  MAX_PIXEL_X = 160  # Max x pixel value for agent state
  MIN_PIXEL_X = 15  # Min x pixel value for agent state
  MAX_PIXEL_Y = 180  # Max y pixel value for agent state
  MIN_PIXEL_Y = 65  # Min y pixel value for agent state
  MIN_RAM_X = 18  # Inclusive range of RAM x coordinate
  MAX_RAM_X = 145
  MIN_RAM_Y = 17
  MAX_RAM_Y = 39
  NUM_ROOMS = 32.

  def __init__(self, state):
    # When updating this, the size property must also be updated

    bucketed_x = \
        (state.ram_state[63] - self.MIN_RAM_X) / self.X_BUCKET_SIZE * \
        self.X_BUCKET_SIZE + self.X_BUCKET_SIZE / 2
    bucketed_y = \
        (state.ram_state[86] - self.MIN_RAM_Y) / self.Y_BUCKET_SIZE * \
        self.Y_BUCKET_SIZE + self.Y_BUCKET_SIZE / 2

    self._abstract_state = np.array([
        bucketed_x / self.ram_x_range(),  # bucketed agent x-coordinate
        bucketed_y / self.ram_y_range(),  # bucketed agent y-coordinate
        state.ram_state[92] / self.NUM_ROOMS,  # room identifier
        state.ram_state[60] / 64.,  # inventory
        state.ram_state[72] / 32.,  # item history
        state.ram_state[93] / 5.,  # dropped off items
        # TODO: Robbers?
    ]).astype(np.float64)

    self._unbucketed = None
    self._x = state.ram_state[63]
    self._y = state.ram_state[86]

  @staticmethod
  def size():
    return 6

  @property
  def numpy(self):
    return self._abstract_state

  @property
  def unbucketed(self):
    # Lazily calculate as optimization
    if self._unbucketed is None:
      self._unbucketed = np.array([
          (self._x - self.MIN_RAM_X) / self.ram_x_range(),
          (self._y - self.MIN_RAM_Y) / self.ram_y_range(),
          self.numpy[2],
          self.numpy[3],
          self.numpy[4],
          self.numpy[5],
      ]).astype(np.float64)
    return self._unbucketed

  @property
  def ram_x(self):
    return int(self.numpy[0] * self.ram_x_range())

  @property
  def ram_y(self):
    return int(self.numpy[1] * self.ram_y_range())

  @property
  def room_number(self):
    return int(self.numpy[2] * self.NUM_ROOMS)

  @property
  def inventory(self):
    return self.numpy[3]

  @property
  def item_history(self):
    return self.numpy[4]

  @property
  def completed_missions(self):
    return self.numpy[5]

  @property
  def time(self):
    return self.numpy[6]

  @property
  def match_attributes(self):
    return self.numpy[[3, 4, 5]]

  def __str__(self):
    return "PrivateEyeAbstractState({})".format(self.numpy)

  __repr__ = __str__
