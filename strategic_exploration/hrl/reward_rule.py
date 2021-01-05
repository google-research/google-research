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

import abc

# Register all classes to this mapping
RULES = {}


def get_reward_rules(config):
  """Returns list[RewardRule] corresponding to the config."""
  reward_rules = []
  for rule_name, rule_constructor in RULES.items():
    reward = config.get(rule_name, None)
    if reward is not None:
      reward_rules.append(rule_constructor(reward, 0.))
  return reward_rules


def get_done_rules(config):
  """Returns list [DoneRule] correspoding to the config."""
  done_rules = []
  for rule_name, rule_constructor in RULES.items():
    if config.get(rule_name, None) is not None:
      done_rules.append(rule_constructor(True, False))
  return done_rules


class Rule(object):
  """Defines custom value for transition (abstract_state,

    next_abstract_state).
    """
  __metaclass__ = abc.ABCMeta

  def __init__(self, match_value, default_value=0.):
    self._match_value = match_value
    self._default_value = default_value

  @abc.abstractmethod
  def __call__(self, abstract_state, next_abstract_state):
    """Returns float reward for this transition."""
    raise NotImplementedError()


class RoomObjectRule(Rule):

  def __call__(self, abstract_state, next_abstract_state):
    rooms_match = abstract_state.room_number == self.room_num and \
            next_abstract_state.room_number == self.room_num
    objects_match = \
            abstract_state.room_objects & self.objects_bit_mask and \
            not (next_abstract_state.room_objects & self.objects_bit_mask)
    if rooms_match and objects_match:
      return self._match_value

    return self._default_value

  @abc.abstractproperty
  def objects_bit_mask(self):
    raise NotImplementedError()

  @abc.abstractproperty
  def room_num(self):
    raise NotImplementedError()


class Room1LeftDoor(RoomObjectRule):

  @property
  def objects_bit_mask(self):
    return 8

  @property
  def room_num(self):
    return 1


RULES["room1leftdoor"] = Room1LeftDoor


class Room1RightDoor(RoomObjectRule):

  @property
  def objects_bit_mask(self):
    return 4

  @property
  def room_num(self):
    return 1


RULES["room1rightdoor"] = Room1RightDoor


class Room1Key(RoomObjectRule):

  @property
  def objects_bit_mask(self):
    return 1

  @property
  def room_num(self):
    return 1


RULES["room1key"] = Room1Key


class Room8(Rule):

  def __call__(self, abstract_state, next_abstract_state):
    if abstract_state.room_number == 9 and \
            next_abstract_state.room_number == 8:
      return self._match_value

    return self._default_value


RULES["room8"] = Room8


class Room13Monster(RoomObjectRule):

  @property
  def objects_bit_mask(self):
    return 1

  @property
  def room_num(self):
    return 13


RULES["room13monster"] = Room13Monster


class InventoryChange(Rule):

  def __call__(self, abstract_state, next_abstract_state):
    room_changed = \
            abstract_state.room_number != next_abstract_state.room_number
    inv_change = \
            abstract_state.inventory != next_abstract_state.inventory
    if not room_changed and inv_change:
      return self._match_value

    return self._default_value


RULES["invchange"] = InventoryChange


class Room14Key(RoomObjectRule):

  @property
  def objects_bit_mask(self):
    return 1

  @property
  def room_num(self):
    return 14


RULES["room14key"] = Room14Key
