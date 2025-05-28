# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Common utilities."""

import dataclasses
import random
from typing import Optional, Sequence, Union
import frozendict
import numpy as np
from shapely import geometry

from aux_tasks.puddle_world import puddle_world

NUM_ACTIONS = 4


@dataclasses.dataclass
class DiscreteState:
  true_state: geometry.Point
  bin_idx: int
  row: int
  col: int


@dataclasses.dataclass
class DiscreteTransition:
  state: DiscreteState
  action: puddle_world.Action
  reward: float
  next_state: DiscreteState
  is_terminal: bool


class DiscretizedPuddleWorld:
  """A discretizing wrapper for PuddleWorld."""

  def __init__(
      self, pw, num_bins = 10):
    self._pw = pw
    self._num_bins = num_bins
    self.num_states = self._num_bins * self._num_bins

    # Since bins are square, size = width = height.
    self._bin_size = 1.0 / self._num_bins

    self.bin_idx_by_row_col = dict()
    self.row_col_by_bin_idx = dict()

    idx = 0
    for row in range(self._num_bins):
      for col in range(self._num_bins):
        self.bin_idx_by_row_col[(row, col)] = idx
        self.row_col_by_bin_idx[idx] = (row, col)
        idx += 1

    self.bin_idx_by_row_col = frozendict.frozendict(self.bin_idx_by_row_col)
    self.row_col_by_bin_idx = frozendict.frozendict(self.row_col_by_bin_idx)

  def get_bin_corners_by_bin_idx(
      self, bin_idx):
    """Gets the bottom left and top right points of a bin given its index."""
    row, col = self.row_col_by_bin_idx[bin_idx]

    left_x = col * self._bin_size
    right_x = left_x + self._bin_size
    bottom_y = row * self._bin_size
    top_y = bottom_y + self._bin_size

    return geometry.Point((left_x, bottom_y)), geometry.Point((right_x, top_y))

  def sample_state_in_bin(self, bin_idx):
    """Samples a state from a specific bin."""
    row, col = self.row_col_by_bin_idx[bin_idx]
    bottom_left, top_right = self.get_bin_corners_by_bin_idx(bin_idx)

    x = random.uniform(bottom_left.x, top_right.x)
    y = random.uniform(bottom_left.y, top_right.y)

    return DiscreteState(
        true_state=geometry.Point((x, y)),
        bin_idx=bin_idx,
        row=row,
        col=col)

  def transition(self,
                 state,
                 action):
    """Computes a transition with discrete observations."""
    t = self._pw.transition(state.true_state, action)

    # We generally consider a bin to contain [low, high), but this causes
    # an error when x or y are at the edge of the arena. In this case,
    # we just push them down to the previous bin.
    col = min(int(t.next_state.x * self._num_bins), self._num_bins - 1)
    row = min(int(t.next_state.y * self._num_bins), self._num_bins - 1)

    lb_col = col * self._bin_size
    ub_col = lb_col + self._bin_size
    lb_row = row * self._bin_size
    ub_row = lb_row + self._bin_size

    # We add a small delta to the bounds to account for rounding errors.
    assert lb_col - 1e-8 <= t.next_state.x <= ub_col + 1e-8
    assert lb_row - 1e-8 <= t.next_state.y <= ub_row + 1e-8

    return DiscreteTransition(
        state,
        action,
        t.reward,
        DiscreteState(
            t.next_state,
            self.bin_idx_by_row_col[(row, col)],
            row,
            col),
        t.is_terminal)


def generate_rollout(
    dpw,
    n,
    state = None):
  """Generates a random rollout starting from a random init."""
  if state is None:
    initial_state = random.randrange(dpw.num_states)
    state = dpw.sample_state_in_bin(initial_state)

  rollout = list()
  for _ in range(n):
    action = random.randrange(NUM_ACTIONS)
    transition = dpw.transition(state, action)

    rollout.append(transition)
    state = transition.next_state

  return rollout


def calculate_empricial_successor_representation(
    dpw,
    rollout,
    gamma):
  result = np.zeros(dpw.num_states, dtype=np.float32)
  current_gamma = 1.0
  for transition in rollout:
    result[transition.next_state.bin_idx] += current_gamma
    current_gamma *= gamma
  return result
