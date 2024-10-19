# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Utilities for adroit environments."""

import enum


class VisibleState(enum.Enum):
  IMAGE = 1
  ROBOT = 2
  FULL = 3


def get_visible_features_for_task(task, visible_state):
  """Defines names of robot state or full state features for task."""
  features = []
  if isinstance(visible_state, str):
    try:
      visible_state = VisibleState[visible_state.upper()]
    except KeyError:
      # Custom subset of state features.
      features = visible_state.split(',')
  if visible_state == VisibleState.ROBOT:
    features = ['qpos', 'palm_pos', 'tactile']
    if task != 'relocate':
      features = ['qpos', 'qvel', 'palm_pos', 'tactile']
  elif visible_state == VisibleState.FULL:
    features = ['original_obs']
  return features
