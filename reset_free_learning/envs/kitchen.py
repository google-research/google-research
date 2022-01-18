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

# Lint as: python3
"""Kitchen environment."""

import numpy as np
from robel.franka.kitchen_multitask import KitchenTaskRelaxV1

ELEMENT_INDICES_LL = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8],  # Arm
    [9, 10],  # Burners
    [11, 12],  # Burners
    [13, 14],  # Burners
    [15, 16],  # Burners
    [17, 18],  # Lightswitch
    [19],  # Slide
    [20, 21],  # Hinge
    [22],  # Microwave
    [23, 24, 25, 26, 27, 28, 29]  # Kettle
]

initial_state = np.array([[
    -0.56617326,
    -1.6541005,
    1.4447045,
    -2.4378936,
    0.71086496,
    1.3657048,
    0.80830157,
    0.019943988,
    0.019964991,
    2.456005e-05,
    2.9547007e-07,
    2.4559975e-05,
    2.954692e-07,
    2.4559975e-05,
    2.954692e-07,
    2.4559975e-05,
    2.954692e-07,
    2.161876e-05,
    5.0806757e-06,
    0.0,
    0.0,
    0.0,
    0.0,
    -0.269,
    0.35,
    1.6192839,
    1.0,
    -8.145112e-19,
    -1.1252103e-05,
    -2.8055027e-19,
    -0.44,
    0.152,
    2.226,
    0.65359545,
    -0.65307516,
    -0.2703603,
    -0.27057564,
]])

# be careful about which initial state is being used when creating the goal
goal_list = {}
goal_list['open_microwave'] = initial_state[0].copy()
goal_list['open_microwave'][22] = -0.7


class Kitchen(KitchenTaskRelaxV1):

  def __init__(self, task='open_microwave'):
    super().__init__()
    self._task = task

  def get_next_goal(self):
    return goal_list[self._task]

  def reset_goal(self, goal=None):
    if goal is None:
      goal = self.get_next_goal()

    self.goal = goal

  def _reset(self):
    super()._reset()
    self.reset_goal()
