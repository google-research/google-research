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
"""Compute MC returns."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import itertools as it
import numpy as np


def max_q_iteration(memory, gamma):
  q_values = copy.deepcopy(memory.rewards)
  for i in range(len(q_values)):
    for j in range(len(q_values[i])-2, -1, -1):
      q_values[i][j] += gamma*q_values[i][j+1]
  return np.array(list(it.chain.from_iterable(q_values)))
