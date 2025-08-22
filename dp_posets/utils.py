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

"""Utility functions for poset experiments."""

import numpy as np


def compute_average_squared_l2_norm(points):
  """Returns average squared l_{2} norm of given points.

  Args:
    points: list of vectors.
  """
  return np.mean(np.square(np.linalg.norm(points, axis=1)))


def compute_linf_average_squared_l2_norm(d):
  """Returns average squared l_{2} norm of d-dim l_inf unit ball.

  See Lemma 4.1 in the paper.

  Args:
    d: dimension of l_inf unit ball.
  """
  return d / 3
