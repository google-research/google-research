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

"""Helper function for randomly shuffling."""


import numpy as np


def random_shuffle(matrix, vector):
  """Returns random shuffle of matrix and vector (using the same shuffle).

  Args:
    matrix: Matrix to shuffle row-wise.
    vector: Vector to shuffle.
  """
  (n, _) = matrix.shape
  order = np.random.permutation(n)
  return matrix[order], vector[order]
