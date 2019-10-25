# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""General utility functions for M-Theory investigations.
"""

import numpy


def get_symmetric_traceless_basis(n):
  """Computes a basis for symmetric-traceless matrices."""
  num_matrices = n * (n + 1) // 2 - 1
  # Basis for symmetric-traceless 5x5 matrices.
  b = numpy.zeros([num_matrices, n, n])
  # First (n-1) matrices are diag(1, -1, 0, ...), diag(0, 1, -1, 0, ...).
  # These are not orthogonal to one another.
  for k in range(n - 1):
    b[k, k, k] = 1
    b[k, k + 1, k + 1] = -1
  i = n - 1
  for j in range(n):
    for k in range(j + 1, n):
      b[i, j, k] = b[i, k, j] = 1
      i += 1
  return b
