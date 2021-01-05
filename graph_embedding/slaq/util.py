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

# Lint as: python3
"""Utility functions for SLaQ."""
import numpy as np
import scipy.sparse
from scipy.sparse.base import spmatrix


def laplacian(adjacency, normalized = True):
  """Computes the sparse Laplacian matrix given sparse adjacency matrix as input.

  Args:
    adjacency (spmatrix): Input adjacency matrix of a graph.
    normalized (bool): If True, return the normalized version of the Laplacian.

  Returns:
    spmatrix: Sparse Laplacian matrix of the graph.
  """
  degree = np.squeeze(np.asarray(adjacency.sum(axis=1)))
  if not normalized:
    return scipy.sparse.diags(degree) - adjacency
  with np.errstate(divide='ignore'):  # Ignore the warning for divide by 0 case.
    degree = 1. / np.sqrt(degree)
  degree[degree == np.inf] = 0
  degree = scipy.sparse.diags(degree)
  return scipy.sparse.eye(
      adjacency.shape[0], dtype=np.float32) - degree @ adjacency @ degree
