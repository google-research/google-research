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

"""Utilities for data loading and preprocessing."""

from typing import Any
from typing import Callable
import numpy as np
import scipy.io
import scipy.sparse as sps


def preprocess_adjacency(
    adjacency,
    *,
    convert_to_csr = True,
    convert_to_unweighted = True,
    remove_self_loops = True,
    remove_isolated_nodes = True,
):
  """Pre-processes input adjacency matrix.

  Args:
    adjacency: Input adjacency matrix.
    convert_to_csr: Whether to convert the input matrix to the CSR format with
      fast matrix multiplications.
    convert_to_unweighted: Whether to discard input weights.
    remove_self_loops: Whether to remove self-loops from the graph.
    remove_isolated_nodes: Whether to remove isolated nodes from the graph.

  Returns:
    Clean adjacency matrix.
  """
  if adjacency.ndim != 2:
    raise ValueError(
        f'Adjacency matrix should be a 2D tensor, got {adjacency.ndim}'
    )
  if adjacency.shape[0] != adjacency.shape[1]:
    raise ValueError(
        f'Adjacency matrix should be square, got {adjacency.shape}'
    )
  if convert_to_csr:
    adjacency = adjacency.tocsr()
  if convert_to_unweighted:
    adjacency.data = np.ones_like(adjacency.data)
  if remove_self_loops:
    adjacency = adjacency - sps.diags(adjacency.diagonal())
  if remove_isolated_nodes:
    nonzero_rows = (adjacency.sum(0) != 0).A1
    adjacency = adjacency[nonzero_rows, :][:, nonzero_rows]
  return adjacency


def load_matfile(
    filepath,
    matfile_variable_name = 'network',
    convert_to_unweighted = True,
    open_fn = open,
):
  with open_fn(filepath, 'rb') as inf:
    data = scipy.io.loadmat(inf)
    adjacency = data[matfile_variable_name].tocsr()
  if convert_to_unweighted:
    adjacency.data = np.ones_like(adjacency.data)
  return adjacency
