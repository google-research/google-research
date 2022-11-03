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

"""Example usage of the methods described in our paper.

Computes approximation error of NetLSD and VNGE for the Karate club graph.
"""

import numpy as np
import scipy.linalg
import scipy.sparse
from scipy.sparse.base import spmatrix
from graph_embedding.slaq.slaq import netlsd
from graph_embedding.slaq.slaq import vnge
from graph_embedding.slaq.util import laplacian


def read_graph(filename):
  """Reads the graph in the adjacency list format into a sparse symmetric matrix.

  Expects the graph in the following format:
    - First line is the number of nodes.
    - Other lines have two numbers separated by space or tabs; the numbers
    correspond to the node IDs of the source and target vertices. The IDs are
    assume to be from [0;n_nodes).
  Example file is available in 'data/karate.txt'.

  Args:
    filename (str): File path of the graph file. Can not be a glob.

  Returns:
    spmatrix: Sparse adjacency matrix of the graph in the CSR format.
  """
  with open(filename, 'r') as file:
    n_nodes = int(file.readline())
    adjacency = adjacency = scipy.sparse.dok_matrix((n_nodes, n_nodes),
                                                    dtype=np.float32)
    for line in file:
      split = line.strip().split()
      source, target = int(split[0]), int(split[1])
      adjacency[source, target] = 1
      adjacency[target, source] = 1
    # Convert to the compressed sparse row format for efficiency.
    adjacency = adjacency.tocsr()
    adjacency.data = np.ones(
        adjacency.data.shape,
        dtype=np.float32)  # Set all elements to one in case of duplicate rows.
    return adjacency


def netlsd_naive(
    adjacency, timescales = np.logspace(-2, 2, 256)
):
  """Computes NetLSD with full eigendecomposition, in a naïve way.

  Args:
    adjacency (spmatrix): Input sparse adjacency matrix of a graph.
    timescales (np.ndarray): A 1D array with the timescale parameter of NetLSD.
      Default value is the one used in both NetLSD and SLaQ papers.

  Returns:
    np.ndarray: NetLSD descriptors of the graph.
  """
  lap = laplacian(adjacency)
  lambdas, _ = scipy.linalg.eigh(lap.todense())
  return np.exp(-np.outer(timescales, lambdas)).sum(axis=-1)


def vnge_naive(adjacency):
  """Computes Von Neumann Graph Entropy (VNGE) with full eigendecomposition.

  Args:
    adjacency (spmatrix): Input sparse adjacency matrix of a graph.

  Returns:
    float: Von Neumann entropy of the graph.
  """
  density = laplacian(adjacency, normalized=False)
  density.data /= np.sum(density.diagonal())
  eigenvalues, _ = scipy.linalg.eigh(density.todense())
  return -np.where(eigenvalues > 0, eigenvalues * np.log(eigenvalues), 0).sum()


if __name__ == '__main__':
  fname = 'graph_embedding/slaq/data/karate.txt'
  graph = read_graph(fname)

  lsd_full = netlsd_naive(graph)
  lsd_slaq = netlsd(graph)

  vnge_full = vnge_naive(graph)
  vnge_slaq = vnge(graph)

  print('NetLSD approximation error:',
        np.linalg.norm(lsd_full - lsd_slaq) / np.linalg.norm(lsd_full))
  print('VNGE approximation error:',
        np.linalg.norm(vnge_full - vnge_slaq) / np.linalg.norm(vnge_full))
