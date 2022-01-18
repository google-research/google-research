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
"""Main SLaQ interface for approximating graph descritptors NetLSD and VNGE."""
import numpy as np
from scipy.sparse.base import spmatrix
from graph_embedding.slaq.slq import slq
from graph_embedding.slaq.util import laplacian


def _slq_red_var_netlsd(matrix, lanczos_steps, nvectors,
                        timescales):
  """Computes unnormalized NetLSD signatures of a given matrix.

  Uses the control variates method to reduce the variance of NetLSD estimation.

  Args:
    matrix (sparse matrix): Input adjacency matrix of a graph.
    lanczos_steps (int): Number of Lanczos steps.
    nvectors (int): Number of random vectors for stochastic estimation.
    timescales (np.ndarray): Timescale parameter for NetLSD computation. Default
      value is the one used in both NetLSD and SLaQ papers.

  Returns:
    np.ndarray: Approximated NetLSD descriptors.
  """
  functions = [np.exp, lambda x: x]
  traces = slq(matrix, lanczos_steps, nvectors, functions, -timescales)
  subee = traces[0, :] - traces[1, :] / np.exp(timescales)
  sub = -timescales * matrix.shape[0] / np.exp(timescales)
  return np.array(subee + sub)


def _slq_red_var_vnge(matrix, lanczos_steps,
                      nvectors):
  """Approximates Von Neumann Graph Entropy (VNGE) of a given matrix.

  Uses the control variates method to reduce the variance of VNGE estimation.

  Args:
    matrix (sparse matrix): Input adjacency matrix of a graph.
    lanczos_steps (int): Number of Lanczos steps.
    nvectors (int): Number of random vectors for stochastic estimation.

  Returns:
    float: Approximated von Neumann graph entropy.
  """
  functions = [lambda x: -np.where(x > 0, x * np.log(x), 0), lambda x: x]
  traces = slq(matrix, lanczos_steps, nvectors, functions).ravel()
  return traces[0] - traces[1] + 1


def vnge(adjacency,
         lanczos_steps = 10,
         nvectors = 100):
  """Computes Von Neumann Graph Entropy (VNGE) using SLaQ.

  Args:
    adjacency (scipy.sparse.base.spmatrix): Input adjacency matrix of a graph.
    lanczos_steps (int): Number of Lanczos steps. Setting lanczos_steps=10 is
      the default from SLaQ.
    nvectors (int): Number of random vectors for stochastic estimation. Setting
      nvectors=10 is the default values from the SLaQ paper.

  Returns:
    float: Approximated VNGE.
  """
  if adjacency.nnz == 0:  # By convention, if x=0, x*log(x)=0.
    return 0
  density = laplacian(adjacency, False)
  density.data /= np.sum(density.diagonal()).astype(np.float32)
  return _slq_red_var_vnge(density, lanczos_steps, nvectors)


def netlsd(adjacency,
           timescales = np.logspace(-2, 2, 256),
           lanczos_steps = 10,
           nvectors = 100,
           normalization = None):
  """Computes NetLSD descriptors using SLaQ.

  Args:
    adjacency (sparse matrix): Input adjacency matrix of a graph.
    timescales (np.ndarray): Timescale parameter for NetLSD computation. Default
      value is the one used in both NetLSD and SLaQ papers.
    lanczos_steps (int): Number of Lanczos steps. Setting lanczos_steps=10 is
      the default from SLaQ.
    nvectors (int): Number of random vectors for stochastic estimation. Setting
      nvectors=10 is the default values from the SLaQ paper.
    normalization (str): Normalization type for NetLSD.

  Returns:
    np.ndarray: Approximated NetLSD descriptors.
  """
  lap = laplacian(adjacency, True)
  hkt = _slq_red_var_netlsd(lap, lanczos_steps, nvectors,
                            timescales)  # Approximated Heat Kernel Trace (hkt).
  if normalization is None:
    return hkt
  n = lap.shape[0]
  if normalization == 'empty':
    return hkt / n
  elif normalization == 'complete':
    return hkt / (1 + (n - 1) * np.exp(-timescales))
  elif normalization is None:
    return hkt
  else:
    raise ValueError(
        "Unknown normalization type: expected one of [None, 'empty', 'complete'], got",
        normalization)
