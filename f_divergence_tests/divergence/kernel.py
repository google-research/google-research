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

"""Utils to compute kernel distances."""

import abc
import enum

import jax
import jax.numpy as jnp

from f_divergence_tests import testing_typing


class Norm(enum.Enum):
  """Supported norms for the kernel distances."""

  L1 = "l1"
  L2 = "l2"
  NONE = "none"


class Kernel(enum.Enum):
  """Supported kernels for the fuse method and their corresponding norm."""

  GAUSSIAN = ("gaussian", Norm.L2)
  LAPLACE = ("laplace", Norm.L1)
  EMPTY = ("empty", Norm.NONE)

  @property
  def kernel_type(self):
    return self.value[0]

  @property
  def norm(self):
    return self.value[1]


def get_distances(
    x,
    y,
    norm,
    min_memory,
):
  """Returns a matrix of pairwise distances using the specified norm.

  Code adapted from
  https://github.com/antoninschrab/dpkernel-paper/blob/master/kernel.py

  Args:
    x: array where each row is a sample.
    y: array where each row is a sample. Must have the same number of columns
      (dimension of samples) as x.
    norm: "l1" or "l2".
    min_memory: Whether to minimize memory usage by computing kernel values
      sequentially or in parallel.

  Returns:
    A matrix `all_distances`of shape (m, n) or (m + n, ) containing the pairwise
    distances
    between the samples in x and y. When return_matrix is True,
    all_distances[i,j] corresponds to the distance between x[i] and y[j]. When
    return_matrix is False, the upper triangular part of the matrix is returned
    as a vector.
  """
  if min_memory:
    if norm == Norm.L1.value:
      distance_vector = lambda x_i: jnp.sum(jnp.abs(y - x_i), 1)
    elif norm == Norm.L2.value:
      distance_vector = lambda x_i: jnp.sqrt(jnp.sum(jnp.square(y - x_i), 1))
    else:
      raise ValueError("`norm` must be either 'l1' or 'l2'.")
    all_distances = jax.lax.map(distance_vector, x)
  else:
    if norm == Norm.L1.value:
      distance_pointwise = lambda x, y: jnp.sum(jnp.abs(x - y))
    elif norm == Norm.L2.value:
      distance_pointwise = lambda x, y: jnp.sqrt(jnp.sum(jnp.square(x - y)))
    else:
      raise ValueError("Value of 'l' must be either 'l1' or 'l2'.")
    vmap_distance_y = jax.vmap(distance_pointwise, in_axes=(None, 0))
    vmap_distance_xy = jax.vmap(vmap_distance_y, in_axes=(0, None))
    all_distances = jnp.squeeze(vmap_distance_xy(x, y))
  return all_distances


def compute_bandwidths(distances, num_bandwidths):
  """Returns a sequence of bandwidths for adaptive kernel tests.

  Given an (m,n) array of pairwise distances, compute a sequence of bandwidths
  that covers distances in [q_min/2, q_max * 2], where q_min is the 0.05
  quantile of the distances, and q_max is the 0.95 quantile.

  Args:
    distances: (m+n, m+n) matrix of pairwise distances.
    num_bandwidths: number of bandwidths to compute.
  """
  median = jnp.median(distances)
  distances = distances + (distances == 0) * median
  sorted_distances = jnp.sort(distances)
  min_bw = (
      sorted_distances[(jnp.floor(len(sorted_distances) * 0.05).astype(int))]
      / 2
  )
  max_bw = (
      sorted_distances[(jnp.floor(len(sorted_distances) * 0.95).astype(int))]
      * 2
  )
  return jnp.linspace(min_bw, max_bw, num_bandwidths)


def get_kernel_matrix(
    pairwise_distance_matrix, kernel, bandwidth
):
  """Returns kernel evaluated on a pairwise distance matrix for a given kernel.

  Args:
    pairwise_distance_matrix: matrix of pairwise distances between samples of
      two distributions for a kernel test.
    kernel: kernel to use for the kernel test.
    bandwidth: bandwidth for the kernel.
  """
  bandwidth = jnp.array(bandwidth, dtype=jnp.float32)

  d = pairwise_distance_matrix / bandwidth
  if kernel.kernel_type == "gaussian":
    return jnp.exp(-(d**2) / 2)
  elif kernel.kernel_type == "laplace":
    return jnp.exp(-d * jnp.sqrt(2))
  else:
    raise ValueError(f"kernel_type={kernel.kernel_type} is not supported.")


class KernelDivergence(metaclass=abc.ABCMeta):
  """Base class for kernel divergence estimators.

  Attributes:
    has_vectorized_kernel: Whether the divergence estimator has a vectorized
      kernel implementation. This is leveraged to optimize the computation of
      fuse adaptive tests.
  """

  def __init__(self, has_vectorized_kernel):
    self.has_vectorized_kernel = has_vectorized_kernel

  @abc.abstractmethod
  def __call__(
      self, k_xx, k_yy, k_xy
  ):
    """Returns the divergence between two distributions.

    Args:
      k_xx: kernel matrix of samples from the first distribution.
      k_yy: kernel matrix of samples from the second distribution.
      k_xy: kernel matrix of samples from the pair of distributions.

    Returns:
      divergence: divergence between the two distributions.
      is_finite: whether the divergence is finite. Some divergences may have
        unstable implementations.
    """

    raise NotImplementedError
