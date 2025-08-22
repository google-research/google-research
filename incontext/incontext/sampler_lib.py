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

# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sampler library for data generation."""
import functools
from typing import Callable, Optional, Tuple

import numpy as np
import numpy.matlib as npm


def str_to_distribution_fn(distribution_str):
  """Convert string representaiton to function."""
  mixtures = distribution_str.split(",")
  if len(mixtures) > 1:
    fns = [str_to_distribution_fn(mixture) for mixture in mixtures]

    def mixture_sample_fn(*args, **kwargs):
      samples = [fn(*args, **kwargs) for fn in fns]
      samples = np.stack(samples, axis=0)
      flat = samples.reshape(samples.shape[0], -1)
      indexer = np.random.randint(flat.shape[0], size=flat.shape[-1])
      flat = flat[indexer, np.arange(flat.shape[-1])]
      return flat.reshape(*samples.shape[1:])

    return mixture_sample_fn
  else:
    distribution_type, beta = distribution_str.split("+")
    distribution_type, alpha = distribution_type.split("*")
    alpha = float(alpha)
    beta = float(beta)
    if distribution_type == "uniform":
      distribution_fn = np.random.rand
    elif distribution_type == "normal":
      distribution_fn = np.random.randn
    else:
      raise ValueError("Unknown distribution type.")

    def distribution_fn_scaled(*args, **kwargs):
      return distribution_fn(*args, **kwargs) * alpha + beta

    return distribution_fn_scaled


class Sampler(object):
  """Samples linear regression data from specified distributions."""

  def __init__(
      self,
      length,
      dim,
      hidden_size,
      x_distribution_fn = np.random.randn,
      w_distribution_fn = np.random.randn,
      noise_std = 0.0,
  ):
    """Initializes the sampler.

    Args:
      length (int): Number of examplers to generate.
      dim (int): dimension of the x vectors.
      hidden_size (int): dimension of the generated vectors
      x_distribution_fn (Callable): random function to sample x vector units
      w_distribution_fn (Callable): random function to sample w vector units
      noise_std (float): adds gaussian noise if the value > 0.0. Default is 0.0
    """
    self.length = length
    self.dim = dim
    self.hidden_size = hidden_size
    self.x_distribution_fn = x_distribution_fn
    self.w_distribution_fn = w_distribution_fn
    self.noise_std = noise_std

  def sample_x(self, n = 1):
    """Generates a random x vector.

    Args:
      n (int, optional): number of samples. Defaults to 1.

    Returns:
      Tuple[np.array, np.array]: x vector, x vector with paddings
    """
    x = self.x_distribution_fn(n, self.dim)  # - 0.5
    x_vec = np.concatenate(
        (
            np.zeros((n, 1)),
            x,
            #            np.zeros((n, self.hidden_size - self.dim - 1)),
        ),
        axis=1,
    )
    return x, x_vec

  def calculate_y(self, x,
                  coefficients):
    """Calculates the y vector from the x vector and the coefficients.

    Args:
      x (np.array): x vector.
      coefficients (np.array): weights of the linear regressor

    Returns:
      np.array: y vector
    """
    y = np.einsum("bi,bi->b", coefficients, x)[:, None]
    if self.noise_std > 0:
      y += self.noise_std * np.random.randn(*y.shape)
    y_vec = np.concatenate((y, np.zeros((x.shape[0], self.dim))), axis=1)
    return y, y_vec

  def sample_coefficients(self, n = 1, alpha = 1.0):
    """Generates a random coefficients vector for the linear regressor.

    Args:
      n (int, optional): batch size. Defaults to 1. alpha(float, optional):
        additional sscale. Defaults to 1.0
      alpha (float, optional): scale distribution. Defaults to 1

    Returns:
      np.array: coefficients vector
    """
    return self.w_distribution_fn(n, self.dim) * alpha

  def get_delimiter_vector(self, n = 1):
    """Generates a constant delimiter vector."""
    return np.zeros((n, self.hidden_size))

  @functools.lru_cache(maxsize=10)
  def get_precision(self,):
    # x = self.x_distribution_fn(10000, self.dim)
    # return np.linalg.inv(x.T @ x) * x.shape[0]
    return np.eye(self.dim)

  def sample(
      self,
      n = 1,
      alpha = 1.0,
      coefficients = None,
  ):
    """Generates a random sequence of x and y vector comes from a linear regressor.

    Args:
      n (int, optional): batch size. Defaults to 1.
      alpha (float, optional): scale distribution. Defaults to 1
      coefficients (Optional[np.ndarray], optional): weights of the regressor.
        Defaults to None.

    Returns:
      Tuple[np.array, np.array]: x,y sequences, weights of the regressor
    """
    if coefficients is None:
      coefficients = self.sample_coefficients(n, alpha)
    else:
      coefficients = npm.repmat(coefficients, n, 1)
    out = []
    xs = []
    ys = []
    for _ in range(self.length):
      x, x_vec = self.sample_x(coefficients.shape[0])
      y, y_vec = self.calculate_y(x, coefficients)
      out.append(x_vec)
      out.append(y_vec)
      xs.append(x)
      ys.append(y)
      # out.append(self.get_delimiter_vector(coefficients.shape[0]))
    out = np.stack(out, axis=1)
    xs = np.stack(xs, axis=1)
    ys = np.stack(ys, axis=1)
    return out, coefficients, xs, ys
