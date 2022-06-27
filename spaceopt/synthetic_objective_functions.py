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

"""Synthetic objective functions."""

import jax.numpy as jnp
import numpy as np


def branin(x):
  """Return the output of Branin function at input x.

  https://www.sfu.ca/~ssurjano/branin.html

  Args:
    x: (n, 2) shaped array of n x values in 2d.
  Returns:
    (n, 1) shaped array of Branin values at x
  """
  pi = jnp.pi
  a = 1
  b = 5.1/(4*(pi**2))
  c = 5/pi
  r = 6
  s = 10
  t = 1/(8*pi)

  y = a * (x[:, 1] - b * (x[:, 0]**2) + c * x[:, 0] -
           r)**2 + s * (1 - t) * jnp.cos(x[:, 0]) + s
  additional_info = {}
  return y[:, None], additional_info


def hartman6d(x):
  """Return the output of Hartman function at input x.

  https://www.sfu.ca/~ssurjano/hart6.html

  Args:
    x: (n, 6) shaped array of n x values in 6d.
  Returns:
    (n, 1) shaped array of Hartman values at x.
  """
  assert x.shape[1] == 6
  n = x.shape[0]
  y = np.zeros(n)
  for i in range(n):
    alpha = jnp.array([1.0, 1.2, 3.0, 3.2])
    a = jnp.array([[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14],
                   [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]])
    p = 1e-4 * jnp.array([[1312, 1696, 5569, 124, 8283, 5886],
                          [2329, 4135, 8307, 3736, 1004, 9991],
                          [2348, 1451, 3522, 2883, 3047, 6650],
                          [4047, 8828, 8732, 5743, 1091, 381]])

    outer = 0
    for ii in range(4):
      inner = 0
      for jj in range(6):
        xj = x[i, jj]
        aij = a[ii, jj]
        pij = p[ii, jj]
        inner = inner + aij*(xj-pij)**2

      new = alpha[ii] * jnp.exp(-inner)
      outer = outer + new

    y[i] = -outer
  additional_info = {}
  return y.reshape((n, 1)), additional_info
