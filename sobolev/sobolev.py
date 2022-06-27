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

# pylint: disable=invalid-name
"""Module to evaluate Chebyshev polynomials."""
from jax import numpy as jnp
from sobolev import chebyshev


def eval_sobolev_chebyt(n, x, low,
                        high):
  """Evaluate the Sobolev-Chebyshev polynomial of the first kind."""
  if not isinstance(n, int):
    raise NotImplementedError(
        "This function is currently implemented only for " +
        "integer values of n")
  if n == 0:
    return jnp.ones_like(x)
  elif n == 1:
    return jnp.ones_like(x) - 2 * x / (high + low)
  s_0 = - (high + low) / (high - low)
  if n == 2:
    alpha = 1.
    xi = 0.
  else:
    xi = chebyshev.eval_chebyt(n - 2, s_0) / chebyshev.eval_chebyt(n, s_0)
    xi *= n / (n - 2)
    gamma_n = -xi / (1 + 2 * (n - 2) * (n - 2) * n)
    alpha = 1 / (1 - xi - gamma_n)
  s_x = 2 * x / (high - low) + s_0
  P_n = chebyshev.eval_chebyt(n, s_x) / chebyshev.eval_chebyt(n, s_0)
  P_n2 = chebyshev.eval_chebyt(n - 2, s_x) / chebyshev.eval_chebyt(n - 2, s_0)
  beta = (1 - alpha + xi * alpha)
  S_n2 = eval_sobolev_chebyt(n - 2, x, low, high)
  return beta * S_n2 + alpha * P_n - xi * alpha * P_n2
