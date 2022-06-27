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

"""Utility functions used throughout the library."""
from typing import Iterator, Tuple
from jax import numpy as jnp


def eval_three_term(degree, recurrence,
                    x):
  """Evaluate a polynomial from its three term recurrence.

  The three-term recurrence is given by

      P_{t+1}(x) = (a_t + b_t x) P_t(x) + c_t P_{t-1}(x)

  with initial conditions P_{-1}(x) = 0, P_0(x) = 1.

  Args:
    degree: int, maximum degree of the polynomial
    recurrence: generator yields coefficients of the polynomials three-term
      recurrence.
    x: array-like

  Returns:
    p_cur : float, value of the MP polynomial at x
  """

  p_cur = jnp.ones_like(x)
  p_prev = jnp.zeros_like(x)
  for _ in range(degree):
    a_t, b_t, c_t = next(recurrence)
    tmp = p_cur
    p_cur = (a_t + b_t * x) * p_cur + c_t * p_prev
    p_prev = tmp
  return p_cur


def eval_four_term(degree, recurrence,
                   x):
  """Evaluate a polynomial from a four term term recurrence.

  The three-term recurrence is given by

      P_{t+1}(x) = (a_t + b_t x + c_t x^2) P_t(x) + d_t P_{t-1}(x)

  with initial conditions P_{-1}(x) = 0, P_0(x) = 1.

  Args:
    degree: int, maximum degree of the polynomial
    recurrence: generator yields coefficients of the polynomials three-term
      recurrence.
    x: array-like

  Returns:
    p_cur : float, value of the MP polynomial at x
  """

  p_cur = jnp.ones_like(x)
  p_prev = jnp.zeros_like(x)
  for _ in range(degree):
    a_t, b_t, c_t, d_t = next(recurrence)
    tmp = p_cur
    p_cur = (a_t + b_t * x + c_t * x * x) * p_cur + d_t * p_prev
    p_prev = tmp
  return p_cur
