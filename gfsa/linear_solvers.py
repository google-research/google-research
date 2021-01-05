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
"""Iterative linear solvers over pytrees."""

from typing import Callable, TypeVar

import jax

# Any nested structure of NDArrays, viewed as a vector space (with some extra
# structure).
VectorPytree = TypeVar("VectorPytree")


def richardson_solve(matvec,
                     b, iterations):
  """Solve a linear system using the Richardson iteration method.

  This function approximates the solution to Ax = b by iterating the equations

    x_0 = b
    x_{k+1} = x_k + b - A x_k = b + (I - A) x_k

  until convergence or a given maximum number of iterations. This method will
  converge if ||I - A|| < 1 (for any induced matrix norm).

  The Richardson method is especially appropriate for systems involving the
  transition matrix of an absorbing Markov chain. In particular, solving for
  the number of visits to each state involves computing (I - P)^{-1} x, where
  A = I - P. In this case, I - A = P, so each estimate x_k is exactly the
  expected number of visits to each state after k timesteps.

  If the provided system has ones along the diagonal, note that this method
  is equivalent to the Jacobi method.

  (Numerical stability note: If matvec is already of the form (I-P), then
  (I - A) is (I - (I - P)) which is equivalent to just P. This doesn't cause
  any significant numerical stability issues though, since the only time this
  would reduce output precision is when x is large and P(x) is small. In this
  case, (I-(I-P))(x) will be low precision after one step, but then the new
  estimate of x will be small so another iteration will restore the missing
  precision.)

  Args:
    matvec: Linear function from (pytrees of) input vectors to (pytrees of)
      output vectors of the same shape, representing the matrix to solve.
    b: Dependent variable for the solve, as a (pytree of) vector(s).
    iterations: Number of steps to iterate.

  Returns:
    Approximate solution to the linear system Ax = b, with gradients defined
    via implicit differentiation (applying Richardson iteration in reverse).
  """

  def do_solve(a_fn, b):
    """Helper function to iterate the system."""

    # Iterate x_k = b + (I-A) x_{k-1}
    def fixedpt_fn(_, x):
      return jax.tree_multimap(lambda bi, xi, ai: bi + xi - ai, b, x, a_fn(x))

    return jax.lax.fori_loop(0, iterations, fixedpt_fn, b)

  # Use custom_linear_solve to get correct derivatives
  return jax.lax.custom_linear_solve(
      matvec, b, solve=do_solve, transpose_solve=do_solve)
