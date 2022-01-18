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

r"""Library for reconstructing Jacobian J \in R^{m x n}.


A library for reconstructing the approximate version of the Jacobian matrix from
the set of local noisy linear measurements.
"""
# pylint: disable=g-doc-return-or-yield,missing-docstring,g-doc-args,line-too-long,invalid-name,pointless-string-statement, super-init-not-called, unused-argument

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import flags

import cvxpy as cp
import numpy as np

FLAGS = flags.FLAGS


def general_regularized_regression_loss(A, b, x, regularization_parameter,
                                        loss_norm, regularization_norm):
  r"""Function implementing general regularized regression loss.

  Implements general regularized regression objective function.
  The optimization problem is defined as follows:
                  min_x ||A*x - b||^{2}_{p} +
                        regularization_parameter*||x||_{q}

  where: p = loss_norm, q = regularization_norm.

  Args:
    A: see the definition of the optimization problem above
    b: see the definition of the optimization problem above
    x: see the definition of the optimization problem above
    regularization_parameter: see description of the optimization problem above
    loss_norm: see the definition of the optimization problem above
    regularization_norm: see the definition of the optimization problem above

  Returns:
    The general regularized loss function.
  """

  def loss_fn(A, b, x):
    k = len(A)
    b_reshaped = b.reshape((k))
    return cp.pnorm(cp.matmul(A, x) - b_reshaped, p=loss_norm)**loss_norm

  def regularizer(x):
    return cp.pnorm(x, p=regularization_norm)**regularization_norm

  return loss_fn(A, b, x) + regularization_parameter * regularizer(x)


def vector_decoding_function(A, b, optimization_parameters, loss_function):
  r"""Function decoding a vector from the set of liner mesurements.

  Decodes the vector from the set of linear measurements along the directions
  defined by the rows of matrix A. Linear measurements are encoded by a
  vector b. The decoding is done by minimizing a specific loss_function
  parametrized by <optimization_parameters>.

  Args:
    A: matrix with rows defining directions along which vector to be
      reconstructed is beign sensed
    b: vector of linear measurements obtained by the above sensing
    optimization_parameters: see description of the optimization problem above
    loss_function: see description of the optimization problem above

  Returns:
    The approximate recovered vector x \in R^{n}
  """
  n = len(np.transpose(A))
  x = cp.Variable(n)
  regularization_parameter = cp.Parameter(nonneg=True)
  regularization_parameter.value = optimization_parameters
  problem = cp.Problem(
      cp.Minimize(loss_function(A, b, x, regularization_parameter)))
  problem.solve(solver=cp.ECOS)
  result = x.value
  res_list = []
  for i in range(n):
    res_list.append(result[i])
  return np.array(res_list)


def general_jacobian_decoder(atranspose, yprime, optimization_parameters,
                             loss_function):
  r"""The wrapper around gradient decoder that serves to decode entire Jacobian.


  Decodes the rows of the Jacobian matrix J \in R^{m x n} and then puts them
  together to reconstruct the entire Jacobian. Each row r_{i} is reconstructed
  from the collection of the noisy dot products: a_{j}^{T}r \sim y_{j},
  where a_{j} is the jth column of the matrix of samples atranspose and y_{j} is
  the jth row of the matrix of measurements yprime \in R^{k x m}.
  The reconstruction of each row is handled by function by a certain decoding
  function that uses parameters defined in <optimization_parameters> and
  which goal is to minimize a specific loss_function parametrized by this set
  of parameters.

  Args:
    atranspose: see description of the optimization problem above
    yprime: see description of the optimization problem above
    optimization_parameters: see description of the optimization problem above
    loss_function: see description of the optimization problem above

  Returns:
    The approximate Jacobian J \in R^{m x n}
  """
  n = len(atranspose)
  m = len(np.transpose(yprime))
  k = len(atranspose[0])

  final_solutions = []
  for i in range(m):
    yprime_row = (np.transpose(yprime))[i]
    yprime_row_reshaped = (yprime_row.reshape((k, 1))).astype(np.double)
    amatrix = (np.transpose(atranspose)).astype(np.double)
    res = vector_decoding_function(amatrix, yprime_row_reshaped,
                                   optimization_parameters, loss_function)
    list_res = []
    for j in range(n):
      list_res.append(res[j])
    final_solutions.append(np.float32(list_res))
  return np.array(final_solutions)


def l1_regression_loss(A, b, x, regularization_parameter):
  return general_regularized_regression_loss(A, b, x, 0.0, 1, 1)


def l1_jacobian_decoder(atranspose, yprime, optimization_parameters):
  return general_jacobian_decoder(atranspose, yprime, optimization_parameters,
                                  l1_regression_loss)


def lasso_regression_loss(A, b, x, regularization_parameter):
  return general_regularized_regression_loss(A, b, x, regularization_parameter,
                                             2, 1)


def lasso_regression_jacobian_decoder(atranspose, yprime,
                                      optimization_parameters):
  return general_jacobian_decoder(atranspose, yprime, optimization_parameters,
                                  lasso_regression_loss)


def ridge_regression_loss(A, b, x, regularization_parameter):
  return general_regularized_regression_loss(A, b, x, regularization_parameter,
                                             2, 2)


def ridge_regression_jacobian_decoder(atranspose, yprime,
                                      optimization_parameters):
  return general_jacobian_decoder(atranspose, yprime, optimization_parameters,
                                  ridge_regression_loss)
