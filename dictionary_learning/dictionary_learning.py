# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""An iterative dictionary learning procedure."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
from numpy import linalg as LA
import scipy as sp
import sklearn.decomposition
import sklearn.linear_model


def run_omp(data, dictionary, sparsity):
  """Solve the orthogonal matching pursuit problem.

  Use sklearn.linear_model.orthogonal_mp to solve the following optimization
  program:
        argmin ||y - X*gamma||^2,
        subject to ||gamma||_0 <= n_{nonzero coefs},
      where
        y is 'data', size = (n_samples, n_targets),
        X is 'dictionary', size = (n_samples, n_features). Columns are assumed
        to have unit norm,
        gamma: sparse coding, size = (n_features, n_targets).

  Args:
    data: the matrix y in the above program,
    dictionary: the matrix X in the above program,
    sparsity: n_{nonzero coefs} in the above program.

  Returns:
    gamma
  """

  print("running sklearn.linear_model.orthogonal_mp ...")

  start_time = time.time()
  code = sklearn.linear_model.orthogonal_mp(
      np.transpose(dictionary), np.transpose(data), n_nonzero_coefs=sparsity)
  code = np.transpose(code)

  print("running time of omp = %s seconds" % (time.time() - start_time))

  return code


def run_knn(data, sparsity, row_percentage):
  """Use kNN to initialize a coding table.

  First, we sample a fraction of
     'row_percentage' rows of 'data'. Then for each row of 'data', we map it to
      the 'sparsity' nearest rows that were sampled.

  Args:
    data: the original matrix
    sparsity: the number rows to which each row of 'data' is mapped
    row_percentage: percent of rows in the sample

  Returns:
    The initial sparse coding table.
  """

  print("Running kNN ...")

  # 'sample_size' should be >= 'sparsity'
  sample_size = int(data.shape[0] * row_percentage)
  if sample_size < sparsity:
    sample_size = sparsity
    print("Reset sample_size to ", sparsity, " in run_knn().")

  print("Sample size = ", sample_size)

  idx = np.random.randint(data.shape[0], size=sample_size)
  sample_table = data[idx, :]

  # Run KNN to compute the coding table.
  tree = sp.spatial.KDTree(sample_table)

  _, indices = tree.query(data, k=sparsity, eps=0)
  code = np.zeros((data.shape[0], sample_size), dtype=float)

  for i in range(data.shape[0]):
    for j in indices[i]:
      code[i][j] = 1

  print("code = ", code)
  print("Done.")

  return code


def build_linear_operator(sparse_matrix, num_columns, sparsity):
  """Builds the linear operator for our least square problem.

  The objective is to solve the least square problem: S ~ M * U, where M is
  a sparse matrix (we are given S and M and try to solve for U). We rewrite this
  as b ~ A * x and use sp.sparse.linalg.lsmr. This function builds and returns
  the linear operator that implicitly represents A. We also need to know
  #columns of U.

  Args:
    sparse_matrix: a sparse matrix
    num_columns: the number of columns of U
    sparsity: the number of non-zero values of each row of 'sparse_matrix'

  Returns:
    A linear operator that represents A.
  """

  m = sparse_matrix.shape[0]
  n = sparse_matrix.shape[1]
  t = num_columns
  nrows = m * t

  row_ind = np.zeros(nrows * sparsity, dtype=int)
  col_ind = np.zeros(nrows * sparsity, dtype=int)
  data = np.zeros(nrows * sparsity, dtype=float)

  cnt = 0

  for i in range(m):
    for col in range(n):
      x = sparse_matrix.item(i, col)
      if abs(x) > 1e-06:
        for j in range(t):
          row = i * t + j
          var_idx = col * t + j

          row_ind[cnt] = row
          col_ind[cnt] = var_idx
          data[cnt] = x
          cnt = cnt + 1

  sparse_matrix = sp.sparse.csr_matrix((data, (row_ind, col_ind)),
                                       shape=(m * t, n * t))

  def matvec(v):
    return sparse_matrix.dot(v)

  def rmatvec(v):
    return sparse_matrix.getH().dot(v)

  return sp.sparse.linalg.LinearOperator((m * t, n * t),
                                         matvec=matvec,
                                         rmatvec=rmatvec)


def fast_least_squared_dictionary(embedding_table, code, sparsity):
  """Apply least-square algorithm to optimize the dictionary.

  Args:
    embedding_table: the original embedding table table
    code: the sparse coding table
    sparsity: # of nonzeros on each row of 'code'

  Returns:
    a dictionary such that ||embedding_table-code*dictionary||_2 is minimized.
  """
  b = np.reshape(embedding_table,
                 embedding_table.shape[0] * embedding_table.shape[1])

  start_time = time.time()
  x = sp.sparse.linalg.lsmr(
      build_linear_operator(code, embedding_table.shape[1], sparsity),
      b,
      maxiter=100000)[0]

  dictionary = x.reshape(code.shape[1], embedding_table.shape[1])

  print("Running time of least-square = %s seconds" %
        (time.time() - start_time))

  return dictionary


def is_zero_vector(a):
  for v in a:
    if v != 0:
      return False
  return True


def dictionary_learning(embedding_table,
                        row_percentage=0.5,
                        col_percentage=0.5,
                        n_iterations=32,
                        seed=15):
  """An iterative algorithm for dictionary learning.

  This algorithm alternatively uses least-square and OMP algorithm to
  optimize the coding table and dictionary, respectively.

  Args:
    embedding_table: the original matrix
    row_percentage: a factor to determine the number of words in the dictionary
    col_percentage: a factor to determine the sparsity of the coding table
    n_iterations: the number of iterations
    seed: a seed number

  Returns:
    A list of 'code' and 'dictionary' such that embedding_table ~ code *
      dictionary, where 'dictionary' has 'row_percentage' * #rows of
      'embedding_table' words, 'code' has sparsity = 'col_percentage' * #columns
      of 'embedding_table' nonzeros per row. Intuitively, each row of
      'embedding_table' will be approximated as a linear combination of
      'sparsity' different words in 'dictionary'.
  """

  np.random.seed(seed)
  errors = np.zeros((n_iterations,), dtype=float)

  for i in range(0, embedding_table.shape[0]):
    if is_zero_vector(embedding_table[i]):
      print("Row ", i, " of embedding_table is a 0-vector.")
      print("Returning 0-matrices for this case ...")
      # TODO(khoatrinh): only do this if 'embedding_table' is a 0-matrix. (Just
      # skip normalizing 0-rows.)

      code = np.zeros((embedding_table.shape[0],
                       int(row_percentage * embedding_table.shape[0])),
                      dtype=float)

      dictionary = np.zeros((int(
          row_percentage * embedding_table.shape[0]), embedding_table.shape[1]),
                            dtype=float)

      return [code, dictionary]

  sparsity = int(col_percentage * embedding_table.shape[1])
  print("sparsity = ", sparsity)

  # Initialize 'code' by knn
  code = run_knn(embedding_table, sparsity, row_percentage)

  for i in range(n_iterations):
    print("Iterator = ", i)

    # Use least-square to optimize for the dictionary.
    dictionary = fast_least_squared_dictionary(embedding_table, code, sparsity)

    # Print the errors
    approx = np.matmul(code, dictionary)
    error = 100.0 * LA.norm(embedding_table - approx,
                            "fro") / LA.norm(embedding_table)
    errors[i] = error
    print("error = ", error)

    dictionary = sklearn.preprocessing.normalize(dictionary, norm="l2")

    # Optimize for the coding table.
    code = run_omp(embedding_table, dictionary, sparsity)

    # Print the errors
    approx = np.matmul(code, dictionary)
    error = 100.0 * LA.norm(embedding_table - approx,
                            "fro") / LA.norm(embedding_table)
    print("error = ", error)

  return [code, dictionary]
