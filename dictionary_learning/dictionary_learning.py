# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

from absl import logging
import numpy as np
from numpy import linalg as LA
import scipy as sp
from scipy import sparse
import sklearn.decomposition
import sklearn.linear_model


def run_lsh_omp_coder(data, dictionary, sparsity, num_buckets=1):
  """Solve the orthogonal matching pursuit problem with LSH bucketing.

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
    data: The matrix y in the above program,
    dictionary: The matrix X in the above program,
    sparsity: n_{nonzero coefs} in the above program.
    num_buckets: number of LSH buckets to use, int.

  Returns:
    gamma.
  """
  logging.info("running LSH based sklearn.linear_model.orthogonal_mp ...")

  indices = lsh_knn_map(
      np.transpose(np.vstack((data, dictionary))), num_buckets, 1)
  logging.info("indices shape is %s", indices.shape)
  data_buckets = [[] for i in range(num_buckets)]
  data_index = [[] for i in range(num_buckets)]
  dict_buckets = [[] for i in range(num_buckets)]
  dict_index = [[] for i in range(num_buckets)]
  for i in range(data.shape[0]):
    data_buckets[indices[i][0]].append(data[i, :])
    data_index[indices[i][0]].append(i)
  for i in range(dictionary.shape[0]):
    dict_buckets[indices[data.shape[0] + i][0]].append(dictionary[i, :])
    dict_index[indices[data.shape[0] + i][0]].append(i)
  code = sparse.lil_matrix((data.shape[0], dictionary.shape[0]))
  for i in range(num_buckets):
    start_time = time.time()
    if len(data_buckets[i]) > 0:  # pylint: disable=g-explicit-length-test
      if len(dict_buckets[i]) == 0:  # pylint: disable=g-explicit-length-test
        logging.error(
            "lsh bucketing failed...empty bucket with no dictionary elements")
      else:
        small_code = sklearn.linear_model.orthogonal_mp(
            np.transpose(np.vstack(dict_buckets[i])),
            np.transpose(np.vstack(data_buckets[i])),
            n_nonzero_coefs=sparsity)
        small_code = np.transpose(small_code)

        row_idx = np.asarray(data_index[i])
        col_idx = np.asarray(dict_index[i])

        code[row_idx[:, None], col_idx] = small_code

    logging.info("running time of OMP for bucket %d = %d seconds",
                 i, time.time() - start_time)
  return code


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
    data: The matrix y in the above program,
    dictionary: The matrix X in the above program,
    sparsity: n_{nonzero coefs} in the above program.

  Returns:
    gamma
  """

  logging.info("running sklearn.linear_model.orthogonal_mp ...")

  start_time = time.time()
  code = sklearn.linear_model.orthogonal_mp(
      np.transpose(dictionary), np.transpose(data), n_nonzero_coefs=sparsity)
  code = np.transpose(code)

  logging.info("running time of omp = %s seconds", time.time() - start_time)

  return code


def run_dot_product_coder(data, dictionary, sparsity, k=3, batch_size=1000):
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
    data: The matrix y in the above program,
    dictionary: The matrix X in the above program,
    sparsity: n_{nonzero coefs} in the above program,
    k: number of rows to use for generating dictionary,
    batch_size: batch size, positive int.

  Returns:
    gamma
  """
  logging.info("running sparse coder sklearn.linear_model.orthogonal_mp ...")
  n, _ = data.shape
  m, _ = dictionary.shape
  index = 0

  start_time = time.time()
  code = sparse.lil_matrix((n, m))

  while index + batch_size < n + 1:
    logging.info("processing batch %d", index // batch_size)
    small_data = np.transpose(data[index:index + batch_size, :])
    prods = np.matmul(dictionary, small_data)
    indices = np.argsort(-abs(prods), axis=0)
    union_of_indices = indices[0:k, :]
    union_of_indices = union_of_indices.flatten()
    union_of_indices = np.unique(union_of_indices)
    logging.info("number of indices = %d", len(union_of_indices))

    small_code = sklearn.linear_model.orthogonal_mp(
        np.transpose(dictionary[union_of_indices, :]),
        small_data,
        n_nonzero_coefs=sparsity,
        precompute=False)
    start_index = index
    end_index = index + batch_size
    code[start_index:end_index, union_of_indices] = np.transpose(small_code)
    index += batch_size

  if index < n:
    small_data = np.transpose(data[index:n, :])
    prods = np.matmul(dictionary, small_data)
    indices = np.argsort(-abs(prods), axis=0)
    union_of_indices = indices[0:k, :]
    union_of_indices = union_of_indices.flatten()
    union_of_indices = np.unique(union_of_indices)

    small_code = sklearn.linear_model.orthogonal_mp(
        np.transpose(dictionary[union_of_indices, :]),
        small_data,
        n_nonzero_coefs=sparsity,
        precompute=False)
    start_index = index
    end_index = n
    code[start_index:end_index, union_of_indices] = np.transpose(small_code)

  print("running time of omp = %s seconds" % (time.time() - start_time))

  return code.tocsr()


def run_batch_omp_coder(data, dictionary, sparsity, batch_size=1000):
  """Solve the orthogonal matching pursuit problem in mini-batch fashion.

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
    data: The matrix y in the above program,
    dictionary: The matrix X in the above program,
    sparsity: n_{nonzero coefs} in the above program,
    batch_size: batch size, positive int.

  Returns:
    gamma
  """

  print("running sparse coder sklearn.linear_model.orthogonal_mp ...")
  [n, _] = data.shape
  [m, _] = dictionary.shape
  index = 0

  start_time = time.time()
  code = sparse.lil_matrix((n, m))

  while index + batch_size < n + 1:  # in range(num_iter):
    logging.info("processing batch")
    small_code = sklearn.linear_model.orthogonal_mp(
        np.transpose(dictionary),
        np.transpose(data[index:index + batch_size, :]),
        n_nonzero_coefs=sparsity,
        precompute=False)
    start_index = index
    end_index = index + batch_size
    code[start_index:end_index, :] = np.transpose(small_code)
    index += batch_size
  if index < n:
    small_code = sklearn.linear_model.orthogonal_mp(
        np.transpose(dictionary),
        np.transpose(data[index:n, :]),
        n_nonzero_coefs=sparsity,
        precompute=False)
    start_index = index
    end_index = n
    code[start_index:end_index, :] = np.transpose(small_code)

  print("running time of omp = %s seconds" % (time.time() - start_time))

  return code.tocsr()


def load_indices_to_csr(indices, dict_size):
  """Load indices into a CSR (compressed sparse row) format indicator matrix.

  Example:

  indices = np.array([[1], [2], [0]])
  dict_size = 4

  sparse_matrix = load_indices_to_csr(indices, dict_size)
  dense_matrix = sparse_matrix.to_dense()
  # dense_matrix = [[0. 1. 0. 0.], [0. 0. 1. 0.], [1. 0. 0. 0.]]

  Args:
    indices: indices array, a numpy 2d array of ints;
    dict_size: size of dictionary, int.

  Returns:
    sparse_matrix: a sparse indicator matrix in the CSR format with dense shape
      (indices.shape[0], dict_size) of floats, with entries at (i, j) equal to
       1.0 for i in range(indices.shape[0]), j in indices[i].
  """
  rows = np.zeros(indices.shape[0] * indices.shape[1], dtype=int)
  cols = np.zeros(indices.shape[0] * indices.shape[1], dtype=int)
  vals = np.zeros(indices.shape[0] * indices.shape[1], dtype=float)

  cnt = 0
  for i in range(indices.shape[0]):
    for j in indices[i]:
      rows[cnt] = i
      cols[cnt] = j
      vals[cnt] = 1.0
      cnt = cnt + 1

  sparse_matrix = sp.sparse.csr_matrix((vals, (rows, cols)),
                                       shape=(indices.shape[0], dict_size))
  return sparse_matrix


def run_knn(data, sparsity, row_percentage, eps=0.9):
  """Use kNN to initialize a coding table.

  First, we sample a fraction of
     'row_percentage' rows of 'data'. Then for each row of 'data', we map it to
      the 'sparsity' nearest rows that were sampled.

  Args:
    data: The original matrix
    sparsity: The number rows to which each row of 'data' is mapped
    row_percentage: Percent of rows in the sample
    eps: approximation tolerance factor, the returned the k-th neighbor is no
         further than (1 + epsilon) times the distance to the true k-th
         neighbor, needs to be nonnegative, float.

  Returns:
    The initial sparse coding table.
  """
  logging.info("Running kNN ...")

  # 'sample_size' should be >= 'sparsity'
  sample_size = int(data.shape[0] * row_percentage + 1)
  if sample_size < sparsity:
    sample_size = sparsity
    logging.info("Reset sample_size to %d in run_knn().", sparsity)

  logging.info("Sample size = %d", sample_size)

  idx = np.random.randint(data.shape[0], size=sample_size)
  sample_table = data[idx, :]

  logging.info("Setting up kd tree.")

  # Run KNN to compute the coding table.
  tree = sp.spatial.KDTree(sample_table)

  logging.info("Querying up kd tree.")

  _, indices = tree.query(data, k=sparsity, eps=eps)
  logging.info("Done querying up kd tree.")

  code = load_indices_to_csr(indices, sample_size)

  logging.info("code.shape = %s.", code.shape)
  logging.info("Done.")

  return code


def run_knn_lsh(data, sparsity, row_percentage):
  """Run Locality Sensitive Hashing to find near neighbors for initialization.

  Args:
    data: The original matrix;
    sparsity: The number rows to which each row of 'data' is mapped;
    row_percentage: Percent of rows in the sample.

  Returns:
    The initial sparse coding table.
  """
  table = np.transpose(data)
  sample_size = int(data.shape[0] * row_percentage + 1)
  indices = lsh_knn_map(table, sample_size, sparsity)

  code = load_indices_to_csr(indices, sample_size)

  logging.info("Done.")

  return code


def lsh_knn_map(table, field_size, k):
  """Returns k nearest neighbors using Locality Sensitive Hashing (LSH).

  Args:
    table: the data matrix, numpy 2d array;
    field_size: integer parameter that controls number of random hyperplanes
      used in LSH;
    k: number of neighbor to return for each row of table, int.

  Returns:
    A matrix in which the i-th row contains the indices of the k neighest
    neighbors of the i-th row of table.
  """
  mean = table.mean(axis=1)
  for i in range(table.shape[1]):
    table[:, i] = table[:, i] - mean

  num_bits_per_word = int(np.ceil(1 + np.log2(field_size)))
  logging.info("num bits per word is table.shape is %d, %d, %d, %s.",
               num_bits_per_word, k, num_bits_per_word * k, table.shape)
  W = np.random.randn(num_bits_per_word * k, table.shape[0])  # pylint: disable=invalid-name
  logging.info("W is  %s, %s.", W.shape, W)
  signs_out = np.sign(np.matmul(W, table)).astype(int)
  # signs_out should be a 0/1 matrix and not -1/1.
  signs_out = ((1 + signs_out) / 2.0).astype(int)

  logging.info("signs out is mean is %s, %s, %s", signs_out.shape,
               signs_out[:10, :30], signs_out.mean(axis=1))
  # matrix to multipily with signs
  transformer = np.zeros((k, k * num_bits_per_word), dtype=np.int32)

  # get powers upto 2^k
  powers = np.power(2, np.arange(num_bits_per_word)).astype(int)
  for i in range(k):
    transformer[i, num_bits_per_word * i:num_bits_per_word * (i + 1)] = powers

  logging.info("Transformer is %s, %s.", transformer.shape, transformer)
  output_map = np.mod(
      np.matmul(transformer, signs_out).astype(int), field_size).astype(int)

  return np.transpose(output_map)


def build_linear_operator(sparse_matrix, num_columns, sparsity):
  """Builds the linear operator for our least square problem.

  The objective is to solve the least square problem: S ~ M * U, where M is
  a sparse matrix (we are given S and M and try to solve for U). We rewrite this
  as b ~ A * x and use sp.sparse.linalg.lsmr. This function builds and returns
  the linear operator that implicitly represents A. We also need to know
  #columns of U.

  Args:
    sparse_matrix: A sparse matrix
    num_columns: The number of columns of U
    sparsity: The number of non-zero values of each row of 'sparse_matrix'

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

  tuples = sparse_matrix.nonzero()
  for (i, col) in zip(tuples[0], tuples[1]):
    x = sparse_matrix[i, col]
    if abs(x) > 1e-06:
      for j in range(t):
        row = i * t + j
        var_idx = col * t + j

        row_ind[cnt] = row
        col_ind[cnt] = var_idx
        data[cnt] = x
        cnt = cnt + 1
  print("Near end of build_linear_operator1")

  sparse_matrix = sp.sparse.csr_matrix((data, (row_ind, col_ind)),
                                       shape=(m * t, n * t))

  def matvec(v):
    return sparse_matrix.dot(v)

  def rmatvec(v):
    return sparse_matrix.getH().dot(v)

  print("Near end of build_linear_operator2")

  return sp.sparse.linalg.LinearOperator((m * t, n * t),
                                         matvec=matvec,
                                         rmatvec=rmatvec)


def fast_least_squared_dictionary(embedding_table, code, sparsity):
  """Apply least-square algorithm to optimize the dictionary.

  Args:
    embedding_table: The original embedding table table
    code: the sparse coding table
    sparsity: The number of nonzeros on each row of 'code'

  Returns:
    a dictionary such that ||embedding_table-code*dictionary||_2 is minimized.
  """
  b = np.reshape(embedding_table,
                 embedding_table.shape[0] * embedding_table.shape[1])

  start_time = time.time()
  x = sp.sparse.linalg.lsmr(
      build_linear_operator(code, embedding_table.shape[1], sparsity),
      b,
      maxiter=50)[0]
  dictionary = x.reshape(code.shape[1], embedding_table.shape[1])

  print("Running time of least-square = %s seconds" %
        (time.time() - start_time))

  return dictionary


def is_zero_vector(a):
  for v in a:
    if v != 0:
      return False
  return True


def build_linear_operator_v2(sparse_matrix, num_columns, sparsity, A, B):  # pylint: disable=invalid-name
  """Builds the linear operator for our least square problem.

  The objective is to solve the least square problem: S ~ M * U, where M is
  a sparse matrix (we are given S and M and try to solve for U). We rewrite this
  as b ~ A * x and use sp.sparse.linalg.lsmr. This function builds and returns
  the linear operator that implicitly represents A. We also need to know
  #columns of U.

  Args:
    sparse_matrix: A sparse matrix
    num_columns: The number of columns of U
    sparsity: The number of non-zero values of each row of 'sparse_matrix',
    A: the A matrix above;
    B: the S matrix above.

  Returns:
    A linear operator that represents A.
  """

  del sparsity  # unused

  print("Near begin of build_linear_operator")
  start_time = time.time()

  m = sparse_matrix.shape[0]
  n = sparse_matrix.shape[1]
  t = num_columns

  S = sparse_matrix.transpose()  # pylint: disable=invalid-name
  S = S.tocsr()  # pylint: disable=invalid-name

  print("Near end of build_linear_operator1")
  print("build op time={}".format(time.time() - start_time))

  def matvec(v):
    M = np.zeros((m, t))  # pylint: disable=invalid-name
    for i in range(t):
      M[:, i] = sparse_matrix.dot(A[i].dot(v))
    return M.flatten(order="F")

  def rmatvec(v):
    vec = np.zeros(n * t)
    for j in range(t):
      r = B[j].dot(S.dot(v[j * m:(j + 1) * m]))
      vec += r

    return vec

  print("Near end of build_linear_operator2")

  return sp.sparse.linalg.LinearOperator((m * t, n * t),
                                         matvec=matvec,
                                         rmatvec=rmatvec)


def fast_least_squared_dictionary_v2(embedding_table, code, sparsity, A, B):  # pylint: disable=invalid-name
  """Apply least-square algorithm to optimize the dictionary.

  Args:
    embedding_table: The original embedding table table;
    code: the sparse coding table;
    sparsity: The number of nonzeros on each row of 'code';
    A: the A matrix above;
    B: the S matrix above.

  Returns:
    a dictionary such that ||embedding_table-code*dictionary||_2 is minimized.
  """

  b = np.reshape(
      embedding_table,
      embedding_table.shape[0] * embedding_table.shape[1],
      order="F")

  print("entered fast_least_squared_dictionary")
  logging.info("entered fast_least_squared_dictionary")
  start_time = time.time()
  x = sp.sparse.linalg.lsmr(
      build_linear_operator_v2(code, embedding_table.shape[1], sparsity, A, B),
      b,
      maxiter=50)[0]
  dictionary = x.reshape(code.shape[1], embedding_table.shape[1])

  print("Running time of least-square = %s seconds" %
        (time.time() - start_time))
  logging.info("Running time of least-square = %s seconds",
               time.time() - start_time)

  return dictionary


def dictionary_learning(embedding_table,
                        row_percentage=0.5,
                        col_percentage=0.5,
                        n_iterations=32,
                        seed=15,
                        use_lsh=False,
                        use_projection=False,
                        projection_dim=100,
                        log_errors=False,
                        return_dense=True,
                        version_num=0,
                        num_buckets=1):
  """An iterative algorithm for dictionary learning.

  This algorithm alternatively uses least-square and OMP algorithm to
  optimize the coding table and dictionary, respectively.

  If use_projection is True, then the embedding_table is first projected down
  to a subspace of dimension `projection_dim` before running DL.

  Args:
    embedding_table: The original matrix
    row_percentage: A factor to determine the number of words in the dictionary
    col_percentage: A factor to determine the sparsity of the coding table. More
      specifically, sparsity is equal to 'col_percentage' times the number of
      columns of 'embedding_table'.
    n_iterations: The number of iterations
    seed: A seed number
    use_lsh: Boolean flag to control whether to use lsh for initialization
    use_projection: Boolean flag to control whether to use projection or not
    projection_dim: The dimension of the projection, int
    log_errors: If True the code will print reconstruction error after each
      update of the code and the dictionary
    return_dense: If True return the code matrix in dense matrix format,
      otherwise return code matrix in CSR format
    version_num: If set to 1 the fast_least_squared_dictionary_v2() will be
      used
    num_buckets: number of buckets to use in run_lsh_omp_coder() function, int.

  Returns:
    A list of 'code' and 'dictionary' such that embedding_table ~ code *
      dictionary, where 'dictionary' has 'row_percentage' * #rows of
      'embedding_table' words, 'code' has sparsity = 'col_percentage' * #columns
      of 'embedding_table' nonzeros per row. Intuitively, each row of
      'embedding_table' will be approximated as a linear combination of
      'sparsity' different words in 'dictionary'. Overall, this approximation of
      'embedding_table' has size equal to ('row_percentage' + 'col_percentage')
      times the size of the original 'embedding_table'.
  """
  # TODO(wanxin): split the dictionary learning function into smaller functions.

  logging.info(
      "Inside run_update_step dictionary_learning: "
      "A shapes is %s A norm is %d: ",
      embedding_table.shape, np.linalg.norm(embedding_table))

  np.random.seed(seed)
  errors = np.zeros((n_iterations,), dtype=float)

  embedding_dims = np.copy(embedding_table.shape)

  # Projection.
  if use_projection:
    projection_matrix = np.random.normal(
        0,
        1.0 / np.sqrt(projection_dim),
        size=(embedding_dims[1], projection_dim))
    embedding_table = np.matmul(embedding_table, projection_matrix)

  dictionary = np.random.normal(0, 1, [
      int(row_percentage * embedding_table.shape[0] + 1),
      embedding_table.shape[1]
  ])
  for i in range(0, embedding_table.shape[0]):
    if is_zero_vector(embedding_table[i]):
      print("Row ", i, " of embedding_table is a 0-vector.")
      print("Returning 0-matrices for this case ...")
      # TODO(khoatrinh): only do this if 'embedding_table' is a 0-matrix. (Just
      # skip normalizing 0-rows.)
      dict_size = int(row_percentage * embedding_table.shape[0] + 1)
      sparsity = int(col_percentage * embedding_table.shape[1] + 1)

      code = sp.sparse.lil_matrix((embedding_table.shape[0], dict_size))
      logging.info("code.shape is %s", code.shape)
      code[:, :sparsity] = 1.0

      dictionary = np.zeros((int(row_percentage * embedding_table.shape[0] + 1),
                             embedding_table.shape[1]),
                            dtype=float)

      if use_projection:
        dictionary = np.matmul(dictionary, np.transpose(projection_matrix))

      code = code.tocsr()
      if return_dense:
        code = code.A

      return [code, dictionary]

  sparsity = int(col_percentage * embedding_table.shape[1] + 1)
  if use_projection:
    sparsity = int(col_percentage * embedding_dims[1] + 1)
  logging.info("sparsity = %d", sparsity)

  # Initialize 'code' by knn
  if use_lsh:
    code = run_knn_lsh(embedding_table, sparsity, row_percentage)
  else:
    code = run_knn(embedding_table, sparsity, row_percentage)

  print("shape of code is ", code.shape)
  if version_num == 1:
    dict_size = int(row_percentage * embedding_table.shape[0] + 1)
    A = [  # pylint: disable=invalid-name
        sp.sparse.lil_matrix((dict_size, dict_size * embedding_table.shape[1]))
        for i in range(embedding_table.shape[1])
    ]
    B = [  # pylint: disable=invalid-name
        sp.sparse.lil_matrix((dict_size, dict_size * embedding_table.shape[1]))
        for i in range(embedding_table.shape[1])
    ]

    for j in range(embedding_table.shape[1]):
      indices = [
          r for r in range(j, dict_size *
                           embedding_table.shape[1], embedding_table.shape[1])
      ]
      A[j][range(dict_size), indices] = np.ones(dict_size)
      A[j] = A[j].tocsr()
      B[j] = A[j].transpose()
      B[j] = B[j].tocsr()

  for i in range(n_iterations):
    logging.info("Iterator = %d", i)

    # Use least-square to optimize for the dictionary.
    if version_num == 0:
      dictionary = fast_least_squared_dictionary(embedding_table, code,
                                                 sparsity)
    else:
      dictionary = fast_least_squared_dictionary_v2(embedding_table, code,
                                                    sparsity, A, B)

    if log_errors:
      # Do sparse matrix multiplication since code matrix is sparse.
      sparse_code = code
      approx = sparse_code.dot(dictionary)
      error = 100.0 * LA.norm(embedding_table - approx,
                              "fro") / LA.norm(embedding_table)
      errors[i] = error
      logging.info("Frobenius norm error at the %d step is %d percent",
                   i, error)

    dictionary = sklearn.preprocessing.normalize(dictionary, norm="l2")

    # Return code from initialization directly if n_iterations is 1.
    if n_iterations != 1:
      code = run_lsh_omp_coder(
          embedding_table, dictionary, sparsity, num_buckets=num_buckets)

    if log_errors:
      # Do sparse matrix multiplication since code matrix is sparse.
      sparse_code = code
      approx = sparse_code.dot(dictionary)

      error = 100.0 * LA.norm(embedding_table - approx,
                              "fro") / LA.norm(embedding_table)
      logging.info("Frobenius norm error at the %d step is %d percent",
                   i, error)

    if use_projection:
      dictionary = np.matmul(dictionary, np.transpose(projection_matrix))

  if return_dense:
    code = code.A

  return [code, dictionary]
