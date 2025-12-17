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

"""Truncated SVD for implicitly-defined matrices."""

from sparse_deferred.implicit import matrix
Tensor = matrix.Tensor


def truncated_svd(
    engine, mat, k,
    n_redundancy = None, n_iter = 10,
    verbose = False):
  """Randomized SVD of Halko et al 2009 on implicitly-defined matrix `mat`.

  Args:
    engine: Compute engine to use.
    mat: Implicit matrix to approximate via truncated SVD.
    k: Rank of decomposition. Returns (approximate) top-k singular values in S
      and their corresponding left- and right- singular vectors in U, V, such
      that, `tf.matmul(U * S, V, transpose_b=True)` is the best rank-k
      approximation of matrix `M` (implicitly) stored in `fn`.
    n_redundancy: rank of "randomized" decomposition of Halko. The analysis of
      Halko provides that if n_redundancy == k, then the rank-k SVD
      approximation is, in expectation, no worse (in frobenius norm) than twice
      of the "true" rank-k SVD compared to the (implicit) matrix represented by
      fn. However, n_redundancy == k is too slow when k is large. Default sets
      it to min(k, 30).
    n_iter: Number of iterations. >=4 gives good results (with 4 passes over the
      data). We set to 10 (slower than 4) to ensure close approximation
      accuracy. The error decays exponentially with n_iter.
    verbose: If True, prints progress.

  Returns:
    U, s, V, s.t. tf.matmul(U*s, V, transpose_b=True) is a rank-k approximation
    of mat.
  """
  if n_redundancy is None:
    n_redundancy = min(k, 30)
  n_random = k + n_redundancy
  n_samples, n_features = mat.shape
  transpose = n_samples < n_features
  if transpose:
    # This is faster
    mat = mat.T

  q = engine.random_normal(shape=(mat.shape[1], n_random))
  q = mat.__matmul__(q)
  iterations = range(n_iter)
  if verbose:
    # iterations = tqdm.tqdm(iterations, desc='SVD')
    print(f'Starting q has shape {q.shape} and rank {engine.matrix_rank(q)}')
  for _ in iterations:
    q = mat.T.__matmul__(q)
    q = _orthonormalize(engine, q)
    q = mat.__matmul__(q)

  if verbose:
    print('SVD: Final step, followed by SVD on small rank matrix.')

  q = _orthonormalize(engine, q)

  b = mat.__rmatmul__(engine.transpose(q))

  u_hat, s, v = engine.svd(b)

  del b
  u = engine.matmul(q, u_hat)

  u, v = _sign_correction(engine, u=u, v=v, u_based_decision=not transpose)

  if transpose:
    return v[:, :k], s[:k], u[:, :k]
  else:
    return u[:, :k], s[:k], v[:, :k]


def _orthonormalize(engine, q):
  return engine.qr(q)[0]


def _sign_correction(engine, u, v, u_based_decision=True):
  m = u if u_based_decision else v
  max_abs_cols = engine.argmax(engine.abs(m), axis=0)
  signs = engine.sign(
      engine.gather_nd(
          m,
          engine.stack(
              [max_abs_cols, engine.range(m.shape[1], dtype='int64')],
              axis=1)))

  return u*signs, v*signs
