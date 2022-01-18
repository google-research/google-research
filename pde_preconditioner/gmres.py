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

# pylint: skip-file

import functools

import jax
from jax import lax
import jax.numpy as np
import jax.ops
from jax.tree_util import Partial
import numpy as onp

# differentiable GMRES


@Partial  # allows for use in @jit
def identity(x):
  return x

def _inner(v, q):
  h_jk = q.conj() @ v
  v = v - h_jk * q
  return (v, h_jk)

@functools.partial(jax.jit, static_argnums=(1,))
def _outer(carray, k):
  ''' combines arnoldi iteration with givens rotation '''
  # https://en.wikipedia.org/wiki/Generalized_minimal_residual_method
  Q, beta, (A, M, cs, sn) = carray
  q = Q[:, k]
  v = A(M(q))
  v, h_col = lax.scan(_inner, v, Q.T)
  v_norm = np.linalg.norm(v)
  Q = Q.at[:, k + 1].set(v / v_norm)
  h_col = h_col.at[k + 1].set(v_norm)
  h1, cs1, sn1 = apply_givens_rotation(h_col, cs, sn, k)
  h_col = h_col.at[:].set(h1)
  cs = cs.at[k].set(cs1)
  sn = sn.at[k].set(sn1)
  beta = beta.at[k + 1].set(-sn1 * beta[k])
  beta = beta.at[k].set(cs1 * beta[k])
  error  = abs(beta[k+1])
  return (Q, beta, (A, M, cs, sn)), (h_col,  error)

@functools.partial(jax.jit, static_argnums=(2,))
def arnoldi_iteration(A, b, n, res_norm, M=identity):
  # https://en.wikipedia.org/wiki/Arnoldi_iteration#The_Arnoldi_iteration
  m = b.shape[0]
  b_norm = np.linalg.norm(b)
  q = b / b_norm
  beta = np.concatenate([np.ones((1,)), np.zeros((m - 1,))])*res_norm
  cs = np.ones((n,))
  sn = np.zeros((n,))
  Q = np.concatenate([q[:, np.newaxis], np.zeros((m, n))], axis=1)
  (Q, beta, _), (h, error) = lax.scan(_outer, (Q, beta, (A, M, cs, sn)), onp.arange(n))
  return Q, h.T, beta, error/b_norm

@jax.jit
def lstsq(a, b):
  return np.linalg.solve(a.T @ a, a.T @ b)

def _givens(carry, i):
  h, cs, sn  = carry
  temp   =  cs[i]*h[i] + sn[i]*h[i+1]
  h = h.at[i + 1].set(-sn[i] * h[i] + cs[i] * h[i + 1])
  h = h.at[i].set(temp)
  return (h, cs, sn), 0

@jax.jit
def apply_givens_rotation(h, cs, sn, k):
  # cs[k:] = 1 at the start
  # sn[k:] = 0 at the start
  # h[k+2:] = 0 at the start (h should be of len=k+1 in matlab)
  # need to only run _givens on 0,...,k-1
  (h, cs, sn), _ = lax.scan(_givens, (h, cs, sn), np.arange(len(h)-1))
  cs_k, sn_k = givens_rotation(h[k], h[k+1])
  h = h.at[k].set(cs_k * h[k] + sn_k * h[k + 1])
  h = h.at[k + 1].set(0.0)
  return h, cs_k, sn_k

@jax.jit
def givens_rotation(v1, v2):
  t = np.sqrt(v1**2+v2**2)
  cs = np.where(abs(v1) < 1e-7, 0, abs(v1)/t)
  sn = np.where(abs(v1) < 1e-7, 1, cs*v2/ v1)
  return cs, sn

@functools.partial(jax.jit, static_argnums=(3,5,))
def _gmres(A, b, x0, n, M, record):
  # https://www-users.cs.umn.edu/~saad/Calais/PREC.pdf
  res_norm = np.linalg.norm(b-A(x0))
  Q, H, beta, error  = arnoldi_iteration(A, b, n, res_norm, M)
  # TODO: add check for being in subspace eps = 1e-6
  K = n
  # H[:K,:K] is upper triangular now
  y = jax.scipy.linalg.solve_triangular(H[:K,:K], beta[:K], overwrite_b=True,
                                        check_finite=False)
  x = x0 + M(Q[:, :K] @ y)
  if record:
    return x, np.concatenate([np.array([res_norm/np.linalg.norm(b)]),error])
  else:
    return x

def gmres(A, b, x0=None, n=5, M=identity, record=False):
  if x0 is None:
    x0 = np.zeros_like(b)
  return _gmres(A, b, x0, n, M, record)
