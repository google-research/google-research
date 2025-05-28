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

"""A library for isotonic regression in JAX with PAV + pure_callback."""

import functools

import jax
import jax.numpy as jnp
from sparse_soft_topk._src import isotonic_dykstra
from sparse_soft_topk._src import isotonic_pav


@functools.partial(jax.jit, static_argnums=(1, 2, 3, 4))
def sparse_soft_topk_mask_pav(x, k, l=1e-1, p=4 / 3, bisect_max_iter=50):
  """Returns a differentiable approximation of the top-k mask operator of x using the PAV algorithm.

  Args:
    x: input for the top-k mask, a nd-array. The topk is applied along the last
      axis.
    k: int k for the top-k.
    l: the regularization parameter l that trades sparsity for smoothness.
    p: a float between 1 and +infinity, which corresponds to the p-norm
      regularizer.
    bisect_max_iter: int, number of iterations in the bisection (only used if p
      != 2 and p != 4/3).

  Returns:
    sol: the relaxed top-k mask of x.
  """
  q = p / (p - 1)
  x_shape = x.shape
  if x.ndim > 1:
    x = jnp.reshape(x, (-1, x_shape[-1]))
  n = x.shape[-1]
  perm = jax.lax.stop_gradient(jnp.argsort(-x, axis=-1))
  P = jax.nn.one_hot(perm, n)
  w = jnp.pad(jnp.ones((k,)), (0, n - k))
  s = jnp.einsum('...ab,...b->...a', P, x)
  out_pav = isotonic_pav.isotonic_mask_pav(
      s, w, l=l, p=p, bisect_max_iter=bisect_max_iter
  )
  out = ((s - out_pav) / l) ** (q - 1)
  return (jnp.einsum('...ba,...b->...a', P, out)).reshape(x_shape)


@functools.partial(jax.jit, static_argnums=(1, 2, 3, 4))
def sparse_soft_topk_mag_pav(x, k, l=1e-1, p=4 / 3, bisect_max_iter=50):
  """Returns a differentiable approximation of the top-k operator of x in magnitude using the PAV algorithm.

  Args:
    x: input for the top-k mask, a nd-array. The topk is applied along the last
      axis.
    k: int k for the top k.
    l: the regularization parameter l that trades sparsity for smoothness.
    p: a float between 1 and +infinity, which corresponds to the p-norm
      regularizer.
    bisect_max_iter: int, number of iterations in the bisection (only used if p
      != 2 and p != 4/3).

  Returns:
    sol: the relaxed top-k in magnitude of x.
  """
  q = p / (p - 1)
  x_shape = x.shape
  if x.ndim > 1:
    x = jnp.reshape(x, (-1, x_shape[-1]))
  n = x.shape[-1]
  perm = jax.lax.stop_gradient(jnp.argsort(-jnp.absolute(x), axis=-1))
  P = jax.nn.one_hot(perm, n)
  w = jnp.pad(jnp.ones((k,)), (0, n - k))
  s = jnp.einsum('...ab,...b->...a', P, jnp.absolute(x))
  out_pav = isotonic_pav.isotonic_mag_pav(
      s, w, l=l, p=p, bisect_max_iter=bisect_max_iter
  )
  out = ((s - out_pav) / l) ** (q - 1)
  perm_out = jnp.einsum('...ba,...b->...a', P, out)
  return (jnp.sign(x) * (perm_out + perm_out ** (1 / (q - 1)) * l)).reshape(
      x_shape
  )


def sparse_soft_topk_mask_dykstra(x, k, l=1e-1, num_iter=500):
  """Returns a differentiable approximation of the top-k mask operator of x using Dykstra's algorithm.

  Args:
    x: input to to the top-k mask, a 1d-array.
    k: int k for the top k.
    l: the regularization parameter l that trades sparsity for smoothness.
    num_iter: int, number of iterations in Dykstra's projection algorithm.

  Returns:
    sol: the relaxed top-k mask of x.
  """
  n = x.shape[0]
  perm = jax.lax.stop_gradient(jnp.argsort(-x))
  P = jax.nn.one_hot(perm, n)
  s = P @ x
  s_w = s - l * jnp.pad(jnp.ones((k,)), (0, n - k))
  out_dykstra = isotonic_dykstra.isotonic_dykstra_mask(s_w, num_iter=num_iter)
  out = (s - out_dykstra) / l
  return P.T @ out


def sparse_soft_topk_mag_dykstra(x, k, l=1e-1, num_iter=500):
  """Returns a differentiable approximation of the top-k operator in magnitude of x using Dykstra's algorithm.

  Args:
    x: input to to the top-k, a 1d-array.
    k: int k for the top k.
    l: the regularization parameter l that trades sparsity for smoothness.
    num_iter: int, number of iterations in Dykstra's projection algorithm.

  Returns:
    sol: the relaxed top-k in magnitude of x.
  """
  n = x.shape[0]
  perm = jax.lax.stop_gradient(jnp.argsort(-jnp.absolute(x)))
  P = jax.nn.one_hot(perm, n)
  s = P @ jnp.absolute(x)
  w = jnp.pad(jnp.ones((k,)), (0, n - k))
  out_dykstra = isotonic_dykstra.isotonic_dykstra_mag(
      s / (1 + l * w), w, l=l, num_iter=num_iter
  )
  out = (s - out_dykstra) / l
  perm_out = P.T @ out
  return jnp.sign(x) * perm_out * (1 + l)


def hard_topk_mask(x, k):
  """Returns the top-k mask of x.

  Args:
    x: input to to the top-k, a 1d-array.
    k: int k for the top k.

  Returns:
    sol: the top-k mask of x.
  """
  _, indices = jax.lax.top_k(x, k)
  return jax.nn.one_hot(indices, x.shape[-1]).sum(axis=0)


def hard_topk_mag(x, k):
  """Returns the top-k in magnitude of x.

  Args:
    x: input to to the top-k, a 1d-array.
    k: int k for the top k.

  Returns:
    sol: the top-k of x in magnitude.
  """
  return x * hard_topk_mask(jnp.absolute(x), k)
