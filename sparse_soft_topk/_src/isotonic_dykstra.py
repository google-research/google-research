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

"""Isotonic regression using Dykstra's algorithm and implicit Differentiation."""

import functools

import jax
import jax.numpy as jnp


@functools.partial(jax.custom_jvp, nondiff_argnums=(1,))
def isotonic_dykstra_mask(s, num_iter=500):
  """Solves an isotonic regression problem using Dykstra's projection algorithm.

  Formally, it approximates the solution of

  argmin_{v_1 >= ... >= v_n} 0.5 ||v - s||^2.

  Args:
    s: input to isotonic regression, 1d-array
    num_iter: the number of alternate steps

  Returns:
    sol: the solution, an array of the same size as y.
  """
  def f(v):
    # Here we assume that v's length is even.
    # Note: for the first reduce, maybe it's worth using
    # a convolution instead, with kernel=(-1, 1), because
    # it's not associative.
    d = jax.lax.reduce_window(
        v,
        0.0,
        lambda x, y: y - x,
        window_dimensions=(2,),
        window_strides=(2,),
        padding=((0, 0),),
    )
    a = jax.lax.reduce_window(
        v,
        0.0,
        jax.lax.add,
        window_dimensions=(2,),
        window_strides=(2,),
        padding=((0, 0),),
    )
    mask = jnp.repeat(d < 0, 2)
    mean = jnp.repeat(a, 2) / 2.0
    return v * mask + mean * (1 - mask)

  def body_fn(_, vpq):
    xk, pk, qk = vpq
    yk = jnp.pad(f(xk[:-1] + pk[:-1]), (0, 1), constant_values=xk[-1] + pk[-1])
    p = xk + pk - yk
    v = jnp.pad(f(yk[1:] + qk[1:]), (1, 0), constant_values=yk[0] + qk[0])
    q = yk + qk - v
    return v, p, q

  # Ensure that the length is odd.
  (n,) = s.shape
  if n % 2 == 0:
    minv = jax.lax.stop_gradient(s.min() - 1)
    s = jnp.pad(s, (0, 1), constant_values=minv)

  v = s.copy()
  p = jnp.zeros(s.shape)
  q = jnp.zeros(s.shape)
  vpq = (v, p, q)
  v, p, q = jax.lax.fori_loop(0, num_iter // 2, body_fn, vpq)
  sol = v

  return sol if n % 2 != 0 else sol[:-1]


def _cumsum_einsum(x, precision=jax.lax.Precision.DEFAULT):
  """A faster cumsum, for vectors of size < 8192."""
  mask = jnp.triu(jnp.ones(x.shape, dtype=jnp.bool_))
  return jnp.einsum("ij,jk", x, mask, precision=precision)


def _jvp_isotonic_mask(solution, vector, eps=1e-4):
  """Jacobian at solution of the isotonic regression times vector product."""
  x = solution
  mask = jnp.pad(jnp.absolute(jnp.diff(x)) <= eps, (1, 0))
  ar = jnp.arange(x.size)

  inds_start = jnp.where(mask == 0, ar, +jnp.inf).sort()

  one_hot_start = jax.nn.one_hot(inds_start, len(vector))
  a = _cumsum_einsum(one_hot_start)
  a = jnp.append(jnp.diff(a[::-1], axis=0)[::-1], a[-1].reshape(1, -1), axis=0)
  return (((a.T * (a @ vector)).T) / (a.sum(1, keepdims=True) + 1e-8)).sum(0)


@isotonic_dykstra_mask.defjvp
def _isotonic_dykstra_mask_jvp(num_iter, primals, tangents):
  """Jacobian at solution of the isotonic regression times vector product."""
  (s,) = primals
  (vector,) = tangents
  primal_out = isotonic_dykstra_mask(s, num_iter)
  tangent_out = _jvp_isotonic_mask(primal_out, vector)
  return primal_out, tangent_out


@functools.partial(jax.custom_jvp, nondiff_argnums=(1, 2, 3))
def isotonic_dykstra_mag(s, w, l=1e-1, num_iter=500):
  """Solves an isotonic regression problem using Dykstra's projection algorithm.

  Formally, it approximates the solution of

  argmin_{v_1 >= ... >= v_n} sum_i(1 + l * w_i)(s_i - v_i)^2.

  Args:
    s: input to isotonic regression, a 1d-array
    w: a 1d array
    l: regularization parameter
    num_iter: the number of alternate steps

  Returns:
    sol: the solution, an array of the same size as y.
  """

  def f(v, u):
    # Here we assume that v's length is even.
    # Note: for the first reduce, maybe it's worth using
    # a convolution instead, with kernel=(-1, 1), because
    # it's not associative.
    d = jax.lax.reduce_window(
        v,
        0.0,
        lambda x, y: y - x,
        window_dimensions=(2,),
        window_strides=(2,),
        padding=((0, 0),),
    )
    s_num = jax.lax.reduce_window(
        v * u,
        0.0,
        jax.lax.add,
        window_dimensions=(2,),
        window_strides=(2,),
        padding=((0, 0),),
    )
    s_den = jax.lax.reduce_window(
        u,
        0.0,
        jax.lax.add,
        window_dimensions=(2,),
        window_strides=(2,),
        padding=((0, 0),),
    )
    mask = jnp.repeat(d < 0, 2)

    mean = jnp.repeat(s_num / s_den, 2)
    return v * mask + mean * (1 - mask)

  u = 1 + l * w

  def body_fn(_, vpq):
    xk, pk, qk = vpq
    yk = jnp.pad(
        f(xk[:-1] + pk[:-1], u[:-1]), (0, 1), constant_values=(xk[-1] + pk[-1])
    )
    p = xk + pk - yk
    v = jnp.pad(
        f(yk[1:] + qk[1:], u[1:]), (1, 0), constant_values=yk[0] + qk[0]
    )
    q = yk + qk - v
    return v, p, q

  # Ensure that the length is odd.
  (n,) = s.shape
  if n % 2 == 0:
    minv = jax.lax.stop_gradient(s.min() - 1)
    s = jnp.pad(s, (0, 1), constant_values=minv)
    u = jnp.pad(u, (0, 1), constant_values=0.0)

  v = s.copy()
  p = jnp.zeros(s.shape)
  q = jnp.zeros(s.shape)
  vpq = (v, p, q)
  v, p, q = jax.lax.fori_loop(0, num_iter // 2, body_fn, vpq)
  sol = v

  return sol if n % 2 != 0 else sol[:-1]


def _jvp_isotonic_mag(solution, vector, w, l, eps=1e-4):
  """Jacobian at solution of the isotonic regression times vector product."""
  x = solution
  mask = jnp.pad(jnp.absolute(jnp.diff(x)) <= eps, (1, 0))
  ar = jnp.arange(x.size)

  inds_start = jnp.where(mask == 0, ar, +jnp.inf).sort()
  u = 1 + l * w
  one_hot_start = jax.nn.one_hot(inds_start, len(vector))
  a = _cumsum_einsum(one_hot_start)
  a = jnp.append(jnp.diff(a[::-1], axis=0)[::-1], a[-1].reshape(1, -1), axis=0)
  return (
      ((a.T * (a @ (vector * u))).T) / ((a * u).sum(1, keepdims=True) + 1e-8)
  ).sum(0)


@isotonic_dykstra_mag.defjvp
def _isotonic_dykstra_mag_jvp(w, l, num_iter, primals, tangents):
  """Jacobian at solution of the isotonic regression times vector product."""
  (s,) = primals
  (vector,) = tangents
  primal_out = isotonic_dykstra_mag(s, w, l=l, num_iter=num_iter)
  tangent_out = _jvp_isotonic_mag(primal_out, vector, w, l)
  return primal_out, tangent_out
