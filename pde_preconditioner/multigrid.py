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

# pylint: skip-file
import functools
import jax
import jax.numpy as np
from jax import random
import jax.ops
from jax.tree_util import Partial
from . import equations
from . import gmres

# helper functions


def helmholtzR(x, k, h, ratio):
  return equations.helmholtz(x, k, h, ratio, equations.make_mask,
                             equations.make_mask_dual)


def helmholtzL(x, k, h, ratio):
  return equations.helmholtz(x, k, h, ratio, equations.make_mask_L,
                             equations.make_mask_L_dual)


@jax.jit
def body_fun2_helmholtzR(tup, i):
  f, x, h, omega, k, ratio = tup
  return (f, (omega * -1/(4/h**2-k**2) * (f - helmholtzR(x, k, h, ratio) - \
      (4/h**2-k**2) * x) + (1 - omega) * x), h, omega, k, ratio), 0


@jax.jit
def body_fun2_helmholtzL(tup, i):
  f, x, h, omega, k, ratio = tup
  return (f, (omega * -1/(4/h**2-k**2) * (f - helmholtzL(x, k, h, ratio) - \
      (4/h**2-k**2) * x) + (1 - omega) * x), h, omega, k, ratio), 0


@functools.partial(jax.jit, static_argnums=(5,))
def smoothing_helmholtz(f, h, x, k, aspect_ratio=1.0, shapeL='R'):
  # f,x are discretized on a given grid
  # weighted jacobi
  num_iter = 3
  omega = 2 / 3
  if shapeL == 'R':
    carry, _ = jax.lax.scan(body_fun2_helmholtzR,
                            (f, x, h, omega, k, aspect_ratio),
                            np.arange(0, num_iter))
  elif shapeL == 'L':
    carry, _ = jax.lax.scan(body_fun2_helmholtzL,
                            (f, x, h, omega, k, aspect_ratio),
                            np.arange(0, num_iter))
  f, x, h, omega, k, aspect_ratio = carry
  return x


@jax.jit
def restriction(r):
  # full weighting restriction
  # assuming n is divisible by 2
  rsmall = r
  kernel = 1 / 16 * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
  lhs = rsmall[np.newaxis, np.newaxis, Ellipsis]
  rhs = kernel[np.newaxis, np.newaxis, Ellipsis]
  result = jax.lax.conv(
      lhs, rhs, window_strides=(2,) * rsmall.ndim, padding='VALID')
  squeezed = np.squeeze(result, axis=(0, 1))
  return squeezed


@jax.jit
def prolongation(r):
  # linear interpolation
  r = np.pad(r, 1)
  n = r.shape[0]
  if r.ndim == 1:
    kernel = np.array([1 / 2, 1 / 2])
    lhs = r[np.newaxis, np.newaxis, Ellipsis]
    rhs = kernel[np.newaxis, np.newaxis, Ellipsis]
    result1 = jax.lax.conv(
        lhs, rhs, window_strides=(1,) * r.ndim, padding='VALID')
    squeezed = np.squeeze(result1, axis=(0, 1))
    answer = np.zeros(n * 2 - 1)
    answer = answer.at[np.arange(1, stop=n * 2 - 2, step=2)].set(squeezed)
    answer = answer.at[np.arange(0, stop=n * 2 - 1, step=2)].set(r)
  if r.ndim == 2:
    kernel1 = np.array([[1 / 2, 1 / 2]])
    kernel2 = np.array([[1 / 2], [1 / 2]])
    kernel3 = np.array([[1 / 4, 1 / 4], [1 / 4, 1 / 4]])
    lhs = r[np.newaxis, np.newaxis, Ellipsis]
    rhs1 = kernel1[np.newaxis, np.newaxis, Ellipsis]
    rhs2 = kernel2[np.newaxis, np.newaxis, Ellipsis]
    rhs3 = kernel3[np.newaxis, np.newaxis, Ellipsis]
    result1 = jax.lax.conv(
        lhs, rhs1, window_strides=(1,) * r.ndim, padding='VALID')
    result2 = jax.lax.conv(
        lhs, rhs2, window_strides=(1,) * r.ndim, padding='VALID')
    result3 = jax.lax.conv(
        lhs, rhs3, window_strides=(1,) * r.ndim, padding='VALID')
    squeezed1 = np.squeeze(result1, axis=(0, 1))
    squeezed2 = np.squeeze(result2, axis=(0, 1))
    squeezed3 = np.squeeze(result3, axis=(0, 1))
    answer = np.zeros((n * 2 - 1, n * 2 - 1))
    answer = answer.at[::2, 1::2].set(squeezed1)
    answer = answer.at[1::2, ::2].set(squeezed2)
    answer = answer.at[1::2, 1::2].set(squeezed3)
    answer = answer.at[::2, ::2].set(r)
  return answer[1:-1, 1:-1]


@functools.partial(
    jax.jit, static_argnums=(
        2,
        3,
    ))
def _V_Cycle(x, f, num_cycle, shapebc='R', k=0, aspect_ratio=1.0):
  # https://en.wikipedia.org/wiki/Multigrid_method
  # Pre-Smoothing
  # bc are not included

  h = 1.0 / (x.shape[0] + 1)
  if shapebc == 'R':
    mask_f = equations.make_mask
    mask_f_dual = equations.make_mask_dual
  elif shapebc == 'L':
    mask_f = equations.make_mask_L
    mask_f_dual = equations.make_mask_L_dual
  x = smoothing_helmholtz(f, h, x, k, aspect_ratio, shapebc)

  # Compute Residual Errors

  # no bc here because we assume they are 0
  r = f - equations.helmholtz(
      x,
      k,
      step=h,
      aspect_ratio=aspect_ratio,
      mask_f=mask_f,
      mask_f_dual=mask_f_dual)
  # Restriction from h to 2h
  rhs = restriction(r)
  eps = np.zeros(rhs.shape)
  mask = mask_f(eps.shape[0], aspect_ratio)
  eps = np.multiply(eps, mask)
  # stop recursion after 3 cycles
  if num_cycle == 3:

    eps = smoothing_helmholtz(rhs, 2 * h, eps, k, aspect_ratio, shapebc)
  else:
    eps = _V_Cycle(
        eps, rhs, num_cycle + 1, shapebc, k=k, aspect_ratio=aspect_ratio)

  # Prolongation and Correction
  x = x + prolongation(eps)
  mask = mask_f(x.shape[0], aspect_ratio)
  x = np.multiply(x, mask)
  # Post-Smoothing
  x = smoothing_helmholtz(f, h, x, k, aspect_ratio, shapebc)

  return x


def _V_Cycle_GMRES(x, f, num_cycle, shapebc='R', k=0, aspect_ratio=1.0):
  # https://en.wikipedia.org/wiki/Multigrid_method
  # Pre-Smoothing
  # bc are not included

  h = 1.0 / (x.shape[0] + 1)
  if shapebc == 'R':
    mask_f = equations.make_mask
    mask_f_dual = equations.make_mask_dual
  elif shapebc == 'L':
    mask_f = equations.make_mask_L
    mask_f_dual = equations.make_mask_L_dual

  r = f - equations.helmholtz(
      x,
      k,
      step=h,
      aspect_ratio=aspect_ratio,
      mask_f=mask_f,
      mask_f_dual=mask_f_dual)

  new_matvec = lambda z: equations.helmholtz(
      z.reshape(x.shape),
      k,
      step=h,
      aspect_ratio=aspect_ratio,
      mask_f=mask_f,
      mask_f_dual=mask_f_dual).ravel()

  A = Partial(new_matvec)
  x = x + gmres.gmres(A, r.ravel(), n=5).reshape(x.shape)

  # Compute Residual Errors

  # no bc here because we assume they are 0
  r = f - equations.helmholtz(
      x,
      k,
      step=h,
      aspect_ratio=aspect_ratio,
      mask_f=mask_f,
      mask_f_dual=mask_f_dual)
  # Restriction from h to 2h
  rhs = restriction(r)
  eps = np.zeros(rhs.shape)
  mask = mask_f(eps.shape[0], aspect_ratio)
  eps = np.multiply(eps, mask)
  # stop recursion after 3 cycles
  if num_cycle == 3:
    r = rhs - equations.helmholtz(
        eps,
        k,
        step=2 * h,
        aspect_ratio=aspect_ratio,
        mask_f=mask_f,
        mask_f_dual=mask_f_dual)

    new_matvec1 = lambda z: equations.helmholtz(
        z.reshape(eps.shape),
        k,
        step=2 * h,
        aspect_ratio=aspect_ratio,
        mask_f=mask_f,
        mask_f_dual=mask_f_dual).ravel()

    A1 = Partial(new_matvec1)
    eps = eps + gmres.gmres(A1, r.ravel(), n=5).reshape(eps.shape)
  else:
    eps = _V_Cycle(
        eps, rhs, num_cycle + 1, shapebc, k=k, aspect_ratio=aspect_ratio)

  # Prolongation and Correction
  x = x + prolongation(eps)
  mask = mask_f(x.shape[0], aspect_ratio)
  x = np.multiply(x, mask)
  # Post-Smoothing
  r = f - equations.helmholtz(
      x,
      k,
      step=h,
      aspect_ratio=aspect_ratio,
      mask_f=mask_f,
      mask_f_dual=mask_f_dual)
  x = x + gmres.gmres(A, r.ravel(), n=5).reshape(x.shape)
  return x
