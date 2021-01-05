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

"""Functions for computing vector Lp norms and dual norms."""

import jax
from jax import numpy as jnp
import numpy as np
import scipy.linalg


def norm_type_to_ord(norm_type):
  if norm_type == 'linf':
    norm_ord = jnp.inf
  else:
    norm_ord = float(norm_type[1:])
  return norm_ord


def norm_type_to_ord_dual(norm_type):
  """Return the dual norm as a float."""
  if norm_type == 'linf':
    norm_ord = 1
  elif norm_type == 'l1':
    norm_ord = jnp.inf
  else:
    # 1/p + 1/q = 1 => q = p/(p-1)
    p = float(norm_type[1:])
    norm_ord = p / (p - 1)
  return norm_ord


def norm_type_dual(norm_type):
  """Return the dual norm as string."""
  if norm_type == 'linf':
    norm_dual = 'l1'
  elif norm_type == 'l1':
    norm_dual = 'linf'
  elif norm_type == 'dft1':
    norm_dual = 'dftinf'
  elif norm_type == 'dftinf':
    norm_dual = 'dft1'
  else:
    # 1/p + 1/q = 1 => q = p/(p-1)
    p = float(norm_type[1:])
    norm_ord = p / (p - 1)
    norm_dual = 'l%g' % norm_ord
  return norm_dual


@jax.custom_jvp
def float_power(x, p):
  return jnp.float_power(x, p)


@float_power.defjvp
def float_power_jvp(primals, tangents):
  # TODO(fartash): Still gives nans
  x, p = primals
  x_dot, p_dot = tangents
  ans = float_power(x, p)
  fdx_xdot = p * jnp.float_power(x, p - 1) * x_dot
  fdp_pdot = jnp.log(jnp.maximum(1e-7, x)) * jnp.float_power(x, p) * p_dot
  ans_dot = fdx_xdot + fdp_pdot
  return ans, ans_dot


def norm_f(x, norm_type):
  """Differentiable implementation of norm handling any Lp norm."""
  if norm_type == 'l2':
    return jnp.float_power(jnp.maximum(1e-7, jnp.sum(x**2)), 0.5)
  if norm_type == 'l1':
    return jnp.sum(jnp.abs(x))
  if norm_type == 'linf':
    return jnp.max(jnp.abs(x))
  if norm_type == 'dft1':
    dft = scipy.linalg.dft(x.shape[0]) / jnp.sqrt(x.shape[0])
    return jnp.sum(jnp.abs(dft @ x))
  if norm_type == 'dftinf':
    dft = scipy.linalg.dft(x.shape[0]) / jnp.sqrt(x.shape[0])
    return jnp.max(jnp.abs(dft @ x))
  p = float(norm_type[1:].split('_')[0])
  q = float(norm_type[1:].split('_')[1]) if '_' in norm_type else 1
  # root(p) becomes gives nan because it's not differentiable at 0
  # The subgradient of the norm is 0 but the directional derivative is inf
  # because 1/x^(p-1) is inf at 0
  # https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html#Enforcing-a-differentiation-convention
  if q != 1:
    return jnp.float_power(
        jnp.maximum(1e-7, jnp.sum(jnp.float_power(jnp.abs(x), p))), 1 / q)
  return jnp.sum(jnp.float_power(jnp.abs(x), p))


def norm_projection(delta, norm_type, eps=1.):
  """Projects to a norm-ball centered at 0.

  Args:
    delta: An array of size dim x num containing vectors to be projected.
    norm_type: A string denoting the type of the norm-ball.
    eps: A float denoting the radius of the norm-ball.

  Returns:
    An array of size dim x num, the projection of delta to the norm-ball.
  """
  if norm_type == 'linf':
    delta = jnp.clip(delta, -eps, eps)
  elif norm_type == 'l2':
    # Euclidean projection: divide all elements by a constant factor
    avoid_zero_div = 1e-12
    norm2 = jnp.sum(delta**2, axis=0, keepdims=True)
    norm = jnp.sqrt(jnp.maximum(avoid_zero_div, norm2))
    # only decrease the norm, never increase
    delta = delta * jnp.clip(eps / norm, a_min=None, a_max=1)
  elif norm_type == 'l1':
    delta = l1_unit_projection(delta / eps) * eps
  elif norm_type == 'dftinf':
    # transform to DFT, project using known projections, then transform back
    dft = np.matrix(scipy.linalg.dft(delta.shape[0]) / np.sqrt(delta.shape[0]))
    dftxdelta = dft @ delta
    # L2 projection of each coordinate to the L2-ball in the complex plane
    dftz = dftxdelta.reshape(1, -1)
    dftz = jnp.concatenate((jnp.real(dftz), jnp.imag(dftz)), axis=0)
    dftz = norm_projection(dftz, 'l2', eps)
    dftz = (dftz[0, :] + 1j * dftz[1, :]).reshape(delta.shape)
    # project back from DFT
    delta = dft.getH() @ dftz
    # Projected vector can have an imaginary part
    delta = jnp.real(delta)
  return delta


def l1_unit_projection(x):
  """Euclidean projection to L1 unit ball i.e. argmin_{|v|_1<= 1} |x-v|_2.

  Args:
    x: An array of size dim x num.

  Returns:
    An array of size dim x num, the projection to the unit L1 ball.
  """
  # https://dl.acm.org/citation.cfm?id=1390191
  xshape = x.shape
  if len(x.shape) == 1:
    x = x.reshape(-1, 1)
  eshape = x.shape
  v = jnp.abs(x.reshape((-1, eshape[-1])))
  u = jnp.sort(v, axis=0)
  u = u[::-1, :]  # descending
  arange = (1 + jnp.arange(eshape[0])).reshape((-1, 1))
  usum = (jnp.cumsum(u, axis=0) - 1) / arange
  rho = jnp.max(((u - usum) > 0) * arange - 1, axis=0, keepdims=True)
  thx = jnp.take_along_axis(usum, rho, axis=0)
  w = (v - thx).clip(a_min=0)
  w = jnp.where(jnp.linalg.norm(v, ord=1, axis=0, keepdims=True) > 1, w, v)
  x = w.reshape(eshape) * jnp.sign(x)
  return x.reshape(xshape)


def get_prox_op(norm_type):
  """Proximal operator of norm-ball projections."""
  if norm_type == 'l1':
    prox_op = lambda v, lam: jnp.sign(v) * jnp.maximum(0, jnp.abs(v) - lam)
  elif norm_type == 'l2':
    # The prox of L2 (not squared)
    prox_op = lambda v, lam: v * jnp.maximum(0, 1 - lam / jnp.sqrt(  # pylint: disable=g-long-lambda
        jnp.sum(v**2)))
  elif norm_type == 'linf':
    prox_op = lambda v, lam: v - lam * l1_unit_projection(v / lam)
  else:
    dual_norm = norm_type_dual(norm_type)
    prox_op = lambda v, lam: v - lam * norm_projection(  # pylint: disable=g-long-lambda
        v / lam, dual_norm, eps=1)
  return prox_op
