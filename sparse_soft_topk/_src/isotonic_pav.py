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

import warnings

import jax
import jax.numpy as jnp
import numba
import numpy as np

try:
  from numba import njit

  NUMBA_AVAILABLE = True
except ImportError:
  NUMBA_AVAILABLE = False
  # If Numba is not available, we define a dummy 'njit' function.

  def njit(func):
    return func


EPS = 1e-6


def _bisect(low, high, y_s_np, w_s_np, q, l, max_iter):
  """Finds the root by bisection."""
  _ = 0
  # bisection
  while _ < max_iter:
    midpoint = (low + high) / 2.0
    a = (
        np.sign(low - y_s_np) * ((np.absolute(low - y_s_np)) ** (q - 1))
        + (l ** (q - 1)) * w_s_np
    ).sum()
    b = (
        np.sign(midpoint - y_s_np)
        * ((np.absolute(midpoint - y_s_np)) ** (q - 1))
        + (l ** (q - 1)) * w_s_np
    ).sum()
    if a * b > 0:
      low = midpoint
    else:
      high = midpoint
    _ += 1
  return midpoint


def _bisect_mag(low, high, y_s_np, w_s_np, q, l, max_iter):
  """Finds the root by bisection."""
  _ = 0
  # bisection
  while _ < max_iter:
    midpoint = (low + high) / 2.0
    a = (
        np.sign(low - y_s_np) * ((np.absolute(low - y_s_np)) ** (q - 1))
        + (l ** (q - 1)) * w_s_np * low
    ).sum()
    b = (
        np.sign(midpoint - y_s_np)
        * ((np.absolute(midpoint - y_s_np)) ** (q - 1))
        + (l ** (q - 1)) * w_s_np * midpoint
    ).sum()
    if a * b > 0:
      low = midpoint
    else:
      high = midpoint
    _ += 1
  return midpoint


@njit
def _f(a, b, c, d):
  return ((3 * c / a) - (b / a) ** 2) / 3


@njit
def _g(a, b, c, d):
  return (2 * (b / a) ** 3 - (9 * b * c) / a**2 + 27 * d / a) / 27


@njit
def _h(f, g):
  return g**2 / 4 + f**3 / 27


@njit
def _solve_real_root(a, b, c, d):
  """Returns one real root of ax^3 + bx^2 + cx + d.

  Implementation of https://www.1728.org/cubic2.htm
  """
  # Firts, we define useful variables for computing the roots
  f, g = _f(a, b, c, d), _g(a, b, c, d)
  h = _h(f, g)

  if h < 0:  # 3 different real roots.
    # This should never happen as we are minimizing
    # a quartic, so there is only one real root.
    i = ((g**2 / 4) - h) ** (1 / 2)
    j = np.sign(i) * np.absolute(i) ** (1 / 3)
    k = np.arccos(-g / (2 * i))
    return 2 * j * np.cos(k / 3) - (b / (3 * a))

  elif h > 0:  # only one real root.
    r = -(g / 2) + h ** (1 / 2)
    s = np.sign(r) * np.absolute(r) ** (1 / 3)
    t = -(g / 2) - h ** (1 / 2)
    u = np.sign(t) * np.absolute(t) ** (1 / 3)
    return (s + u) - (b / (3 * a))

  else:  # all roots are real and equal.
    d_a = d / a
    return -(np.sign(d_a) * np.absolute(d_a) ** (1 / 3))


@njit(
    numba.float32[::1](
        numba.types.Array(numba.types.float32, 1, "C", readonly=True))
)
def _isotonic_l2_mask_pav_numba_1d(s):
  n = s.shape[0]
  s = s.astype(np.float64)
  target = np.arange(n)
  c = np.ones(n)
  sums = np.zeros(n)
  sol = np.zeros(n)

  # target describes a list of blocks.  At any time, if [i..j] (inclusive) is
  # an active block, then target[i] := j and target[j] := i.

  for i in range(n):
    sol[i] = s[i]
    sums[i] = s[i]

  i = 0
  while i < n:
    k = target[i] + 1
    if k == n:
      break
    if sol[i] > sol[k]:
      i = k
      continue
    sum_s = sums[i]
    sum_c = c[i]
    while True:
      # We are within an increasing subsequence.
      prev_s = sol[k]
      sum_s += sums[k]
      sum_c += c[k]
      k = target[k] + 1
      if k == n or prev_s > sol[k]:
        # Non-singleton increasing subsequence is finished,
        # update first entry.
        sol[i] = sum_s / sum_c
        sums[i] = sum_s
        c[i] = sum_c
        target[i] = k - 1
        target[k - 1] = i
        if i > 0:
          # Backtrack if we can.  This makes the algorithm
          # single-pass and ensures O(n) complexity.
          i = target[i - 1]
        # Otherwise, restart from the same point.
        break

  # Reconstruct the solution.
  i = 0
  while i < n:
    k = target[i] + 1
    sol[i + 1 : k] = sol[i]
    i = k
  return sol.astype(np.float32)


@njit(
    numba.float32[::1](
        numba.types.Array(numba.types.float32, 1, "C", readonly=True),
        numba.types.Array(numba.types.float32, 1, "C", readonly=True),
        numba.types.float32,
    )
)
def _isotonic_l4_mask_pav_numba_1d(s, w, l=1e-1):
  """Solves an isotonic regression problem using PAV."""
  n = s.shape[0]
  s = s.astype(np.float64)
  # target describes a list of blocks.  At any time, if [i..j] (inclusive) is
  # an active block, then target[i] := j and target[j] := i.
  target = np.arange(n)
  sums = np.zeros(n)
  sol = np.zeros(n)
  a_3 = np.ones(n)
  a_2 = -3 * s
  a_1 = 3 * s**2
  a_0 = -(s**3) + l**3 * w
  sums_a_3 = np.zeros(n)
  sums_a_2 = np.zeros(n)
  sums_a_1 = np.zeros(n)
  sums_a_0 = np.zeros(n)
  # avoid extra computation if w is binary
  b = np.count_nonzero((w != 0) & (w != 1)) == 0
  if b:
    for i in range(n):
      sol[i] = s[i] - l * w[i]
      sums_a_3[i] = a_3[i]
      sums_a_2[i] = a_2[i]
      sums_a_1[i] = a_1[i]
      sums_a_0[i] = a_0[i]
  else:
    for i in range(n):
      sol[i] = s[i] - l * w[i] ** (1 / (3))
      sums_a_3[i] = a_3[i]
      sums_a_2[i] = a_2[i]
      sums_a_1[i] = a_1[i]
      sums_a_0[i] = a_0[i]
  i = 0
  while i < n:
    p = target[i] + 1
    if p == n:
      break
    if sol[i] > sol[p]:
      i = p
      continue
    sum_a_3 = sums_a_3[i]
    sum_a_2 = sums_a_2[i]
    sum_a_1 = sums_a_1[i]
    sum_a_0 = sums_a_0[i]
    while True:
      # We are within an increasing subsequence.
      prev_s = sol[p]
      sum_a_3 += sums_a_3[p]
      sum_a_2 += sums_a_2[p]
      sum_a_1 += sums_a_1[p]
      sum_a_0 += sums_a_0[p]
      p = target[p] + 1
      if p == n or prev_s > sol[p]:
        # Non-singleton increasing subsequence is finished,
        # update first entry.

        sol[i] = _solve_real_root(sum_a_3, sum_a_2, sum_a_1, sum_a_0)
        sums_a_3[i] = sum_a_3
        sums_a_2[i] = sum_a_2
        sums_a_1[i] = sum_a_1
        sums_a_0[i] = sum_a_0
        target[i] = p - 1
        target[p - 1] = i
        if i > 0:
          # Backtrack if we can.  This makes the algorithm
          # single-pass and ensures O(n) complexity.
          i = target[i - 1]
        # Otherwise, restart from the same point.
        break

  # Reconstruct the solution.
  i = 0
  while i < n:
    p = target[i] + 1
    sol[i + 1 : p] = sol[i]
    i = p
  return sol.astype(np.float32)


def _isotonic_lp_mask_pav_1d(s, w, l=1e-1, p=4 / 3, bisect_max_iter=50):
  """Solves an isotonic regression problem using PAV."""
  n = s.shape[0]
  s = s.astype(np.float64)
  target = np.arange(n)
  s_list = []
  w_list = []
  sol = np.zeros(n)
  q = p / (p - 1)

  # target describes a list of blocks.  At any time, if [i..j] (inclusive) is
  # an active block, then target[i] := j and target[j] := i.

  for i in range(n):
    sol[i] = s[i] - l * w[i] ** (1 / (q - 1))
    s_list.append([s[i]])
    w_list.append([w[i]])

  i = 0
  while i < n:
    j = target[i] + 1
    if j == n:
      break
    if sol[i] > sol[j]:
      i = j
      continue
    s_s = s_list[i]
    w_s = w_list[i]
    while True:
      # We are within an increasing subsequence.
      prev_s = sol[j]
      s_s += s_list[j]
      w_s += w_list[j]
      j = target[j] + 1
      if j == n or prev_s > sol[j]:
        # Non-singleton increasing subsequence is finished,
        # update first entry.
        s_s_np = np.array(s_s, dtype=np.float64)
        w_s_np = np.array(w_s, dtype=np.float64)
        low = np.min(s_s_np) - l * np.max(w_s_np) ** (1 / (q - 1))
        high = s.max()
        sol[i] = _bisect(low, high, s_s_np, w_s_np, q, l, bisect_max_iter)
        s_list[i] = s_s
        w_list[i] = w_s
        target[i] = j - 1
        target[j - 1] = i
        if i > 0:
          # Backtrack if we can.  This makes the algorithm
          # single-pass and ensures O(n) complexity.
          i = target[i - 1]
        # Otherwise, restart from the same point.
        break

  # Reconstruct the solution.
  i = 0
  while i < n:
    j = target[i] + 1
    sol[i + 1 : j] = sol[i]
    i = j
  return sol.astype(np.float32)

@njit(
    numba.float32[:, ::1](
        numba.types.Array(numba.types.float32, 2, "C", readonly=True),
    ), parallel=True
)
def _isotonic_l2_mask_pav_numba_2d(s):
  """Solves an isotonic regression problem using PAV."""
  batch_shape = s.shape[:-1]
  s = s.reshape((-1, s.shape[-1]))
  y = np.zeros_like(s)
  for i in numba.prange(s.shape[0]):
    y[i] = _isotonic_l2_mask_pav_numba_1d(s[i])
  y = y.reshape(batch_shape + (-1,))
  return y


def _isotonic_l2_mask_pav_numba(s):
  """Solves an isotonic regression problem using PAV."""
  if s.ndim == 1:
    return _isotonic_l2_mask_pav_numba_1d(s)
  else:
    return _isotonic_l2_mask_pav_numba_2d(s)


@njit(
    numba.float32[:, ::1](
        numba.types.Array(numba.types.float32, 2, "C", readonly=True),
        numba.types.Array(numba.types.float32, 1, "C", readonly=True),
        numba.types.float32,
    ), parallel=True
)
def _isotonic_l4_mask_pav_numba_2d(s, w, l=1e-1):
  """Solves an isotonic regression problem using PAV."""
  batch_shape = s.shape[:-1]
  s = s.reshape((-1, s.shape[-1]))
  y = np.zeros_like(s)
  for i in numba.prange(s.shape[0]):
    y[i] = _isotonic_l4_mask_pav_numba_1d(s[i], w, l=l)
  y = y.reshape(batch_shape + (-1,))
  return y


def _isotonic_l4_mask_pav_numba(s, w, l=1e-1):
  """Solves an isotonic regression problem using PAV."""
  if s.ndim == 1:
    return _isotonic_l4_mask_pav_numba_1d(s, w, l=l)
  else:
    return _isotonic_l4_mask_pav_numba_2d(s, w, l=l)


def _isotonic_lp_mask_pav_2d(s, w, l=1e-1, p=4 / 3, bisect_max_iter=50):
  """Solves an isotonic regression problem using PAV."""
  batch_shape = s.shape[:-1]
  s = s.reshape((-1, s.shape[-1]))
  y = np.zeros_like(s)
  for i in range(s.shape[0]):
    y[i] = _isotonic_lp_mask_pav_1d(
        s[i], w, l=l, p=p, bisect_max_iter=bisect_max_iter
    )
  y = y.reshape(batch_shape + (-1,))
  return y


def _isotonic_lp_mask_pav(s, w, l=1e-1, p=4 / 3, bisect_max_iter=50):
  """Solves an isotonic regression problem using PAV."""
  if s.ndim == 1:
    return _isotonic_lp_mask_pav_1d(s, w, l=l, p=p, bisect_max_iter=50)
  else:
    return _isotonic_lp_mask_pav_2d(s, w, l=l, p=p, bisect_max_iter=50)


def _isotonic_mask_pav(s, w, l=1e-1, p=4 / 3, bisect_max_iter=50):
  """Solves an isotonic regression problem using PAV."""
  l = float(l)
  p = float(p)
  if abs(p - 2) < EPS:
    return _isotonic_l2_mask_pav_numba(s - l * w)
  elif abs(p - 4 / 3) < EPS:
    return _isotonic_l4_mask_pav_numba(s, w, l=l)
  else:
    return _isotonic_lp_mask_pav(
        s, w, l=l, p=p, bisect_max_iter=bisect_max_iter
    )


@jax.custom_vjp
def isotonic_mask_pav(s, w, l=1e-1, p=4 / 3, bisect_max_iter=50):
  """Solves an isotonic regression problem using PAV.

  It solves argmin_{v_1 >= ... >= v_n} 1/(q * l **(q-1)) * ||s - v||_q^q
  + sum_i w_i v_i.
  with q = p / (p-1)

  Args:
    s: input to isotonic regression, a 1d-array or a 2d-array. If 2d array, then
      the function is applied along the second axis.
    w: a 1d-array with positive entries
    l: regularization parameter
    p: a float between 1 and +infinity
    bisect_max_iter: int, number of iterations in the bisection (only used if p
      != 2 and p != 4/3)

  Returns:
    sol: the solution an array of the same size as y
  """
  if not NUMBA_AVAILABLE:
    warnings.warn(
        "Numba could not be imported. Code will run much more slowly."
        " To install, run 'pip install numba'."
    )
  # Define the expected shape & dtype of output.
  shape_dtype = jax.ShapeDtypeStruct(
      shape=s.shape,
      dtype=s.dtype,
      sharding=jax.sharding.NamedSharding(
          jax.sharding.Mesh(jax.devices(), "x"), jax.sharding.PartitionSpec()),
  )
  sol = jax.pure_callback(
      _isotonic_mask_pav,
      shape_dtype,
      s,
      w,
      l,
      p,
      bisect_max_iter,
      vectorized=False,
  )
  return sol


@njit
def _partition(solution):
  """Returns partition corresponding to solution."""
  sizes = np.zeros(len(solution))
  sizes[0] = 1
  part_idx = 0
  for i in range(1, len(solution)):
    if abs(solution[i] - solution[i - 1]) <= 1e-9:
      sizes[part_idx] += 1
    else:
      part_idx += 1
      sizes[part_idx] += 1
  return sizes.astype(np.int32)


@njit(
    numba.float32[::1](
        numba.types.Array(numba.types.float32, 1, "C", readonly=True),
        numba.types.Array(numba.types.float32, 1, "C", readonly=True),
        numba.types.Array(numba.types.float32, 1, "C", readonly=True),
    ), parallel=True
)
def _vjp_mask_numba_l2(s, solution, vector):
  start = 0
  return_value = np.zeros_like(solution)
  for size in _partition(solution):
    if size > 0:
      end = start + size
      val = 0
      for i in range(start, end):
        val = val + vector[i]
      for i in range(start, end):
        return_value[i] = val / size
      start = end
  return return_value


@njit(
    numba.float32[::1](
        numba.types.Array(numba.types.float32, 1, "C", readonly=True),
        numba.types.float32,
        numba.types.Array(numba.types.float32, 1, "C", readonly=True),
        numba.types.Array(numba.types.float32, 1, "C", readonly=True),
    )
)
def _vjp_mask_numba_general(s, p, solution, vector):
  start = 0
  return_value = np.zeros_like(solution)
  q = p / (p - 1)
  diff = (np.absolute(solution - s)) ** (q - 2)
  for size in _partition(solution):
    if size > 0:
      end = start + size
      den = 1e-15
      v_sum = 0
      for i in range(start, end):
        den = den + diff[i]
        v_sum = v_sum + vector[i]

      for i in range(start, end):
        return_value[i] = diff[i] * v_sum / den
      start = end
  return return_value


def _vjp_mask_numba_1d(s, p, solution, vector):
  p = float(p)
  if abs(p - 2) < EPS:
    return _vjp_mask_numba_l2(s, solution, vector)
  else:
    return _vjp_mask_numba_general(s, p, solution, vector)


@njit(
    numba.float32[:, ::1](
        numba.types.Array(numba.types.float32, 2, "C", readonly=True),
        numba.types.Array(numba.types.float32, 2, "C", readonly=True),
        numba.types.Array(numba.types.float32, 2, "C", readonly=True),
    )
)
def _vjp_mask_numba_l2_2d(s, solution, vector):
  batch_shape = s.shape[:-1]
  s = s.reshape((-1, s.shape[-1]))
  y = np.zeros_like(s)
  for i in numba.prange(s.shape[0]):
    y[i] = _vjp_mask_numba_l2(s[i], solution[i], vector[i])
  y = y.reshape(batch_shape + (-1,))
  return y

@njit(
    numba.float32[:, ::1](
        numba.types.Array(numba.types.float32, 2, "C", readonly=True),
        numba.types.float32,
        numba.types.Array(numba.types.float32, 2, "C", readonly=True),
        numba.types.Array(numba.types.float32, 2, "C", readonly=True),
    ), parallel=True
)
def _vjp_mask_numba_general_2d(s, p, solution, vector):
  batch_shape = s.shape[:-1]
  s = s.reshape((-1, s.shape[-1]))
  y = np.zeros_like(s)
  for i in numba.prange(s.shape[0]):
    y[i] = _vjp_mask_numba_general(s[i], p, solution[i], vector[i])
  y = y.reshape(batch_shape + (-1,))
  return y


def _vjp_mask_numba_2d(s, p, solution, vector):
  p = float(p)
  if abs(p - 2) < EPS:
    return _vjp_mask_numba_l2_2d(s, solution, vector)
  else:
    return _vjp_mask_numba_general_2d(s, p, solution, vector)


def _vjp_mask_numba(s, p, solution, vector):
  if s.ndim == 1:
    return _vjp_mask_numba_1d(s, p, solution, vector)
  else:
    return _vjp_mask_numba_2d(s, p, solution, vector)


def _isotonic_mask_pav_fwd(s, w, l=1e-1, p=4 / 3, bisect_max_iter=50):
  # Returns primal output and residuals to be used in backward pass by f_bwd.
  sol = isotonic_mask_pav(s, w, l=l, p=p, bisect_max_iter=bisect_max_iter)
  return sol, (s, p, sol)


def _isotonic_mask_pav_bwd(res, g, l=1e-1, p=4 / 3, bisect_max_iter=50):
  s, p, sol = res  # Gets residuals computed in f_fwd
  shape_dtype = jax.ShapeDtypeStruct(shape=g.shape, dtype=sol.dtype)
  output = jax.pure_callback(
      _vjp_mask_numba, shape_dtype, s, p, sol, g, vectorized=False
  )
  return (output, None, None, None, None)


isotonic_mask_pav.defvjp(_isotonic_mask_pav_fwd, _isotonic_mask_pav_bwd)


@njit(
    numba.float32[::1](
        numba.types.Array(numba.types.float32, 1, "C", readonly=True),
        numba.types.Array(numba.types.float32, 1, "C", readonly=True),
        numba.types.Array(numba.types.float32, 0, "C", readonly=True),
    )
)
def _isotonic_l2_mag_pav_numba_1d(s, w, l=1e-1):
  n = s.shape[0]
  s = s.astype(np.float64)
  target = np.arange(n)
  sums = np.zeros(n)
  sums_w = np.zeros(n)
  sol = np.zeros(n)

  # target describes a list of blocks.  At any time, if [i..j] (inclusive) is
  # an active block, then target[i] := j and target[j] := i.

  for i in range(n):
    sol[i] = s[i] / (l * w[i] + 1)
    sums[i] = s[i]
    sums_w[i] = l * w[i] + 1

  i = 0
  while i < n:
    k = target[i] + 1
    if k == n:
      break
    if sol[i] > sol[k]:
      i = k
      continue
    sum_s = sums[i]
    sum_w = sums_w[i]
    while True:
      # We are within an increasing subsequence.
      prev_s = sol[k]
      sum_s += sums[k]
      sum_w += sums_w[k]
      k = target[k] + 1
      if k == n or prev_s > sol[k]:
        # Non-singleton increasing subsequence is finished,
        # update first entry.
        sol[i] = sum_s / sum_w
        sums[i] = sum_s
        sums_w[i] = sum_w
        target[i] = k - 1
        target[k - 1] = i
        if i > 0:
          # Backtrack if we can.  This makes the algorithm
          # single-pass and ensures O(n) complexity.
          i = target[i - 1]
        # Otherwise, restart from the same point.
        break

  # Reconstruct the solution.
  i = 0
  while i < n:
    k = target[i] + 1
    sol[i + 1 : k] = sol[i]
    i = k
  return sol.astype(np.float32)


@njit
def _simple_root(a, b):  # root of (x - a)**3 = -bx
  if np.absolute(b) < 1e-10:
    return a
  else:
    num_1 = -((2 / 3) ** (1 / 3)) * b
    den_1 = (
        np.sqrt(3) * np.sqrt(27 * (a * b) ** 2 + 4 * b**3) - 9 * a * b
    ) ** (1 / 3)
    den_2 = 2 ** (1 / 3) * 3 ** (2 / 3)
    return num_1 / den_1 + den_1 / den_2 + a


@njit(
    numba.float32[::1](
        numba.types.Array(numba.types.float32, 1, "C", readonly=True),
        numba.types.Array(numba.types.float32, 1, "C", readonly=True),
        numba.types.Array(numba.types.float32, 0, "C", readonly=True),
    )
)
def _isotonic_l4_mag_pav_numba_1d(s, w, l=1e-1):
  """Solves an isotonic regression problem using PAV."""
  n = s.shape[0]
  s = s.astype(np.float64)
  target = np.arange(n)
  sol = np.zeros(n)
  a_3 = np.ones(n)
  a_2 = -3 * s
  a_1 = 3 * s**2 + l**3 * w
  a_0 = -(s**3)
  sums_a_3 = np.zeros(n)
  sums_a_2 = np.zeros(n)
  sums_a_1 = np.zeros(n)
  sums_a_0 = np.zeros(n)

  # target describes a list of blocks.  At any time, if [i..j] (inclusive) is
  # an active block, then target[i] := j and target[j] := i.

  for i in range(n):
    sol[i] = _simple_root(s[i], w[i] * (l**3))
    sums_a_3[i] = a_3[i]
    sums_a_2[i] = a_2[i]
    sums_a_1[i] = a_1[i]
    sums_a_0[i] = a_0[i]
  i = 0
  while i < n:
    p = target[i] + 1
    if p == n:
      break
    if sol[i] > sol[p]:
      i = p
      continue
    sum_a_3 = sums_a_3[i]
    sum_a_2 = sums_a_2[i]
    sum_a_1 = sums_a_1[i]
    sum_a_0 = sums_a_0[i]
    while True:
      # We are within an increasing subsequence.
      prev_s = sol[p]
      sum_a_3 += sums_a_3[p]
      sum_a_2 += sums_a_2[p]
      sum_a_1 += sums_a_1[p]
      sum_a_0 += sums_a_0[p]
      p = target[p] + 1
      if p == n or prev_s > sol[p]:
        # Non-singleton increasing subsequence is finished,
        # pdate first entry.
        sol[i] = _solve_real_root(sum_a_3, sum_a_2, sum_a_1, sum_a_0)
        sums_a_3[i] = sum_a_3
        sums_a_2[i] = sum_a_2
        sums_a_1[i] = sum_a_1
        sums_a_0[i] = sum_a_0
        target[i] = p - 1
        target[p - 1] = i
        if i > 0:
          # Backtrack if we can.  This makes the algorithm
          # single-pass and ensures O(n) complexity.
          i = target[i - 1]
        # Otherwise, restart from the same point.
        break

  # Reconstruct the solution.
  i = 0
  while i < n:
    p = target[i] + 1
    sol[i + 1 : p] = sol[i]
    i = p
  return sol.astype(np.float32)


def _isotonic_lp_mag_pav_1d(s, w, l=1e-1, p=4 / 3, bisect_max_iter=50):
  """Solves an isotonic regression problem using PAV."""
  n = s.shape[0]
  s = s.astype(np.float64)
  target = np.arange(n)
  s_list = []
  w_list = []
  sol = np.zeros(n)
  q = p / (p - 1)

  # target describes a list of blocks.  At any time, if [i..j] (inclusive) is
  # an active block, then target[i] := j and target[j] := i.
  low = np.min([np.min(s) - l * np.max(w) ** (1 / (q - 1)), 0.5])
  high = s.max()
  for i in range(n):
    sol[i] = _bisect_mag(low, high, s[i], w[i], q, l, bisect_max_iter)
    s_list.append([s[i]])
    w_list.append([w[i]])

  i = 0
  while i < n:
    j = target[i] + 1
    if j == n:
      break
    if sol[i] > sol[j]:
      i = j
      continue
    s_s = s_list[i]
    w_s = w_list[i]
    while True:
      # We are within an increasing subsequence.
      prev_s = sol[j]
      s_s += s_list[j]
      w_s += w_list[j]
      j = target[j] + 1
      if j == n or prev_s > sol[j]:
        # Non-singleton increasing subsequence is finished,
        # update first entry.
        s_s_np = np.array(s_s, dtype=np.float64)
        w_s_np = np.array(w_s, dtype=np.float64)
        low = np.min(s_s_np) - l * np.max(w_s_np) ** (1 / (q - 1))
        high = s.max()
        sol[i] = _bisect_mag(low, high, s_s_np, w_s_np, q, l, bisect_max_iter)
        s_list[i] = s_s
        w_list[i] = w_s
        target[i] = j - 1
        target[j - 1] = i
        if i > 0:
          # Backtrack if we can.  This makes the algorithm
          # single-pass and ensures O(n) complexity.
          i = target[i - 1]
        # Otherwise, restart from the same point.
        break

  # Reconstruct the solution.
  i = 0
  while i < n:
    j = target[i] + 1
    sol[i + 1 : j] = sol[i]
    i = j
  return sol.astype(np.float32)


@njit(
    numba.float32[:, ::1](
        numba.types.Array(numba.types.float32, 2, "C", readonly=True),
        numba.types.Array(numba.types.float32, 1, "C", readonly=True),
        numba.types.Array(numba.types.float32, 0, "C", readonly=True),
    )
)
def _isotonic_l2_mag_pav_numba_2d(s, w, l=1e-1):
  """Solves an isotonic regression problem using PAV."""
  batch_shape = s.shape[:-1]
  s = s.reshape((-1, s.shape[-1]))
  y = np.zeros_like(s)
  for i in numba.prange(s.shape[0]):
    y[i] = _isotonic_l2_mag_pav_numba_1d(s[i],w, l=l)
  y = y.reshape(batch_shape + (-1,))
  return y


def _isotonic_l2_mag_pav_numba(s, w, l=1e-1):
  """Solves an isotonic regression problem using PAV."""
  if s.ndim == 1:
    return _isotonic_l2_mag_pav_numba_1d(s, w, l=l)
  else:
    return _isotonic_l2_mag_pav_numba_2d(s, w, l=l)


@njit(
    numba.float32[:, ::1](
        numba.types.Array(numba.types.float32, 2, "C", readonly=True),
        numba.types.Array(numba.types.float32, 1, "C", readonly=True),
        numba.types.Array(numba.types.float32, 0, "C", readonly=True),
    )
)
def _isotonic_l4_mag_pav_numba_2d(s, w, l=1e-1):
  """Solves an isotonic regression problem using PAV."""
  batch_shape = s.shape[:-1]
  s = s.reshape((-1, s.shape[-1]))
  y = np.zeros_like(s)
  for i in numba.prange(s.shape[0]):
    y[i] = _isotonic_l4_mag_pav_numba_1d(s[i], w, l=l)
  y = y.reshape(batch_shape + (-1,))
  return y


def _isotonic_l4_mag_pav_numba(s, w, l=1e-1):
  """Solves an isotonic regression problem using PAV."""
  if s.ndim == 1:
    return _isotonic_l4_mag_pav_numba_1d(s, w, l=l)
  else:
    return _isotonic_l4_mag_pav_numba_2d(s, w, l=l)


def _isotonic_lp_mag_pav_2d(s, w, l=1e-1, p=4 / 3, bisect_max_iter=50):
  """Solves an isotonic regression problem using PAV."""
  batch_shape = s.shape[:-1]
  s = s.reshape((-1, s.shape[-1]))
  y = np.zeros_like(s)
  for i in range(s.shape[0]):
    y[i] = _isotonic_lp_mag_pav_1d(
        s[i], w, l=l, p=p, bisect_max_iter=bisect_max_iter
    )
  y = y.reshape(batch_shape + (-1,))
  return y


def _isotonic_lp_mag_pav(s, w, l=1e-1, p=4 / 3, bisect_max_iter=50):
  """Solves an isotonic regression problem using PAV."""
  if s.ndim == 1:
    return _isotonic_lp_mag_pav_1d(s, w, l=l, p=p, bisect_max_iter=50)
  else:
    return _isotonic_lp_mag_pav_2d(s, w, l=l, p=p, bisect_max_iter=50)


def _isotonic_mag_pav(s, w, l=1e-1, p=4 / 3, bisect_max_iter=50):
  """Solves an isotonic regression problem using PAV."""
  if abs(p - 2) < EPS:
    return _isotonic_l2_mag_pav_numba(s, w, l=l)
  elif abs(p - 4 / 3) < EPS:
    return _isotonic_l4_mag_pav_numba(s, w, l=l)
  else:
    return _isotonic_lp_mag_pav(
        s, w, l=l, p=p, bisect_max_iter=bisect_max_iter
    )


@jax.custom_vjp
def isotonic_mag_pav(s, w, l=1e-1, p=4 / 3, bisect_max_iter=50):
  """Solves an isotonic regression problem using PAV."""
  if not NUMBA_AVAILABLE:
    warnings.warn(
        "Numba could not be imported. Code will run much more slowly."
        " To install, run 'pip install numba'."
    )
  # Define the expected shape & dtype of output.
  shape_dtype = jax.ShapeDtypeStruct(
      shape=s.shape,
      dtype=s.dtype,
      sharding=jax.sharding.NamedSharding(
          jax.sharding.Mesh(jax.devices(), "x"), jax.sharding.PartitionSpec()),
  )
  l = jnp.array(l, float)
  sol = jax.pure_callback(
      _isotonic_mag_pav,
      shape_dtype,
      s,
      w,
      l,
      p,
      bisect_max_iter,
      vectorized=False,
  )
  return sol


@njit
def _vjp_mag_numba_l2(s, w, l, solution, vector):
  start = 0
  return_value = np.zeros_like(solution)
  for size in _partition(solution):
    if size > 0:
      end = start + size
      val = np.sum(vector[start:end]) / np.sum(1 + l * w[start:end])
      return_value[start:end] = val
      start = end
  return return_value


@njit
def _vjp_mag_numba_general(s, w, l, p, solution, vector):
  start = 0
  return_value = np.zeros_like(solution)
  q = p / (p - 1)
  for size in _partition(solution):
    if size > 0:
      end = start + size
      den = np.sum(
          (q - 1) * np.absolute((solution[start:end] - s[start:end])) ** (q - 2)
          + l ** (q - 1) * w[start:end]
      )
      v_sum = vector[start:end].sum()
      return_value[start:end] = (
          (q - 1)
          * np.absolute((solution - s)[start:end]) ** (q - 2)
          * v_sum
          / den
      )
      start = end
  return return_value


def _vjp_mag_numba_1d(s, w, l, p, solution, vector):
  if abs(p - 2) < EPS:
    return _vjp_mag_numba_l2(s, w, l, solution, vector)
  else:
    return _vjp_mag_numba_general(s, w, l, p, solution, vector)


@njit
def _vjp_mag_numba_l2_2d(s, w, l, solution, vector):
  batch_shape = s.shape[:-1]
  s = s.reshape((-1, s.shape[-1]))
  y = np.zeros_like(s)
  for i in numba.prange(s.shape[0]):
    y[i] = _vjp_mag_numba_l2(s[i], w, l, solution[i], vector[i])
  y = y.reshape(batch_shape + (-1,))
  return y


@njit
def _vjp_mag_numba_general_2d(s, w, l, p, solution, vector):
  batch_shape = s.shape[:-1]
  s = s.reshape((-1, s.shape[-1]))
  y = np.zeros_like(s)
  for i in numba.prange(s.shape[0]):
    y[i] = _vjp_mag_numba_general(s[i], w, l, p, solution[i], vector[i])
  y = y.reshape(batch_shape + (-1,))
  return y


def _vjp_mag_numba_2d(s, w, l, p, solution, vector):
  if abs(p - 2) < EPS:
    return _vjp_mag_numba_l2_2d(s, w, l, solution, vector)
  else:
    return _vjp_mag_numba_general_2d(s, w, l, p, solution, vector)


def _vjp_mag_numba(s, w, l, p, solution, vector):
  if s.ndim == 1:
    return _vjp_mag_numba_1d(s, w, l, p, solution, vector)
  else:
    return _vjp_mag_numba_2d(s, w, l, p, solution, vector)


def _isotonic_mag_pav_fwd(s, w, l=1e-1, p=4 / 3, bisect_max_iter=50):
  # Returns primal output and residuals to be used in backward pass by f_bwd.
  sol = isotonic_mag_pav(s, w, l=l, p=p, bisect_max_iter=bisect_max_iter)
  return sol, (s, w, l, p, sol)


def _isotonic_mag_pav_bwd(res, g, l=1e-1, p=4 / 3, bisect_max_iter=50):
  s, w, l, p, sol = res  # Gets residuals computed in f_fwd

  shape_dtype = jax.ShapeDtypeStruct(
      shape=jnp.broadcast_shapes(sol.shape, g.shape), dtype=sol.dtype
  )
  output = jax.pure_callback(
      _vjp_mag_numba, shape_dtype, s, w, l, p, sol, g, vectorized=False
  )
  return (output, None, None, None, None)


isotonic_mag_pav.defvjp(_isotonic_mag_pav_fwd, _isotonic_mag_pav_bwd)
