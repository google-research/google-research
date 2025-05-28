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

# pylint: skip-file
"""Mathy utility functions."""

import functools
import jax
import jax.numpy as jnp
import numpy as np

tiny_val = np.float32(np.finfo(np.float32).tiny)
min_val = np.float32(np.finfo(np.float32).min)
max_val = np.float32(np.finfo(np.float32).max)


def abs(x):
  return jnp.abs(x)


def laplace_cdf(x, beta):
  alpha = 1 / beta
  return alpha * (0.5 + 0.5 * safe_sign(x) * (jnp.exp(-jnp.abs(x) / beta) - 1))


def scaled_softplus(x, scale=100.0):
  return (1.0 / scale) * jax.nn.softplus(scale * x)


def power_3(x, exponent=3.0):
  return jnp.power(jnp.abs(x), exponent) * safe_sign(x)


def matmul(a, b):
  """jnp.matmul defaults to bfloat16, but this helper function doesn't."""
  return jnp.matmul(a, b, precision=jax.lax.Precision.HIGHEST)


@jax.custom_jvp
def plus_eps(x):
  return jnp.where(
      jnp.abs(x) < tiny_val, tiny_val, jnp.nextafter(jnp.float32(x), jnp.inf)
  )


@jax.custom_jvp
def minus_eps(x):
  return jnp.where(
      jnp.abs(x) < tiny_val, -tiny_val, jnp.nextafter(jnp.float32(x), -jnp.inf)
  )


@plus_eps.defjvp
def plus_eps_jvp(primals, tangents):
  """Make plus_eps()'s gradient a no-op (nextafter's gradient is undefined)."""
  return plus_eps(*primals), tangents[0]


@minus_eps.defjvp
def minus_eps_jvp(primals, tangents):
  """Make minus_eps()'s gradient a no-op (nextafter's gradient is undefined)."""
  return minus_eps(*primals), tangents[0]


def safe_trig_helper(x, fn, t=100 * jnp.pi):
  """Helper function used by safe_cos/safe_sin: mods x before sin()/cos()."""
  return fn(jnp.nan_to_num(jnp.where(jnp.abs(x) < t, x, x % t)))


def safe_cos(x):
  """jnp.cos() on a TPU may NaN out for large values."""
  return safe_trig_helper(x, jnp.cos)


def safe_sin(x):
  """jnp.sin() on a TPU may NaN out for large values."""
  return safe_trig_helper(x, jnp.sin)


def generate_clip_nograd_fn(a_min, a_max):
  """Generates a function that clips to [a_min, a_max] with no grad effects."""

  @jax.custom_jvp
  def clip_nograd(a):
    """Clamps `a` from above and below."""
    return jnp.clip(a, a_min, a_max)

  @clip_nograd.defjvp
  def clip_nograd_jvp(primals, tangents):
    """Override clips()'s gradient to be a no-op."""
    return clip_nograd(primals[0]), tangents[0]

  return clip_nograd


clip_finite_nograd = generate_clip_nograd_fn(min_val, max_val)

clip_pos_finite_nograd = generate_clip_nograd_fn(tiny_val, max_val)


def clip_pos(x):
  """Clamps `x` from below to be positive."""
  return jnp.maximum(tiny_val, x)


def safe_sign(x):
  """jnp.sign(x) except x=0 is assumed to have a sign of +1, not 0."""
  return jnp.where(x < 0, -1, +1)


def remove_zero(x):
  """Shifts `x` away from 0."""
  return jnp.where(jnp.abs(x) < tiny_val, tiny_val, x)


@jax.custom_vjp
def safe_div(n, d):
  """Divide `n` by `d` but the value and gradient never nan out."""
  return safe_div_fwd(n, d)[0]


def safe_div_fwd(n, d):
  r = jnp.clip(n / remove_zero(d), min_val, max_val)
  return jnp.where(jnp.abs(d) < tiny_val, 0, r), (d, r)


def safe_div_bwd(res, g):
  d, r = res
  dn = jnp.clip(g / remove_zero(d), min_val, max_val)
  dd = jnp.clip(-g * r / remove_zero(d), min_val, max_val)
  return dn, dd


safe_div.defvjp(safe_div_fwd, safe_div_bwd)


def generate_safe_fn(fn, grad_fn, x_range):
  """Generate's a `safe` fn() where inputs are clipped in fwd and bwd passes."""

  @jax.custom_jvp
  def safe_fn(x):
    """fn() with clipped inputs."""
    return fn(jnp.clip(x, *x_range))

  @safe_fn.defjvp
  def safe_fn_jvp(primals, tangents):
    """Backpropagate using the gradient and clipped inputs."""
    (x,) = primals
    (x_dot,) = tangents
    y = safe_fn(x)
    y_dot = grad_fn(jnp.clip(x, *x_range), y, x_dot)
    return y, y_dot

  return safe_fn


# These safe_* functions need to be wrapped in no-op function definitions for
# gin to recognize them, otherwise they could just be calls to generate_safe_fn.


def safe_log(x):
  fn = generate_safe_fn(
      jnp.log,
      lambda x, _, x_dot: x_dot / x,
      (tiny_val, max_val),
  )
  return fn(x)


def safe_exp(x):
  fn = generate_safe_fn(
      jnp.exp,
      lambda _, y, x_dot: y * x_dot,
      (min_val, np.nextafter(80.0, np.float32(0))),
  )
  return fn(x)


def safe_sqrt(x):
  fn = generate_safe_fn(
      jnp.sqrt,
      lambda x, _, x_dot: 0.5 * x_dot / jnp.sqrt(jnp.maximum(tiny_val, x)),
      (0, max_val),
  )
  return fn(x)


safe_log1p = generate_safe_fn(
    jnp.log1p,
    lambda x, _, x_dot: x_dot / (1 + x),
    (np.nextafter(np.float32(-1), np.float32(0)), max_val),
)


def safe_expm1(x):
  fn = generate_safe_fn(
      jnp.expm1,
      lambda x, _, x_dot: jnp.exp(x) * x_dot,
      (min_val, np.nextafter(np.log1p(max_val), np.float32(0))),
  )
  return fn(x)


def safe_arccos(x):
  """jnp.arccos(x) where x is clipped to [-1, 1]."""
  y = jnp.arccos(jnp.clip(x, plus_eps(-1), minus_eps(1)))
  return jnp.where(x >= 1, 0, jnp.where(x <= -1, jnp.pi, y))


def apply_fn_to_grad(grad_fn):
  """Applies a scalar `grad_fn` function to the gradient of the input."""

  @jax.custom_vjp
  def fn_out(x):
    return x

  fn_out.defvjp(lambda x: (x, None), lambda _, y: (grad_fn(y),))
  return fn_out


# Turns NaNs into zeros for the gradients propagating onto `x`.
nangrad_to_zero = apply_fn_to_grad(jnp.nan_to_num)


def cholesky3(a, symmetrize_input=True):
  """A closed form Cholesky decomposition for only 3x3 matrices."""
  if any([s != 3 for s in a.shape[-2:]]):
    raise ValueError(f'input shape must be (3, 3), but is {a.shape[-2:]}.')
  a11, a12, a13, a21, a22, a23, a31, a32, a33 = jnp.moveaxis(
      a.reshape(a.shape[:-2] + (9,)), -1, 0
  )
  if symmetrize_input:
    a12 = (a12 + a21) / 2
    a13 = (a13 + a31) / 2
    a23 = (a23 + a32) / 2

  l11 = safe_sqrt(a11)
  l21 = safe_div(a21, l11)
  l22 = safe_sqrt(a22 - safe_div(a21, l11) ** 2)
  l31 = safe_div(a31, l11)
  l32 = safe_div(a32 - l31 * l21, l22)
  l33 = safe_sqrt(
      a33 - safe_div(a31**2, a11) - safe_div((a32 - l31 * l21), l22) ** 2
  )
  z = jnp.zeros_like(a11)

  return jnp.stack([l11, z, z, l21, l22, z, l31, l32, l33], axis=-1).reshape(
      a.shape
  )


def safe_cholesky(a, **kwargs):
  if all([s == 3 for s in a.shape[-2:]]):
    fn = cholesky3
  else:
    fn = jax.lax.linalg.cholesky
  return jnp.nan_to_num(fn(nangrad_to_zero(a), **kwargs))


def select(cond_pairs, default):
  """A helpful wrapper around jnp.select() that is easier to read."""
  return jnp.select(*zip(*cond_pairs), default)


def power_ladder_max_output(p):
  """The limit of power_ladder(x, p) as x goes to infinity."""
  return select(
      [
          (p == -jnp.inf, 1),
          (p >= 0, jnp.inf),
      ],
      safe_div(p - 1, p),
  )


def power_ladder(x, p, premult=None, postmult=None):
  """Tukey's power ladder, with a +1 on x, some scaling, and special cases."""
  # Compute sign(x) * |p - 1|/p * ((|x|/|p-1| + 1)^p - 1)
  if premult is not None:
    x = x * premult
  xp = jnp.abs(x)
  xs = xp / jnp.maximum(tiny_val, jnp.abs(p - 1))
  p_safe = clip_finite_nograd(remove_zero(p))
  y = safe_sign(x) * select(
      [
          (p == 1, xp),
          (p == 0, safe_log1p(xp)),
          (p == -jnp.inf, -safe_expm1(-xp)),
          (p == jnp.inf, safe_expm1(xp)),
      ],
      clip_finite_nograd(
          jnp.abs(p_safe - 1) / p_safe * ((xs + 1) ** p_safe - 1)
      ),
  )
  if postmult is not None:
    y = y * postmult
  return y


def inv_power_ladder(y, p, premult=None, postmult=None):
  """The inverse of `power_ladder()`."""
  if postmult is not None:
    y /= postmult
  yp = jnp.abs(y)
  p_safe = clip_finite_nograd(remove_zero(p))
  y_max = minus_eps(power_ladder_max_output(p))
  yp = override_gradient(jnp.clip(yp, -y_max, y_max), yp)  # Clip val, not grad.
  x = safe_sign(y) * select(
      [
          (p == 1, yp),
          (p == 0, safe_expm1(yp)),
          (p == -jnp.inf, -safe_log1p(-yp)),
          (p == jnp.inf, safe_log1p(yp)),
      ],
      jnp.abs(p_safe - 1)
      * (
          ((safe_div(p_safe, jnp.abs(p_safe - 1)) * yp + 1)) ** (1 / p_safe) - 1
      ),
  )
  if premult is not None:
    x /= premult
  return x


def power_iteration(a_mat, n):
  """Do n rounds of power iteration to get a_mat's top eigenvalue/vector."""
  # Equivalent to initializing vec = 𝟙/||𝟙|| and multiplying by A.
  vec = jnp.sum(a_mat, axis=-1) / jnp.sqrt(a_mat.shape[-1])
  for i in range(n):
    if i > 0:
      vec = matmul(a_mat, vec[Ellipsis, None])[Ellipsis, 0]
    val = jnp.sqrt(jnp.sum(vec**2, axis=-1))
    vec /= val[Ellipsis, None]
  return val, vec


def log_lerp(t, v0, v1):
  """Interpolate log-linearly from `v0` (t=0) to `v1` (t=1)."""
  if v0 <= 0 or v1 <= 0:
    raise ValueError(f'Interpolants {v0} and {v1} must be positive.')
  lv0 = jnp.log(v0)
  lv1 = jnp.log(v1)
  return jnp.exp(jnp.clip(t, 0, 1) * (lv1 - lv0) + lv0)


def approx_erf(x):
  """An approximation of erf() that is accurate to within 0.007."""
  return jnp.sign(x) * jnp.sqrt(1 - jnp.exp(-(4 / jnp.pi) * x**2))


def create_learning_rate_decay(**kwargs):
  """A partial evaluation of learning rate decay that can be used with gin."""
  return functools.partial(learning_rate_decay, **kwargs)


def learning_rate_decay(
    step, lr_init, lr_final, max_steps, lr_delay_steps=0, lr_delay_mult=1
):
  """Continuous learning rate decay function.

  The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
  is log-linearly interpolated elsewhere (equivalent to exponential decay).
  If lr_delay_steps>0 then the learning rate will be scaled by some smooth
  function of lr_delay_mult, such that the initial learning rate is
  lr_init*lr_delay_mult at the beginning of optimization but will be eased back
  to the normal learning rate when steps>lr_delay_steps.

  Args:
    step: int, the current optimization step.
    lr_init: float, the initial learning rate.
    lr_final: float, the final learning rate.
    max_steps: int, the number of steps during optimization.
    lr_delay_steps: int, the number of steps to delay the full learning rate.
    lr_delay_mult: float, the multiplier on the rate when delaying it.

  Returns:
    lr: the learning for current step 'step'.
  """
  if lr_init == 0.0 and lr_final == 0.0:
    return 0.0

  if lr_delay_steps > 0:
    # A kind of reverse cosine decay.
    delay_rate = lr_delay_mult + (1 - lr_delay_mult) * jnp.sin(
        0.5 * jnp.pi * jnp.clip(step / lr_delay_steps, 0, 1)
    )
  else:
    delay_rate = 1.0

  return delay_rate * log_lerp(step / max_steps, lr_init, lr_final)


def sorted_lookup(x, xp, fps, device_is_tpu):
  """Lookup `x` into locations `xp` , return indices and each `[fp]` value."""
  if not isinstance(fps, tuple):
    raise ValueError(f'Input `fps` must be a tuple, but is {type(fps)}.')

  if device_is_tpu:
    # Identify the location in `xp` that corresponds to each `x`.
    # The final `True` index in `mask` is the start of the matching interval.
    mask = x[Ellipsis, None, :] >= xp[Ellipsis, :, None]

    def find_interval(x):
      # Grab the value where `mask` switches from True to False, and vice versa.
      # This approach takes advantage of the fact that `x` is sorted.
      x0 = jnp.max(jnp.where(mask, x[Ellipsis, None], x[Ellipsis, :1, None]), -2)
      x1 = jnp.min(jnp.where(~mask, x[Ellipsis, None], x[Ellipsis, -1:, None]), -2)
      return x0, x1

    idx0, idx1 = find_interval(jnp.arange(xp.shape[-1]))
    vals = [find_interval(fp) for fp in fps]
  else:
    # jnp.searchsorted() has slightly different conventions for boundary
    # handling than the rest of this codebase.
    idx = jnp.vectorize(
        lambda a, v: jnp.searchsorted(a, v, side='right'),
        signature='(n),(m)->(m)',
    )(xp, x)
    idx1 = jnp.minimum(idx, xp.shape[-1] - 1)
    idx0 = jnp.maximum(idx - 1, 0)
    vals = []
    for fp in fps:
      fp0 = jnp.take_along_axis(fp, idx0, axis=-1)
      fp1 = jnp.take_along_axis(fp, idx1, axis=-1)
      vals.append((fp0, fp1))
  return (idx0, idx1), vals


def sorted_interp(
    x, xp, fp, device_is_tpu, eps=jnp.finfo(jnp.float32).eps ** 2
):
  """A version of interp() where xp and fp must be sorted."""
  (xp0, xp1), (fp0, fp1) = sorted_lookup(
      x, xp, (xp, fp), device_is_tpu=device_is_tpu
  )[1]
  offset = jnp.clip((x - xp0) / jnp.maximum(eps, xp1 - xp0), 0, 1)
  ret = fp0 + offset * (fp1 - fp0)
  return ret


def searchsorted(a, v, device_is_tpu):
  """Behaves like jnp.searchsorted, excluding boundary conditions."""
  return sorted_lookup(v, a, (), device_is_tpu=device_is_tpu)[0]


def override_gradient(fval, bval):
  """Use `fval` in the forward pass but `bval` in the backward pass."""
  # Note that the parentheses are needed to avoid catastrophic cancellation.
  return jax.lax.stop_gradient(fval) + (bval - jax.lax.stop_gradient(bval))


def average_across_multisamples(x):
  """Function that averages grid query results across the multisample dimension."""
  return jnp.mean(x, axis=-2)


def normalize(x):
  """Normalization helper function."""
  return x / jnp.linalg.norm(x, axis=-1, keepdims=True)


def dot(x, y, axis=-1, keepdims=True):
  """Normalization helper function."""
  return (x * y).sum(axis=axis, keepdims=keepdims)


def random_sample_sphere(key, sh):
  return normalize(jax.random.normal(key, sh))


def random_sample_hemisphere(key, sh):
  samples = normalize(jax.random.normal(key, sh))
  return jnp.concatenate(
      [
          samples[Ellipsis, 0:1],
          samples[Ellipsis, 1:2],
          jnp.abs(samples[Ellipsis, 2:3]),
      ],
      axis=-1,
  )


def cosine_sample_hemisphere(key, sh):
  samples = random_sample_sphere(key, sh)

  ones = jnp.ones(sh[:-1] + (1,))
  zeros = jnp.zeros(sh[:-1] + (1,))

  return normalize(samples + jnp.concatenate([zeros, zeros, ones], axis=-1))


def cosine_sample_hemisphere_normal(key, sh, normal):
  samples = random_sample_sphere(key, sh)
  return normalize(samples + normal)


def cosine_sample_weights(samples, normal):
  return dot(samples, normal) / jnp.pi


def samples_to_frame(samples, forward):
  ones = jnp.ones_like(forward[Ellipsis, :1])
  zeros = jnp.zeros_like(forward[Ellipsis, :1])

  x = jnp.concatenate([ones, zeros, zeros], axis=-1)
  y = jnp.concatenate([zeros, ones, zeros], axis=-1)

  up = jnp.where(
      jnp.repeat(jnp.abs(forward[Ellipsis, 1:2]) < 0.9, 3, axis=-1),
      y,
      x,
  )

  right = jnp.cross(forward, up)
  up = jnp.cross(forward, right)

  return jnp.concatenate(
      [
          samples[Ellipsis, 0:1] * right,
          samples[Ellipsis, 1:2] * up,
          samples[Ellipsis, 2:3] * forward,
      ],
      axis=-1,
  )
