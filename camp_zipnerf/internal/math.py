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

"""Mathy utility functions."""

import functools

import jax
import jax.numpy as jnp
import numpy as np


tiny_val = np.float32(np.finfo(np.float32).tiny)
min_val = np.float32(np.finfo(np.float32).min)
max_val = np.float32(np.finfo(np.float32).max)


def laplace_cdf(x, beta):
  alpha = 1 / beta
  return alpha * (0.5 + 0.5 * safe_sign(x) * (jnp.exp(-jnp.abs(x) / beta) - 1))


def scaled_softplus(x, scale=100.0):
  return (1.0 / scale) * jax.nn.softplus(scale * x)


def matmul(a, b):
  """jnp.matmul defaults to bfloat16, but this helper function doesn't."""
  return jnp.matmul(a, b, precision=jax.lax.Precision.HIGHEST)


def unstack(x, axis=0):
  return tuple(
      jnp.squeeze(z, axis=axis) for z in jnp.split(x, x.shape[axis], axis=axis)
  )


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


@jax.custom_jvp
def expm1(x):
  """jnp.expm1() has inaccurate gradients when x << 0, this doesn't."""
  return jnp.expm1(x)


@expm1.defjvp
def expm1_jvp(primals, tangents):
  return expm1(*primals), tangents[0] * jnp.exp(primals[0])


def safe_trig_helper(x, fn, t=100 * jnp.pi):
  """Helper function used by safe_cos/safe_sin: mods x before sin()/cos()."""
  return fn(jnp.nan_to_num(jnp.where(jnp.abs(x) < t, x, x % t)))


def safe_cos(x):
  """jnp.cos() on a TPU may NaN out for large values."""
  return safe_trig_helper(x, jnp.cos)


def safe_sin(x):
  """jnp.sin() on a TPU may NaN out for large values."""
  return safe_trig_helper(x, jnp.sin)


@jax.custom_vjp
def safe_arctan2(x1, x2):
  return safe_arctan2_fwd(x1, x2)[0]


def safe_arctan2_fwd(x1, x2):
  return jnp.arctan2(x1, x2), (x1, x2)


def safe_arctan2_bwd(res, g):
  x1, x2 = res
  denom = remove_zero(x1**2 + x2**2)
  d1 = g * (x2 / denom)
  d2 = g * (-x1 / denom)
  return d1, d2


safe_arctan2.defvjp(safe_arctan2_fwd, safe_arctan2_bwd)


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


def clip_finite(x):
  return jnp.clip(x, min_val, max_val)


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
  return generate_safe_fn(
      jnp.log,
      lambda x, _, x_dot: x_dot / x,
      (tiny_val, max_val),
  )(x)


def safe_exp(x):
  return generate_safe_fn(
      jnp.exp,
      lambda _, y, x_dot: y * x_dot,
      (min_val, np.nextafter(np.log(max_val), np.float32(0))),
  )(x)


def safe_sqrt(x):
  return generate_safe_fn(
      jnp.sqrt,
      lambda x, _, x_dot: 0.5 * x_dot / jnp.sqrt(jnp.maximum(tiny_val, x)),
      (0, max_val),
  )(x)


def safe_log1p(x):
  return generate_safe_fn(
      jnp.log1p,
      lambda x, _, x_dot: x_dot / (1 + x),
      (np.nextafter(np.float32(-1), np.float32(0)), max_val),
  )(x)


def safe_expm1(x):
  return generate_safe_fn(
      expm1,  # Note that we wrap around our more accurate expm1.
      lambda x, _, x_dot: jnp.exp(x) * x_dot,
      (min_val, np.nextafter(np.log1p(max_val), np.float32(0))),
  )(x)


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


def noop(x):
  return x


@jax.custom_jvp
def fake_clip(a, a_min, a_max):
  """jnp.clip() but the gradient doesn't get clipped on the backward pass."""
  return jnp.clip(a, a_min, a_max)


@fake_clip.defjvp
def fake_clip_jvp(primals, tangents):
  """Override fake_clip()'s gradient so that it's a no-op."""
  return jnp.clip(*primals), tangents[0]


@jax.jit
def general_lossfun(x, alpha, scale):
  r"""This implements the rho(x, \alpha, c) function described in "A General and
  Adaptive Robust Loss Function", Jonathan T. Barron,
  https://arxiv.org/abs/1701.03077.

  Args:
    x: The residual for which the loss is being computed. x can have any shape,
      and alpha and scale will be broadcasted to match x's shape if necessary.
    alpha: The shape parameter of the loss (\alpha in the paper), where more
      negative values produce a loss with more robust behavior (outliers "cost"
      less), and more positive values produce a loss with less robust behavior
      (outliers are penalized more heavily). Alpha can be any value in
      [-infinity, infinity], but the gradient of the loss with respect to alpha
      is 0 at -infinity, infinity, 0, and 2. Varying alpha allows for smooth
      interpolation between several discrete robust losses:
        alpha=-Infinity: Welsch/Leclerc Loss.
        alpha=-2: Geman-McClure loss.
        alpha=0: Cauchy/Lortentzian loss.
        alpha=1: Charbonnier/pseudo-Huber loss.
        alpha=2: L2 loss.
    scale: The scale parameter of the loss. When |x| < scale, the loss is an
      L2-like quadratic bowl, and when |x| > scale the loss function takes on a
      different shape according to alpha.

  Returns:
    The losses for each element of x, in the same shape as x.
  """
  eps = jnp.finfo(jnp.float32).eps
  maxval = 1e15

  # A "safe" versions of expm1 that will not NaN-out on large inputs.
  expm1_safe = lambda x: jnp.expm1(jnp.minimum(x, 43))

  # `scale` must be > 0.
  scale = jnp.maximum(eps, scale)

  # Large values of |x| can cause non-finite gradients.
  x = fake_clip(x, -maxval, maxval)

  # The loss when alpha == 2. This will get reused repeatedly.
  loss_two = 0.5 * (x / scale)**2

  # Clamp |alpha| to be >= machine epsilon so that it's safe to divide by.
  a = jnp.where(alpha >= 0, jnp.ones_like(alpha),
                -jnp.ones_like(alpha)) * jnp.maximum(eps, jnp.abs(alpha))

  # Clamp |2-alpha| to be >= machine epsilon so that it's safe to divide by.
  b = jnp.maximum(eps, jnp.abs(a - 2))

  # The loss when not in one of the special casess.
  loss_ow = (b / a) * ((loss_two / (0.5 * b) + 1)**(0.5 * a) - 1)

  # Select which of the cases of the loss to return as a function of alpha.
  return jnp.where(
      alpha == -jnp.inf, -expm1_safe(-loss_two),
      jnp.where(
          alpha == 0, jnp.log1p(loss_two),
          jnp.where(alpha == 2, loss_two,
                    jnp.where(alpha == jnp.inf, expm1_safe(loss_two),
                              loss_ow))))
