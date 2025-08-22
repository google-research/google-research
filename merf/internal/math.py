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

import jax
import jax.numpy as jnp


def matmul(a, b):
  """jnp.matmul defaults to bfloat16, but this helper function doesn't."""
  return jnp.matmul(a, b, precision=jax.lax.Precision.HIGHEST)


def safe_trig_helper(x, fn, t=100 * jnp.pi):
  """Helper function used by safe_cos/safe_sin: mods x before sin()/cos()."""
  return fn(jnp.nan_to_num(jnp.where(jnp.abs(x) < t, x, x % t)))


def safe_cos(x):
  """jnp.cos() on a TPU may NaN out for large values."""
  return safe_trig_helper(x, jnp.cos)


def safe_sin(x):
  """jnp.sin() on a TPU may NaN out for large values."""
  return safe_trig_helper(x, jnp.sin)


@jax.custom_jvp
def safe_exp(x):
  """jnp.exp() but with finite output and gradients for large inputs."""
  return jnp.exp(jnp.minimum(x, 88.0))  # jnp.exp(89) is infinity.


@safe_exp.defjvp
def safe_exp_jvp(primals, tangents):
  """Override safe_exp()'s gradient so that it's large when inputs are large."""
  [x] = primals
  [x_dot] = tangents
  exp_x = safe_exp(x)
  exp_x_dot = exp_x * x_dot
  return exp_x, exp_x_dot


@jax.custom_jvp
def safe_log(x):
  """jnp.log() but with finite outputs/gradients for negative/huge inputs."""
  return jnp.log(jnp.clip(x, 1e-37, 1e37))  # jnp.log(1e-38) is -infinity.


@safe_log.defjvp
def safe_log_jvp(primals, tangents):
  """Override safe_log()'s gradient to always be finite."""
  [x] = primals
  [x_dot] = tangents
  log_x = safe_log(x)
  log_x_dot = x_dot / jnp.maximum(1e-37, x)
  return log_x, log_x_dot


def log_lerp(t, v0, v1):
  """Interpolate log-linearly from `v0` (t=0) to `v1` (t=1)."""
  if v0 <= 0 or v1 <= 0:
    raise ValueError(f'Interpolants {v0} and {v1} must be positive.')
  lv0 = jnp.log(v0)
  lv1 = jnp.log(v1)
  return jnp.exp(jnp.clip(t, 0, 1) * (lv1 - lv0) + lv0)


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
    idx = jax.vmap(lambda a, v: jnp.searchsorted(a, v, side='right'))(
        xp.reshape([-1, xp.shape[-1]]), x.reshape([-1, x.shape[-1]])
    ).reshape(x.shape)
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


def density_to_alpha(x, step_size):
  return 1.0 - safe_exp(-x * step_size)


def density_activation(x):
  return safe_exp(x - 1.0)


def normalize(x, v_min, v_max):
  """[v_min, v_max] -> [0, 1]."""
  return (x - v_min) / (v_max - v_min)


def denormalize(x, v_min, v_max):
  """[0, 1] -> [v_min, v_max]."""
  return v_min + x * (v_max - v_min)


def as_mib(x):
  """Computes size of array in Mebibyte (MiB)."""
  return x.size / (1024**2)


@jax.custom_jvp
def fake_clip(a, a_min, a_max):
  """jnp.clip() but the gradient doesn't get clipped on the backward pass."""
  return jnp.clip(a, a_min, a_max)


@fake_clip.defjvp
def fake_clip_jvp(primals, tangents):
  """Override fake_clip()'s gradient so that it's a no-op."""
  return jnp.clip(*primals), tangents[0]


@jax.jit
def lossfun(x, alpha, scale):
  r"""Implements the general form of the loss.

  This implements the rho(x, \alpha, c) function described in "A General and
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
      interpolation between several discrete robust losses: alpha=-Infinity:
      Welsch/Leclerc Loss. alpha=-2: Geman-McClure loss. alpha=0:
      Cauchy/Lortentzian loss. alpha=1: Charbonnier/pseudo-Huber loss. alpha=2:
      L2 loss.
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
  loss_two = 0.5 * (x / scale) ** 2

  # Clamp |alpha| to be >= machine epsilon so that it's safe to divide by.
  a = jnp.where(
      alpha >= 0, jnp.ones_like(alpha), -jnp.ones_like(alpha)
  ) * jnp.maximum(eps, jnp.abs(alpha))

  # Clamp |2-alpha| to be >= machine epsilon so that it's safe to divide by.
  b = jnp.maximum(eps, jnp.abs(a - 2))

  # The loss when not in one of the special casess.
  loss_ow = (b / a) * ((loss_two / (0.5 * b) + 1) ** (0.5 * a) - 1)

  # Select which of the cases of the loss to return as a function of alpha.
  return jnp.where(
      alpha == -jnp.inf,
      -expm1_safe(-loss_two),
      jnp.where(
          alpha == 0,
          jnp.log1p(loss_two),
          jnp.where(
              alpha == 2,
              loss_two,
              jnp.where(alpha == jnp.inf, expm1_safe(loss_two), loss_ow),
          ),
      ),
  )
