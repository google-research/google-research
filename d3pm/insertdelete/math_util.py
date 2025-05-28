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

"""Math utilities."""

import functools

import jax
import jax.numpy as jnp


def safe_logaddexp(x1, x2, *xs):
  """jnp.logaddexp, but safe if x1 and x2 are both -inf.

  We redefine the gradient of logaddexp such that if both are -inf, the
  gradient is zero. Strictly speaking, this may not be mathematically correct,
  but it works if we are taking the gradient of something like

    logaddexp(logaddexp(-inf + x, -inf + y), c + z)

  with respect to [x, y, z]. Essentially, as long as the things we are taking
  gradients for are finite, -inf implies that we decided this option is
  impossible, so it should have zero gradient flowing to the conditional for
  that option.

  Args:
    x1: First value.
    x2: Second value.
    *xs: Optional additional values.

  Returns:
    Log-sum-exp of the arguments.
  """
  if xs:
    return functools.reduce(safe_logaddexp, [x1, x2, *xs])

  impossible = (jnp.maximum(x1, x2) == -jnp.inf)
  x1 = jnp.where(impossible, jax.lax.stop_gradient(x1), x1)
  x2 = jnp.where(impossible, jax.lax.stop_gradient(x2), x2)
  return jnp.logaddexp(x1, x2)


def safe_logsumexp(a, axis=None, *args, **kwargs):  # pylint: disable=keyword-arg-before-vararg
  """jax.scipy.special.logsumexp, but safe if all values are -inf.

  See safe_logaddexp.

  Args:
    a: Array to compute logsumexp over.
    axis: Axis to reduce over.
    *args: Additional arguments to jax.scipy.special.logsumexp.
    **kwargs: Additional keyword arguments to jax.scipy.special.logsumexp.

  Returns:
    Equivalent output as jax.scipy.special.logsumexp.
  """
  impossible = (jnp.max(a, axis, keepdims=True) == -jnp.inf)
  a = jnp.where(impossible, jax.lax.stop_gradient(a), a)
  return jax.scipy.special.logsumexp(a, axis, *args, **kwargs)


def safe_sub_or_ninf(case_log_prob, total_log_prob):
  """Return a-b or -inf if total_log_prob is -inf."""
  return jnp.where(total_log_prob == -jnp.inf, -jnp.inf,
                   case_log_prob - total_log_prob)


def safe_exp_weighted(log_gate, log_result):
  """Return exp(log_gate) * log_result, but safe if log_gate = -np.inf."""
  # Where log_gate == -np.inf, this is impossible.
  log_result = jnp.where(log_gate == -jnp.inf,
                         jax.lax.stop_gradient(log_result), log_result)
  # log_gate may still be so large that this is effectively impossible, but
  # in that case we should still propagate -inf in the result.
  if_possible = jnp.where(log_result == -jnp.inf, -jnp.inf,
                          jnp.exp(log_gate) * log_result)
  return jnp.where(log_gate == -jnp.inf, 0.0, if_possible)


def logmeanexp(a, axis=None):
  total = safe_logsumexp(a, axis=axis)
  return total + jnp.log(total.size) - jnp.log(a.size)


def log_not_any(*log_probs):
  total = sum(jnp.exp(v) for v in log_probs)
  return jnp.where(total >= 1, -jnp.inf, jnp.log1p(-total))


def log_not(log_prob):
  return log_not_any(log_prob)
