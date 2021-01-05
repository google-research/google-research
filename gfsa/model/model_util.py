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

# Lint as: python3
"""Utilities for defining loss functions."""

from typing import Optional

import flax
import jax
import jax.numpy as jnp
import numpy as np

from gfsa import jax_util


def forward_clip(v,
                 minval = None,
                 maxval = None):
  """Clip in the forward pass, but behave as identity backward.

  The primary use case is when an NDArray mathematically must lie within some
  given bounds, but due to numerical issues might be outside it. In this case,
  `forward_clip` ensures the output can still be passed into functions that
  assume it is in those bounds (i.e. jnp.log) without preventing gradient flow
  when those errors occur.

  Args:
    v: Value to clip.
    minval: Minimum value to clip to in the forwards pass.
    maxval: Maximum value to clip to in the forwards pass.

  Returns:
    Clipped version of the value.
  """
  res = v
  if minval is not None:
    res = res - jnp.minimum(0, jax.lax.stop_gradient(v - minval))
  if maxval is not None:
    res = res + jnp.minimum(0, jax.lax.stop_gradient(maxval - v))
  return res


def safe_logit(prob,
               epsilon_multiplier = 1.0e4):
  """Compute the log-odds (logit) of a probability, in a numerically safe way.

  This function converts inputs from probability space into log-odds space.
  Mathematically,

    safe_logit(p) = log(p/(1-p)).

  It is the inverse of the logistic sigmoid function, and has domain/range
  (0,1) -> (-inf, inf).

  The values in v are assumed to be within the range (0, 1), and we clip them
  into that range (with an epsilon buffer) to avoid nuemrical stability issues
  when `prob` has noise. (Note that gradients will be meaningful (and nonzero)
  even if clipping occurs.) Since values may be very close to 0 or 1, we
  divide it into two cases, and use `where` expressions to avoid NaNs in the
  gradients.

  For values far from 0 or 1, this function returns the same thing as
  scipy.special.logit.

  Args:
    prob: Probability, as a float between 0 and 1.
    epsilon_multiplier: `prob` will be clamped to `epsilon_multiplier` times the
      smallest usable positive floating point number. This should be large
      enough that computing gradients doesn't overflow (if this is too close to
      1, the gradient of the loss with respect to `prob` will be the largest
      representable float, so intermediate computations while computing `prob`
      could cause the intermediate gradients to overflow).

  Returns:
    Float NDArray of same shape as `prob` containing log-odds.
  """
  # `tiny` is the smallest usable number greater than 0.
  # (Note that there are smaller numbers, but they are "denormal" and XLA
  # rounds them to 0 after most operations.)
  # For float32, tiny is around 1e-38.
  # We then scale this clipping value up to prevent overflow in the case where:
  #  1. the model confidently predicts False (i.e. prob ~= tiny), possibly
  #     because of numerical accuracy, or because of overconfidence / bad
  #     initialization.
  #  2. the answer is actually True (so the un-clipped loss is log(tiny) and the
  #     gradient is 1/tiny).
  #  3. computing gradients through `prob` tries to scale the incoming gradient
  #     by some factor larger than one, overflowing it to inf.
  epsilon_near_zero = np.finfo(prob.dtype).tiny * epsilon_multiplier

  # Smallest number such that 1 - epsilon_near_one != 1.
  # For float32, epsneg is about 6e-8.
  epsilon_near_one = np.finfo(prob.dtype).epsneg

  # When computing log(p), clip to epsilon_near_zero.
  log_true = jnp.log(forward_clip(prob, minval=epsilon_near_zero))

  # When computing log(1-p), use different implementations based on the size.
  prob_is_big = prob > 0.5
  # When prob is near 1, clip it to the closest value less than 1.
  log_false_where_big = jnp.log(forward_clip(1 - prob, minval=epsilon_near_one))
  # When prob is near 0, use log1p for extra precision (and skip this
  # branch of the computation if prob is near 1 to avoid
  # https://github.com/google/jax/issues/1052)
  log_false_where_small = jnp.log1p(-jnp.where(prob_is_big, 0, prob))
  log_false = jnp.where(prob_is_big, log_false_where_big, log_false_where_small)

  # Combine these to compute log odds.
  log_odds = log_true - log_false
  return log_odds


def binary_logit_cross_entropy(logits,
                               targets):
  """Compute cross entropy between logits and a binary target.

  The model distribution is interpreted as a Bernoulli distribution with
  logits = log(p / (1-p)), i.e. p = sigmoid(logits).

  As described in
  https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits,
  we compute this as

    max(logits, 0) - logits * targets + log(1 + exp(-abs(logits)))

  Args:
    logits: Logits, or log-odds, of the model distribution.
    targets: Target output, as a boolean.

  Returns:
    Float NDArray of same shape as `logits` containing negative log-likelihoods
    of the targets.
  """
  return (jax.nn.relu(logits) -
          jnp.where(targets, logits, jnp.zeros_like(logits)) +
          jnp.log1p(jnp.exp(-jnp.abs(logits))))


def linear_cross_entropy(prob,
                         target,
                         epsilon_multiplier = 1.0e4):
  """Compute the cross entropy between a probability and a target boolean.

  This function is designed for inputs in probability space, NOT
  log-probability (logit) space. Since values may be very close to 0 or 1, we
  divide it into two cases, and use `where` expressions to avoid NaNs in the
  gradients.

  The values in v are assumed to be within the range (0, 1), and we clip them
  into that range (with an epsilon buffer) to avoid nuemrical stability issues
  when `prob` has noise. (Note that gradients will be meaningful (and nonzero)
  even if clipping occurs.)

  Args:
    prob: Output probability, as a float.
    target: Target output, as a boolean.
    epsilon_multiplier: `prob` will be clamped to `epsilon_multiplier` times the
      smallest usable positive floating point number. This should be large
      enough that computing gradients doesn't overflow (if this is too close to
      1, the gradient of the loss with respect to `prob` will be the largest
      representable float, so intermediate computations while computing `prob`
      could cause the intermediate gradients to overflow).

  Returns:
    Float NDArray of same shape as `prob` containing negative log-likelihoods.
  """
  return binary_logit_cross_entropy(
      safe_logit(prob, epsilon_multiplier), target)


class ScaleAndShift(flax.nn.Module):
  """Learnable function f(x) = ax + b where a and b are scalars."""

  def apply(self, x):
    """Apply the scale and shift layer.

    Args:
      x: Float ndarray of arbitrary shape.

    Returns:
      Array of the same shape with a scale and shift applied.
    """
    scale = self.param("scale", shape=(), initializer=jax.nn.initializers.ones)
    shift = self.param("shift", shape=(), initializer=jax.nn.initializers.zeros)
    return x * scale + shift
