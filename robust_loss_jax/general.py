# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

r"""Implements the general form of the loss.

This is the simplest way of using this loss. No parameters will be tuned
automatically, it's just a simple function that takes in parameters (likely
hand-tuned ones) and return a loss. For an adaptive loss, look at adaptive.py
or distribution.py.
"""

import jax
import jax.numpy as jnp


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

  # `scale` must be > 0.
  scale = jnp.maximum(eps, scale)

  # The loss when alpha == 2. This will get reused repeatedly.
  loss_two = 0.5 * (x / scale)**2

  # "Safe" versions of log1p and expm1 that will not NaN-out.
  log1p_safe = lambda x: jnp.log1p(jnp.minimum(x, 3e37))
  expm1_safe = lambda x: jnp.expm1(jnp.minimum(x, 87.5))

  # The loss when not in one of the special casess.
  # Clamp |alpha| to be >= machine epsilon so that it's safe to divide by.
  a = jnp.where(alpha >= 0, jnp.ones_like(alpha),
                -jnp.ones_like(alpha)) * jnp.maximum(eps, jnp.abs(alpha))
  # Clamp |2-alpha| to be >= machine epsilon so that it's safe to divide by.
  b = jnp.maximum(eps, jnp.abs(alpha - 2))
  loss_ow = (b / a) * ((loss_two / (0.5 * b) + 1)**(0.5 * alpha) - 1)

  # Select which of the cases of the loss to return as a function of alpha.
  return jnp.where(
      alpha == -jnp.inf, -expm1_safe(-loss_two),
      jnp.where(
          alpha == 0, log1p_safe(loss_two),
          jnp.where(alpha == 2, loss_two,
                    jnp.where(alpha == jnp.inf, expm1_safe(loss_two),
                              loss_ow))))
