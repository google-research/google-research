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

"""Define optimizer facilities."""

from typing import Any

import flax
import jax
import jax.numpy as jnp
import optax


@flax.struct.dataclass
class MaybeSkipGradientUpdateState:
  inner_state: Any


def maybe_skip_gradient_update(
    inner,
    gradient_norm_skip_threshold,
):
  """A function that wraps an optimiser to skip updates under some condition.

  The purpose of this function is to prevent any optimisation to happen if the
  gradients contain NaNs, Infs, or if its norm is higher than a certain
  threshold. That is, when a NaN of Inf, is detected in the gradients or when
  the norm of the gradient is higher than the threshold, the wrapped optimiser
  ignores that gradient update.

  Args:
    inner: Inner transformation to be wrapped.
    gradient_norm_skip_threshold: float,

  Returns:
    New GradientTransformation.
  """

  def init(params):
    return MaybeSkipGradientUpdateState(inner_state=inner.init(params))

  def update(updates, state, params=None):
    inner_state = state.inner_state
    # Compute gradient norm and clip gradient if necessary
    gradient_norm = optax.global_norm(updates)
    flat_updates = jax.tree_flatten(updates)[0]
    isfinite = jnp.all(
        jnp.array([jnp.all(jnp.isfinite(p)) for p in flat_updates]))
    islowerthan = gradient_norm < gradient_norm_skip_threshold

    def do_update(_):
      return inner.update(updates, inner_state, params)

    def reject_update(_):
      return (jax.tree_map(jnp.zeros_like, updates), inner_state)

    updates, new_inner_state = jax.lax.cond(
        jnp.logical_and(isfinite, islowerthan),
        do_update,
        reject_update,
        operand=None)

    return updates, MaybeSkipGradientUpdateState(inner_state=new_inner_state)

  return optax.GradientTransformation(init=init, update=update)
