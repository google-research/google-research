# coding=utf-8
# Copyright 2023 The Google Research Authors.
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

"""Common effect handler implementations."""

from typing import Any, Sequence

import jax
import jax.numpy as jnp


def choose_grad(lr, params, k, lk):
  """Gradient-based parameter update."""
  grads = jax.grad(lk, argnums=(1))(lr, params)
  new_params = jax.tree_map(lambda p, g: p - lr * g, params, grads)
  return k(lr, new_params)


def choose_enumerate(options, k, lk):
  """Enumerative search."""
  if not isinstance(options, Sequence):
    raise ValueError(
        f'options {options} must be a Sequence; instead got {type(options)}'
    )

  losses = jax.vmap(lk)(jnp.asarray(options))
  best_loss_index = jnp.argmin(losses)
  best_option = jnp.asarray(options).at[best_loss_index].get()
  return k(best_option)
