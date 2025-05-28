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

"""Training utils for NDP models."""

import functools

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp

from hct.common import utils


def create_ndp_train_state(model,
                           key,
                           learning_rate,
                           weight_decay,
                           batch_images,
                           batch_hf_obs):
  """Create an NDP training state."""
  init_params = model.init(key, batch_images, batch_hf_obs)
  return utils.TrainStateBN.create(
      apply_fn=functools.partial(model.apply,
                                 method=model.compute_augmented_flow),
      params=init_params["params"],
      batch_stats=init_params["batch_stats"],
      tx=utils.make_optax_adam(learning_rate, weight_decay))


@functools.partial(jax.pmap, axis_name="batch")
def optimize_ndp(state,
                 images,
                 hf_obs,
                 u_true):
  """Do a step of training with the NDP model.

  Args:
    state: batch-norm enabled trainstate, with augmented_flow as the apply_fn.
    images: (num_devices, batch_size, ....): images
    hf_obs: (num_devices, batch_size, x_dim): concurrent hf observations
    u_true: (num_devices, batch_size, num_actions, u_dim): observed controls

  Returns:
    loss: training loss
    state: updated train-state.
  """
  def loss_fn(params):
    (u_preds, losses), updates = state.apply_fn(
        {"params": params, "batch_stats": state.batch_stats},
        images, hf_obs, u_true, train=True, mutable=["batch_stats"])
    return jnp.mean(losses), (u_preds, updates)

  # Get gradients
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, (_, updates)), grads = grad_fn(state.params)

  # Aggregate across devices
  loss = jax.lax.pmean(loss, axis_name="batch")
  grads = jax.lax.pmean(grads, axis_name="batch")
  batch_stats = jax.lax.pmean(updates["batch_stats"], axis_name="batch")

  # Update state
  state = state.apply_gradients(grads=grads)
  state = state.replace(batch_stats=batch_stats)
  return loss, state
