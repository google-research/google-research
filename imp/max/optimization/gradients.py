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

"""Base modules for updating models during optimization."""

from typing import Any

from flax import struct
from flax.core import scope
import jax
import jax.numpy as jnp

from imp.max.core import constants
from imp.max.utils import tree as mtu
from imp.max.utils import typing


PROBES = constants.FlaxCollection.PROBES
AUX_LOSS = constants.FlaxCollection.AUX_LOSS


@struct.dataclass
class _MicroBatchInput:
  data: typing.Data
  rngs: dict[str, jax.Array]


@struct.dataclass
class _MicroBatchCarry:
  mutables: typing.NestedDict
  grads: typing.NestedDict
  loss: jax.Array


@struct.dataclass
class _MicroBatchOutput:
  probes: typing.NestedDict
  metrics: typing.NestedDict


def _split_batch_for_microbatching(
    array,
    microbatch_steps,
):
  """Splits the batch dimension of the array for microbatching."""
  batch_size = array.shape[0]
  if batch_size % microbatch_steps != 0:
    raise ValueError(
        f'`batch_size` must be divisible by number of total microbatch steps. '
        f'Instead, received {batch_size=} and {microbatch_steps=}.')

  # First split the batch dimension into microbatch splits
  microbatch_size = batch_size // microbatch_steps
  split_batch_array = jnp.reshape(
      array, (microbatch_size, microbatch_steps, *array.shape[1:]))

  # Then swap axes because JAX automatically continues sharding the leading dim
  split_batch_array = jnp.moveaxis(split_batch_array, 0, 1)
  return split_batch_array


def _replicate_rngs_for_microbatching(
    array,
    microbatch_steps,
):
  """Replicates the rngs for microbatching."""
  return jnp.tile(array, (microbatch_steps, *([1] * array.ndim)))


def apply_model_and_calculate_loss(
    params,
    mutables,
    model,
    data,
    rngs,
    obj_fn,
    mutable,
):
  """Calculates the loss value by performing a forward step."""

  # Merge model params with the mutables from previous step
  variables = {**params, **mutables}

  # Forward call while assuring certain collections are not mutable
  model_data, mutables = model.apply(
      rngs=rngs,
      variables=variables,
      data=data,
      deterministic=False,
      mutable=mutable,
  )

  loss, metrics = obj_fn(model_data)

  # Fetch probed information (if any)
  probes = mutables.pop(PROBES, {})

  # Add auxiliary losses (if any) to the objecive value
  auxiliary_losses = mutables.pop(AUX_LOSS, {})
  if auxiliary_losses:
    loss += mtu.tree_aggregate_array(auxiliary_losses, jnp.sum)

  return loss, (mutables, probes, metrics)


def apply_value_and_grad_fn_by_microbatching(
    value_and_grad_fn,
    params,
    mutables,
    model,
    data,
    rngs,
    obj_fn,
    mutable,
    microbatch_steps,
):
  """Calculates the loss and grads by accumulating via microbatching."""
  if microbatch_steps <= 1:
    raise ValueError(
        '`microbatch_steps` must be greater than 1. Instead, received '
        f'{microbatch_steps=}'
    )

  def microbatch_step_value_and_grad_fn(
      carry,
      inputs,
  ):
    step_values, step_grads = value_and_grad_fn(
        params, carry.mutables, model,
        inputs.data, inputs.rngs, obj_fn, mutable)
    (step_loss, (step_mutables, step_probes, step_metrics)) = step_values
    carry_update_fn = lambda old, new: old.astype(new) + new
    accumulated_grads, accumulated_loss = jax.tree.map(
        carry_update_fn,
        (carry.grads, carry.loss),
        (step_grads, step_loss),
    )
    return (
        _MicroBatchCarry(
            mutables=step_mutables,
            grads=accumulated_grads,
            loss=accumulated_loss,
        ),
        _MicroBatchOutput(
            probes=step_probes,
            metrics=step_metrics,
        ),
    )

  # Initialize inputs and carry
  inputs = _MicroBatchInput(
      data=jax.tree.map(
          lambda d: _split_batch_for_microbatching(d, microbatch_steps),
          data,
      ),
      rngs=jax.tree.map(
          lambda r: _replicate_rngs_for_microbatching(r, microbatch_steps),
          rngs,
      ),
  )
  carry = _MicroBatchCarry(
      mutables=mutables,
      grads=jax.tree.map(jnp.zeros_like, params),
      loss=jnp.zeros(()),
  )

  # Perform scan
  carry, outputs = jax.lax.scan(
      microbatch_step_value_and_grad_fn,
      carry, inputs, length=microbatch_steps,
  )

  # Normalize the accumulated losses and grads
  norm_denominator = jnp.asarray(microbatch_steps, dtype=jnp.int32)
  loss_and_grads_normalizer = lambda x: x / norm_denominator.astype(x.dtype)
  grads = jax.tree.map(loss_and_grads_normalizer, carry.grads)
  loss = jax.tree.map(loss_and_grads_normalizer, carry.loss)

  # Reduce probes and metrics over the first axis. These have a shape of
  # [microbatch_steps, ...] since scan yields probes and metrics out of
  # the value_and_grad_fn at each micro-step
  probes_reducer = lambda x: x[-1, :]
  metrics_reducer = lambda x: jnp.mean(x, axis=0)
  probes = jax.tree.map(probes_reducer, outputs.probes)
  metrics = jax.tree.map(metrics_reducer, outputs.metrics)
  mutables = carry.mutables
  return (loss, (mutables, probes, metrics)), grads


def apply_model_and_calculate_loss_and_grads(
    params,
    mutables,
    model,
    data,
    rngs,
    obj_fn,
    mutable,
    microbatch_steps,
):
  """Calculates the loss and grads by performing forward & bacward steps."""
  # We only take grads wrt the first dimension (i.e. params)
  value_and_grad_fn = jax.value_and_grad(
      apply_model_and_calculate_loss, argnums=0, has_aux=True)
  if microbatch_steps == 1:
    values, grads = value_and_grad_fn(
        params, mutables, model, data, rngs, obj_fn, mutable)

  else:
    values, grads = apply_value_and_grad_fn_by_microbatching(
        value_and_grad_fn, params, mutables, model, data,
        rngs, obj_fn, mutable, microbatch_steps)

  return values, grads
