# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Helper utilities for training."""

import functools

from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from flax import optim

import jax
from jax import random
import jax.experimental.optimizers
import jax.numpy as jnp

# Type Stubs
PRNGKey = Any


def create_learning_rate_scheduler(
    factors = "constant * linear_warmup * cosine_decay",
    base_learning_rate = 0.5,
    warmup_steps = 1000,
    decay_steps = 100000):
  """Creates learning rate schedule.

  Interprets factors in the factors string which can consist of:
  * constant: Interpreted as the constant value.
  * linear_warmup: Interpreted as linear warmup until warmup_steps.
  * rsqrt_decay: Divide by square root of max(step, warmup_steps).
  * linear_decay: Linearly decays to zero from maximum rate.
  * cosine_decay: Cyclic cosine decay, uses decay_steps parameter.

  Args:
    factors: Factors separated by "*" that defines the schedule.
    base_learning_rate: The starting constant for the learning rate schedule.
    warmup_steps: How many steps to warm up for in the warmup schedule.
    decay_steps: Number of steps over which to decay rate to zero from maximum
      rate (following warm-up), when using linear or cosine decay.

  Returns:
    The step-dependent learning rate function.

  Raises:
    ValueError: If unrecognized factor is passed in, or the warm-up factor is
      specified with 0 warm-up steps.
  """
  factors = [n.strip() for n in factors.split("*")]

  def step_fn(step):
    """Step to learning rate function."""
    ret = 1.0
    for name in factors:
      if name == "constant":
        ret *= base_learning_rate
      elif name == "linear_warmup":
        if warmup_steps <= 0:
          raise ValueError(
              "Specified 'linear_warmup' factor with warmup_steps=0.")
        ret *= jnp.minimum(1.0, step / warmup_steps)
      elif name == "rsqrt_decay":
        ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
      elif name == "rsqrt_normalized_decay":
        ret *= jnp.sqrt(warmup_steps)
        ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
      elif name == "linear_decay":
        # TODO(b/181607810): Handle case where step - warmup_steps > decay_steps
        progress = jnp.maximum(0.0, (step - warmup_steps) / float(decay_steps))
        ret *= 1.0 - (progress % 1.0)
      elif name == "cosine_decay":
        progress = jnp.maximum(0.0, (step - warmup_steps) / float(decay_steps))
        ret *= jnp.maximum(0.0,
                           0.5 * (1.0 + jnp.cos(jnp.pi * (progress % 1.0))))
      else:
        raise ValueError("Unknown factor %s." % name)
    return jnp.asarray(ret, dtype=jnp.float32)

  return step_fn


def train_step(
    optimizer,
    batch,
    loss_and_metrics_fn,
    learning_rate_fn,
    rng = None,
    clipped_grad_norm = None
):
  """Performs a single training step.

  We typically parallelize this training computation across multiple devices.

  Args:
    optimizer: Underlying model and model state.
    batch: Current batch of training examples.
    loss_and_metrics_fn: Given a batch of examples, a model and a PRNGKey, this
      function returns the model loss and metrics.
    learning_rate_fn: Function mapping training step to learning rate.
    rng: Random number generator key.
    clipped_grad_norm: If set, clip the gradient norm to this value.

  Returns:
    New optimizer (with updated state), training metrics and refreshed PRNGKey.
  """
  # We handle PRNG splitting inside the top pmap to improve efficiency.
  rng, new_rng = random.split(rng)

  loss_fn = functools.partial(loss_and_metrics_fn, batch=batch, rng=rng)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (unused_loss, metrics), grad = grad_fn(optimizer.target)
  grad = jax.lax.pmean(grad, "batch")
  grad, metrics = _measure_and_maybe_clip_grad(grad, metrics, clipped_grad_norm)

  step = optimizer.state.step

  new_optimizer = optimizer.apply_gradient(
      grad, learning_rate=learning_rate_fn(step))

  return new_optimizer, metrics, new_rng


def eval_step(
    params, batch,
    metric_fn
):
  """Performs a single evaluation step.

  We use this wrapper for parallelizing the evaluation computations across
  multiple devices.

  Args:
    params: Current model state.
    batch: Current batch of evaluation examples.
    metric_fn: Function that maps the model state and batch to output model
      metrics.

  Returns:
    Model metrics for given inputs.
  """
  return metric_fn(params, batch)


def _measure_and_maybe_clip_grad(grad,
                                 metrics,
                                 clipped_grad_norm = None):
  """Records and optionally clips gradient."""
  grad_l2_sum = sum([jnp.sum(x**2) for x in jax.tree_leaves(grad)])
  metrics["unclipped_grad_l2_sum"] = grad_l2_sum

  if clipped_grad_norm is not None:
    # Clip gradients after pmean aggregation
    grad = jax.experimental.optimizers.clip_grads(grad, clipped_grad_norm)
    metrics["clipped_grad_l2_sum"] = sum(
        [jnp.sum(x**2) for x in jax.tree_leaves(grad)])
  else:
    # Clipped grad same as unclipped grad
    metrics["clipped_grad_l2_sum"] = grad_l2_sum

  return grad, metrics
