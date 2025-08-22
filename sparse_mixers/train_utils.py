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

"""Helper utilities for training."""

import functools
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import flax
from flax import serialization
from flax import traverse_util
from flax.training import train_state as train_state_lib
import jax
from jax import random
import jax.example_libraries.optimizers
import jax.numpy as jnp
import ml_collections
import numpy as np

from sparse_mixers import core_utils
from sparse_mixers.layers import DiversityMetrics
from sparse_mixers.models import ClassificationStats
from sparse_mixers.models import PretrainingStats

# Type Stubs
Batch = Mapping[str, jnp.ndarray]
Gradients = flax.core.FrozenDict
Loss = float
Params = flax.core.FrozenDict
PRNGKey = Any
Stats = Union[PretrainingStats, ClassificationStats]


class FlaxTrainState(train_state_lib.TrainState):
  """Flax train state subclass with support for restoring state."""

  def restore_state(self, state_dict):
    """Restore parameter and optimizer state from state dictionary.

    Adapted from
    https://github.com/google-research/t5x/blob/main/t5x/optimizers.py. Includes
    support to handle `optax.EmptyState`.

    Args:
      state_dict: Contains desired new parameters and optimizer state

    Returns:
      Updated train state.
    """
    params = serialization.from_state_dict(self.params, state_dict["params"])

    # Get all the possible keys in the reference optimizer state.
    flat_ref_opt_state_dict = traverse_util.flatten_dict(
        serialization.to_state_dict(self.opt_state),
        keep_empty_nodes=True,
        sep="/")

    flat_src_opt_state_dict = dict(
        traverse_util.flatten_dict(state_dict["opt_state"], sep="/"))
    # Adding the empty paths back to flat_src_opt_state_dict.
    for k, v in flat_ref_opt_state_dict.items():
      if k in flat_src_opt_state_dict:
        continue
      # The key is not in the input state dict, presumably because it
      # corresponds to an empty dict.
      if v != traverse_util.empty_node:
        raise ValueError(
            f"Failed to restore optimizer state, path {k} is not present "
            "in the input optimizer state dict.")
      flat_src_opt_state_dict[k] = v

    # Restore state from the enhanced state dict.
    opt_state = serialization.from_state_dict(
        self.opt_state,
        traverse_util.unflatten_dict(flat_src_opt_state_dict, sep="/"))
    return self.replace(params=params, opt_state=opt_state)


def validate_config(config):
  """Validates training config.

  Args:
    config: Training configuration file.

  Raises:
    ValueError if any necessary config fields are missing or the config is
      inconsistent (internally or with the available devices).
  """
  n_hosts = jax.process_count()
  n_devices_per_host = jax.local_device_count()
  n_devices = jax.device_count()

  if config.train_batch_size % n_devices > 0:
    raise ValueError(
        "Training batch size must be divisible by the total number of devices, "
        "but train_batch_size = %d, while total number of devices = %d "
        "(%d hosts, each with %d devices)" %
        (config.train_batch_size, n_devices, n_hosts, n_devices_per_host))

  if config.eval_batch_size % n_devices > 0:
    raise ValueError(
        "Eval batch size must be divisible by the total number of devices, "
        "but eval_batch_size = %d, while total number of devices = %d "
        "(%d hosts, each with %d devices)" %
        (config.eval_batch_size, n_devices, n_hosts, n_devices_per_host))

  if (config.gradient_accum_steps and
      config.train_batch_size % config.gradient_accum_steps != 0):
    raise ValueError(
        "If gradient_accum_steps is set, then the training batch size must be "
        "divisible by gradient_accum_steps, however the train_batch_size = %d, while "
        "gradient_accum_steps = %d" %
        (config.train_batch_size, config.gradient_accum_steps))

  if 1 < config.num_experts < n_devices:
    raise ValueError(
        "When training MoE models under pmap, the number of experts must be "
        ">= the number of available devices, but num_experts = %d, while total "
        "number of devices = %d." % (config.num_experts, n_devices))


def collect_metrics(stats):
  """Concatenates a sequence of model stats onto a single stats object.

  Args:
    stats: [N_STEPS] array of model outputs. Each field on each stat contains a
      [BATCH_SIZE] array of values.

  Returns:
    Single stats object wherein each field is populated by a
    [N_STEPS * BATCH_SIZE] array.
  """
  stats_np = jax.device_get(stats)
  concat_args = lambda *args: np.concatenate(args) if isinstance(  # pylint: disable=g-long-lambda
      args, list) else np.asarray(args)
  result = jax.tree.map(concat_args, *stats_np)
  return result


def compute_pretraining_metrics(stats,
                                record_grad_norm = True
                               ):
  """Computes pre-training loss and accuracy metrics.

  Args:
    stats: Raw model predictions and example labels.
    record_grad_norm: Whether or not to record the L2-norm of the gradient.
      Typically only performed during training.

  Returns:
    Model loss and accuracy metrics.
  """
  metrics = {
      "masked_lm_loss":
          jnp.sum(stats.masked_lm_loss) /
          jnp.sum(stats.masked_lm_normalization),
      "next_sentence_loss":
          jnp.sum(stats.next_sentence_loss) /
          jnp.sum(stats.num_next_sentence_labels),
      "masked_lm_accuracy":
          jnp.sum(stats.masked_lm_correct) / jnp.sum(stats.masked_lm_total),
      "next_sentence_accuracy":
          jnp.sum(stats.next_sentence_correct) /
          jnp.sum(stats.num_next_sentence_labels)
  }
  metrics["loss"] = metrics["masked_lm_loss"] + metrics["next_sentence_loss"]

  if record_grad_norm:
    metrics.update({"grad_l2_norm": jnp.sqrt(jnp.sum(stats.grad_l2_sum))})

  if stats.expert_metrics:
    # Mixture of experts specific metrics are averaged across experts/devices.
    metrics["auxiliary_loss"] = jnp.mean(stats.expert_metrics.auxiliary_loss)
    metrics["router_z_loss"] = jnp.mean(stats.expert_metrics.router_z_loss)
    metrics["loss"] += metrics["auxiliary_loss"] + metrics["router_z_loss"]
    metrics["fraction_tokens_left_behind"] = jnp.mean(
        stats.expert_metrics.fraction_tokens_left_behind)
    metrics["expert_usage"] = jnp.mean(stats.expert_metrics.expert_usage)
    metrics["router_confidence"] = jnp.mean(
        stats.expert_metrics.router_confidence)

  return metrics  # pytype: disable=bad-return-type  # jax-types


def compute_classification_metrics(
    stats, is_regression_task):
  """Computes classification loss and accuracy metrics.

  Args:
    stats: Raw model predictions and example labels.
    is_regression_task: Whether or not the current task is a regression task.

  Returns:
    Model loss and accuracy metrics.
  """
  metrics = {
      "loss": jnp.sum(stats.batch_loss) / jnp.sum(stats.num_labels),
  }
  if not is_regression_task:
    metrics["accuracy"] = jnp.sum(stats.correct_predictions) / jnp.sum(
        stats.num_labels)

  if stats.expert_metrics:
    # Mixture of experts specific metrics are averaged across experts/devices.
    metrics["auxiliary_loss"] = jnp.mean(stats.expert_metrics.auxiliary_loss)
    metrics["router_z_loss"] = jnp.mean(stats.expert_metrics.router_z_loss)
    metrics["loss"] += metrics["auxiliary_loss"] + metrics["router_z_loss"]
    metrics["fraction_tokens_left_behind"] = jnp.mean(
        stats.expert_metrics.fraction_tokens_left_behind)
    metrics["expert_usage"] = jnp.mean(stats.expert_metrics.expert_usage)
    metrics["router_confidence"] = jnp.mean(
        stats.expert_metrics.router_confidence)

  return metrics  # pytype: disable=bad-return-type  # jnp-type


def summarize_expert_metrics(state, auxiliary_loss_factor,
                             router_z_loss_factor):
  """Summarizes per-layer expert diversity metrics for the entire encoder.

  Args:
    state: Encoder parameters.
    auxiliary_loss_factor: Factor by which to scale auxiliary load balancing
      loss for mixture of experts models. The raw auxiliary losses will be
      summed and then scaled by this factor.
    router_z_loss_factor: Factor by which to scale router z-loss for mixture of
      experts models.

  Returns:
    Expert diversity metrics extracted from input state.
  """
  diversity_metrics = [
      m for m in traverse_util.flatten_dict(flax.core.unfreeze(state)).values()
      if isinstance(m, DiversityMetrics)
  ]

  total_aux_loss = jnp.sum(
      jnp.asarray([m.auxiliary_loss for m in diversity_metrics]))
  total_aux_loss *= auxiliary_loss_factor
  total_router_z_loss = jnp.sum(
      jnp.asarray([m.router_z_loss for m in diversity_metrics]))
  total_router_z_loss *= router_z_loss_factor
  avg_fraction_tokens_left_behind = jnp.mean(
      jnp.asarray([m.fraction_tokens_left_behind for m in diversity_metrics]))
  avg_expert_usage = jnp.mean(
      jnp.asarray([m.expert_usage for m in diversity_metrics]))
  avg_confidence = jnp.mean(
      jnp.asarray([m.router_confidence for m in diversity_metrics]))

  return DiversityMetrics(total_aux_loss, total_router_z_loss,  # pytype: disable=wrong-arg-types  # jnp-type
                          avg_fraction_tokens_left_behind, avg_expert_usage,
                          avg_confidence)


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


def pmap_train_step(
    train_state,
    batch,
    rng,
    loss_and_metrics_fn,
    axis_name = "batch",
    sharded_match_fn = None,
    gradient_accum_steps = None
):
  """Performs a single training step.

  This training step computation assumes that the training will be parallelized
  using jax.pmap.

  Args:
    train_state: Training state holding model params, state and gradient.
    batch: Current batch of training examples.
    rng: Random number generator key.
    loss_and_metrics_fn: Given the current model parameters, a batch of
      examples, and a PRNGKey, this function returns the model loss and metrics.
    axis_name: Axis name used by JAX for SPMD. This should match the axis name
      of jax.pmap.
    sharded_match_fn: Filter function for distinguishing sharded (mixture of
      expert) parameters from replicated parameters.
    gradient_accum_steps: Number of mini-steps over which to split batch and
      accumulate the gradient. If None or 1, no gradient accumulation is used.

  Returns:
    New optimizer (with updated state), training metrics and refreshed PRNGKey.
  """
  # Bind the rng key to the device id (which is unique across hosts).
  rng = random.fold_in(rng, jax.lax.axis_index(axis_name))

  # We handle PRNG splitting inside the top pmap to improve efficiency.
  rng, new_rng = random.split(rng)

  loss_fn = functools.partial(loss_and_metrics_fn, rng=rng)
  grads, metrics = _accumulate_gradient(train_state.params, batch, loss_fn,
                                        gradient_accum_steps)
  # Average gradients among replicas of each parameter.
  grads = _pmean_with_sharded_params(grads, sharded_match_fn)

  # Record L2 norm of gradients.
  metrics = metrics.replace(
      grad_l2_sum=sum([jnp.sum(x**2) for x in jax.tree.leaves(grads)]))

  new_train_state = train_state.apply_gradients(grads=grads)

  return new_train_state, metrics, new_rng


def _accumulate_gradient(
    params,
    batch,
    loss_fn,
    accum_steps = None):
  """Accumulate gradient over multiple steps to save on memory."""
  grad_fn = jax.grad(loss_fn, has_aux=True)

  if accum_steps and accum_steps > 1:
    split_fn = functools.partial(
        jnp.split, indices_or_sections=accum_steps, axis=0)
    mini_batches = jax.tree.map(lambda x: jnp.asarray(split_fn(x)), batch)

    def get_mini_batch(big_batch, step):
      """Extracts mini-batch for specified step."""
      return {k: v[step] for k, v in big_batch.items()}

    def accumulate(step, state):
      """Updates current state with loss, grads and metrics for current step."""
      mini_grad, mini_metrics = grad_fn(
          params, batch=get_mini_batch(mini_batches, step))
      old_grad, old_metrics = state
      new_grad = jax.tree.map(jnp.add, old_grad, mini_grad)
      new_metrics = jax.tree.map(jnp.add, old_metrics, mini_metrics)
      return new_grad, new_metrics

    start_grad, start_metrics = grad_fn(
        params, batch=get_mini_batch(mini_batches, 0))
    accumulated_state = jax.lax.fori_loop(1, accum_steps, accumulate,
                                          (start_grad, start_metrics))
    return jax.tree.map(lambda x: x / accum_steps, accumulated_state)
  else:
    return grad_fn(params, batch)


def _pmean_with_sharded_params(grads,
                               sharded_match_fn,
                               axis_name = "batch"):
  """Computes pmeans of non-sharded params; leaves sharded params untouched."""
  if sharded_match_fn is None:
    return jax.lax.pmean(grads, axis_name=axis_name)

  names_and_grads, tree_def = core_utils.tree_flatten_with_names(grads)
  non_sharded_grads_mean = jax.lax.pmean(([
      None if sharded_match_fn(name) else grad for name, grad in names_and_grads
  ]),
                                         axis_name=axis_name)
  grads_mean = tree_def.unflatten([
      grad if sharded_match_fn(name) else grad_mean
      for (name,
           grad), grad_mean in zip(names_and_grads, non_sharded_grads_mean)
  ])
  return grads_mean
