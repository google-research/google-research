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

"""Common loss functions and other training utilities."""

import functools
import os
import time
from typing import Any, Callable, Dict, Sequence, Tuple, Union

from absl import logging
from clu import checkpoint
from clu import metric_writers
from clu import parameter_overview
from clu import periodic_actions
import flax
from flax.deprecated import nn
import jax
import jax.numpy as jnp
from lib import utils  # pytype: disable=import-error
from lib.typing import LossOrMetric  # pytype: disable=import-error
import ml_collections
import optax
import tensorflow as tf


@flax.struct.dataclass
class TrainState:
  """Data structure for checkpoint of the model."""
  step: int
  model_state: nn.Collection
  model_params: nn.Collection
  optimizer_state: Any


@jax.jit
def cross_entropy(logits, labels, stats):
  del stats
  logp = jax.nn.log_softmax(logits)
  loglik = jnp.take_along_axis(logp, labels[:, None], axis=1)
  return -jnp.mean(loglik)


@jax.jit
def binary_cross_entropy(logits, labels, stats):
  del stats
  x, y = logits, labels  # Makes the math more readable.
  max_val = jnp.clip(logits, 0, None)
  loss = x - x * y + max_val + jnp.log(
      jnp.exp(-max_val) + jnp.exp((-x - max_val)))
  return jnp.mean(loss)


@jax.jit
def multilabel_accuracy(logits, labels, stats):
  """Strict multilabel classification accuracy (all labels have to be correct).

  Args:
    logits: Tensor, shape [batch_size, num_classes]
    labels: Tensor, shape [batch_size, num_classes], values in {0, 1}
    stats: Dict of statistics output by the model.

  Returns:
    per_sample_success: Tensor of shape [batch_size]
  """
  del stats
  error = jnp.abs(labels - jnp.round(nn.sigmoid(logits)))
  return 1. - jnp.max(error, axis=-1)


@jax.jit
def accuracy(logits, labels, stats):
  del stats
  predictions = jnp.argmax(logits, axis=-1)
  return jnp.mean(predictions == labels)


@functools.partial(
    jax.pmap, axis_name="batch", static_broadcasted_argnums=(2, 3, 4, 5))
def train_step(
    state,
    batch,
    module,
    loss_fn,
    optimizer,
    metrics_dict,
    rng
):
  """Perform a single training step.

  Args:
    state: Current training state. Updated training state will be returned.
    batch: Training inputs for this step.
    module: Module function.
    loss_fn: Loss function that takes logits and labels as input.
    optimizer: Optax optimizer to compute updates from gradients.
    metrics_dict: A dictionary of metrics, mapping names to metric functions.
    rng: Jax pseudo-random number generator key.

  Returns:
    Tuple of updated state, dictionary with metrics, and updated PRNG key.
  """

  rng, new_rng = jax.random.split(rng)

  def impl_loss_fn(model_params):
    with nn.stochastic(rng), nn.stateful(state.model_state) as new_model_state:
      logits, stats = module.call(model_params, batch["image"])
    losses = loss_fn if isinstance(loss_fn, (list, tuple)) else [loss_fn]
    loss = sum(l(logits, batch["label"], stats) for l in losses)
    return loss, (logits, new_model_state, stats)

  grad_fn = jax.value_and_grad(impl_loss_fn, has_aux=True)
  with nn.stochastic(rng):
    (_, loss_aux), grad = grad_fn(state.model_params)
  logits, new_model_state, stats = loss_aux
  # Compute average gradient across multiple workers.
  grad = jax.lax.pmean(grad, axis_name="batch")
  updates, new_opt_state = optimizer.update(grad, state.optimizer_state,
                                            params=state.model_params)
  new_model_params = optax.apply_updates(state.model_params, updates)
  metrics = {m: fn(logits, batch["label"], stats)
             for (m, fn) in metrics_dict.items()}
  metrics = jax.lax.all_gather(metrics, axis_name="batch")
  stats = jax.lax.all_gather(stats, axis_name="batch")
  stats = jax.tree_map(lambda x: x[0], stats)
  new_state = state.replace(  # pytype: disable=attribute-error
      step=state.step + 1,
      optimizer_state=new_opt_state,
      model_state=new_model_state,
      model_params=new_model_params)
  return new_state, grad, updates, metrics, stats, new_rng


@functools.partial(
    jax.pmap, axis_name="batch", static_broadcasted_argnums=(1, 3))
def eval_step(
    state,
    module,
    batch,
    metrics_dict,
    rng):
  """Compute the metrics for the given model in inference mode.

  The model is applied to the inputs using all devices on the host. Afterwards
  metrics are averaged across *all* devices (of all hosts).

  Args:
    state: Replicated model state.
    module: Model function.
    batch: Inputs that should be evaluated.
    metrics_dict: A dictionary of metrics, mapping names to metric functions.
    rng: Jax pseudo-random number generator key.

  Returns:
    Dictionary of replicated metrics, stats output by the model and updated PRNG
      key.
  """
  rng, new_rng = jax.random.split(rng)
  with nn.stochastic(rng), flax.nn.stateful(state.model_state, mutable=False):
    logits, stats = module.call(state.model_params, batch["image"], train=False)
  metrics = {m: fn(logits, batch["label"], stats)
             for (m, fn) in metrics_dict.items()}
  metrics = jax.lax.all_gather(metrics, axis_name="batch")
  stats = jax.lax.all_gather(stats, axis_name="batch")
  return metrics, stats, new_rng


def evaluate(
    state,
    module,
    eval_ds,
    metrics_dict,
    eval_rngs
):
  """Evaluate the model on the given dataset."""
  metrics = utils.Means()
  start = time.time()
  is_first = True
  for batch in eval_ds:
    # Use ._numpy() to avoid copy.
    batch = jax.tree_map(lambda x: x._numpy(), batch)  # pylint: disable=protected-access

    device_metrics, stats, eval_rngs = eval_step(state, module, batch,
                                                 metrics_dict, eval_rngs)
    metrics.append(flax.jax_utils.unreplicate(device_metrics))
    if is_first:
      first_batch_stats = flax.jax_utils.unreplicate(stats)
      first_batch_stats = jax.tree_map(lambda x: x[0], first_batch_stats)
      is_first = False

  elapsed = time.time() - start
  logging.info("Evaluation took %.1fs", elapsed)
  metrics_result = metrics.result()
  metrics_result["eval_timing"] = elapsed
  return metrics_result, first_batch_stats, eval_rngs


def create_model(module, input_shape, rng):
  """Instanciates the model."""
  model_rng, init_rng = jax.random.split(rng)
  with nn.stochastic(model_rng), nn.stateful() as init_state:
    x = jnp.ones(input_shape, dtype=jnp.float32)
    _, init_params = module.init(init_rng, x)
  model = nn.Model(module, init_params)
  return model, init_params, init_state


def training_loop(
    *,
    module,
    rng,
    train_ds,
    eval_ds,
    loss_fn,
    optimizer,
    train_metrics_dict,
    eval_metrics_dict,
    stats_aggregators,
    config,
    workdir,
):
  """Runs a training and evaluation loop.

  Args:
    module: The module that should be trained.
    rng: A jax pseudo-random number generator key.
    train_ds: Dataset used for training.
    eval_ds: Dataset used for evaluation.
    loss_fn: Loss function to use for training.
    optimizer: Optax optimizer to use for training.
    train_metrics_dict: Collection of metrics to be collected during training.
    eval_metrics_dict: Collection of metrics to be collected during evaluation.
    stats_aggregators: Dictionary of statistics aggregator functions to be run
      on the first evaluation batch. These functions ingest the stats returned
      by the model and output a Dict[str, image/scalar] that will be logged.
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.

  Raises:
    RuntimeError: If a training metric is NaN or inf.

  Returns:
    Training state.
  """
  rng, model_rng = jax.random.split(rng)
  input_shape = tuple(train_ds.element_spec["image"].shape[1:])
  model, init_params, init_state = create_model(module, input_shape, model_rng)
  parameter_overview.log_parameter_overview(model.params)

  # Load a pretrained model parameters and state. Ignore the step and the
  # optimizer state in the checkpoint.
  pretrained_path = config.get("pretrained_checkpoint", "")
  if pretrained_path:
    logging.info("Load pretrained weights from '%s'", pretrained_path)
    state_dict = checkpoint.load_state_dict(pretrained_path)
    flatten_model_params = utils.flatten_dict(state_dict["model_params"],
                                              sep="/")
    model_state = state_dict["model_state"]

    # A prefix can be used to replace only a subpart of the network (e.g the
    # encoder). Prepend the prefix (if any) to model parameters and states.
    prefix = config.get("pretrained_prefix", "")
    if prefix:
      flatten_model_params = utils.add_prefix_to_dict_keys(
          flatten_model_params, f"{prefix}/")
      model_state = utils.add_prefix_to_dict_keys(
          model_state, f"/{prefix}")

    # Merge the params/state from the checkpoint into the initial params/state.
    flatten_init_params = utils.flatten_dict(init_params, sep="/")
    flatten_init_params, ignored_params = utils.override_dict(
        flatten_init_params, flatten_model_params)
    init_params = utils.unflatten_dict(flatten_init_params, delimiter="/")
    init_state, _ = utils.override_dict(init_state, model_state)

    if ignored_params:
      logging.warning("%d/%d parameters from the pretrained checkpoint "
                      "were ignored: %s", len(ignored_params),
                      len(flatten_init_params), ignored_params)

  optimizer_state = optimizer.init(init_params)

  state = TrainState(
      step=1,
      model_params=init_params,
      model_state=init_state,
      optimizer_state=optimizer_state)  # type: ignore
  # Do not keep a copy of the initial model.
  del init_params, init_state, optimizer_state

  train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types
  checkpoint_dir = os.path.join(workdir, "checkpoints")

  ckpt = checkpoint.MultihostCheckpoint(checkpoint_dir)
  state = ckpt.restore_or_initialize(state)
  initial_step = int(state.step)
  # Replicate our parameters.
  state = flax.jax_utils.replicate(state)

  writer = metric_writers.create_default_writer(
      workdir, just_logging=jax.host_id() > 0)
  step_timer = utils.StepTimer(
      batch_size=config.batch_size, initial_step=initial_step)

  # Write config to the summary files. This makes the hyperparameters available
  # in TensorBoard and makes comparison of runs with tensorboard/ easier.
  if initial_step == 1:
    writer.write_hparams(utils.flatten_dict(config.to_dict()))

  # Generate per-device PRNG keys for the training loop.
  rng, train_rng = jax.random.split(rng)
  train_rngs = jax.random.split(train_rng, jax.local_device_count())

  # Generate per-device PRNG keys for model evaluation.
  rng, eval_rng = jax.random.split(rng)
  eval_rngs = jax.random.split(eval_rng, jax.local_device_count())

  logging.info("Starting training loop at step %d.", initial_step)
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=config.num_train_steps, writer=writer)
  train_metrics = utils.Means()

  do_eval_only = config.get("do_eval_only", False)
  if do_eval_only:
    config.num_train_steps = 1

  debug_enabled = config.get("debug", False)
  previous_grads = grads = None
  previous_updates = updates = None
  previous_state = None
  for step in range(initial_step, config.num_train_steps + 1):
    is_last_step = step == config.num_train_steps
    if debug_enabled:
      previous_grads = grads
      previous_updates = updates
      previous_state = state

    # Skip the training if only do the eval.
    if not do_eval_only:
      # Use ._numpy() to avoid copy.
      batch = jax.tree_map(lambda x: x._numpy(), next(train_iter))  # pylint: disable=protected-access
      state, grads, updates, metrics, training_stats, train_rngs = train_step(
          state, batch, module, loss_fn, optimizer, train_metrics_dict,
          train_rngs)
      train_metrics.append(flax.jax_utils.unreplicate(metrics))

      # Update topk temperature with linearly decreasing schedule if enabled.
      if (config.get("linear_decrease_perturbed_sigma", False) and
          config.get("selection_method", "") == "perturbed-topk"):

        model_state = state.model_state.as_dict()

        if "/PatchNet_0" in model_state:
          net_str = "/PatchNet_0"
        else:
          net_str = "/"

        progress = step / config.num_train_steps
        sigma_multiplier = 1. - progress
        previous_mult = model_state[net_str]["sigma_mutiplier"]
        sigma_multiplier = sigma_multiplier + jnp.zeros_like(previous_mult)
        model_state[net_str]["sigma_mutiplier"] = sigma_multiplier
        state = state.replace(model_state=nn.Collection(model_state))

      if debug_enabled:
        if utils.has_any_inf_or_nan(metrics):
          # Save checkpoint
          if previous_state:
            ckpt.save(flax.jax_utils.unreplicate(previous_state))
          ckpt.save(flax.jax_utils.unreplicate(state))

          # Log gradients and updates.
          if previous_grads or previous_updates:
            write_gradient_histogram(writer, step,
                                     grads=previous_grads,
                                     updates=previous_updates)
          write_gradient_histogram(writer, step + 1,
                                   grads=grads, updates=updates)

          raise RuntimeError("A training metric took an invalid value: "
                             f"{metrics}.")

      logging.log_first_n(logging.INFO, "Finished training step %d.", 3, step)
      report_progress(step)

      if step % config.log_loss_every_steps == 0 or is_last_step:
        results = train_metrics.result()
        writer.write_scalars(step, results)
        writer.write_scalars(step, step_timer.get_and_reset(step))
        if utils.has_any_inf_or_nan(results):
          raise ValueError("A training metric took an invalid value.")
        train_metrics.reset()

    if (step % config.checkpoint_every_steps == 0 or is_last_step):
      with step_timer.paused():
        ckpt.save(flax.jax_utils.unreplicate(state))

    # Evaluation
    if step % config.eval_every_steps == 0 or is_last_step:
      with step_timer.paused():
        eval_metrics, first_batch_stats, eval_rngs = evaluate(
            state, module, eval_ds, eval_metrics_dict, eval_rngs)

      if jax.host_id() == 0:
        log_histograms = config.get("log_histograms", False)
        log_images = config.get("log_images", True)
        # Log the last gradients and updates histograms.
        if not do_eval_only:
          write_stats_results(writer, step, training_stats, stats_aggregators,
                              prefix="train/", log_images=log_images)
          if log_histograms:
            write_gradient_histogram(writer, step, grads=grads, updates=updates)

        write_stats_results(writer, step, first_batch_stats,
                            stats_aggregators, prefix="eval/",
                            log_images=log_images)

        # write patch representation histograms
        if (log_histograms and first_batch_stats and
            "patch_representations" in first_batch_stats):
          patch_representations = first_batch_stats["patch_representations"]
          writer.write_histograms(step, {
              "patch_representations": patch_representations
          })

        if eval_metrics:
          writer.write_scalars(step, eval_metrics)

  writer.flush()
  return state


def write_stats_results(writer, step, stats, stats_aggregators, *,
                        prefix=None, log_images=True):
  """Computes and logs aggregated stats."""
  prefix = prefix or ""

  # Computes aggregated stats results and split scalars and images for
  # logging with metric_writers.
  # pylint: disable=g-complex-comprehension
  stats_results = {name: result
                   for agg in stats_aggregators
                   for name, result in agg(stats).items()}
  scalar_results = {n: v.reshape()
                    for n, v in stats_results.items()
                    if v.size == 1}
  # pylint: enable=g-complex-comprehension

  if log_images:
    image_results = {k[len("image_"):]: v.reshape((-1,) + v.shape[-3:])
                     for k, v in stats_results.items()
                     if k.startswith("image_")}

    if image_results:
      writer.write_images(step,
                          utils.add_prefix_to_dict_keys(image_results, prefix))
  if scalar_results:
    writer.write_scalars(step,
                         utils.add_prefix_to_dict_keys(scalar_results, prefix))

  return stats_results


def write_gradient_histogram(writer, step, *, grads=None, updates=None):
  """Log computed gradients and/or updates histograms."""
  histograms = {"grad": grads, "update": updates}
  histograms = {k: v for k, v in histograms.items() if v is not None}
  if not histograms: return

  # Transpose a histograms dict from
  # {"grad": {"param1": Tensor}, "update": {"param1": Tensor}} to
  # {"param1": {"grad": Tensor, "update": Tensor}} such that the gradient and
  # the transformed updates appear next to each other in tensorboard.
  histograms = jax.tree_transpose(
      jax.tree_structure({k: 0 for k in histograms.keys()}),
      jax.tree_structure(next(iter(histograms.values()))),
      histograms)

  histograms = utils.flatten_dict(histograms, sep=".")
  writer.write_histograms(step, histograms)
