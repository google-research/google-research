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

"""Function to train the rendering model."""

import functools
from typing import Any, Callable, Tuple

from absl import logging

from clu import metric_writers
from clu import metrics
from clu import periodic_actions
import flax
import flax.jax_utils as flax_utils
import flax.linen as nn
from flax.training import checkpoints
from flax.training import train_state
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np

from omnimatte3D.src import datasets
from omnimatte3D.src import models
from omnimatte3D.src.utils import file_utils
from omnimatte3D.src.utils import schedule_utils
from omnimatte3D.src.utils import train_utils


def train_step(
    model,
    rng,
    state,
    batch,
    alpha_fn_dict,
    learning_rate_fn,
    weight_decay,
    metric_collector,
):
  """Perform a single training step.

  Args:
    model: Flax module for the model. The apply method must take input images
      and a boolean argument indicating whether to use training or inference
      mode.
    rng: random number generator.
    state: State of the model (optimizer and state).
    batch: Training inputs for this step.
    alpha_fn_dict: to store the alpha values for the loss terms.
    learning_rate_fn: Function that computes the learning rate given the step
      number.
    weight_decay: Weighs L2 regularization term.
    metric_collector: To store logging metrics.

  Returns:
    The new model state and dictionary with metrics.
  """
  logging.info("train_step(batch=%s)", batch)

  step = state.step + 1
  lr = learning_rate_fn(step)
  alpha_dict = jax.tree.map(lambda fn: fn(step), alpha_fn_dict)

  def loss_fn(params):
    variables = {"params": params}
    out = model.apply(variables, batch)

    # ------------------------------------------------------------------------
    # Compute the loss.
    pred_loss, stat_dict = out.compute_total_loss(batch, alpha_dict)

    # ------------------------------------------------------------------------
    # Weight Regularization
    weight_penalty_params = jax.tree_util.tree_leaves(variables["params"])
    weight_l2 = sum(
        [jnp.sum(x**2) for x in weight_penalty_params if x.ndim > 1]
    )
    weight_penalty = weight_decay * 0.5 * weight_l2
    # ------------------------------------------------------------------------

    total_loss = pred_loss + weight_penalty
    stat_dict["weight_l2"] = weight_l2

    return total_loss, stat_dict

  # ------------------------------------------------------------------------
  # Compute graidents
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, stat_dict), grad = grad_fn(state.params)

  # Compute average gradient across multiple workers.
  grad = jax.lax.pmean(grad, axis_name="batch")

  # ------------------------------------------------------------------------
  # Update States
  new_state = state.apply_gradients(grads=grad)

  metrics_update = metric_collector.gather_from_model_output(
      total_loss=loss,
      learning_rate=lr,
      **stat_dict,
  )
  return new_state, metrics_update, rng


def train_and_evaluate(config, workdir):
  """Runs a training and evaluation loop.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """
  if config.dataset.batch_size % jax.device_count() != 0:
    raise ValueError("Batch size must be divisible by the number of devices.")

  file_utils.makedirs(workdir)
  # Deterministic training.
  rng = jax.random.PRNGKey(config.seed)
  # Shift the numpy random seed by process_index() to shuffle data loaded
  # by different hosts
  np.random.seed(20201473 + jax.process_index())

  # ----------------------------------------------------------------------------
  # Build input pipeline.
  # ----------------------------------------------------------------------------
  rng, data_rng = jax.random.split(rng)
  data_rng = jax.random.fold_in(data_rng, jax.process_index())

  train_iter, _, _ = datasets.create_datasets(config)
  example_batch = train_iter.peek()
  train_iter = flax.jax_utils.prefetch_to_device(train_iter, 6)

  config.dataset.image_height = example_batch["rgb"].shape[-3]
  config.dataset.image_width = example_batch["rgb"].shape[-2]

  # ----------------------------------------------------------------------------
  # Learning rate schedule.
  num_train_steps = config.train.max_steps
  if num_train_steps == -1:
    raise ValueError

  steps_per_epoch = num_train_steps // config.train.num_epochs
  logging.info(
      "num_train_steps=%d, steps_per_epoch=%d", num_train_steps, steps_per_epoch
  )

  learning_rate_fn = train_utils.create_learning_rate_fn(config)

  # ----------------------------------------------------------------------------
  # Initialize model.
  rng, model_rng = jax.random.split(rng)
  model, state, metric_collector = models.create_train_state(
      config,
      model_rng,
      learning_rate_fn=learning_rate_fn,
      example_batch=example_batch,
  )

  # ----------------------------------------------------------------------------
  # Set up checkpointing of the model and the input pipeline.

  # check if the job was stopped and relaunced
  state = checkpoints.restore_checkpoint(workdir, state)

  initial_step = int(state.step) + 1
  if config.dev_run:
    jnp.set_printoptions(precision=2)
    np.set_printoptions(precision=2)

  # ----------------------------------------------------------------------------
  # Get the multiplier dictionary.
  alpha_dict = dict(config.loss)
  alpha_fn_dict = {}
  for key, value in alpha_dict.items():
    if key in [
        "fg_mask_alpha",
        "mask_layer_alpha",
        "shadow_smooth_alpha",
    ]:
      alpha_fn_dict[key] = schedule_utils.cons_then_decay(
          value, config.train.switch_steps, config.train.max_steps
      )
    elif key in [
        "disp_layer_alpha",
        "mask_l0_alpha",
        "disp_smooth_alpha",
    ]:  # "disp_layer_alpha"]:
      alpha_fn_dict[key] = schedule_utils.cons(value)
    else:
      alpha_fn_dict[key] = schedule_utils.warmup_then_cons(
          value, config.train.switch_steps
      )

  # Distribute training.
  state = flax_utils.replicate(state)
  p_train_step = jax.pmap(
      functools.partial(
          train_step,
          model=model,
          learning_rate_fn=learning_rate_fn,
          alpha_fn_dict=alpha_fn_dict,
          weight_decay=config.train.weight_decay,
          metric_collector=metric_collector,
      ),
      axis_name="batch",
  )

  # ----------------------------------------------------------------------------
  # Prepare Metric Writers
  writer = metric_writers.create_default_writer(
      workdir, just_logging=jax.process_index() > 0
  )
  if initial_step == 1:
    writer.write_hparams(dict(config))

  logging.info("Starting training loop at step %d.", initial_step)
  hooks = []
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=num_train_steps, writer=writer
  )
  if jax.process_index() == 0:
    hooks += [
        report_progress,
    ]
  train_metrics = None

  n_local_devices = jax.local_device_count()
  rng = rng + jax.process_index()  # Make random seed separate across hosts.
  keys = jax.random.split(rng, n_local_devices)  # For pmapping RNG keys.

  with metric_writers.ensure_flushes(writer):
    for step in range(initial_step, num_train_steps + 1):
      # `step` is a Python integer. `state.step` is JAX integer on the GPU/TPU
      # devices.
      is_last_step = step == num_train_steps
      with jax.profiler.StepTraceAnnotation("train", step_num=step):
        batch = jax.tree.map(np.asarray, next(train_iter))
        state, metrics_update, keys = p_train_step(
            rng=keys, state=state, batch=batch
        )
        metric_update = flax_utils.unreplicate(metrics_update)
        train_metrics = (
            metric_update
            if train_metrics is None
            else train_metrics.merge(metric_update)
        )
      # Quick indication that training is happening.
      logging.log_first_n(logging.INFO, "Finished training step %d.", 5, step)
      for h in hooks:
        h(step)
      if step % config.train.log_loss_every_steps == 0 or is_last_step:
        writer.write_scalars(step, train_metrics.compute())
        train_metrics = None

      if (jax.process_index() == 0) and (
          step % config.train.checkpoint_every_steps == 0 or is_last_step
      ):
        with report_progress.timed("checkpoint"):
          state_to_save = jax.device_get(jax.tree.map(lambda x: x[0], state))
          checkpoints.save_checkpoint(workdir, state_to_save, step, keep=100)

  logging.info("Finishing training at step %d", num_train_steps)
