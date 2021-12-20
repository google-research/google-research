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

"""Function to train the rendering model."""

import functools
import json
import os
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
from skimage import metrics as skmetrics
import tensorflow as tf

from light_field_neural_rendering.src import datasets
from light_field_neural_rendering.src import models
from light_field_neural_rendering.src.utils import data_types
from light_field_neural_rendering.src.utils import file_utils
from light_field_neural_rendering.src.utils import model_utils
from light_field_neural_rendering.src.utils import render_utils
from light_field_neural_rendering.src.utils import train_utils


def train_step(
    model, rng, state,
    batch, learning_rate_fn,
    weight_decay,
    config):
  """Perform a single training step.

  Args:
    model: Flax module for the model. The apply method must take input images
      and a boolean argument indicating whether to use training or inference
      mode.
    rng: random number generator.
    state: State of the model (optimizer and state).
    batch: Training inputs for this step.
    learning_rate_fn: Function that computes the learning rate given the step
      number.
    weight_decay: Weighs L2 regularization term.
    config: experiment config dict.

  Returns:
    The new model state and dictionary with metrics.
  """
  logging.info("train_step(batch=%s)", batch)

  step = state.step + 1
  lr = learning_rate_fn(step)
  rng, key_0, key_1 = jax.random.split(rng, 3)

  def loss_fn(params):
    variables = {"params": params}
    ret = model.apply(
        variables, key_0, key_1, batch, randomized=config.model.randomized)
    if len(ret) not in (1, 2):
      raise ValueError(
          "ret should contain either 1 set of output (coarse only), or 2 sets"
          "of output (coarse as ret[0] and fine as ret[1]).")
    #------------------------------------------------------------------------
    # Main prediction
    # The main prediction is always at the end of the ret list.
    rgb, unused_disp, unused_acc = ret[-1]
    batch_pixels = model_utils.uint2float(batch.target_view.rgb)
    loss = ((rgb - batch_pixels[Ellipsis, :3])**2).mean()
    psnr = model_utils.compute_psnr(loss)

    #------------------------------------------------------------------------
    # Coarse / Regularization Prediction
    if len(ret) > 1:
      # If there are both coarse and fine predictions, we compute the loss for
      # the coarse prediction (ret[0]) as well.
      rgb_c, unused_disp_c, unused_acc_c = ret[0]
      loss_c = ((rgb_c - batch_pixels[Ellipsis, :3])**2).mean()
      psnr_c = model_utils.compute_psnr(loss_c)
    else:
      loss_c = 0.
      psnr_c = 0.

    #------------------------------------------------------------------------
    # Weight Regularization
    weight_penalty_params = jax.tree_leaves(variables["params"])
    weight_l2 = sum(
        [jnp.sum(x**2) for x in weight_penalty_params if x.ndim > 1])
    weight_penalty = weight_decay * 0.5 * weight_l2

    #------------------------------------------------------------------------
    # Compute total loss and wrap the stats
    total_loss = loss + loss_c + weight_penalty
    stats = train_utils.Stats(
        loss=loss, psnr=psnr, loss_c=loss_c, psnr_c=psnr_c, weight_l2=weight_l2)
    return total_loss, stats

  #------------------------------------------------------------------------
  # Compute Graidents
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, stats), grad = grad_fn(state.params)

  # Compute average gradient across multiple workers.
  grad = jax.lax.pmean(grad, axis_name="batch")

  #------------------------------------------------------------------------
  # Update States
  new_state = state.apply_gradients(grads=grad)

  metrics_update = train_utils.TrainMetrics.gather_from_model_output(
      total_loss=loss,
      loss=stats.loss,
      psnr=stats.psnr,
      loss_c=stats.loss_c,
      psnr_c=stats.psnr_c,
      weight_l2=stats.weight_l2,
      learning_rate=lr)
  return new_state, metrics_update, rng


def eval_step(state, rng, batch,
              render_pfn, config):
  """Compute the metrics for the given model in inference mode.

  The model is applied to the inputs with train=False using all devices on the
  host. Afterwards metrics are averaged across *all* devices (of all hosts).
  Args:
    state: Replicate model state.
    rng: random number generator.
    batch: data_types.Batch. Inputs that should be evaluated.
    render_pfn: pmaped render function.
    config: exepriment config.

  Returns:
    Dictionary of the replicated metrics.
  """
  logging.info("eval_step=================")
  variables = {
      "params": jax.device_get(jax.tree_map(lambda x: x[0], state)).params,
  }
  pred_color, pred_disp, pred_acc = render_utils.render_image(
      functools.partial(render_pfn, variables),
      batch,
      rng,
      render_utils.normalize_disp(config.dataset.name),
      chunk=config.eval.chunk)

  return pred_color, pred_disp, pred_acc


def train_and_evaluate(config, workdir):
  """Runs a training and evaluation loop.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """
  if config.dataset.batch_size % jax.device_count() != 0:
    raise ValueError("Batch size must be divisible by the number of devices.")

  tf.io.gfile.makedirs(workdir)
  # Deterministic training.
  rng = jax.random.PRNGKey(config.seed)
  # Shift the numpy random seed by process_index() to shuffle data loaded
  # by different hosts
  np.random.seed(20201473 + jax.process_index())

  #----------------------------------------------------------------------------
  # Build input pipeline.
  rng, data_rng = jax.random.split(rng)
  data_rng = jax.random.fold_in(data_rng, jax.process_index())
  config.dataset.data_dir = os.path.join(config.dataset.base_dir,
                                         config.dataset.scene)
  train_ds, eval_ds = datasets.create_dataset(config)
  example_batch = train_ds.peek()

  #----------------------------------------------------------------------------
  # Learning rate schedule.
  num_train_steps = config.train.max_steps
  if num_train_steps == -1:
    num_train_steps = train_ds.size()
  steps_per_epoch = num_train_steps // config.train.num_epochs
  logging.info("num_train_steps=%d, steps_per_epoch=%d", num_train_steps,
               steps_per_epoch)

  learning_rate_fn = train_utils.create_learning_rate_fn(config)

  #----------------------------------------------------------------------------
  # Initialize model.
  rng, model_rng = jax.random.split(rng)
  model, state = models.create_train_state(
      config,
      model_rng,
      learning_rate_fn=learning_rate_fn,
      example_batch=example_batch,
  )

  #----------------------------------------------------------------------------
  # Set up checkpointing of the model and the input pipeline.
  state = checkpoints.restore_checkpoint(workdir, state)
  initial_step = int(state.step) + 1

  #----------------------------------------------------------------------------
  # Distribute training.
  state = flax_utils.replicate(state)
  p_train_step = jax.pmap(
      functools.partial(
          train_step,
          model=model,
          learning_rate_fn=learning_rate_fn,
          weight_decay=config.train.weight_decay,
          config=config,
      ),
      axis_name="batch",
  )

  # Get distributed rendering function
  render_pfn = render_utils.get_render_function(
      model=model,
      config=config,
      randomized=False,  # No randomization for evaluation.
  )

  #----------------------------------------------------------------------------
  # Prepare Metric Writers
  writer = metric_writers.create_default_writer(
      workdir, just_logging=jax.process_index() > 0)
  if initial_step == 1:
    writer.write_hparams(dict(config))

  logging.info("Starting training loop at step %d.", initial_step)
  hooks = []
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=num_train_steps, writer=writer)
  if jax.process_index() == 0:
    hooks += [
        report_progress,
    ]
  train_metrics = None

  # Prefetch_buffer_size = 6 x batch_size
  ptrain_ds = flax.jax_utils.prefetch_to_device(train_ds, 6)
  n_local_devices = jax.local_device_count()
  rng = rng + jax.process_index()  # Make random seed separate across hosts.
  keys = jax.random.split(rng, n_local_devices)  # For pmapping RNG keys.

  with metric_writers.ensure_flushes(writer):
    for step in range(initial_step, num_train_steps + 1):
      # `step` is a Python integer. `state.step` is JAX integer on the GPU/TPU
      # devices.
      is_last_step = step == num_train_steps
      with jax.profiler.StepTraceAnnotation("train", step_num=step):
        batch = next(ptrain_ds)
        state, metrics_update, keys = p_train_step(
            rng=keys, state=state, batch=batch)
        metric_update = flax_utils.unreplicate(metrics_update)
        train_metrics = (
            metric_update
            if train_metrics is None else train_metrics.merge(metric_update))
      # Quick indication that training is happening.
      logging.log_first_n(logging.INFO, "Finished training step %d.", 5, step)
      for h in hooks:
        h(step)

      if step % config.train.log_loss_every_steps == 0 or is_last_step:
        writer.write_scalars(step, train_metrics.compute())
        train_metrics = None

      if step % config.train.render_every_steps == 0 or is_last_step:
        test_batch = next(eval_ds)
        test_pixels = model_utils.uint2float(
            test_batch.target_view.rgb)  # extract for evaluation
        with report_progress.timed("eval"):
          pred_color, pred_disp, pred_acc = eval_step(state, keys[0],
                                                      test_batch, render_pfn,
                                                      config)
        #------------------------------------------------------------------
        # Log metrics and images for host 0
        #------------------------------------------------------------------
        if jax.process_index() == 0:
          psnr = model_utils.compute_psnr(
              ((pred_color - test_pixels)**2).mean())
          ssim = skmetrics.structural_similarity(
              pred_color.astype(np.float32),
              test_pixels.astype(np.float32),
              win_size=11,
              multichannel=True,
              gaussian_weight=True)
          writer.write_scalars(step, {
              "train_eval/test_psnr": psnr,
              "train_eval/test_ssim": ssim,
          })
          writer.write_images(
              step, {
                  "test_pred_color": pred_color[None, :],
                  "test_target": test_pixels[None, :]
              })
          if pred_disp is not None:
            writer.write_images(step, {"test_pred_disp": pred_disp[None, :]})
          if pred_acc is not None:
            writer.write_images(step, {"test_pred_acc": pred_acc[None, :]})
        #------------------------------------------------------------------

      if (jax.process_index()
          == 0) and (step % config.train.checkpoint_every_steps == 0 or
                     is_last_step):
        # Write final metrics to file
        with file_utils.open_file(
            os.path.join(workdir, "train_logs.json"), "w") as f:
          log_dict = metric_update.compute()
          for k, v in log_dict.items():
            log_dict[k] = v.item()
          f.write(json.dumps(log_dict))
        with report_progress.timed("checkpoint"):
          state_to_save = jax.device_get(jax.tree_map(lambda x: x[0], state))
          checkpoints.save_checkpoint(workdir, state_to_save, step, keep=100)

  logging.info("Finishing training at step %d", num_train_steps)
