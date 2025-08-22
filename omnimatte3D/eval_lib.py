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

"""Function to evaluate the model and save predictions."""

import functools
import os
from typing import Dict

from absl import logging

from clu import metric_writers
from clu import metrics
import flax
import flax.jax_utils as flax_utils
import flax.linen as nn
from flax.training import checkpoints
from flax.training import train_state
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import tensorflow as tf

from omnimatte3D.src import datasets
from omnimatte3D.src import models
from omnimatte3D.src.utils import file_utils
from omnimatte3D.src.utils import model_utils
from omnimatte3D.src.utils import train_utils


def eval_step(
    config,
    model,
    state,
    batch,
):
  """Compute the metrics for the given model in inference mode.

  Args:
    config: exp config.
    model: Flax module for the model. The apply method must take input images
      and a boolean argument indicating whether to use training or inference
      mode.
    state: Replicate model state.
    batch: Inputs that should be evaluated.

  Returns:
    Dictionary of the replicated metrics.
  """
  logging.info("eval_step(batch=%s)", batch)
  variables = {"params": state.params}

  out = model.apply(variables, batch)
  # Compute Loss
  alpha_dict = dict(config.loss)
  _, loss_dict = out.compute_total_loss(batch, alpha_dict)

  log_dict = out.get_log_dict(batch)

  return log_dict, train_utils.EvalMetrics.gather_from_model_output(**loss_dict)


def evaluate(config, workdir):
  """Evalution function."""

  # Hide the GPUs and TPUs from TF so it does not reserve memory on them for
  # LPIPS computation or dataset loading.
  tf.config.experimental.set_visible_devices([], "GPU")
  tf.config.experimental.set_visible_devices([], "TPU")

  # Currently used because orbax checkpointing produces broken checkpoints
  flax.config.update("flax_use_orbax_checkpointing", False)

  rng = jax.random.PRNGKey(config.seed)

  # Currently only supports single image batch eval.
  config.dataset.batch_size = 1
  assert (
      jax.local_device_count() == 1
  ), "Currently does not support multi device training."

  # ----------------------------------------------------------------------------
  # Build input pipeline.
  rng, data_rng = jax.random.split(rng)
  data_rng = jax.random.fold_in(data_rng, jax.process_index())

  train_iter, eval_iter, _ = datasets.create_datasets(config)
  eval_ds = eval_iter
  example_batch = train_iter.peek()
  train_iter = flax.jax_utils.prefetch_to_device(train_iter, 6)

  config.dataset.image_height = example_batch["rgb"].shape[-3]
  config.dataset.image_width = example_batch["rgb"].shape[-2]
  # ----------------------------------------------------------------------------
  # Initialize model.
  learning_rate_fn = train_utils.create_learning_rate_fn(config)
  rng, model_rng = jax.random.split(rng)
  model, state, _ = models.create_train_state(
      config,
      model_rng,
      learning_rate_fn=learning_rate_fn,
      example_batch=example_batch,
  )

  # ----------------------------------------------------------------------------
  # Get the eval step.
  p_eval_step = jax.pmap(
      functools.partial(eval_step, config=config, model=model),
      axis_name="batch",
  )

  last_step = 0
  out_dir = os.path.join(workdir, "test_preds")
  out_dir_rgb = os.path.join(workdir, "test_preds", "rgb")
  out_dir_rgb_tgt = os.path.join(workdir, "test_preds", "rgb_tgt")
  out_dir_fg = os.path.join(workdir, "test_preds", "shadow")

  if not config.eval.eval_once:
    # Prepare Metric Writers
    writer = metric_writers.create_default_writer(
        os.path.join(workdir, "eval"), just_logging=jax.process_index() > 0
    )

  while True:
    state = checkpoints.restore_checkpoint(workdir, state)
    step = int(state.step)

    if step <= last_step:
      # Check if a new checkpoint has been saved. If not, load the latest
      # checkpoint again. Repeat until a new checkpoint is available.
      continue

    if config.eval.save_output:
      file_utils.makedirs(out_dir)
      file_utils.makedirs(out_dir_rgb)
      file_utils.makedirs(out_dir_rgb_tgt)
      file_utils.makedirs(out_dir_fg)

    if not config.eval.eval_once:
      showcase_index = np.random.randint(0, eval_ds.cardinality())

    state = flax_utils.replicate(state)
    eval_metrics = None
    for idx in range(eval_ds.cardinality()):
      batch = next(eval_iter)
      if config.dev_run and idx >= 2:
        # Stop early for developer run.
        break

      logging.info("Evaluating [%d / %d].", idx, eval_ds.cardinality())
      batch = jax.tree.map(np.asarray, batch)
      render_dict, metrics_update = flax_utils.unreplicate(
          p_eval_step(state=state, batch=batch)
      )

      eval_metrics = (
          metrics_update
          if eval_metrics is None
          else eval_metrics.merge(metrics_update)
      )

      if jax.process_index() != 0:
        # Only log from the main process.
        continue

      viz_batch = jax.tree.map(lambda x: x[0], batch)
      test_pixels = viz_batch["rgb"]

      # -----------------------------------------------------------
      # Save generated image
      if config.eval.save_output:
        model_utils.save_img(
            test_pixels[0],
            os.path.join(out_dir_rgb_tgt, "{:03d}_rgb_tgt.png".format(idx)),
        )

        render_dict = jax.tree_util.tree_map(np.array, render_dict)

        if "tgt_images/pred" in render_dict.keys():
          model_utils.save_img(
              render_dict["tgt_images/pred"],
              os.path.join(
                  out_dir_rgb_tgt, "{:03d}_rgb_pred_tgt.png".format(idx)
              ),
          )

        # -----------------------------------------------------------------
        # RGB
        if "rgb_layers_0/pred" in render_dict.keys():
          for i in range(config.model.num_ldi_layers):
            model_utils.save_img(
                render_dict["rgb_layers_0/pred"][i],
                os.path.join(
                    out_dir_rgb, "{:03d}_rgb_layer{}_pred.png".format(idx, i)
                ),
            )
        if "rgb_layers_1/pred" in render_dict.keys():
          for i in range(config.model.num_ldi_layers):
            model_utils.save_img(
                render_dict["rgb_layers_1/pred"][i],
                os.path.join(
                    out_dir_rgb,
                    "{:03d}_rgb_layer{}_pred.png".format(idx + 2, i),
                ),
            )

        if "fg_layers_0/pred" in render_dict.keys():
          num_layers = render_dict["fg_layers_0/pred"].shape[0]
          for i in range(num_layers):
            model_utils.save_img(
                render_dict["fg_layers_0/pred"][i],
                os.path.join(
                    out_dir_fg,
                    "{:03d}_fg_layer{}_pred.png".format(idx, i),
                ),
            )
        if "fg_layers_1/pred" in render_dict.keys():
          num_layers = render_dict["fg_layers_0/pred"].shape[0]
          for i in range(num_layers):
            model_utils.save_img(
                render_dict["fg_layers_1/pred"][i],
                os.path.join(
                    out_dir_fg,
                    "{:03d}_fg_layer{}_pred.png".format(idx + 2, i),
                ),
            )

        if "fg_alphas_0/pred" in render_dict.keys():
          num_layers = render_dict["fg_layers_0/pred"].shape[0]
          for i in range(num_layers):
            model_utils.save_img(
                render_dict["fg_alphas_0/pred"][i][Ellipsis, 0],
                os.path.join(
                    out_dir_fg,
                    "{:03d}_fg_alphas{}_pred.png".format(idx, i),
                ),
            )
        if "fg_alphas_1/pred" in render_dict.keys():
          num_layers = render_dict["fg_layers_0/pred"].shape[0]
          for i in range(num_layers):
            model_utils.save_img(
                render_dict["fg_alphas_1/pred"][i][Ellipsis, 0],
                os.path.join(
                    out_dir_fg,
                    "{:03d}_fg_alphas{}_pred.png".format(idx + 2, i),
                ),
            )

        if "src_images/pred" in render_dict.keys():
          for sidx in range(2):
            model_utils.save_img(
                render_dict["src_images/pred"][sidx],
                os.path.join(
                    out_dir_rgb_tgt,
                    "{:03d}_src_recon_rgb{}.png".format(idx, sidx),
                ),
            )
      # -----------------------------------------------------------
      # Get showcase example for logging
      if not config.eval.eval_once and idx == showcase_index:
        showcase_dict = render_dict

    eval_metrics_cpu = jax.tree.map(np.array, eval_metrics.compute())
    # -----------------------------------------------------------
    if (not config.eval.eval_once) and (jax.process_index() == 0):
      writer.write_images(step, showcase_dict)

      writer.write_scalars(step, eval_metrics_cpu)

    # -----------------------------------------------------------
    # Save the metric to file
    if config.eval.save_output and (jax.process_index() == 0):
      for key, item in eval_metrics_cpu.items():
        with file_utils.open_file(
            os.path.join(out_dir, "{}_{}.txt".format(key, step)), "w"
        ) as f:
          f.write("{}".format(item.item()))

    if config.eval.eval_once:
      break
    if int(step) >= config.train.max_steps:
      break
    last_step = step

  logging.info("Finishing evaluation at step %d", last_step)
