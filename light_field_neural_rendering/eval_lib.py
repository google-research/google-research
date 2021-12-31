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
import os

from absl import logging

from flax.metrics import tensorboard
from flax.training import checkpoints
import jax
import ml_collections
import numpy as np
from skimage import metrics as skmetrics
import tensorflow as tf

from light_field_neural_rendering.src import datasets
from light_field_neural_rendering.src import models
from light_field_neural_rendering.src.utils import file_utils
from light_field_neural_rendering.src.utils import model_utils
from light_field_neural_rendering.src.utils import render_utils
from light_field_neural_rendering.src.utils import train_utils


def evaluate(config, workdir):
  """Evalution function."""

  # Hide the GPUs and TPUs from TF so it does not reserve memory on them for
  # LPIPS computation or dataset loading.
  tf.config.experimental.set_visible_devices([], "GPU")
  tf.config.experimental.set_visible_devices([], "TPU")

  rng = jax.random.PRNGKey(config.seed)

  #----------------------------------------------------------------------------
  # Build input pipeline.
  rng, data_rng = jax.random.split(rng)
  data_rng = jax.random.fold_in(data_rng, jax.process_index())
  config.dataset.data_dir = os.path.join(config.dataset.base_dir,
                                         config.dataset.scene)
  train_ds, test_ds = datasets.create_dataset(config)
  example_batch = train_ds.peek()

  rng, key = jax.random.split(rng)

  #----------------------------------------------------------------------------
  # Initialize model.
  learning_rate_fn = train_utils.create_learning_rate_fn(config)
  model, state = models.create_train_state(
      config,
      key,
      learning_rate_fn=learning_rate_fn,
      example_batch=example_batch)

  #----------------------------------------------------------------------------
  # Get the rendering function. Renderig is forced ot be deterministic even if
  # trainin is randomized
  render_pfn = render_utils.get_render_function(model, config, randomized=False)

  last_step = 0
  out_dir = os.path.join(
      workdir, "path_renders" if config.dataset.render_path else "test_preds")

  if not config.eval.eval_once:
    # Prepare Metric Writers
    summary_writer = tensorboard.SummaryWriter(os.path.join(workdir, "eval"))

  while True:
    state = checkpoints.restore_checkpoint(workdir, state)
    step = int(state.step)
    if step <= last_step:
      continue

    if config.eval.save_output and (not file_utils.isdir(out_dir)):
      file_utils.makedirs(out_dir)

    psnr_values = []
    ssim_values = []

    if not config.eval.eval_once:
      showcase_index = np.random.randint(0, test_ds.size)

    for idx in range(test_ds.size):
      logging.info("Evaluating [%d / %d].", idx, test_ds.size)
      batch = next(test_ds)
      test_pixels = batch.target_view.rgb
      if test_pixels is not None:
        test_pixels = model_utils.uint2float(test_pixels)

      #-----------------------------------------------------------
      # Render Image
      variables = {"params": state.params}
      pred_color, pred_disp, pred_acc = render_utils.render_image(
          functools.partial(render_pfn, variables),
          batch,
          rng,
          render_utils.normalize_disp(config.dataset.name),
          chunk=config.eval.chunk)

      if jax.process_index() != 0:
        continue

      #-----------------------------------------------------------
      # Get showcase example for logging
      if not config.eval.eval_once and idx == showcase_index:
        showcase_color = pred_color
        showcase_disp = pred_disp
        showcase_acc = pred_acc
        if not config.dataset.render_path:
          showcase_gt = test_pixels
      #-----------------------------------------------------------
      # If get pixels available, evaluate
      if not config.dataset.render_path:
        psnr = model_utils.compute_psnr(((pred_color - test_pixels)**2).mean())
        ssim = skmetrics.structural_similarity(
            pred_color.astype(np.float32),
            test_pixels.astype(np.float32),
            win_size=11,
            multichannel=True,
            gaussian_weights=True)
        logging.info(f"PSNR = {psnr:.4f}, SSIM = {ssim:.4f}")  # pylint: disable=logging-format-interpolation
        psnr_values.append(float(psnr))
        ssim_values.append(float(ssim))

      #-----------------------------------------------------------
      # Save generated image
      if config.eval.save_output:
        model_utils.save_img(pred_color,
                             os.path.join(out_dir, "{:03d}.png".format(idx)))
        if pred_disp is not None:
          model_utils.save_img(
              pred_disp[Ellipsis, 0],
              os.path.join(out_dir, "disp_{:03d}.png".format(idx)))
    #-----------------------------------------------------------
    if (not config.eval.eval_once) and (jax.process_index() == 0):
      summary_writer.image("pred_color", showcase_color, step)
      if showcase_disp is not None:
        summary_writer.image("pred_disp", showcase_disp, step)
      if showcase_acc is not None:
        summary_writer.image("pred_acc", showcase_acc, step)
      if not config.dataset.render_path:
        summary_writer.scalar("eval_metric/psnr",
                              np.mean(np.array(psnr_values)), step)
        summary_writer.scalar("eval_metric/ssim",
                              np.mean(np.array(ssim_values)), step)
        summary_writer.image("target", showcase_gt, step)

    #-----------------------------------------------------------
    # Save the metric to file
    if config.eval.save_output and (not config.dataset.render_path) and (
        jax.process_index() == 0):
      with file_utils.open_file(
          os.path.join(out_dir, f"psnrs_{step}.txt"), "w") as f:
        f.write(" ".join([str(v) for v in psnr_values]))
      with file_utils.open_file(
          os.path.join(out_dir, f"ssims_{step}.txt"), "w") as f:
        f.write(" ".join([str(v) for v in ssim_values]))
      with file_utils.open_file(os.path.join(out_dir, "psnr.txt"), "w") as f:
        f.write("{}".format(np.mean(np.array(psnr_values))))
      with file_utils.open_file(os.path.join(out_dir, "ssim.txt"), "w") as f:
        f.write("{}".format(np.mean(np.array(ssim_values))))
    if config.eval.eval_once:
      break
    if int(step) >= config.train.max_steps:
      break
    last_step = step

  logging.info("Finishing evaluation at step %d", last_step)
