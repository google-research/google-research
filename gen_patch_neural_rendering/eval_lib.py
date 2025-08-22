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

"""Function to evaluate the rendering model."""

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

from gen_patch_neural_rendering.src import datasets
from gen_patch_neural_rendering.src import models
from gen_patch_neural_rendering.src.utils import file_utils
from gen_patch_neural_rendering.src.utils import model_utils
from gen_patch_neural_rendering.src.utils import render_utils
from gen_patch_neural_rendering.src.utils import train_utils


def evaluate(config, workdir):
  """Evalution function."""

  # Hide the GPUs and TPUs from TF so it does not reserve memory on them for
  # LPIPS computation or dataset loading.
  tf.config.experimental.set_visible_devices([], "GPU")
  tf.config.experimental.set_visible_devices([], "TPU")

  rng = jax.random.PRNGKey(config.seed)

  if config.eval.return_coarse:
    sfx = "coarse"
  else:
    sfx = ""

  #----------------------------------------------------------------------------
  # Build input pipeline.
  rng, data_rng = jax.random.split(rng)
  data_rng = jax.random.fold_in(data_rng, jax.process_index())

  train_ds, test_ds_dict = datasets.create_eval_dataset(config)
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
  out_dir_dict = {}
  for key in test_ds_dict:
    out_dir_dict[key] = os.path.join(
        workdir, "path_renders" +
        sfx if config.dataset.render_path else "test_preds" + sfx, key)

  if not config.eval.eval_once:
    summary_writer_dict = {}
    for key in test_ds_dict:
      # Prepare Metric Writers
      summary_writer_dict[key] = tensorboard.SummaryWriter(
          os.path.join(workdir, "eval" + sfx, key))
    summary_writer_dict["all"] = tensorboard.SummaryWriter(
        os.path.join(workdir, "eval" + sfx))

  while True:
    state = checkpoints.restore_checkpoint(
        workdir,
        state,
        step=None
        if config.eval.checkpoint_step == -1 else config.eval.checkpoint_step)
    step = int(state.step)
    if step <= last_step:
      continue
    total_psnr = 0

    for scene_name, test_ds in test_ds_dict.items():
      out_dir = out_dir_dict[scene_name]
      if config.eval.save_output and (not file_utils.isdir(out_dir)):
        file_utils.makedirs(out_dir)

      psnr_values = []
      ssim_values = []

      if not config.eval.eval_once:
        showcase_index = np.random.randint(0, test_ds.size)

      for idx in range(test_ds.size):
        logging.info("Evaluating scene %s [%d / %d].", scene_name, idx,
                     test_ds.size)
        batch = next(test_ds)
        test_pixels = batch.target_view.rgb
        test_mask = batch.target_view.mask
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
            chunk=config.eval.chunk,
            return_coarse=config.eval.return_coarse,
        )

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
          if config.eval.mvsn_style:
            h_crop, w_crop = np.array(pred_color.shape[:2]) // 10
            pred_color = pred_color[h_crop:-h_crop, w_crop:-w_crop]
            test_pixels = test_pixels[h_crop:-h_crop, w_crop:-w_crop]

          if test_mask is not None:
            psnr = model_utils.compute_psnr(
                ((pred_color[test_mask] - test_pixels[test_mask])**2).mean())
          else:
            psnr = model_utils.compute_psnr(
                ((pred_color - test_pixels)**2).mean())
          ssim = skmetrics.structural_similarity(
              pred_color.astype(np.float32),
              test_pixels.astype(np.float32),
              win_size=11,
              multichannel=True,
              gaussian_weights=True)
          logging.info(f"PSNR = {psnr:.4f}, SSIM = {ssim:.4f}")  # pylint: disable=logging-fstring-interpolation
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
        summary_writer_dict[scene_name].image(
            "eval/{}_pred_color".format(scene_name), showcase_color, step)
        if showcase_disp is not None:
          summary_writer_dict[scene_name].image(
              "eval/{}_pred_disp".format(scene_name), showcase_disp, step)
        if showcase_acc is not None:
          summary_writer_dict[scene_name].image(
              "eval/{}_pred_acc".format(scene_name), showcase_acc, step)
        if not config.dataset.render_path:
          summary_writer_dict[scene_name].scalar("eval_metric/psnr",
                                                 np.mean(np.array(psnr_values)),
                                                 step)
          summary_writer_dict[scene_name].scalar("eval_metric/ssim",
                                                 np.mean(np.array(ssim_values)),
                                                 step)
          summary_writer_dict[scene_name].image(
              "eval/{}_target".format(scene_name), showcase_gt, step)

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
      total_psnr += np.mean(np.array(psnr_values))

    if not config.eval.eval_once:
      summary_writer_dict["all"].scalar(
          "eval_metric/avg_psnr_{}".format(config.dataset.eval_dataset),
          total_psnr / len(test_ds_dict.keys()), step)

    if config.eval.eval_once:
      break
    if int(step) >= config.train.max_steps:
      break
    last_step = step

  logging.info("Finishing evaluation at step %d", last_step)
