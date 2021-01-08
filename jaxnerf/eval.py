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

# Lint as: python3
"""Evaluation script for Nerf."""
from os import path

from absl import app
from absl import flags
from flax import optim
from flax.metrics import tensorboard
from flax.training import checkpoints
import jax
from jax import random
import numpy as np

from jaxnerf.nerf import datasets
from jaxnerf.nerf import model_utils
from jaxnerf.nerf import models
from jaxnerf.nerf import utils

FLAGS = flags.FLAGS

utils.define_flags()


def main(unused_argv):
  rng = random.PRNGKey(20200823)

  if FLAGS.config is not None:
    utils.update_flags(FLAGS)
  if FLAGS.train_dir is None:
    raise ValueError("train_dir must be set. None set now.")
  if FLAGS.data_dir is None:
    raise ValueError("data_dir must be set. None set now.")
  FLAGS.randomized = False
  dataset = datasets.get_dataset("test", FLAGS)
  rng, key = random.split(rng)
  init_model, init_state = models.get_model(key, dataset.peek(), FLAGS)
  optimizer_def = optim.Adam(FLAGS.lr_init)
  optimizer = optimizer_def.create(init_model)

  def render_fn(key_0, key_1, model, rays):
    # Note rng_keys are useless in eval mode since there's no randomness.
    return jax.lax.all_gather(model(key_0, key_1, *rays), axis_name="batch")

  # pmap over only the data input.
  render_pfn = jax.pmap(
      render_fn,
      in_axes=(None, None, None, 0),
      donate_argnums=3,
      axis_name="batch",
  )

  state = model_utils.TrainState(
      step=0, optimizer=optimizer, model_state=init_state)
  del init_model, init_state

  last_step = 0
  out_dir = path.join(FLAGS.train_dir,
                      "path_renders" if FLAGS.render_path else "test_preds")
  if not FLAGS.eval_once:
    summary_writer = tensorboard.SummaryWriter(
        path.join(FLAGS.train_dir, "eval"))
  while True:
    state = checkpoints.restore_checkpoint(FLAGS.train_dir, state)
    if state.step <= last_step:
      continue
    if FLAGS.save_output and (not utils.isdir(out_dir)):
      utils.makedirs(out_dir)
    psnrs = []
    if not FLAGS.eval_once:
      showcase_index = np.random.randint(0, dataset.size)
    for idx in range(dataset.size):
      print(f"Evaluating {idx+1}/{dataset.size}")
      batch = next(dataset)
      pred_color, pred_disp, pred_acc = utils.render_image(
          state,
          batch["rays"],
          render_pfn,
          rng,
          FLAGS.dataset == "llff",
          chunk=FLAGS.chunk)
      if jax.host_id() != 0:  # Only record via host 0.
        continue
      if not FLAGS.eval_once and idx == showcase_index:
        showcase_color = pred_color
        showcase_disp = pred_disp
        showcase_acc = pred_acc
        if not FLAGS.render_path:
          showcase_gt = batch["pixels"]
      if not FLAGS.render_path:
        psnr = utils.compute_psnr(((pred_color - batch["pixels"])**2).mean())
        print(f"  PSNR = {psnr:.4f}")
        psnrs.append(float(psnr))
      if FLAGS.save_output:
        utils.save_img(pred_color, path.join(out_dir, "{:03d}.png".format(idx)))
        utils.save_img(pred_disp[Ellipsis, 0],
                       path.join(out_dir, "disp_{:03d}.png".format(idx)))
    if (not FLAGS.eval_once) and (jax.host_id() == 0):
      summary_writer.image("pred_color", showcase_color, state.step)
      summary_writer.image("pred_disp", showcase_disp, state.step)
      summary_writer.image("pred_acc", showcase_acc, state.step)
      if not FLAGS.render_path:
        summary_writer.scalar("psnr", np.mean(np.array(psnrs)), state.step)
        summary_writer.image("target", showcase_gt, state.step)
    if FLAGS.save_output and (not FLAGS.render_path) and (jax.host_id() == 0):
      with utils.open_file(path.join(out_dir, "psnr.txt"), "w") as pout:
        pout.write("{}".format(np.mean(np.array(psnrs))))
    if FLAGS.eval_once:
      break
    if int(state.step) >= FLAGS.max_steps:
      break
    last_step = state.step


if __name__ == "__main__":
  app.run(main)
