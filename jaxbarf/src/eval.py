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

"""Evaluation script for Nerf."""
import functools
from os import path

from absl import app
from absl import flags
import flax
from flax.metrics import tensorboard
from flax.training import checkpoints
import jax
from jax import random
import numpy as np
import tensorflow as tf
import tensorflow_hub as tf_hub
import pickle
import optax

from jaxbarf.src import datasets
from jaxbarf.src import models
from jaxbarf.src import utils
from jaxbarf.src import camera

FLAGS = flags.FLAGS
utils.define_flags()
LPIPS_TFHUB_PATH = "@neural-rendering/lpips/distance/1"

def compute_lpips(image1, image2, model):
  """Compute the LPIPS metric."""
  # The LPIPS model expects a batch dimension.
  return model(
      tf.convert_to_tensor(image1[None, Ellipsis]),
      tf.convert_to_tensor(image2[None, Ellipsis]))[0]


def main(unused_argv):
  """Entry point for evaluation binary."""
  tf.config.experimental.set_visible_devices([], "GPU")
  tf.config.experimental.set_visible_devices([], "TPU")
  rng = random.PRNGKey(20200823)

  if FLAGS.config is not None:
    utils.update_flags(FLAGS)
  if FLAGS.train_dir is None:
    raise ValueError("train_dir must be set. None set now.")
  if FLAGS.data_dir is None:
    raise ValueError("data_dir must be set. None set now.")

  # load train dataset to get GT poses
  #with utils.open_file(FLAGS.init_poses_file, "rb") as f: # load init poses
  #  poses_train_init = pickle.load(f)
  dataset_train = datasets.get_dataset("train", FLAGS, train_mode=False)
  poses_train = utils.to_device(dataset_train.get_all_poses())

  rng, key = random.split(rng)
  model, variables = models.get_model(key, dataset_train.peek(), FLAGS)
  # Set up seperate optimizer and LR schedule for pose and MLP parameters
  params = variables["params"]
  learning_rate_fn_mlp = functools.partial(
      utils.learning_rate_decay,
      lr_init=FLAGS.lr_init,
      lr_final=FLAGS.lr_final,
      max_steps=FLAGS.max_steps,
      lr_delay_steps=FLAGS.lr_delay_steps,
      lr_delay_mult=FLAGS.lr_delay_mult)
  learning_rate_fn_pose = functools.partial(
      utils.learning_rate_decay,
      lr_init=FLAGS.lr_init_pose,
      lr_final=FLAGS.lr_final_pose,
      max_steps=FLAGS.max_steps,
      lr_delay_steps=FLAGS.lr_delay_steps_pose,
      lr_delay_mult=FLAGS.lr_delay_mult_pose)
  pose_params = flax.traverse_util.ModelParamTraversal(
      lambda path, _: "POSE" in path)
  mlp_params = flax.traverse_util.ModelParamTraversal(
      lambda path, _: "MLP" in path)
  all_false = jax.tree_util.tree_map(lambda _: False, params)
  pose_mask = pose_params.update(lambda _: True, all_false)
  mlp_mask = mlp_params.update(lambda _: True, all_false)
  optimizer = optax.chain(
          optax.scale_by_adam(),
          optax.masked(optax.scale_by_schedule(learning_rate_fn_pose), pose_mask),
          optax.masked(optax.scale_by_schedule(learning_rate_fn_mlp), mlp_mask),
          optax.scale(-1),
  )
  optimizer_state = optimizer.init(params)
  state = utils.TrainState(optimizer_state=optimizer_state, params=params, step=0)
  del params, optimizer_state


  # Rendering is forced to be deterministic even if training was randomized, as
  # this eliminates "speckle" artifacts.
  def render_fn(variables, key_0, key_1, rays, step):
    """Render function (no learned pose refinement if train_mode=False.)"""
    return jax.lax.all_gather(
        model.apply({"params":variables}, key_0, key_1, rays,
                    False, train_mode=False, step=step),
        axis_name="batch")

  # pmap over only the data input.
  render_pfn = jax.pmap(
      render_fn,
      in_axes=(None, None, None, 0, None),
      donate_argnums=(3,),
      axis_name="batch",
  )

  # Compiling to the CPU because it's faster and more accurate.
  ssim_fn = jax.jit(
      functools.partial(utils.compute_ssim, max_val=1.), backend="cpu")
  lpips_model = tf_hub.load(LPIPS_TFHUB_PATH)

  last_step = 0
  out_dir = path.join(FLAGS.train_dir, "test_preds")
  if not utils.isdir(out_dir):
    utils.makedirs(out_dir)

  summary_writer = tensorboard.SummaryWriter(path.join(FLAGS.train_dir, "eval"))

  while True:
    state = checkpoints.restore_checkpoint(FLAGS.train_dir, state)
    step = int(state.step * FLAGS.max_steps)
    if step <= last_step:
      continue

    poses_refine_se3 = state.params["POSE_0"]["delta_se3"]
    poses_refine_se3exp = camera.se3_exp(poses_refine_se3)
    poses_train_pred = camera.compose([poses_refine_se3exp,
                                       poses_train["poses_init"]])
    poses_train_aligned, sim3 = camera.prealign_cameras(poses_train_pred,
                                                        poses_train["poses_gt"])
    r_error, t_error = camera.evaluate_camera(poses_train_pred,
                                              poses_train["poses_gt"])

    psnr_values = []
    ssim_values = []
    lpips_values = []

    # Every time we load a new checkpoint, we need to update poses
    dataset = datasets.get_dataset("test", FLAGS,
                                   calib_matrix=sim3,
                                   train_mode=False)
    for idx in range(8):
      print(f"Evaluating {idx+1}/{dataset.size}")
      batch = next(dataset)
      pred_color, pred_disp, pred_acc = utils.render_image(
          functools.partial(render_pfn, state.params),
          batch["rays"],
          rng,
          FLAGS.dataset == "llff",
          chunk=FLAGS.chunk,
          step=step/FLAGS.max_steps)
      if jax.host_id() != 0:  # Only record via host 0.
        continue

      psnr = utils.compute_psnr(((pred_color - batch["pixels"])**2).mean())
      ssim = ssim_fn(pred_color, batch["pixels"])
      lpips = compute_lpips(pred_color, batch["pixels"], lpips_model)
      psnr_values.append(float(psnr))
      ssim_values.append(float(ssim))
      lpips_values.append(float(lpips))

      utils.save_img(pred_color, path.join(out_dir,
                                           "pred_{:03d}_{}.png".format(idx, step)))
      utils.save_img(batch["pixels"], path.join(out_dir,
                                           "gt_{:03d}_{}.png".format(idx, step)))
      summary_writer.image("val_pred_color", pred_color, step)
      summary_writer.image("val_gt_color", batch["pixels"], step)
    summary_writer.scalar("val_psnr", np.mean(np.array(psnr_values)), step)

    with utils.open_file(path.join(out_dir, f"{step}.txt"), "w") as f:
      f.write("Trainset: num {}, R_error: {:.3f}, t_error: {:.3f}\n".format(
          len(r_error), np.mean(r_error)*180/np.pi, np.mean(t_error)))
      f.write("Average over {} validation images\n".format(len(psnr_values)))
      f.write("Mean PSNR: {:.2f}\n".format(np.mean(np.array(psnr_values))))
      f.write("Mean SSIM: {:.2f}\n".format(np.mean(np.array(ssim_values))))
      f.write("Mean LPIPS: {:.2f}\n".format(np.mean(np.array(lpips_values))))
      f.write("Mean PSNR (first 8): {:.2f}\n".format(
          np.mean(np.array(psnr_values)[:8])))
      f.write("Mean SSIM (first 8): {:.2f}\n".format(
          np.mean(np.array(ssim_values)[:8])))
      f.write("Mean LPIPS (first 8): {:.2f}\n".format(
          np.mean(np.array(lpips_values)[:8])))

    if int(step) >= FLAGS.max_steps:
      break
    last_step = step


if __name__ == "__main__":
  app.run(main)
