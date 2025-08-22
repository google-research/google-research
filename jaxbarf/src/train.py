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

"""Training script for Barf."""
import functools
import gc
import time
import os
from absl import app
from absl import flags
import flax
from flax.metrics import tensorboard
from flax.training import checkpoints
import jax
from jax import config
from jax import random
import jax.numpy as jnp
import numpy as np
import io
import optax
import pickle

from jaxbarf.src import datasets
from jaxbarf.src import models
from jaxbarf.src import utils
from jaxbarf.src import utils_vis
from jaxbarf.src import camera

FLAGS = flags.FLAGS

utils.define_flags()
config.parse_flags_with_absl()

def train_step(model, optimizer, poses_ref, rng, state, batch, step):
  """Train step."""
  rng, key_0, key_1 = random.split(rng, 3)

  def loss_fn(variables):
    """Loss function."""
    rays = batch["rays"] # [B, 3]
    ret = model.apply({"params":variables}, key_0, key_1, rays,
                      FLAGS.randomized, train_mode=True, step=step)
    rgb, unused_disp, unused_acc = ret[-1]
    loss = ((rgb - batch["pixels"][Ellipsis, :3])**2).mean()
    psnr = utils.compute_psnr(loss)
    if len(ret) > 1:
      rgb_c, unused_disp_c, unused_acc_c = ret[0]
      loss_c = ((rgb_c - batch["pixels"][Ellipsis, :3])**2).mean()
      psnr_c = utils.compute_psnr(loss_c)
    else:
      loss_c = 0.
      psnr_c = 0.
    # Evaluate the camera pose optimization
    poses_refine_se3 = variables["POSE_0"]["delta_se3"]
    poses_refine_se3exp = camera.se3_exp(poses_refine_se3)
    poses_pred = camera.compose([poses_refine_se3exp, poses_ref["poses_init"]])
    poses_aligned, _ = camera.prealign_cameras(
        poses_pred, poses_ref["poses_gt"])
    r_error, t_error = camera.evaluate_camera(
        poses_aligned, poses_ref["poses_gt"])
    # L2 regularization on the network weights
    def tree_sum_fn(fn):
      """Tree sum."""
      return jax.tree_util.tree_reduce(
          lambda x, y: x + fn(y), variables, initializer=0)
    weight_l2 = (
        tree_sum_fn(lambda z: jnp.sum(z**2)) /
        tree_sum_fn(lambda z: jnp.prod(jnp.array(z.shape))))
    # Collect training losses and pose optimization metrics
    stats = utils.Stats(loss=loss, psnr=psnr, loss_c=loss_c,
                        psnr_c=psnr_c, weight_l2=weight_l2,
                        r_error=r_error.mean(), t_error=t_error.mean())
    return loss + loss_c + FLAGS.weight_decay_mult * weight_l2, stats

  (_, stats), grad = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
  grad = jax.lax.pmean(grad, axis_name="batch")
  stats = jax.lax.pmean(stats, axis_name="batch")

  # Update the learning rate for pose parameters and MLP
  updates, optimizer_state = optimizer.update(grad, state.optimizer_state)
  params = optax.apply_updates(state.params, updates)
  new_state = state.replace(  # pytype: disable=attribute-error
                              optimizer_state=optimizer_state,
                              params=params,
                              step=step)
  return new_state, stats, rng


def main(unused_argv):
  """Entry point for training binary."""

  # Load hyperparameters defined in configs/*yaml files
  if FLAGS.config is not None:
    utils.update_flags(FLAGS)
  if FLAGS.batch_size % jax.device_count() != 0:
    raise ValueError("Batch size must be divisible by the number of devices.")
  if FLAGS.train_dir is None:
    raise ValueError("train_dir must be set. None set now.")
  if FLAGS.data_dir is None:
    raise ValueError("data_dir must be set. None set now.")

  rng = random.PRNGKey(FLAGS.random_seed_jax)
  # Generate consistent camera pertubation across hosts
  np.random.seed(FLAGS.random_seed_np)
  # Shift the numpy random seed by host_id() to
  # shuffle data loaded by different hosts.
  np.random.seed(FLAGS.random_seed_np + jax.host_id())

  # Randomly sample rays if train_mode=True, otherwise, sample rays per pixel
  # We use trainset to visualize the pose optimization procedure
  #with utils.open_file(FLAGS.init_poses_file, "rb") as f:
  #  init_poses = pickle.load(f)
  dataset = datasets.get_dataset("train", FLAGS, train_mode=True)
  test_dataset = datasets.get_dataset("train", FLAGS, train_mode=False)

  poses_ref = dataset.get_all_poses()
  poses_ref = utils.to_device(poses_ref)

  # Construct model and get model parameters
  rng, key = random.split(rng)
  model, variables = models.get_model(key, dataset.peek(), FLAGS)

  # Initial poses
  poses_init_aligned, _ = camera.prealign_cameras(
      poses_ref["poses_init"], poses_ref["poses_gt"])
  camera_init_fig = utils_vis.plot_poses(poses_init_aligned,
                                         poses_ref["poses_gt"],
                                         0)

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
      optax.masked(
          optax.scale_by_schedule(learning_rate_fn_pose), pose_mask),
      optax.masked(optax.scale_by_schedule(learning_rate_fn_mlp), mlp_mask),
      optax.scale(-1),
  )
  optimizer_state = optimizer.init(params)
  state = utils.TrainState(
      optimizer_state=optimizer_state, params=params, step=0)
  del params, optimizer_state

  # Train step
  train_pstep = jax.pmap(
      functools.partial(train_step, model, optimizer, poses_ref),
      axis_name="batch",
      in_axes=(0, 0, 0, None),
      donate_argnums=(2,))

  def render_fn(variables, key_0, key_1, rays, step):
    # Train_mode=True: apply learned pose refinement on the intial poses
    return jax.lax.all_gather(
        model.apply({"params":variables}, key_0, key_1, rays,
                    FLAGS.randomized, train_mode=True, step=step),
        axis_name="batch")

  render_pfn = jax.pmap(
      render_fn,
      in_axes=(None, None, None, 0, None),  # Only distribute the data input.
      donate_argnums=(3,),
      axis_name="batch",
  )

  # Compiling to the CPU because it's faster and more accurate.
  ssim_fn = jax.jit(
      functools.partial(utils.compute_ssim, max_val=1.), backend="cpu")

  if not utils.isdir(FLAGS.train_dir):
    utils.makedirs(FLAGS.train_dir)
  # Resume training a the step of the last checkpoint.
  # state = checkpoints.restore_checkpoint(FLAGS.train_dir, state)
  # init_step = state.optimizer_state.step + 1
  # state = ckpt.restore_or_initialize(state)
  init_step = int(state.step) + 1
  state = flax.jax_utils.replicate(state)

  if jax.host_id() == 0:
    summary_writer = tensorboard.SummaryWriter(FLAGS.train_dir)

  # Prefetch_buffer_size = 3 x batch_size
  pdataset = flax.jax_utils.prefetch_to_device(dataset, 3)
  n_local_devices = jax.local_device_count()
  rng = rng + jax.host_id()  # Make random seed separate across hosts.
  keys = random.split(rng, n_local_devices)  # For pmapping RNG keys.
  gc.disable()  # Disable automatic garbage collection for efficiency.
  stats_trace = []
  reset_timer = True
  for step, batch in zip(range(init_step, FLAGS.max_steps + 1), pdataset):
    # batch['pixel'].shape = (1,4096,3), so as other fields
    if reset_timer:
      t_loop_start = time.time()
      reset_timer = False
    lr = learning_rate_fn_mlp(step)
    lr_pose = learning_rate_fn_pose(step)
    state, stats, keys = train_pstep(keys, state, batch, step/FLAGS.max_steps)
    if jax.host_id() == 0:
      stats_trace.append(stats)
    if step % FLAGS.gc_every == 0:
      gc.collect()

    # Log training summaries. This is put behind a host_id check because in
    # multi-host evaluation, all hosts need to run inference even though we
    # only use host 0 to record results.
    if jax.host_id() == 0:

      if step % FLAGS.print_every == 0:
        summary_writer.scalar("train_loss", stats.loss[0], step)
        summary_writer.scalar("train_psnr", stats.psnr[0], step)
        summary_writer.scalar("train_loss_coarse", stats.loss_c[0], step)
        summary_writer.scalar("train_psnr_coarse", stats.psnr_c[0], step)
        summary_writer.scalar("weight_l2", stats.weight_l2[0], step)
        summary_writer.scalar("R_error", stats.r_error[0], step)
        summary_writer.scalar(
            "R_error_degree", stats.r_error[0]*180/jnp.pi, step)
        summary_writer.scalar("t_error", stats.t_error[0], step)
        avg_loss = np.mean(np.concatenate([s.loss for s in stats_trace]))
        avg_psnr = np.mean(np.concatenate([s.psnr for s in stats_trace]))
        stats_trace = []
        summary_writer.scalar("train_avg_loss", avg_loss, step)
        summary_writer.scalar("train_avg_psnr", avg_psnr, step)
        summary_writer.scalar("learning_rate_mlp", lr, step)
        summary_writer.scalar("learning_rate_pose", lr_pose, step)
        steps_per_sec = FLAGS.print_every / (time.time() - t_loop_start)
        reset_timer = True
        rays_per_sec = FLAGS.batch_size * steps_per_sec
        summary_writer.scalar("train_steps_per_sec", steps_per_sec, step)
        summary_writer.scalar("train_rays_per_sec", rays_per_sec, step)
        precision = int(np.ceil(np.log10(FLAGS.max_steps))) + 1
        print(("{:" + "{:d}".format(precision) + "d}").format(step) +
              f"/{FLAGS.max_steps:d}: " + f"i_loss={stats.loss[0]:0.4f}, " +
              f"avg_loss={avg_loss:0.4f}, " +
              f"R_error_deg={stats.r_error[0]*180/jnp.pi:0.4f}, " +
              f"t_error={stats.t_error[0]:0.4f}, " +
              f"weight_l2={stats.weight_l2[0]:0.2e}, " +
              f"lr_mlp={lr:0.2e}, " +
              f"lr_pose={lr_pose:0.2e}, " +
              f"{rays_per_sec:0.0f} rays/sec")
      if step % FLAGS.save_every == 0:
        state_to_save = jax.device_get(jax.tree.map(lambda x: x[0], state))
        checkpoints.save_checkpoint(FLAGS.train_dir, state_to_save,
                                    int(step),
                                    keep=10,
                                    overwrite=True)

    # Test-set evaluation.
    if FLAGS.render_every > 0 and step % FLAGS.render_every == 0:
      # We reuse the same random number generator from the optimization step
      # here on purpose so that the visualization matches what happened in
      # training.
      t_eval_start = time.time()
      # Visualize draw camera
      eval_variables = jax.device_get(
          jax.tree.map(lambda x: x[0], state)).params
      poses_refine_se3 = eval_variables["POSE_0"]["delta_se3"]
      poses_refine_se3exp = camera.se3_exp(poses_refine_se3)
      poses_pred = camera.compose([poses_refine_se3exp,
                                   poses_ref["poses_init"]])
      poses_aligned, _ = camera.prealign_cameras(
          poses_pred, poses_ref["poses_gt"])
      camera_fig = utils_vis.plot_poses(
          poses_aligned, poses_ref["poses_gt"], step)

      test_case = next(test_dataset)
      pred_color, pred_disp, pred_acc = utils.render_image(
          functools.partial(render_pfn, eval_variables),
          test_case["rays"],
          keys[0],
          FLAGS.dataset == "llff",
          chunk=FLAGS.chunk,
          step = step/FLAGS.max_steps)

      # Log eval summaries on host 0.
      if jax.host_id() == 0:
        psnr = utils.compute_psnr(
            ((pred_color - test_case["pixels"])**2).mean())
        ssim = ssim_fn(pred_color, test_case["pixels"])
        eval_time = time.time() - t_eval_start
        num_rays = jnp.prod(jnp.array(test_case["rays"].directions.shape[:-1]))
        rays_per_sec = num_rays / eval_time
        summary_writer.scalar("test_rays_per_sec", rays_per_sec, step)
        print(f"Eval {step}: {eval_time:0.3f}s., {rays_per_sec:0.0f} rays/sec")
        summary_writer.scalar("test_psnr", psnr, step)
        summary_writer.scalar("test_ssim", ssim, step)
        summary_writer.image("test_pred_color", pred_color, step)
        summary_writer.image("test_pred_disp", pred_disp, step)
        summary_writer.image("test_pred_acc", pred_acc, step)
        summary_writer.image("test_target", test_case["pixels"], step)
        summary_writer.image("train_poses", camera_fig[:, :, :3], step)
        summary_writer.image(
            "train_poses_init", camera_init_fig[:, :, :3], step)
  if FLAGS.max_steps % FLAGS.save_every != 0:
    state = jax.device_get(jax.tree.map(lambda x: x[0], state))
    checkpoints.save_checkpoint(
        FLAGS.train_dir, state, int(FLAGS.max_steps), keep=10)


if __name__ == "__main__":
  app.run(main)
