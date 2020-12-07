# coding=utf-8
# Copyright 2020 The Google Research Authors.
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
"""Training script for Nerf."""

import functools
import gc
import time
from absl import app
from absl import flags
from flax import jax_utils
from flax import nn
from flax import optim
from flax.metrics import tensorboard
from flax.training import checkpoints
import jax
from jax import config
from jax import random
import numpy as np

from jaxnerf.nerf import datasets
from jaxnerf.nerf import model_utils
from jaxnerf.nerf import models
from jaxnerf.nerf import utils

FLAGS = flags.FLAGS

utils.define_flags()
config.parse_flags_with_absl()


def train_step(rng_key, state, batch, lr):
  """One optimization step.

  Args:
    rng_key: jnp.ndarray, random number generator.
    state: modle_utils.TrainState, state of model and optimizer.
    batch: dict. A mini-batch of data for training.
    lr: float, real-time learning rate.

  Returns:
    new_state: model_utils.TrainState, new training state.
    stats: list. [(loss, psnr), (loss_coarse, psnr_coarse)].
    rng_key: jnp.ndarray, updated random number generator.
  """
  rng_key, key_0, key_1 = random.split(rng_key, 3)

  def loss_fn(model):
    with nn.stateful(state.model_state) as new_model_state:
      ret = model(key_0, key_1, batch["rays"])
    if len(ret) not in (1, 2):
      raise ValueError(
          "ret should contain either 1 set of output (coarse only), or 2 sets"
          "of output (coarse as ret[0] and fine as ret[1]).")
    # The main prediction is always at the end of the ret list.
    rgb, unused_disp, unused_acc = ret[-1]
    loss = ((rgb - batch["pixels"][Ellipsis, :3])**2).mean()
    psnr = utils.compute_psnr(loss)
    stats = [utils.Stats(loss=loss, psnr=psnr)]
    if len(ret) > 1:
      # If there are both coarse and fine predictions, we compuate the loss for
      # the coarse prediction (ret[0]) as well.
      rgb_c, unused_disp_c, unused_acc_c = ret[0]
      loss_c = ((rgb_c - batch["pixels"][Ellipsis, :3])**2).mean()
      psnr_c = utils.compute_psnr(loss_c)
      stats.append(utils.Stats(loss=loss_c, psnr=psnr_c))
    else:
      loss_c = 0.
      psnr_c = 0.
    return loss + loss_c, (new_model_state, stats)

  step = state.step
  optimizer = state.optimizer
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (unused_loss, (new_model_state, stats)), grad = grad_fn(optimizer.target)
  grad = jax.lax.pmean(grad, axis_name="batch")
  stats = jax.lax.pmean(stats, axis_name="batch")
  new_optimizer = optimizer.apply_gradient(grad, learning_rate=lr)
  new_state = state.replace(
      step=step + 1, optimizer=new_optimizer, model_state=new_model_state)
  return new_state, stats, rng_key


def main(unused_argv):
  rng = random.PRNGKey(20200823)
  # Shift the numpy random seed by host_id() to shuffle data loaded by different
  # hosts.
  np.random.seed(20201473 + jax.host_id())

  if FLAGS.config is not None:
    utils.update_flags(FLAGS)
  if FLAGS.batch_size % jax.device_count() != 0:
    raise ValueError("Batch size must be divisible by the number of devices.")
  if FLAGS.train_dir is None:
    raise ValueError("train_dir must be set. None set now.")
  if FLAGS.data_dir is None:
    raise ValueError("data_dir must be set. None set now.")
  dataset = datasets.get_dataset("train", FLAGS)
  test_dataset = datasets.get_dataset("test", FLAGS)
  test_render_fn = jax.pmap(
      # Note rng_keys are useless in eval mode since there's no randomness.
      # pylint: disable=g-long-lambda
      lambda key_0, key_1, model, rays: jax.lax.all_gather(
          model(key_0, key_1, rays), axis_name="batch"),
      in_axes=(None, None, None, 0),  # Only distribute the data input.
      donate_argnums=3,
      axis_name="batch",
  )
  rng, key = random.split(rng)
  init_model, init_state = models.get_model(key, FLAGS)
  optimizer_def = optim.Adam(FLAGS.lr)
  optimizer = optimizer_def.create(init_model)
  state = model_utils.TrainState(
      step=0, optimizer=optimizer, model_state=init_state)
  if not utils.isdir(FLAGS.train_dir):
    utils.makedirs(FLAGS.train_dir)
  state = checkpoints.restore_checkpoint(FLAGS.train_dir, state)
  offset = state.step + 1
  state = jax_utils.replicate(state)
  del init_model, init_state

  if jax.host_id() == 0:
    summary_writer = tensorboard.SummaryWriter(FLAGS.train_dir)
  t_loop_start = time.time()
  learning_rate_fn = functools.partial(
      utils.learning_rate_decay,
      init_lr=FLAGS.lr,
      decay_steps=FLAGS.lr_decay * 1000,
      decay_rate=0.1)
  ptrain_step = jax.pmap(
      train_step, axis_name="batch", in_axes=(0, 0, 0, None), donate_argnums=2)
  # Prefetch_buffer_size = 3 x batch_size
  pdataset = jax_utils.prefetch_to_device(dataset, 3)
  n_local_deices = jax.local_device_count()
  rng = rng + jax.host_id()  # Make random seed separate across hosts.
  keys = random.split(rng, n_local_deices)  # For pmapping RNG keys.
  gc.disable()  # Disable automatic garbage collection for efficiency.
  for step, batch in zip(range(offset, FLAGS.max_steps + 1), pdataset):
    lr = learning_rate_fn(step)
    state, stats, keys = ptrain_step(keys, state, batch, lr)
    if step % FLAGS.gc_every == 0:
      gc.collect()
    # --- Train logs start ---
    # Put the training time visualization before the host_id check as in
    # multi-host evaluation, all hosts need to run inference even though we
    # only use host 0 to record results.
    if FLAGS.render_every > 0 and step % FLAGS.render_every == 0:
      # We reuse the same random number generator from the optimization step
      # here on purpose so that the visualization matches what happened in
      # training.
      state_to_eval = jax.device_get(jax.tree_map(lambda x: x[0], state))
      test_case = next(test_dataset)
      pred_color, pred_disp, pred_acc = utils.render_image(
          state_to_eval, test_case, test_render_fn, keys[0], FLAGS.chunk)
      if jax.host_id() == 0:
        summary_writer.image("pred_color", pred_color, step)
        summary_writer.image("pred_disp", pred_disp, step)
        summary_writer.image("pred_acc", pred_acc, step)
        summary_writer.image("target", test_case["pixels"], step)
    if jax.host_id() != 0:  # Only log via host 0.
      continue
    if step % 100 == 0:
      steps_per_sec = 100. / (time.time() - t_loop_start)
      t_loop_start = time.time()
      summary_writer.scalar("loss", stats[0].loss[0], step)
      summary_writer.scalar("psnr", stats[0].psnr[0], step)
      summary_writer.scalar("learning_rate", lr, step)
      if len(stats) > 1:
        summary_writer.scalar("loss_coarse", stats[1].loss[0], step)
        summary_writer.scalar("psnr_coarse", stats[1].psnr[0], step)
      summary_writer.scalar("step/sec", steps_per_sec, step)
    if step % FLAGS.save_every == 0:
      state_to_save = jax.device_get(jax.tree_map(lambda x: x[0], state))
      checkpoints.save_checkpoint(
          FLAGS.train_dir, state_to_save, state_to_save.step, keep=100)
    # --- Train logs end ---

  if FLAGS.max_steps % FLAGS.save_every != 0:
    state = jax.device_get(jax.tree_map(lambda x: x[0], state))
    checkpoints.save_checkpoint(
        FLAGS.train_dir, state, int(state.step), keep=100)


if __name__ == "__main__":
  app.run(main)
