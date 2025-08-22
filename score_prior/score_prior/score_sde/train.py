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

"""Train a score model with `score_sde` library.

Please see https://github.com/yang-song/score_sde/blob/main/run_lib.py
for the official training implementation.
"""

import functools
import logging
import os

from absl import app
from absl import flags
import flax
from flax.metrics import tensorboard
from flax.training import checkpoints
import jax
from ml_collections.config_flags import config_flags
import numpy as np
from score_sde import sampling
import tensorflow as tf

from score_prior import datasets
from score_prior import utils
from score_prior.score_sde import losses

_CONFIG = config_flags.DEFINE_config_file('config', None, 'Score-model config.')
_WORKDIR = flags.DEFINE_string(
    'workdir', None, 'Working directory.', required=True)


def get_datasets_and_scalers():
  """Get train and eval datasets and data scaler and inverse scaler."""
  config = _CONFIG.value
  train_ds, eval_ds, _ = datasets.get_dataset(
      config,
      additional_dim=config.training.n_jitted_steps,
      uniform_dequantization=config.data.uniform_dequantization)
  # `scaler` assumes images are originally [0, 1] and scales to
  # [0, 1] or [-1, 1].
  scaler = datasets.get_data_scaler(config)
  # `inverse_scaler` rescales to images that are [0, 1].
  inverse_scaler = datasets.get_data_inverse_scaler(config)
  return (train_ds, eval_ds), (scaler, inverse_scaler)


def save_checkpoint(ckpt_dir, pstate, rng, **kwargs):
  """Unreplicate `pstate` and save it as a checkpoint."""
  saved_state = flax.jax_utils.unreplicate(pstate)
  saved_state = saved_state.replace(rng=rng)
  path = checkpoints.save_checkpoint(ckpt_dir, saved_state, **kwargs)
  return path


def main(_):
  config = _CONFIG.value
  workdir = _WORKDIR.value

  # Create working directory and its subdirectories.
  workdir = _WORKDIR.value
  ckpt_dir = os.path.join(workdir, 'checkpoints')
  tb_dir = os.path.join(workdir, 'tensorboard')
  sample_dir = os.path.join(workdir, 'samples')
  tf.io.gfile.makedirs(ckpt_dir)
  tf.io.gfile.makedirs(tb_dir)
  tf.io.gfile.makedirs(sample_dir)

  # Create TensorBoard writer.
  writer = tensorboard.SummaryWriter(tb_dir)

  if utils.is_coordinator():
    # Save config.
    with tf.io.gfile.GFile(os.path.join(workdir, 'config.txt'), 'w') as f:
      f.write(str(config))

  # Get data.
  (train_ds, eval_ds), (scaler, inverse_scaler) = get_datasets_and_scalers()
  train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types
  eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types

  # Initialize training state.
  state, score_model, tx = utils.initialize_training_state(config)

  # Load checkpoint.
  state = checkpoints.restore_checkpoint(ckpt_dir, state)
  if utils.is_coordinator():
    logging.info('Starting training at step %d', state.step)

  # Get SDE.
  sde, t0_eps = utils.get_sde(config)

  # Build training and eval functions.
  optimize_fn = losses.optimization_manager(config)
  train_step_fn = losses.get_step_fn(
      sde,
      score_model,
      optimizer=tx,
      train=True,
      optimize_fn=optimize_fn,
      reduce_mean=config.training.reduce_mean,
      continuous=config.training.continuous,
      likelihood_weighting=config.training.likelihood_weighting)
  eval_step_fn = losses.get_step_fn(
      sde,
      score_model,
      optimizer=tx,
      train=False,
      optimize_fn=optimize_fn,
      reduce_mean=config.training.reduce_mean,
      continuous=config.training.continuous,
      likelihood_weighting=config.training.likelihood_weighting)

  # Build sampling function.
  sampling_shape = (
      int(config.training.batch_size // jax.device_count()),
      config.data.image_size, config.data.image_size,
      config.data.num_channels)
  sampling_fn = sampling.get_sampling_fn(
      config, sde, score_model, sampling_shape, inverse_scaler, t0_eps)

  # Pmap and JIT multiple training/eval steps together for faster running.
  p_train_step = jax.pmap(
      functools.partial(jax.lax.scan, train_step_fn), axis_name='batch',
      donate_argnums=1)
  p_eval_step = jax.pmap(
      functools.partial(jax.lax.scan, eval_step_fn), axis_name='batch',
      donate_argnums=1)

  # Replicate training state to run on multiple devices.
  pstate = flax.jax_utils.replicate(state)

  # Keep track of best eval loss.
  best_eval_loss = np.inf
  best_eval_ckpt_path = None

  init_step = state.step
  n_steps = config.training.n_iters
  n_jitted_steps = config.training.n_jitted_steps
  # Create different random states for different processes in a
  # multi-host environment (e.g., TPU pods).
  rng = jax.random.fold_in(state.rng, jax.process_index())
  for step in range(init_step, n_steps + 1, n_jitted_steps):
    # Convert data to NumPy arrays and normalize them.
    batch = jax.tree.map(lambda x: scaler(x._numpy()), next(train_iter))  # pylint: disable=protected-access

    # Update RNG.
    rng, step_rngs = utils.psplit(rng)

    # Train step.
    (_, pstate), ploss = p_train_step((step_rngs, pstate), batch)
    loss = flax.jax_utils.unreplicate(ploss).mean()

    if step % config.training.log_freq == 0 and utils.is_coordinator():
      # Log training loss.
      logging.info('[step %d] training loss: %.5e', step, loss)
      writer.scalar('training_loss', loss, step)

    if ((step != 0 and step % config.training.snapshot_freq == 0 or
         step == n_steps) and utils.is_coordinator()):
      # Save model checkpoint.
      save_checkpoint(ckpt_dir, pstate, rng, step=step, keep=np.inf)

      # Save samples.
      rng, sample_rngs = utils.psplit(rng)
      sample, _ = sampling_fn(sample_rngs, pstate)
      image_grid = sample.reshape(-1, *sample.shape[2:])
      nrow = int(np.sqrt(image_grid.shape[0]))
      with tf.io.gfile.GFile(
          os.path.join(sample_dir, f'samples_{step}.png'), 'wb') as fout:
        utils.save_image_grid(image_grid, fout, nrow=nrow, padding=2)

    if step % config.training.eval_freq == 0:
      # Eval step.
      eval_batch = jax.tree.map(lambda x: scaler(x._numpy()), next(eval_iter))  # pylint: disable=protected-access
      rng, next_rngs = utils.psplit(rng)
      (_, _), peval_loss = p_eval_step((next_rngs, pstate), eval_batch)

      eval_loss = flax.jax_utils.unreplicate(peval_loss).mean()
      if utils.is_coordinator():
        logging.info('[step %d] eval loss: %.5e', step, eval_loss)
        writer.scalar('eval_loss', eval_loss, step)

      if eval_loss < best_eval_loss and utils.is_coordinator():
        if best_eval_ckpt_path is not None:
          # We want to keep only the best checkpoint,
          # so delete the previous one.
          tf.io.gfile.remove(best_eval_ckpt_path)

        best_eval_loss = eval_loss
        best_eval_ckpt_path = save_checkpoint(
            ckpt_dir, pstate, rng,
            step=step, prefix=f'best_eval_{eval_loss:.3f}_', overwrite=True)

if __name__ == '__main__':
  app.run(main)
