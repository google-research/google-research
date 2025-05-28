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

"""Train DPI. Runs on 1 TPU device."""
import datetime
import logging
import os
import time
from typing import Any, Callable

from absl import app
from absl import flags
import flax
from flax.training import checkpoints
import jax
from jaxtyping import PyTree
from ml_collections.config_flags import config_flags
import numpy as np
import optax
from PIL import Image
from score_sde import utils as sutils
import tensorflow as tf

from score_prior import datasets
from score_prior import inference_utils
from score_prior import utils
from score_prior.posterior_sampling import losses
from score_prior.posterior_sampling import model_utils

# DPI config.
_CONFIG = config_flags.DEFINE_config_file('config', None, 'DPI config.')
# Score-model config.
_SCORE_MODEL_CONFIG = config_flags.DEFINE_config_file(
    'score_model_config', None, 'SCore-model config.')
# Working directory.
_WORKDIR = flags.DEFINE_string(
    'workdir', None,
    'Directory to write checkpoints, samples, and TensorBoard logs to.',
    required=True)


def save_configs():
  """Save grad ascent and score-model configs."""
  workdir = _WORKDIR.value
  config = _CONFIG.value
  score_model_config = _SCORE_MODEL_CONFIG.value

  with tf.io.gfile.GFile(os.path.join(workdir, 'config.txt'), 'w') as f:
    f.write(str(config))
  with tf.io.gfile.GFile(
      os.path.join(workdir, 'score_model_config.txt'), 'w') as f:
    f.write(str(score_model_config))


def save_true_and_naive_images(true_image, naive_image):
  """Save true image and naive image as NumPy arrays and .png files.

  Assumes images are scaled [0, 1].

  Args:
    true_image: Original image underlying the measurement.
    naive_image: Naive (least-squares) reconstruction from the measurement.
  """
  config = _CONFIG.value
  workdir = _WORKDIR.value

  # Save images as NumPy arrays.
  with tf.io.gfile.GFile(
      os.path.join(workdir, 'true_image.npy'), 'wb') as f:
    np.save(f, true_image)
  with tf.io.gfile.GFile(
      os.path.join(workdir, 'naive_image.npy'), 'wb') as f:
    np.save(f, naive_image)

  # Save true image and naive image as PNG images.
  true_image = (true_image * 255).astype(np.uint8)
  naive_image = (naive_image * 255).astype(np.uint8)

  if config.data.num_channels == 1:
    true_image = true_image[:, :, 0]
    naive_image = naive_image[:, :, 0]

  with tf.io.gfile.GFile(
      os.path.join(workdir, 'true_image.png'), 'wb') as f:
    Image.fromarray(true_image).save(f)
  with tf.io.gfile.GFile(
      os.path.join(workdir, 'naive_image.png'), 'wb') as f:
    Image.fromarray(naive_image).save(f)


# pylint:disable=line-too-long
def train(train_step_fn,
          params,
          model_state,
          optimizer,
          max_iters,
          reinit_opt_state = False):
# pylint:enable=line-too-long
  """Train DPI.

  This function starts training from a checkpoint. If no checkpoint is found,
  it starts training from scratch.

  `train_step_fn` will be pjitted, so expect the inputs and outputs of
  `train_step_fn` to have an additional local-device axis.

  Args:
    train_step_fn: The function that takes `rng, state` and returns
      `new_state, (loss, loss_data, loss_prior, loss_entropy), samples)`.
    params: `PyTree` of current model params.
    model_state: `PyTree` of current model state, including `batch_stats`.
    optimizer: Optax optimizer.
    max_iters: Keep training until `max_iters` steps are done.
    reinit_opt_state: If `True`, start the training loop from the initialized
      optimizer state. If `False`, start with the optimizer state found in
      the checkpoint.
  """
  config = _CONFIG.value
  image_shape = (
      config.data.image_size, config.data.image_size, config.data.num_channels)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  ckpt_dir = os.path.join(_WORKDIR.value, 'checkpoints')
  sample_dir = os.path.join(_WORKDIR.value, 'samples')
  tf.io.gfile.makedirs(ckpt_dir)
  tf.io.gfile.makedirs(sample_dir)

  if utils.is_coordinator():
    # Create summary writer.
    now = datetime.datetime.now()
    tb_dir = os.path.join(
        _WORKDIR.value, 'tensorboard', now.strftime('%Y%m%d-%H%M%S'))
    writer = tf.summary.create_file_writer(tb_dir)

  # Construct training state.
  opt_state = optimizer.init(params)
  state = model_utils.State(
      step=0,
      opt_state=opt_state,
      params=params,
      model_state=model_state,
      data_weight=config.optim.lambda_data,
      prior_weight=config.optim.lambda_prior,
      entropy_weight=config.optim.lambda_entropy,
      rng=jax.random.PRNGKey(config.seed + 1)
  )

  # Load checkpoint.
  state = checkpoints.restore_checkpoint(ckpt_dir, state)
  init_step = state.step
  if init_step > max_iters:
    return
  if utils.is_coordinator():
    logging.info('Starting training at step %d', state.step)

  if reinit_opt_state:
    # Start from the initialized optimizer state.
    state = state.replace(opt_state=opt_state)

  p_train_step = jax.pmap(
      jax.jit(train_step_fn), axis_name='batch', donate_argnums=(1,))
  pstate = flax.jax_utils.replicate(state)
  # Create different random states for different processes in a
  # multi-host environment (e.g., TPU pods).
  rng = jax.random.fold_in(state.rng, jax.process_index())
  for step in range(init_step, max_iters + 1):
    # Track the amount of time it takes per step.
    step_start_time = time.perf_counter()

    # Update data weight.
    data_weight = losses.data_weight_fn(
        step,
        start_order=config.optim.lambda_data_start_order,
        decay_rate=config.optim.lambda_data_decay_steps,
        final_data_weight=config.optim.lambda_data)
    pstate = pstate.replace(
        data_weight=flax.jax_utils.replicate(data_weight))

    rng, step_rngs = utils.psplit(rng)
    pstate, (ploss, ploss_data, ploss_prior,
             ploss_entropy), psamples = p_train_step(step_rngs, pstate)

    loss = flax.jax_utils.unreplicate(ploss).item()
    loss_data = flax.jax_utils.unreplicate(ploss_data).item()
    loss_prior = flax.jax_utils.unreplicate(ploss_prior).item()
    loss_entropy = flax.jax_utils.unreplicate(ploss_entropy).item()

    if (((step + 1) % config.training.log_freq == 0 or step == init_step) and
        utils.is_coordinator()):
      step_time = time.perf_counter() - step_start_time
      logging.info(
          'step %d: %.0f seconds (data weight = %.1e)',
          step + 1, step_time, data_weight)
      with writer.as_default(step=step + 1):
        tf.summary.scalar('total', loss)
        tf.summary.scalar('likelihood', loss_data)
        tf.summary.scalar('prior', loss_prior)
        tf.summary.scalar('entropy', loss_entropy)

    if ((step + 1) % config.training.snapshot_freq == 0 and
        utils.is_coordinator()):
      # Save checkpoint.
      checkpoints.save_checkpoint(
          ckpt_dir,
          flax.jax_utils.unreplicate(pstate),
          step=step + 1,
          keep=config.training.n_saved_checkpoints)

      # Save .npy file of all samples.
      samples = psamples.reshape(
          config.training.batch_size // jax.process_count(), *image_shape)
      with tf.io.gfile.GFile(
          os.path.join(sample_dir, f'x_{step + 1:06}.npy'), 'wb') as f:
        np.save(f, samples)
      # Save .png file of all samples.
      images = inverse_scaler(samples)
      with tf.io.gfile.GFile(
          os.path.join(sample_dir, f'x_{step + 1:06}.png'), 'wb') as f:
        sutils.save_image(images[:9], f, nrow=3, padding=2)


def main(_):
  config = _CONFIG.value
  score_model_config = _SCORE_MODEL_CONFIG.value
  inverse_scaler = datasets.get_data_inverse_scaler(config)
  sde, t0 = utils.get_sde(score_model_config)

  if utils.is_coordinator():
    tf.io.gfile.makedirs(_WORKDIR.value)
    save_configs()
    logging.info('[INFO] local device count = %d', jax.local_device_count())
    logging.info('[INFO] device count = %d', jax.device_count())
    logging.info('[INFO] process count = %d', jax.process_count())

  # Get likelihood module.
  likelihood = inference_utils.get_likelihood(config)

  # Get measurements.
  true_image, y = inference_utils.get_measurement(config, likelihood)
  naive_image = np.array(likelihood.invert_measurement(y)[0])

  if utils.is_coordinator():
    # Save true and naive images.
    save_true_and_naive_images(
        inverse_scaler(true_image), inverse_scaler(naive_image))

  # Initialize generator model.
  # `params` is a dict of trainable parameters.
  # `model_state` is a dict of mutable states, e.g., `batch_stats`.
  model, model_state, params = model_utils.get_model_and_init_params(
      config, image_flux=np.sum(true_image), train=True)

  # Create optimizer.
  optimizer = optax.adam(
      learning_rate=config.optim.learning_rate,
      b1=config.optim.adam_beta1, b2=config.optim.adam_beta2,
      eps=config.optim.adam_eps)

  # Data loss function.
  data_loss_fn = losses.get_data_loss_fn(likelihood, y)

  # Get `ProbabilityFlow` module.
  if config.optim.prior == 'score':
    prob_flow = inference_utils.get_prob_flow(
        config, score_model_config)
  else:
    prob_flow = None

  # Get step function.
  prior_loss_fn = losses.get_prior_loss_fn(
      config, prob_flow, t0=t0, t1=sde.T, dt0=config.prob_flow.dt0)
  train_step_fn = losses.get_train_step_fn(
      config, model, optimizer,
      data_loss_fn, prior_loss_fn,
      use_score_fn=False)

  train(
      train_step_fn, params, model_state, optimizer,
      max_iters=config.training.n_iters)


if __name__ == '__main__':
  app.run(main)
