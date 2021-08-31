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

"""training a VDVAE."""

import functools
import importlib
import os
from typing import NamedTuple, Any

from absl import logging

from clu import checkpoint
from clu import metric_writers
from clu import parameter_overview
from clu import periodic_actions
import flax
import flax.jax_utils as flax_utils
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import tensorflow.compat.v2 as tf

import tensorflow_datasets as tfds

from vdvae_flax import optimizers
from vdvae_flax import vdvae_utils

AUTOTUNE = tf.data.experimental.AUTOTUNE
OptState = NamedTuple


@flax.struct.dataclass
class TrainState:
  step: int
  params: Any
  ema_params: Any
  opt_state: OptState


class Experiment:
  """VAE experiment."""

  def __init__(self, mode, config):
    """Initializes experiment."""

    self.mode = mode
    self.config = config
    vdvae = importlib.import_module(
        f'.{self.config.model_name}'
    )
    self.vdvae = vdvae.Vdvae(config)
    self.rng = jax.random.PRNGKey(config.seed)

    # setup eval
    self._eval_input = self._build_eval_input()
    self.rng, eval_rng = jax.random.split(self.rng)
    self._eval_batch = functools.partial(self._eval_batch, base_rng=eval_rng)
    self._eval_batch = jax.pmap(self._eval_batch, axis_name='batch')
    self._num_eval_batch = int(
        np.ceil(self.config.evaluation.num_data /
                self.config.evaluation.batch_size))
    self._test_input_batch = next(self._eval_input)

    if mode == 'train':
      self.rng, init_rng = jax.random.split(self.rng)

      init_rng, sample_rng = jax.random.split(init_rng)
      params = self.vdvae.init(
          init_rng, sample_rng, self._test_input_batch.shape[-4],
          jnp.ones(self._test_input_batch.shape[-4:], dtype=jnp.uint8))
      parameter_overview.log_parameter_overview(params)
      # Use the same rng to init with the same params.
      ema_params = jax.tree_map(jnp.array, params)

      opt_init, _ = self.optimizer(
          learning_rate=self.config.optimizer.base_learning_rate)
      opt_state = opt_init(params)

      # create train state
      self._train_state = TrainState(
          step=0, params=params, ema_params=ema_params, opt_state=opt_state)

      self._train_input = vdvae_utils.py_prefetch(self._build_train_input)
      self.rng, update_rng = jax.random.split(self.rng)
      self._update_func = functools.partial(self._update_func, update_rng)
      self._update_func = functools.partial(jax.lax.scan, self._update_func)
      self._update_func = jax.pmap(self._update_func, axis_name='batch')

  def learning_rate(self, global_step):
    """Learning rate scheduler.

    Args:
      global_step: the current model step

    Returns:
      A f32 lr scalar.
    """
    learning_rate = self.config.optimizer.base_learning_rate
    warmup_steps = self.config.training.warmup_iters
    if warmup_steps > 0:
      learning_rate = learning_rate * jnp.minimum(
          1.,
          global_step / warmup_steps,
      )

    return learning_rate

  def optimizer(self, learning_rate):
    """Construct optimizer."""
    clip = optax.clip_by_global_norm(self.config.optimizer.gradient_clip_norm)
    optimizer = getattr(optax, self.config.optimizer.name)(
        learning_rate,
        **self.config.optimizer.args,
    )
    optim_step = optax.chain(clip, optimizer)
    optim_step = optimizers.maybe_skip_gradient_update(
        optim_step,
        self.config.optimizer.gradient_skip_norm,
    )

    return optim_step

  #  _             _
  # | |_ _ __ __ _(_)_ __
  # | __| '__/ _` | | '_ \
  # | |_| | | (_| | | | | |
  #  \__|_|  \__,_|_|_| |_|
  #

  def train_and_evaluate(self, workdir):
    """Runs a training and evaluation loop.

    Args:
      workdir: Working directory for checkpoints and TF summaries. If this
        contains checkpoint training will be resumed from the latest checkpoint.
    """

    tf.io.gfile.makedirs(workdir)
    config = self.config
    substeps = config.training.substeps

    # Learning rate schedule.
    num_train_steps = config.training.num_train_steps
    logging.info('num_train_steps=%d', num_train_steps)

    # Get train state
    state = self._train_state

    # Set up checkpointing of the model and the input pipeline.
    checkpoint_dir = os.path.join(workdir, 'checkpoints')
    ckpt = checkpoint.MultihostCheckpoint(checkpoint_dir, max_to_keep=5)
    state = ckpt.restore_or_initialize(state)
    initial_step = int(state.step)

    # Distribute training.
    state = flax_utils.replicate(state)

    writer = metric_writers.create_default_writer(
        workdir, just_logging=jax.process_index() > 0)
    if initial_step == 0:
      writer.write_hparams(dict(config))

    logging.info('Starting training loop at step %d.', initial_step)
    hooks = []
    report_progress = periodic_actions.ReportProgress(
        num_train_steps=num_train_steps, writer=writer)
    if jax.process_index() == 0:
      hooks += [
          report_progress,
          periodic_actions.Profile(num_profile_steps=5, logdir=workdir)
      ]
    step = initial_step
    with metric_writers.ensure_flushes(writer):
      while step < num_train_steps:
        # `step` is a Python integer. `state.step` is JAX integer on the GPU/TPU
        # devices.
        is_last_step = step + substeps >= num_train_steps

        with jax.profiler.StepTraceAnnotation('train', step_num=step):
          inputs = next(self._train_input)
          state, outputs = self._update_func(state, inputs)

        # Quick indication that training is happening.
        logging.log_first_n(logging.INFO, 'Finished training step %d.', 5, step)
        for h in hooks:
          h(step)

        new_step = int(state.step[0])
        assert new_step == step + substeps
        step = new_step

        if step % config.logs.log_loss_every_steps == 0 or is_last_step:

          def avg_over_substeps(x):
            assert x.shape[0] == substeps
            return float(x.mean(axis=0))

          # Extract scalars and images.
          outputs = flax_utils.unreplicate(outputs)
          outputs = jax.tree_map(avg_over_substeps, outputs)
          scalars = outputs['scalars']
          writer.write_scalars(step, scalars)

        if step % config.logs.eval_batch_every_steps == 0 or is_last_step:
          with report_progress.timed('eval_batch'):
            outputs = self._eval_batch(
                params=state.ema_params,
                inputs=self._test_input_batch,
            )
            outputs = flax_utils.unreplicate(outputs)
            scalars, images = outputs['scalars'], outputs['images']
            writer.write_scalars(step, scalars)
            writer.write_images(step, images)

        if step % config.logs.eval_full_every_steps == 0 or is_last_step:
          with report_progress.timed('eval_full'):
            # eval_epoch_fn = functools.partial(self._eval_epoch, rng=rng)
            # outputs = jax.tree_map(np.array, eval_epoch_fn())
            outputs = self._eval_epoch(params=state.ema_params)
            outputs = flax_utils.unreplicate(outputs)
            scalars, images = outputs['scalars'], outputs['images']
            writer.write_scalars(step, scalars)
            # writer.write_images(step, images[:10])

        if step % config.logs.checkpoint_every_steps == 0 or is_last_step:
          with report_progress.timed('checkpoint'):
            ckpt.save(flax_utils.unreplicate(state))

    logging.info('Finishing training at step %d', num_train_steps)

  def _preprocess_cifar10(self, images, labels):
    """Helper to extract images from dict."""
    assert labels is not None
    return images

  def _build_train_input(self):
    """See base class."""
    num_devices = jax.device_count()
    total_batch_size = self.config.training.batch_size
    per_device_batch_size, ragged = divmod(total_batch_size, num_devices)

    if ragged:
      raise ValueError(
          f'Global batch size {total_batch_size} must be divisible by the '
          f'total number of devices {num_devices}')

    preprocess_fn, ds = self._get_ds_and_preprocess_fn(split='train')
    ds = ds.shard(jax.process_count(), jax.process_index())
    # Shuffle before repeat ensures all examples seen in an epoch.
    # See https://www.tensorflow.org/guide/data_performance#repeat_and_shuffle.
    ds = ds.shuffle(buffer_size=10000)
    ds = ds.repeat()
    ds = ds.map(preprocess_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(per_device_batch_size, drop_remainder=True)
    ds = ds.batch(self.config.training.substeps, drop_remainder=True)
    ds = ds.batch(jax.local_device_count(), drop_remainder=True)
    ds = ds.prefetch(AUTOTUNE)

    return iter(tfds.as_numpy(ds))

  def _update_func(
      self,
      base_rng,
      state,
      inputs,
  ):
    """Computes loss and updates model parameters."""
    step = state.step
    rng = jax.random.fold_in(base_rng, jax.lax.axis_index('batch'))
    rng = jax.random.fold_in(rng, step)
    grad_loss_fn = jax.value_and_grad(self._loss_fn, has_aux=True)
    (_, loss_dict), scaled_grads = grad_loss_fn(state.params, inputs, rng)
    grads = jax.lax.psum(scaled_grads, axis_name='batch')
    grad_norm = optax.global_norm(grads)
    loss_dict['scalars']['grad_norm'] = grad_norm

    # Compute and apply updates via our optimizer.
    learning_rate = self.learning_rate(state.step)
    loss_dict['scalars']['learning_rate'] = learning_rate
    _, opt_apply = self.optimizer(learning_rate)
    updates, new_opt_state = opt_apply(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)

    # Update ema params
    ema_rate = self.config.evaluation.ema_rate
    new_ema_params = jax.tree_multimap(
        lambda x, y: x + (1 - ema_rate) * (y - x),
        state.ema_params,
        new_params,
    )
    new_state = state.replace(
        step=step + 1,
        params=new_params,
        ema_params=new_ema_params,
        opt_state=new_opt_state)

    # Rescale loss dict and return
    loss_dict['scalars'] = jax.tree_map(
        lambda x: jax.lax.psum(x, axis_name='batch') / jax.device_count(),
        loss_dict['scalars'],
    )
    return new_state, loss_dict

  def _generate_image_grids(self, images):
    """Simple helper to generate a single image from a mini batch."""

    def image_grid(nrow, ncol, imagevecs, imshape):
      """Reshape a stack of image vectors into an image grid for plotting.

      Args:
        nrow: number of desired rows.
        ncol: number of desired columns.
        imagevecs: array of images.
        imshape: shape of image, [W, H, C]

      Returns:
        A single, non batched jnp.array of for the image grid.

      """
      images = iter(imagevecs.reshape((-1,) + imshape))
      return jnp.squeeze(
          jnp.vstack([
              jnp.hstack([next(images)
                          for _ in range(ncol)][::-1])
              for _ in range(nrow)
          ]))

    batch_size = images.shape[0]
    grid_size = int(np.floor(np.sqrt(batch_size)))

    image_shape = images.shape[1:]
    return image_grid(
        nrow=grid_size,
        ncol=grid_size,
        imagevecs=images[0:grid_size**2],
        imshape=image_shape,
    )

  def _loss_fn(self, params, inputs, rng, for_evaluation=False):
    """Computes the variational lower bound."""
    # Forward pass the VDVAE and get the predictions.
    rng, sample_rng = jax.random.split(rng)
    vdvae_output = self.vdvae.apply(
        variables=params,
        sample_rng=sample_rng,
        num_samples_to_generate=inputs.shape[0],
        inputs=inputs,
    )

    nb_image_dim = np.prod(inputs.shape[1:])
    loss = jnp.mean(vdvae_output.elbo) / nb_image_dim  # for optimizing.

    # Aggregate the scalar values we want to monitor.
    scalar_dict = {
        'elbo': loss,
        'kld': vdvae_output.kl_per_decoder_block.sum(axis=0).mean(),
        'reconstruction_loss': vdvae_output.reconstruction_loss.mean(),
        'max_inputs': inputs.max(),
        'min_inputs': inputs.min(),
    }
    images_dict = {}

    if for_evaluation:
      rng, sample_rng = jax.random.split(rng)
      num_samples_to_generate = self.config.evaluation.batch_size // jax.device_count(
      )
      vdvae_uncond_output = self.vdvae.apply(
          variables=params,
          sample_rng=sample_rng,
          num_samples_to_generate=num_samples_to_generate,
          inputs=None)
      sample_grid = vdvae_utils.allgather_and_reshape(
          vdvae_uncond_output.samples)
      # Create a single image out of a minibatch for flatboard visualization.
      input_grid = vdvae_utils.allgather_and_reshape(inputs)
      recon_grid = vdvae_utils.allgather_and_reshape(vdvae_output.samples)

      # Add image reconstruction
      images_dict = {
          'samples': sample_grid,
          'reconstructions': recon_grid,
          'inputs': input_grid,
      }

    # Loss contains both the scalars and the images.
    loss_dict = {'scalars': scalar_dict, 'images': images_dict}

    # return the scaled elbo for optimization, but others for visualization.
    scaled_loss = loss / jax.device_count()
    return scaled_loss, loss_dict

  #                  _
  #   _____   ____ _| |
  #  / _ \ \ / / _` | |
  # |  __/\ V / (_| | |
  #  \___| \_/ \__,_|_|
  #

  def evaluate(self, workdir, dir_name='eval', ckpt_name=None):
    """Perform one evaluation."""
    checkpoint_dir = os.path.join(workdir, 'checkpoints-0')
    ckpt = checkpoint.Checkpoint(checkpoint_dir)
    state_dict = ckpt.restore_dict(os.path.join(checkpoint_dir, ckpt_name))
    ema_params = flax.core.FrozenDict(state_dict['ema_params'])
    step = int(state_dict['step'])

    # Distribute training.
    ema_params = flax_utils.replicate(ema_params)

    eval_logdir = os.path.join(workdir, dir_name)
    tf.io.gfile.makedirs(eval_logdir)
    writer = metric_writers.create_default_writer(
        eval_logdir, just_logging=jax.process_index() > 0)

    outputs = self._eval_epoch(params=ema_params)
    outputs = flax_utils.unreplicate(outputs)
    scalars, images = outputs['scalars'], outputs['images']
    writer.write_scalars(step, scalars)
    writer.write_images(step, images)

  def _get_ds_and_preprocess_fn(self, split):
    """Helper to get the right reprocessing function and dataset."""
    if self.config.data.task == 'cifar10':
      preprocess_fn = self._preprocess_cifar10
      ds = tfds.load(
          name='cifar10',
          split=split,
          as_supervised=True,
      )
    else:
      raise NotImplementedError(
          'task {} not implemented'.format(self.config.data.task),)

    return preprocess_fn, ds

  def _build_eval_input(self):
    """See base class."""
    num_devices = jax.device_count()
    total_batch_size = self.config.evaluation.batch_size
    per_device_batch_size, ragged = divmod(total_batch_size, num_devices)

    if ragged:
      raise ValueError(
          f'Global batch size {total_batch_size} must be divisible by the '
          f'total number of devices {num_devices}')

    preprocess_fn, ds = self._get_ds_and_preprocess_fn(
        split=self.config.evaluation.subset)
    ds = ds.shard(jax.process_count(), jax.process_index())
    ds = ds.repeat()
    ds = ds.map(preprocess_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(per_device_batch_size, drop_remainder=True)
    ds = ds.batch(jax.local_device_count(), drop_remainder=True)
    ds = ds.prefetch(AUTOTUNE)
    return iter(tfds.as_numpy(ds))

  def _eval_batch(self, base_rng, params, inputs, step=0):
    """Evaluates a batch."""
    rng = jax.random.fold_in(base_rng, jax.lax.axis_index('batch'))
    rng = jax.random.fold_in(rng, step)
    _, loss_dict = self._loss_fn(params, inputs, rng, for_evaluation=True)
    loss_dict['scalars'] = jax.tree_map(
        lambda x: jax.lax.psum(x, axis_name='batch') / jax.device_count(),
        loss_dict['scalars'],
    )
    loss_dict['images'] = jax.tree_map(
        lambda x: self._generate_image_grids(x)[None, :, :, :],
        loss_dict['images'])

    return loss_dict

  def _eval_epoch(self, params):
    """Evaluates an epoch."""
    num_minibatches_seen = 0.
    summed_scalars = None
    concat_images = None

    for num_minibatches_seen in range(self._num_eval_batch):
      inputs = next(self._eval_input)
      outputs = self._eval_batch(
          params=params,
          inputs=inputs,
          step=flax_utils.replicate(num_minibatches_seen),
      )
      scalars = outputs['scalars']
      images = outputs['images']

      # Accumulate the sum of scalars for each step.
      if summed_scalars is None:
        summed_scalars = scalars
        concat_images = images
      else:
        summed_scalars = jax.tree_multimap(jnp.add, summed_scalars, scalars)
        concat_images = jax.tree_multimap(
            lambda x, y: jnp.concatenate((x, y), axis=1), concat_images, images)

    mean_scalars = jax.tree_map(lambda x: x / (num_minibatches_seen + 1),
                                summed_scalars)

    return {'scalars': mean_scalars, 'images': concat_images}
