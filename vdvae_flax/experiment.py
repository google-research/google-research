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

from vdvae_flax import dataset
from vdvae_flax import optimizers
from vdvae_flax import vdvae
from vdvae_flax import vdvae_utils

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
    self.vdvae = vdvae.Vdvae(config)
    self.rng = jax.random.PRNGKey(config.seed)

    # setup eval
    _, self._eval_ds = dataset.create_eval_dataset(
        config.data.task,
        config.evaluation.batch_size,
        config.evaluation.subset)
    self.rng, eval_rng = jax.random.split(self.rng)
    self._eval_batch = functools.partial(self._eval_batch, base_rng=eval_rng)
    self._eval_batch = jax.pmap(self._eval_batch, axis_name='batch')

    if mode == 'train':
      self.rng, data_rng = jax.random.split(self.rng)
      data_rng = jax.random.fold_in(data_rng, jax.process_index())
      _, train_ds = dataset.create_train_dataset(
          config.data.task,
          config.training.batch_size,
          config.training.substeps,
          data_rng)
      self._train_iter = iter(train_ds)

      self.rng, init_rng, sample_rng = jax.random.split(self.rng, num=3)
      input_shape = tuple(train_ds.element_spec.shape[2:])
      params = self.vdvae.init(
          init_rng, sample_rng, input_shape[0],
          jnp.ones(input_shape, dtype=jnp.uint8))
      parameter_overview.log_parameter_overview(params)
      # Use the same rng to init with the same params.
      ema_params = jax.tree_map(jnp.array, params)

      opt_init, _ = self.optimizer(
          learning_rate=self.config.optimizer.base_learning_rate)
      opt_state = opt_init(params)

      # create train state
      self._train_state = TrainState(
          step=0, params=params, ema_params=ema_params, opt_state=opt_state)

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
          inputs = jax.tree_map(np.asarray, next(self._train_iter))
          state, outputs = self._update_func(state, inputs)

        # Quick indication that training is happening.
        logging.log_first_n(logging.INFO, 'Finished training step %d.', 5, step)
        for h in hooks:
          h(step)

        new_step = int(state.step[0])
        assert new_step == step + substeps
        step = new_step

        is_eval = step % config.logs.eval_full_every_steps == 0 or is_last_step
        if step % config.logs.log_loss_every_steps == 0 and not is_eval:

          def avg_over_substeps(x):
            assert x.shape[0] == substeps
            return float(x.mean(axis=0))

          # Extract scalars and images.
          outputs = flax_utils.unreplicate(outputs)
          outputs = jax.tree_map(avg_over_substeps, outputs)
          scalars = outputs['scalars']
          writer.write_scalars(step, scalars)

        if is_eval:
          with report_progress.timed('eval_full'):
            outputs = self._eval_epoch(params=state.ema_params)
            outputs = flax_utils.unreplicate(outputs)
            scalars = outputs['scalars']
            writer.write_scalars(step, scalars)

        if step % config.logs.checkpoint_every_steps == 0 or is_last_step:
          with report_progress.timed('checkpoint'):
            ckpt.save(flax_utils.unreplicate(state))

    logging.info('Finishing training at step %d', num_train_steps)

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

  def _eval_batch(self, base_rng, params, inputs, step):
    """Evaluates a batch."""
    rng = jax.random.fold_in(base_rng, jax.lax.axis_index('batch'))
    rng = jax.random.fold_in(rng, step)
    _, loss_dict = self._loss_fn(params, inputs, rng, for_evaluation=True)
    loss_dict['scalars'] = jax.tree_map(
        lambda x: jax.lax.psum(x, axis_name='batch') / jax.device_count(),
        loss_dict['scalars'],
    )
    loss_dict['images'] = jax.tree_map(
        lambda x: vdvae_utils.generate_image_grids(x)[None, :, :, :],
        loss_dict['images'])

    return loss_dict

  def _eval_epoch(self, params):
    """Evaluates an epoch."""
    summed_scalars = None
    concat_images = None
    i = 0

    for i, inputs in enumerate(self._eval_ds):
      inputs = jax.tree_map(np.asarray, inputs)
      outputs = self._eval_batch(
          params=params,
          inputs=inputs,
          step=flax_utils.replicate(i),
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

    mean_scalars = jax.tree_map(lambda x: x / (i + 1),
                                summed_scalars)

    return {'scalars': mean_scalars, 'images': concat_images}
