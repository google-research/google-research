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

"""Training worker for generative model."""

import functools
import os
import time
from typing import Any, Dict, Tuple

from absl import logging
from clu import metric_writers
import flax
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import ml_collections
import numpy as onp
import tensorflow.compat.v2 as tf

from d3pm.images import datasets
from d3pm.images import utils


@flax.struct.dataclass
class TrainState:
  step: int
  optimizer: flax.optim.Optimizer
  ema_params: Any


class TrainableModel:
  """Generic trainable model with training and eval/sampling workers.

  config: reads config.seed, config.train, config.eval, config.dataset
  """

  def __init__(self, config, dataset=None):
    self.config = config

    if dataset is not None:
      self.dataset = dataset
    else:
      self.dataset = getattr(datasets,
                             config.dataset.name)(**config.dataset.args)

    self._eval_step = None

  # override
  def make_init_params(self, global_rng):
    raise NotImplementedError

  # override
  def loss_fn(self, rng, train, batch,
              params):
    """Loss function."""
    raise NotImplementedError

  # override
  def samples_fn(self, params, rng,
                 samples_shape):
    """Generate samples in [0, 255] for evaluation."""
    raise NotImplementedError

  def _gen_samples(self, params, rng, samples_shape):
    """Generate samples (unnormalized, in [0, 255])."""

    # retries to guard against rare nans
    max_retries = 10

    def cond_fun(val):
      i, x = val
      return jnp.logical_and(i < max_retries,
                             jnp.logical_not(jnp.all(jnp.isfinite(x))))

    def body_fun(val):
      i, _ = val
      return (i + 1,
              self.samples_fn(params, jax.random.fold_in(rng, i),
                              samples_shape))

    _, unnormalized_samples = jax.lax.while_loop(
        cond_fun,
        body_fun,
        init_val=(0, jnp.full(samples_shape, jnp.nan, dtype=jnp.float32)))
    # assert unnormalized_samples.shape == device_batch_shape
    assert unnormalized_samples.shape == samples_shape

    return unnormalized_samples

  def get_model_samples(self, params, rng):
    """Generate one batch of samples."""
    rng = utils.RngGen(rng)
    samples = self.p_gen_samples(params, rng.split(jax.local_device_count()))
    return samples

  # possibly override
  def step_fn(self, base_rng, train, state,
              batch):
    """One training/eval step."""
    config = self.config

    # RNG for this step on this host
    step = state.step
    rng = jax.random.fold_in(base_rng, jax.lax.axis_index('batch'))
    rng = jax.random.fold_in(rng, step)
    rng = utils.RngGen(rng)

    # Loss and gradient
    loss_fn = functools.partial(self.loss_fn, next(rng), train, batch)

    if train:
      # Training mode
      (_, metrics), grad = jax.value_and_grad(
          loss_fn, has_aux=True)(
              state.optimizer.target)

      # Average grad across shards
      grad_clip = metrics['grad_clip'] = config.train.grad_clip
      grad, metrics['gnorm'] = utils.clip_by_global_norm(
          grad, clip_norm=grad_clip)
      grad = jax.lax.pmean(grad, axis_name='batch')

      # Learning rate
      if config.train.learning_rate_warmup_steps > 0:
        learning_rate = config.train.learning_rate * jnp.minimum(
            jnp.float32(step) / config.train.learning_rate_warmup_steps, 1.0)
      else:
        learning_rate = config.train.learning_rate
      metrics['lr'] = learning_rate

      # Update optimizer and EMA params
      new_optimizer = state.optimizer.apply_gradient(
          grad, learning_rate=learning_rate)
      new_ema_params = utils.apply_ema(
          decay=jnp.where(step == 0, 0.0, config.train.ema_decay),
          avg=state.ema_params,
          new=new_optimizer.target)
      new_state = state.replace(  # pytype: disable=attribute-error
          step=step + 1,
          optimizer=new_optimizer,
          ema_params=new_ema_params)
      if config.train.get('enable_update_skip', True):
        # Apply update if the new optimizer state is all finite
        ok = jnp.all(
            jnp.asarray([
                jnp.all(jnp.isfinite(p)) for p in jax.tree_leaves(new_optimizer)
            ]))
        new_state_no_update = state.replace(step=step + 1)
        state = jax.tree_multimap(lambda a, b: jnp.where(ok, a, b), new_state,
                                  new_state_no_update)
      else:
        logging.info('Update skipping disabled')
        state = new_state

    else:
      # Eval mode with EMA params
      _, metrics = loss_fn(state.ema_params)

    # Average metrics across shards
    metrics = jax.lax.pmean(metrics, axis_name='batch')
    # check that v.shape == () for all v in metric.values()
    assert all(not v.shape for v in metrics.values())
    metrics = {  # prepend prefix to names of metrics
        f"{'train' if train else 'eval'}/{k}": v for k, v in metrics.items()
    }
    return (state, metrics) if train else metrics

  def make_optimizer_def(self):
    """Make the optimizer def."""
    config = self.config

    optimizer_kwargs = {}
    if config.train.weight_decay > 0.:
      optimizer_kwargs['weight_decay'] = config.train.weight_decay

    if config.train.optimizer == 'adam':
      optimizer_def = flax.optim.Adam(
          **optimizer_kwargs,
          beta1=config.train.get('adam_beta1', 0.9),
          beta2=config.train.get('adam_beta2', 0.999))
    elif config.train.optimizer == 'momentum':
      optimizer_def = flax.optim.Momentum(
          **optimizer_kwargs, beta=config.train.optimizer_beta)
    elif config.train.optimizer == 'nesterov':
      optimizer_def = flax.optim.Momentum(
          **optimizer_kwargs, beta=config.train.optimizer_beta, nesterov=True)
    else:
      raise NotImplementedError(f'Unknown optimizer: {config.train.optimizer}')

    return optimizer_def

  ##### ##### #####

  def make_init_state(self):
    """Make an initial TrainState."""
    # Init model params (same rng across hosts)
    init_params = self.make_init_params(
        global_rng=jax.random.PRNGKey(self.config.seed))
    logging_string = (f'Param shapes: '
                      f'{jax.tree_map(lambda a: a.shape, init_params)}')
    logging.info(logging_string)
    logging.info('Number of trainable parameters: %d',
                 utils.count_params(init_params))

    # Make the optimizer
    optimizer_def = self.make_optimizer_def()

    # For ema_params below, copy so that pmap buffer donation doesn't donate the
    # same buffer twice
    return TrainState(
        step=0,
        optimizer=optimizer_def.create(init_params),
        ema_params=utils.copy_pytree(init_params))

  def _calc_eval_metrics(self, *, state, eval_iter, eval_steps, eval_base_rng,
                         total_bs):
    """Calculate eval metrics."""
    logging.info('Calculating eval metrics...')

    # Eval step (does not modify parameters; no substeps)
    if self._eval_step is None:

      def _eval_step(state_, batch_, rng_):
        return self.step_fn(
            base_rng=rng_, train=False, state=state_, batch=batch_)

      self._eval_step = utils.dist(
          _eval_step, accumulate='mean', axis_name='batch')

    rngs = [
        jax.random.split(
            jax.random.fold_in(eval_base_rng, i), jax.local_device_count())
        for i in range(eval_steps)
    ]
    all_eval_metrics = [
        self._eval_step(state, eval_batch, rng_i)
        for rng_i, eval_batch in zip(rngs, eval_iter)
    ]

    assert len(all_eval_metrics) == self.dataset.num_eval // total_bs
    logging.info('Evaluated %d batches (%d examples)', len(all_eval_metrics),
                 len(all_eval_metrics) * total_bs)
    if all_eval_metrics:
      averaged_eval_metrics = {
          key: float(onp.mean([float(m[key]) for m in all_eval_metrics
                              ])) for key in all_eval_metrics[0].keys()
      }
    else:
      warning = (f'No eval batches! num_eval={self.dataset.num_eval} '
                 f'total_bs={total_bs}')
      logging.warning(warning)
      averaged_eval_metrics = None
    return averaged_eval_metrics

  def run_train(self, experiment_dir, work_unit_dir,
                rng):
    """Training loop with fixed number of steps and checkpoint every steps."""
    del experiment_dir  # unused
    tf.io.gfile.makedirs(work_unit_dir)

    config = self.config

    total_bs = config.train.batch_size
    assert total_bs % jax.device_count() == 0, (
        f'num total devices {jax.device_count()} must divide the batch size '
        f'{total_bs}')
    device_bs = total_bs // jax.device_count()
    logging.info('total_bs=%d device_bs=%d', total_bs, device_bs)

    # Logging setup
    writer = metric_writers.create_default_writer(
        work_unit_dir, just_logging=jax.host_id() > 0)
    if jax.host_id() == 0:
      utils.write_config_json(config, os.path.join(work_unit_dir,
                                                   'config.json'))

    # Build input pipeline
    logging.info('Substeps per training step: %d', config.train.substeps)
    train_ds = self.dataset.get_tf_dataset(
        split='train',
        batch_shape=(
            jax.local_device_count(),  # for pmap
            config.train.substeps,  # for lax.scan over multiple substeps
            device_bs,  # batch size per device
        ),
        global_rng=jax.random.PRNGKey(config.seed),
        repeat=True,
        shuffle=True,
        augment=True,
        shard_id=jax.host_id(),
        num_shards=jax.host_count())
    train_iter = utils.numpy_iter(train_ds)
    eval_ds = self.dataset.get_tf_dataset(
        split='eval',
        batch_shape=(jax.local_device_count(), device_bs),
        global_rng=jax.random.PRNGKey(config.seed),
        repeat=True,
        shuffle=True,
        augment=False,
        shard_id=jax.host_id(),
        num_shards=jax.host_count())
    eval_iter = utils.numpy_iter(eval_ds)

    samples_shape = (device_bs, *self.dataset.data_shape)

    self.p_gen_samples = utils.dist(
        functools.partial(self._gen_samples, samples_shape=samples_shape),
        accumulate='concat',
        axis_name='batch')

    # Set up model and training state
    state = jax.device_get(self.make_init_state())
    checkpoint_dir = os.path.join(work_unit_dir, 'checkpoints')
    state = checkpoints.restore_checkpoint(checkpoint_dir, state)
    initial_step = int(state.step)
    state = flax.jax_utils.replicate(state)

    # Training step
    train_step = functools.partial(self.step_fn, next(rng), True)
    train_step = functools.partial(jax.lax.scan, train_step)  # for substeps
    train_step = jax.pmap(train_step, axis_name='batch', donate_argnums=(0,))

    # Eval step (does not modify parameters; no substeps)
    eval_base_rng = next(rng)

    # Training loop
    logging.info('Entering training loop at step %i', initial_step)
    utils.assert_synced(state)
    last_log_time = last_ckpt_time = time.time()
    prev_step = initial_step

    with metric_writers.ensure_flushes(writer):
      for batch in train_iter:

        state, metrics = train_step(state, batch)
        new_step = int(state.step[0])
        assert new_step == prev_step + config.train.substeps

        # Quick indication that training is happening.
        logging.log_first_n(logging.INFO, 'Finished training step %d', 5,
                            new_step)
        # Log metrics
        if new_step % config.train.log_loss_every_steps == 0:
          # Unreplicate metrics, average over substeps, and cast to python float
          metrics = jax.device_get(flax.jax_utils.unreplicate(metrics))

          def avg_over_substeps(x):
            assert x.shape[0] == config.train.substeps
            return float(x.mean(axis=0))

          metrics = jax.tree_map(avg_over_substeps, metrics)
          metrics['train/steps_per_sec'] = float(
              config.train.log_loss_every_steps / (time.time() - last_log_time))
          writer.write_scalars(new_step, metrics)
          last_log_time = time.time()

        # Eval
        should_eval = new_step % config.train.eval_every_steps == 0
        if prev_step == 0 or should_eval:
          # Samples

          samples_to_log = {
              'eval/samples':
                  self.get_model_samples(
                      params=state.ema_params, rng=next(rng))
          }

          if samples_to_log:
            assert all(v.shape == (total_bs, *self.dataset.data_shape)
                       for v in samples_to_log.values())
            # tf.summary.image asks for a batch, so insert a new axis
            writer.write_images(
                new_step, {
                    k: utils.np_tile_imgs(v.astype('uint8'))[None, :, :, :]
                    for k, v in samples_to_log.items()
                })

          # Eval metrics
          if config.train.get('calc_eval_metrics', True):
            eval_metrics = self._calc_eval_metrics(
                state=state,
                eval_iter=eval_iter,
                eval_steps=config.train.get('eval_number_steps',
                                            self.dataset.num_eval // total_bs),
                eval_base_rng=eval_base_rng,
                total_bs=total_bs)
            if eval_metrics is not None:
              writer.write_scalars(new_step, eval_metrics)

        # Checkpointing: only if checkpoint_every_secs is not None.
        if config.train.checkpoint_every_secs is not None:
          should_ckpt = (
              time.time() - last_ckpt_time >=
              config.train.checkpoint_every_secs)
          should_ckpt = (
              prev_step == 0 or new_step == config.train.num_train_steps or
              should_ckpt)
        else:
          should_ckpt = False

        if should_ckpt and jax.host_id() == 0:
          checkpoints.save_checkpoint(
              checkpoint_dir,
              flax.jax_utils.unreplicate(state),
              step=new_step,
              keep=3)
          last_ckpt_time = time.time()

        # Keep extra checkpoints without removal. Training does not resume
        # from these checkpoints.
        if (('retain_checkpoint_every_steps' in config.train) and
            ((new_step % config.train.retain_checkpoint_every_steps == 0) or
             (new_step == config.train.num_train_steps)) and
            (jax.host_id() == 0)):
          # Below, overwrite=True because training might resume from a
          # checkpoint from an earlier step than the latest retained checkpoint,
          # causing the latest retained checkpoint to be overwritten.
          checkpoints.save_checkpoint(
              os.path.join(work_unit_dir, 'retained_checkpoints'),
              flax.jax_utils.unreplicate(state),
              step=new_step,
              keep=int(1e10),
              overwrite=True)

        prev_step = new_step
        if new_step == config.train.num_train_steps:
          logging.info('Finished training for %d iterations.', new_step)
          break
