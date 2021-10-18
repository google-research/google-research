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

"""Audio generation example.

This script trains AO AR models on audio datasets.
"""

import copy
import functools
import time

from absl import logging
from clu import metric_writers
from clu import periodic_actions
import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax

from autoregressive_diffusion.experiments.audio import input_pipeline_sc09
from autoregressive_diffusion.experiments.audio import utils as train_utils
from autoregressive_diffusion.experiments.audio.arch import diff_wave
from autoregressive_diffusion.experiments.audio.model import arm
from autoregressive_diffusion.experiments.images import checkpoint
from autoregressive_diffusion.experiments.language import language_train_state
from autoregressive_diffusion.model.autoregressive_diffusion import ao_arm
from autoregressive_diffusion.model.autoregressive_diffusion import bit_ao
from autoregressive_diffusion.utils import util_fns


def train_step(rng, batch, state, model, config, learning_rate_fn):
  """Train for a single step."""
  rng_return, rng = jax.random.split(rng)
  rng = jax.random.fold_in(rng, jax.lax.axis_index('batch'))

  def loss_fn(params):
    # Outputs are: (elbo_per_t, ce_value, t) or (acc, None, None).
    elbo_value, *extra = model.elbo(rng, params, batch['inputs'], train=True)
    elbo_value = jnp.mean(elbo_value, axis=0)
    loss = -elbo_value
    if config.model != 'arm':
      elbo_per_t, ce_value, t = extra
      if config.ce_term > 0:
        ce_value = extra[1]
        ce_value = jnp.mean(ce_value, axis=0)
        loss -= config.ce_term * ce_value
      outputs = {'nelbo': -elbo_value,
                 'nelbo_per_t_batch': elbo_per_t,
                 't_batch': t,
                 'ce': -ce_value}
    else:
      acc = extra[0]
      outputs = {'nelbo': -elbo_value, 'acc': acc}
    return loss, outputs

  lr = learning_rate_fn(state.step)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, aux), grads = grad_fn(state.params)
  grads = jax.lax.pmean(grads, axis_name='batch')
  if config.clip_grad > 0:
    grads, grad_norm = util_fns.clip_by_global_norm(
        grads, clip_norm=config.clip_grad)
  else:
    grad_norm = util_fns.global_norm(grads)

  state = state.apply_gradients(
      grads=grads,
      lr=lr,
      ema_momentum=config.ema_momentum)
  metrics = {
      'lr': lr,
      'grad_norm': grad_norm}
  for name, value in aux.items():
    if 'batch' in name:
      metrics[name] = jax.lax.all_gather(value, axis_name='batch')
    else:
      metrics[name] = jax.lax.pmean(value, axis_name='batch')
  metrics['loss'] = jax.lax.pmean(loss, axis_name='batch')
  return state, metrics, rng_return


def eval_step(rng, batch, state, model):
  """Eval a single step."""
  rng_return, rng = jax.random.split(rng)
  rng = jax.random.fold_in(rng, jax.lax.axis_index('batch'))
  elbo_value, *extra = model.elbo(
      rng, state.ema_params, batch['inputs'], train=False)

  outputs = {'nelbo': -elbo_value}

  if model.config.model != 'arm':
    _, ce_value, _ = extra
    outputs['ce'] = -ce_value
  else:
    acc, _, _ = extra
    outputs['acc'] = acc

  # Carefuly account for batch masking.
  batch_mask = batch.get(
      'mask',
      jnp.ones(batch['inputs'].shape[:1], dtype=jnp.bool_))

  outputs = {k: jnp.sum(util_fns.apply_weight(v, batch_mask))
             for k, v in outputs.items()}
  outputs, denom = jax.lax.psum(
      (outputs, jnp.sum(batch_mask)), axis_name='batch')
  return outputs, denom, rng_return


def eval_model(
    p_eval_step,
    rng,
    state,
    it,
    num_steps):
  """Eval for a number of steps."""
  start_time = time.time()
  batch_metrics, batch_denom = [], []

  for step in range(num_steps):
    with jax.profiler.StepTraceAnnotation('eval', step_num=step):
      metrics, denom, rng = p_eval_step(rng, next(it), state)

    # Better to leave metrics on device, and off-load after finishing epoch.
    batch_metrics.append(metrics)
    batch_denom.append(denom)

  # Load to CPU.
  batch_metrics, batch_denom = jax.device_get(
      flax.jax_utils.unreplicate((batch_metrics, batch_denom)))

  # Compute mean of metrics across each batch in epoch.
  denom_np = np.sum(batch_denom)
  metrics_np = {k: np.sum([metrics[k] for metrics in batch_metrics]) / denom_np
                for k in batch_metrics[0] if 'batch' not in k}
  logging.info('Eval took: %.3f seconds', time.time() - start_time)
  return metrics_np, rng


def log_standard_metrics(
    writer, step, *, train_metrics=None, eval_metrics=None):
  """Logs metrics using a metrics writer."""
  metrics_dict = {}
  if train_metrics:
    metrics_dict.update(
        {'train_' + k: v for k, v in train_metrics.items() if 'batch' not in k})
  if eval_metrics:
    metrics_dict.update(
        {'eval_' + k: v for k, v in eval_metrics.items() if 'batch' not in k})
  writer.write_scalars(step, metrics_dict)


def model_setup(init_rng, config):
  """Sets up the model and initializes params."""
  def get_architecture(
      num_input_classes, n_output_channels, num_steps, is_causal=False):
    cfg = copy.deepcopy(config.arch.config)
    cfg.max_time = num_steps
    cfg.num_classes = num_input_classes
    if config.arch.name == 'diff_wave':
      cfg.output_features = n_output_channels
      cfg.is_causal = is_causal
      net = diff_wave.DiffWave(**cfg)
    else:
      raise ValueError(f'Unknown architecture requested: {config.arch.name}.')
    return net

  if config.model == 'ao_arm':
    model = ao_arm.ArbitraryOrderARM.create(
        config, get_architecture, absorbing_state=config.num_classes // 2)
  elif config.model == 'bit_ao':
    model = bit_ao.BitUpscaleAutoregressiveDiffusion.create(
        config, get_architecture)
  elif config.model == 'arm':
    model = arm.ARM.create(config, get_architecture)
  else:
    raise ValueError(f'Unknown model {config.model}.')

  tmp_x, tmp_t = (jnp.ones([1, *config.data_shape], dtype=jnp.int32),
                  jnp.ones([1]))

  @functools.partial(jax.jit, backend='cpu')
  def init():
    return model.init_architecture(init_rng, tmp_x, tmp_t)

  logging.info('Initializing neural network.')
  variables = init()
  return model, variables


def train_and_evaluate(
    config,
    work_dir,
    try_checkpoint=True):
  """Execute model training and evaluation loop.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    work_dir: Directory where the tensorboard summaries are written to.
    try_checkpoint: Should try to load checkpoint (usually enabled, practical
        for debugging purposes to disable).

  Returns:
    The train state (which includes the `.params`).
  """
  # Init rng key.
  rng = jax.random.PRNGKey(config.seed)
  data_rng, rng = jax.random.split(rng)
  is_first_host = jax.process_index() == 0

  if config.dataset.name.endswith('speech_commands09'):
    ds, ds_metadata = input_pipeline_sc09.get_dataset(data_rng, config)
  else:
    raise ValueError(f'Unknown dataset {config.dataset.name}.')

  # Immediately create infinite iterators.
  it = jax.tree_map(util_fns.get_iterator, ds)

  # TODO(agritsenko): Can we fix the ugly nested dicts?
  config.data_shape = ds_metadata['train']['shape']['inputs'][2:]
  config.num_classes = ds_metadata['train']['num_classes']
  config.sample_rate = ds_metadata['train']['sample_rate']

  writer = metric_writers.create_default_writer(
      work_dir, just_logging=jax.process_index() > 0)
  rng, init_rng = jax.random.split(rng)

  model, variables = model_setup(init_rng, config)

  # From now on we want different rng across hosts:
  rng = jax.random.fold_in(rng, jax.process_index())
  def tx_fn(lr):
    return optax.adamw(
        lr, b1=0.9, b2=config.beta2, eps=1e-08, eps_root=0.0,
        weight_decay=config.weight_decay)
  state = language_train_state.TrainState.create(
      params=variables['params'], tx_fn=tx_fn)

  start_step = None
  if try_checkpoint:
    state, start_step = checkpoint.restore_from_path(work_dir, state)
  start_step = start_step or 0

  # Use different rngs for train & eval.
  rng_train, rng_eval, rng_sample = jax.random.split(rng, 3)

  kl_tracker = util_fns.KLTracker(num_steps=model.num_steps)
  kl_history = []

  learning_rate_fn = train_utils.create_learning_rate_scheduler(
      **config.learning_rate)
  p_train_step = jax.pmap(
      functools.partial(
          train_step,
          config=config,
          learning_rate_fn=learning_rate_fn,
          model=model),
      axis_name='batch',
      in_axes=(None, 0, 0),
      out_axes=(0, 0, None),
      donate_argnums=(2,))

  # The only axes that are broadcasted are the in- and output rng key ones. The
  # rng is the first arg, and the last return value.
  p_eval_step = jax.pmap(
      functools.partial(
          eval_step,
          model=model),
      axis_name='batch',
      in_axes=(None, 0, 0),
      out_axes=(0, 0, None))

  # Training length.
  logging.info('Training will start from step %d', start_step)

  # Replicate state.
  state = flax.jax_utils.replicate(state)

  # Setup hooks.
  hooks = []
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=config.num_train_steps, writer=writer)
  if is_first_host:
    hooks += [
        report_progress,
        periodic_actions.Profile(logdir=work_dir, num_profile_steps=5)
    ]

  with metric_writers.ensure_flushes(writer):
    batch_metrics = []
    for step in range(start_step, config.num_train_steps):
      logging.log_first_n(logging.INFO, f'Train step: {step}', 5)
      with jax.profiler.StepTraceAnnotation('train', step_num=step):
        state, metrics, rng_train = p_train_step(
            rng_train,
            next(it['train']),
            state)
      batch_metrics.append(metrics)

      # Cycle though hooks.
      for h in hooks:
        h(step)

      is_last_step = step == config.num_train_steps - 1

      if (step % config.log_every_steps == 0) or is_last_step:
        with report_progress.timed('training_metrics'):
          ################### Process batch metrics ############################
          batch_metrics = jax.device_get(
              flax.jax_utils.unreplicate(batch_metrics))

          if 't_batch' in metrics:
            # TODO(agritsenko): Factor out into a separate function.
            # This processes the loss per t, although two nested for-loops
            # (counting the one inside kl_tracker), it actually does not hurt
            # timing performance meaningfully.
            batch_t = [
                metrics['t_batch'].reshape(-1) for metrics in batch_metrics]
            batch_nelbo_per_t = [
                metrics['nelbo_per_t_batch'].reshape(-1)
                for metrics in batch_metrics]
            for t, nelbo_per_t in zip(batch_t, batch_nelbo_per_t):
              kl_tracker.update(t, nelbo_per_t)

          ################### Process batch metrics ############################
          metrics = {key: np.mean([metrics[key] for metrics in batch_metrics])
                     for key in batch_metrics[0] if 'batch' not in key}

          # Metric logging.
          if is_first_host:
            log_standard_metrics(writer, step, train_metrics=metrics)
          batch_metrics = []

      if config.eval_every_steps and (
          (step % config.eval_every_steps == 0) or is_last_step):
        with report_progress.timed('eval'):
          ####################### Run evaluation ###############################
          metrics, rng_eval = eval_model(
              p_eval_step,
              rng_eval,
              state,
              it['eval'],
              (ds_metadata['eval']['num_batches'] *
               config.get('num_eval_passes', 1)))

          # Metric logging.
          if is_first_host:
            log_standard_metrics(writer, step, eval_metrics=metrics)

        # Track KL (unrelated to the eval, but nice to not do every step).
        kl_values = kl_tracker.get_kl_per_t()
        kl_history.append(np.array(kl_values))
        kl_history = kl_history[-50:]

      if config.sample_every_steps and (
          (step % config.sample_every_steps == 0) or is_last_step):
        with report_progress.timed('sample'):
          ######################### Run sampling ###############################
          chain = model.sample(
              jax.random.fold_in(rng_sample, step),
              state.ema_params,
              config.sample_batch_size,
              chain_out_size=config.get('chain_out_size', model.num_stages))

          if is_first_host:
            chain = jax.device_get(chain)
            long_sample = np.reshape(chain[-1], (1, -1, 1)).astype(np.float32)
            long_sample = (2. * long_sample) / config.num_classes - 1.
            writer.write_audios(
                step, {'samples': long_sample}, sample_rate=config.sample_rate)

      ######################### Checkpointing #################################
      if is_first_host and config.checkpoint_every_steps and (
          (step % config.checkpoint_every_steps == 0) or is_last_step):
        logging.info('Saving checkpoint: step %d', step)
        with report_progress.timed('checkpoint'):
          checkpoint.save_checkpoint(
              work_dir, state=flax.jax_utils.unreplicate(state), step=step)
        logging.info('Finished saving checkpoint: step %d', step)

    return state


def monitor_and_sample(config, work_dir):
  """Monitors `work_dir` for new checkpoints and run sampling on them.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    work_dir: Directory where the tensorboard summaries are written to.
  """
  # Init rng key.
  rng = jax.random.PRNGKey(config.seed)
  data_rng, rng = jax.random.split(rng)
  is_first_host = jax.process_index() == 0

  # TODO(agritsenko): We are loading the datasets just to get the metadata.
  #  Can we be smarter about this?
  if config.dataset.name.endswith('speech_commands09'):
    _, ds_metadata = input_pipeline_sc09.get_dataset(data_rng, config)
  else:
    raise ValueError(f'Unknown dataset {config.dataset.name}.')

  # TODO(agritsenko): Can we fix the ugly nested dicts?
  config.data_shape = ds_metadata['train']['shape']['inputs'][2:]
  config.num_classes = ds_metadata['train']['num_classes']
  config.sample_rate = ds_metadata['train']['sample_rate']

  writer = metric_writers.create_default_writer(
      work_dir, just_logging=jax.process_index() > 0)
  rng, init_rng = jax.random.split(rng)

  model, variables = model_setup(init_rng, config)

  # From now on we want different rng across hosts:
  rng = jax.random.fold_in(rng, jax.process_index())
  rng, rng_sample = jax.random.split(rng)
  def tx_fn(lr):
    return optax.adamw(
        lr, b1=0.9, b2=config.beta2, eps=1e-08, eps_root=0.0,
        weight_decay=config.weight_decay)
  state = language_train_state.TrainState.create(
      params=variables['params'], tx_fn=tx_fn)

  # Wait for checkpoints in an loop.
  ckpt_path_iterator = checkpoint.checkpoints_iterator(work_dir, target=None)

  with metric_writers.ensure_flushes(writer):
    for _ in ckpt_path_iterator:
      state, step = checkpoint.restore_from_path(work_dir, state)
      is_last_step = step == config.num_train_steps - 1
      logging.info('Loaded checkpoint for step: %d', step)

      # Replicate the state
      state = flax.jax_utils.replicate(state)

      ######################### Run sampling ###############################
      chain = model.sample(
          jax.random.fold_in(rng_sample, step),
          state.ema_params,
          config.sample_batch_size,
          chain_out_size=config.get('chain_out_size', model.num_stages))

      if is_first_host:
        chain = jax.device_get(chain)
        long_sample = np.reshape(chain[-1], (1, -1, 1)).astype(np.float32)
        long_sample = (2. * long_sample) / config.num_classes - 1.
        long_sample = long_sample.astype(np.float32)
        writer.write_audios(
            step, {'samples': long_sample}, sample_rate=config.sample_rate)

      if is_last_step:
        break
