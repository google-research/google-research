# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Train file.

Library file which executes the training and evaluation loop.
"""

# pytype: disable=wrong-keyword-args

import functools
import os
import pickle
import time


from absl import logging
from clu import metric_writers
import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import tensorflow as tf

from autoregressive_diffusion.experiments.images import checkpoint
from autoregressive_diffusion.experiments.images import custom_train_state
from autoregressive_diffusion.experiments.images import datasets
from autoregressive_diffusion.experiments.images.architectures import unet
from autoregressive_diffusion.model.autoregressive_diffusion import ao_arm
from autoregressive_diffusion.model.autoregressive_diffusion import bit_ao
from autoregressive_diffusion.utils import util_fns


def train_step(rng, batch, state, model, config):
  """Train for a single step."""
  logging.info('Training step...')
  rng_return, rng = jax.random.split(rng)
  rng = jax.random.fold_in(rng, jax.lax.axis_index('batch'))

  def loss_fn(params):
    elbo_value, elbo_per_t, ce_value, t = model.elbo(
        rng, params, batch['image'], train=True)
    loss = -elbo_value.mean(0) - config.ce_term * ce_value.mean(0)
    return loss, (elbo_value, elbo_per_t, ce_value, t)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, (elbo_value, elbo_per_t, ce_value, t)), grads = grad_fn(state.params)
  grads = jax.lax.pmean(grads, axis_name='batch')
  if config.clip_grad > 0:
    grads, grad_norm = util_fns.clip_by_global_norm(
        grads, clip_norm=config.clip_grad)
  else:
    grad_norm = util_fns.global_norm(grads)

  state = state.apply_gradients(grads=grads)
  metrics = {
      'loss': jax.lax.pmean(loss, axis_name='batch'),
      'nelbo': jax.lax.pmean(-elbo_value, axis_name='batch'),
      'ce': jax.lax.pmean(-ce_value, axis_name='batch'),
      # batch statistics useful for dp and iw sampling:
      'nelbo_per_t_batch': jax.lax.all_gather(-elbo_per_t, axis_name='batch'),
      't_batch': jax.lax.all_gather(t, axis_name='batch'),
      'grad_norm': grad_norm
  }
  return state, metrics, rng_return


def eval_step(rng, batch, state, model):
  """Eval a single step."""
  logging.info('Eval step...')
  rng_return, rng = jax.random.split(rng)
  rng = jax.random.fold_in(rng, jax.lax.axis_index('batch'))
  elbo_value, _, ce_value, _ = model.elbo(
      rng, state.ema_params, batch['image'], train=False)
  metrics = {
      'nelbo': jax.lax.pmean(-elbo_value, axis_name='batch'),
      'ce': jax.lax.pmean(-ce_value, axis_name='batch')
  }
  return metrics, rng_return


def train_epoch(p_train_step, state, train_ds, batch_size, epoch, rng,
                kl_tracker):
  """Train for a single epoch."""
  start_time = time.time()

  batch_metrics = []

  train_ds = util_fns.get_iterator(train_ds)
  with jax.profiler.StepTraceAnnotation('train', step_num=state.step):
    for batch in train_ds:
      state, metrics, rng = p_train_step(rng, batch, state)

      # Better to leave metrics on device, and off-load after finishing epoch.
      batch_metrics.append(metrics)

  # Load to CPU.
  batch_metrics = jax.device_get(flax.jax_utils.unreplicate(batch_metrics))

  # This processes the loss per t, although two nested for-loops (counting the
  # one inside kl_tracker), it actually does not hurt timing performance
  # meaningfully.
  t_batches = [
      metrics['t_batch'].reshape(batch_size) for metrics in batch_metrics]
  nelbo_per_t_batches = [
      metrics['nelbo_per_t_batch'].reshape(batch_size)
      for metrics in batch_metrics]
  for t_batch, nelbo_per_t_batch in zip(t_batches, nelbo_per_t_batches):
    kl_tracker.update(t_batch, nelbo_per_t_batch)

  # Compute mean of metrics across each batch in epoch.
  epoch_metrics = {
      key: np.mean([metrics[key] for metrics in batch_metrics])
      for key in batch_metrics[0] if 'batch' not in key}

  message = f'Epoch took {time.time() - start_time:.1f} seconds.'
  logging.info(message)
  info_string = (
      f'train epoch: {epoch}, loss: {epoch_metrics["loss"]:.4f} '
      f'nelbo: {epoch_metrics["nelbo"]:.4f} ce: {epoch_metrics["ce"]:.4f}'
      )
  logging.info(info_string)

  return state, epoch_metrics, rng


def eval_model(p_eval_step, rng, state, test_ds, epoch):
  """Eval for a single epoch."""
  start_time = time.time()
  batch_metrics = []

  test_ds = util_fns.get_iterator(test_ds)

  for batch in test_ds:
    metrics, rng = p_eval_step(rng, batch, state)

    # Better to leave metrics on device, and off-load after finishing epoch.
    batch_metrics.append(metrics)

  # Load to CPU.
  batch_metrics = jax.device_get(flax.jax_utils.unreplicate(batch_metrics))

  # Compute mean of metrics across each batch in epoch.
  epoch_metrics_np = {
      k: np.mean([metrics[k] for metrics in batch_metrics])
      for k in batch_metrics[0] if 'batch' not in k}

  nelbo = epoch_metrics_np['nelbo']
  message = f'Eval epoch took {time.time() - start_time:.1f} seconds.'
  logging.info(message)
  info_string = f'eval epoch: {epoch}, nelbo: {nelbo:.4f}'
  logging.info(info_string)

  return epoch_metrics_np, rng


# The axes that are broadcasted are the in- and output rng key ones, and the
# model, and the policy. The rng is the first arg, and the last return value.
@functools.partial(
    jax.pmap,
    static_broadcasted_argnums=(3,),
    in_axes=(None, 0, 0, None, None),
    out_axes=(0, None),
    axis_name='batch')
def eval_step_policy(rng, batch, state, model, policy):
  """Eval a single step."""
  rng_return, rng = jax.random.split(rng)
  rng = jax.random.fold_in(rng, jax.lax.axis_index('batch'))
  elbo_value, _, ce_value, _ = model.elbo_with_policy(
      rng, state.ema_params, batch['image'], policy=policy, train=False)
  metrics = {
      'nelbo': jax.lax.pmean(-elbo_value, axis_name='batch'),
      'ce': jax.lax.pmean(-ce_value, axis_name='batch')
  }
  return metrics, rng_return


def eval_policy(policy, rng, state, model, test_ds, epoch):
  """Eval for a single epoch."""
  batch_metrics = []

  policy = flax.jax_utils.unreplicate(flax.jax_utils.replicate(policy))

  # Function is recompiled for this specific policy.
  test_ds = util_fns.get_iterator(test_ds)
  for batch in test_ds:
    metrics, rng = eval_step_policy(rng, batch, state, model, policy)

    # Better to leave metrics on device, and off-load after finishing epoch.
    batch_metrics.append(metrics)

  # Load to CPU.
  batch_metrics = jax.device_get(flax.jax_utils.unreplicate(batch_metrics))
  # Compute mean of metrics across each batch in epoch.
  epoch_metrics_np = {
      k: np.mean([metrics[k] for metrics in batch_metrics])
      for k in batch_metrics[0] if 'batch' not in k}

  nelbo = epoch_metrics_np['nelbo']
  info_string = f'eval policy epoch: {epoch}, nelbo: {nelbo:.4f}'
  logging.info(info_string)

  return epoch_metrics_np


def log_standard_metrics(writer, train_metrics, eval_metrics, epoch):
  metric_dict = {
      'train_loss': train_metrics['loss'],
      'train_nelbo': train_metrics['nelbo'],
      'train_ce': train_metrics['ce'],
      'grad_norm': train_metrics['grad_norm'],
      'eval_nelbo': eval_metrics['nelbo'],
  }
  writer.write_scalars(epoch, metric_dict)


def extensive_eval(config, test_rng, writer,
                   output_path, model, state, kl_history, test_ds, epoch):
  """This function combines all extra eval benchmarks we want to run."""
  # Eval settings.
  is_first_host = jax.process_index() == 0
  max_num_steps = 25000
  max_num_steps_for_policy = 25000
  num_samples = config.num_samples
  n_rows = int(np.sqrt(num_samples))

  return_rng, rng1, rng2, rng3, rng4, rng5 = jax.random.split(test_rng, 6)

  # Plot loss components over time.
  if jax.process_index() == 0:
    fname = f'loss_t_{epoch}.png'
    filename = os.path.join(output_path, 'loss_plots', fname)
    util_fns.plot_loss_components(kl_history, filename, model.num_stages)

  # Sample from the model.
  if model.num_steps < max_num_steps:
    start = time.time()
    chain = model.sample(rng1, state.ema_params, num_samples)
    msg = f'Sampling took {time.time() - start:.2f} seconds'
    logging.info(msg)

    if is_first_host:
      filename = os.path.join(output_path, 'samples', f'chain_epoch{epoch}.gif')
      util_fns.save_chain_to_gif(chain, filename, n_rows)
      util_fns.plot_batch_images(chain[-1], n_rows, config.num_classes)
      writer.write_images(epoch, {'samples': chain[-1]})

    del chain

  # Validate and sample using naive policy.
  if model.policy_support:
    nelbo_policy_naive = eval_policy(model.get_naive_policy(), rng2,
                                     state, model, test_ds,
                                     epoch)['nelbo']
    naive_dict = {'eval_nelbo_policy_naive': nelbo_policy_naive}
    chain_naive = model.sample_with_naive_policy(
        rng3, state.ema_params, num_samples)
    if is_first_host:
      writer.write_scalars(epoch, naive_dict)
      filename = os.path.join(output_path, 'samples_naive',
                              f'chain_epoch_naive_{epoch}.gif')
      util_fns.save_chain_to_gif(
          chain_naive, filename, n_rows)
      util_fns.plot_batch_images(chain_naive[-1], n_rows, config.num_classes)

    del chain_naive

  # Val optimal policies.
  if model.policy_support and model.num_steps < max_num_steps_for_policy:
    # Check 25, 50 & 100 steps, just because they are interesting to see.
    budgets = [50, 100]

    # Compute policies and costs.
    start = time.time()
    policies, costs = model.compute_policies_and_costs(kl_history[-1], budgets)
    msg = f'Computing policy mats took {time.time() - start:.2f} secs'
    logging.info(msg)

    # Evaluate policy for budget 50.
    nelbo_policy_50 = eval_policy(policies[0], rng4, state, model,
                                  test_ds, epoch)['nelbo']
    metric_dict = {'eval_nelbo_policy_50': nelbo_policy_50}
    budget_results_train = {
        f'train_nelbo_steps_{b}': c
        for b, c in zip(budgets, costs)
    }
    metric_dict.update(budget_results_train)
    if jax.process_index() == 0:
      writer.write_scalars(epoch, metric_dict)

    # Sample with lowest policy.
    chain_policy = model.sample_with_policy(
        rng5, state.ema_params, num_samples, policies[0])

    if jax.process_index() == 0:
      filename = os.path.join(output_path, 'samples_policy',
                              f'chain_epoch_policy_{epoch}.gif')
      util_fns.save_chain_to_gif(
          chain_policy, filename, n_rows)
      util_fns.plot_batch_images(chain_policy[-1], n_rows, config.num_classes)

    del chain_policy, policies, costs
  return return_rng


def model_setup(init_rng, config):
  """Sets up the model and initializes params."""
  def get_architecture(num_input_classes, n_output_channels, num_steps):
    net = unet.UNet(
        num_classes=num_input_classes,
        ch=config.architecture.n_channels,
        out_ch=n_output_channels,
        ch_mult=config.architecture.ch_mult,
        num_res_blocks=config.architecture.num_res_blocks,
        full_attn_resolutions=config.architecture.attn_resolutions,
        num_heads=config.architecture.num_heads,
        dropout=config.architecture.dropout,
        max_time=float(num_steps))
    return net

  if config.model == 'ao_arm':
    model = ao_arm.ArbitraryOrderARM.create(
        config, get_architecture, absorbing_state=config.num_classes // 2)
  elif config.model == 'bit_ao':
    model = bit_ao.BitUpscaleAutoregressiveDiffusion.create(
        config, get_architecture)
  else:
    raise ValueError

  tmp_x, tmp_t = (jnp.ones([1, *config.data_shape], dtype=jnp.int32),
                  jnp.ones([1]))

  @functools.partial(jax.jit, backend='cpu')
  def init():
    return model.init_architecture(init_rng, tmp_x, tmp_t)

  logging.info('Initializing neural network')
  variables = init()
  return model, variables


def train_and_evaluate(config,
                       work_dir, try_checkpoint=True):
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
  msg = f'Running with seed {config.seed}.'
  logging.info(msg)
  rng = jax.random.PRNGKey(config.seed)
  data_rng, rng = jax.random.split(rng)
  is_first_host = jax.process_index() == 0

  train_ds, test_ds, shape, num_classes = datasets.get_dataset(config, data_rng)

  # config.mask_shape = mask_shape
  config.data_shape = shape
  config.num_classes = num_classes

  writer = metric_writers.create_default_writer(
      work_dir, just_logging=jax.process_index() > 0)
  rng, init_rng = jax.random.split(rng)

  # Create output directory for saving samples.
  output_path = work_dir
  tf.io.gfile.makedirs(output_path)

  model, variables = model_setup(init_rng, config)

  # From now on we want different rng across hosts:
  rng = jax.random.fold_in(rng, jax.process_index())

  tx = optax.adam(
      config.learning_rate, b1=0.9, b2=config.beta2, eps=1e-08, eps_root=0.0)
  state = custom_train_state.TrainState.create(
      params=variables['params'], tx=tx)

  if try_checkpoint:
    state, start_epoch = checkpoint.restore_from_path(work_dir, state)
    if start_epoch is None:
      start_epoch = 1
  else:
    # For debugging we start at zero, so we immediately do detailed eval.
    start_epoch = 0

  if is_first_host and start_epoch == 1:
    config_dict = dict(config)
    writer.write_hparams(config_dict)

  if is_first_host and start_epoch in (0, 1):
    # Dump config file to work dir for easy model loading.
    config_path = os.path.join(work_dir, 'config')
    with tf.io.gfile.GFile(config_path, 'wb') as fp:
      pickle.dump(config, fp)

  test_rng, train_rng = jax.random.split(rng)

  kl_tracker_train = util_fns.KLTracker(num_steps=model.num_steps)
  kl_history = []

  p_train_step = jax.pmap(
      functools.partial(train_step, model=model, config=config),
      axis_name='batch',
      in_axes=(None, 0, 0),
      out_axes=(0, 0, None),
      donate_argnums=(2,))

  # The only axes that are broadcasted are the in- and output rng key ones. The
  # rng is the first arg, and the last return value.
  p_eval_step = jax.pmap(
      functools.partial(eval_step, model=model),
      axis_name='batch',
      in_axes=(None, 0, 0),
      out_axes=(0, None))

  # Replicate state.
  state = flax.jax_utils.replicate(state)

  with metric_writers.ensure_flushes(writer):
    for epoch in range(start_epoch, config.num_epochs + 1):
      # Train part.
      state, train_metrics, train_rng = train_epoch(p_train_step, state,
                                                    train_ds, config.batch_size,
                                                    epoch, train_rng,
                                                    kl_tracker_train)

      # Val part.
      eval_metrics, test_rng = eval_model(p_eval_step, test_rng, state,
                                          test_ds, epoch)

      # Metric logging.
      if is_first_host:
        log_standard_metrics(writer, train_metrics, eval_metrics, epoch)

      kl_values = kl_tracker_train.get_kl_per_t()
      kl_history.append(np.array(kl_values))

      # Prune to avoid too much memory consumption.
      kl_history = kl_history[-50:]

      if epoch == 15 or epoch % config.detailed_eval_every == 0:
        if is_first_host:
          loss_components_path = os.path.join(work_dir, 'loss_components')
          with tf.io.gfile.GFile(loss_components_path, 'wb') as fp:
            pickle.dump(kl_history[-1], fp)

        test_rng = extensive_eval(config, test_rng, writer, output_path, model,
                                  state, kl_history, test_ds, epoch)

      # Save to checkpoint.
      if is_first_host and epoch % config.save_every == 0:
        # Save to epoch + 1 since current epoch has just been completed.
        logging.info('saving checkpoint')
        checkpoint.save_checkpoint(
            work_dir, state=flax.jax_utils.unreplicate(state), step=epoch + 1,
            keep=2)
        logging.info('finished saving checkpoint')

    return state
