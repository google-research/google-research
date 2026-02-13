# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Main entry for evaluation and training."""

import functools
import os
import sys
import time
import traceback
from typing import Any
from unittest import mock

from absl import app
from absl import flags
from absl import logging
import chex
import flax
from flax.training import checkpoints
import ipdb
import jax
from jax import random
from jax import tree_util
from jax.experimental import pjit
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import tensorflow as tf

from global_metnet import config as config_lib
from global_metnet import dataset_preprocessor
from global_metnet import model as metnet_model
from global_metnet import utils


config = jax.config
_DEBUG = flags.DEFINE_bool('debug', False, 'Disables pmap and jit.')
_POST_MORTEM = flags.DEFINE_enum(
    'post_mortem', None, ['ipdb'], 'Debugger to use for post mortem.'
)
_EXPERIMENT_DIR = flags.DEFINE_string(
    'experiment_dir',
    '/tmp/log_dir',
    'Experiment directory for storing experiment data.',
)


class TrainState(flax.struct.PyTreeNode):
  step: int
  optimizer_state: optax.OptState
  apply_fn: Any
  params: Any


def get_initial_state(
    params, apply_fn, hps, log_dir
):
  """Initialize state, and restore from checkpoints if present."""
  lr_scheduler = utils.make_lr_scheduler(hps)
  optimizer, optimizer_state = make_optimizer(params, hps, lr_scheduler)
  state = TrainState(
      step=0,
      optimizer_state=optimizer_state,
      apply_fn=apply_fn,
      params=params,
  )
  if tree_util.tree_leaves(params):  # If the model has any parameters.
    state = checkpoints.restore_checkpoint(log_dir, state, prefix='checkpoint_')
  step = int(state.step)
  return state, step, optimizer


def get_params_and_apply_fn(hps, num_output_channels, input_specs, rng):
  """Make a model that can make predictions and can be trained."""
  module = metnet_model.GlobalMetNet(
      hps=hps, num_output_channels=num_output_channels
  )

  def wrapped_apply(*args, **kwargs):
    kwargs = {
        k: kwargs[k]
        for k in set(module.input_keys).union(['train', 'rngs']).intersection(
            kwargs.keys()
        )
    }
    output = module.apply(*args, **kwargs)
    return output

  apply_fn = wrapped_apply

  inputs = {k: jnp.zeros(s, d) for k, (s, d) in input_specs.items()}
  jit_init = jax.jit(module.init, static_argnames='train')
  variables = jit_init(rng, **inputs, train=False)
  params = variables['params']
  num_params = sum([x.size for x in tree_util.tree_leaves(params)])
  logging.info('Number of trainable parameters %d', num_params)
  return params, apply_fn


def make_optimizer(params, hps, learning_rate_fn):
  """Construct optimizer."""
  if hps.optimizer == 'adam':
    optimizer_def = optax.adam(learning_rate=learning_rate_fn, b1=0.9, b2=0.999)
  else:
    raise NotImplementedError()

  if hps.get('weight_decay', 0.0) > 0.0:
    optimizer = optax.chain(
        optimizer_def, optax.add_weight_decay(hps.get('weight_decay', 0.0))
    )
  else:
    optimizer = optimizer_def

  optimizer_state = optimizer.init(params)
  return optimizer, optimizer_state


def compute_loss(hps, output, target, mask):
  """Compute the total loss across all targets."""
  def _reshape(x):
    # head.compute_loss expects batched inputs but the batch dimension
    # is consumed by vmap so we add a dummy batch dimension here.
    return x[:, None]

  def compute_loss_fn(head, output, target, mask):
    if output is None:
      return 0
    output = jax.tree.map(_reshape, output)
    target = jax.tree.map(_reshape, target)
    mask = jax.tree.map(_reshape, mask)
    loss_result = jax.vmap(
        functools.partial(head.compute_loss),
        in_axes=(0, 0, 0),
        out_axes=0)(output, target, mask)
    return loss_result

  heads = dict(hps.heads)
  losses = jax.tree.map(compute_loss_fn, heads, output,
                        {k: target[k] for k in heads.keys()},
                        {k: mask[k] for k in heads.keys()})
  weighted_losses = [heads[k].loss_weight * v for k, v in losses.items()]
  weights = [head.loss_weight for head in heads.values()]
  average_weight = sum(weights) / len(weights)
  loss = sum(weighted_losses) / average_weight / len(weights)

  loss = jnp.mean(loss)
  losses = jax.tree.map(jnp.mean, losses)
  return loss, losses


def train_step(hps, state, batch, optimizer):
  """Perform a train step."""
  step = state.step
  device_rng = random.PRNGKey(0)
  rngs = {'dropout': random.fold_in(device_rng, step)}

  def loss_fn(params):
    output = state.apply_fn(
        {'params': params}, **batch['inputs'], train=True, rngs=rngs
    )
    loss, losses = compute_loss(hps, output, batch['target'], batch['mask'])
    loss_no_grad = jax.lax.stop_gradient(loss)
    loss = jnp.nan_to_num(loss, nan=loss_no_grad, posinf=loss_no_grad,
                          neginf=loss_no_grad)
    return loss, losses

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, losses), grad = grad_fn(state.params)

  max_grad_norm = hps.get('max_grad_norm')
  if max_grad_norm is not None:
    grad = utils.clip_grad_norm(grad, max_grad_norm)

  updates, new_optimizer_state = optimizer.update(
      grad, state.optimizer_state, state.params
  )
  new_params = optax.apply_updates(state.params, updates)
  state = state.replace(
      optimizer_state=new_optimizer_state, params=new_params, step=step + 1
  )
  return state, loss, losses


def get_logline(prefix, stats):
  return prefix + ':\t' + ', '.join(
      ['{0}={1:0.3f}'.format(k, v.item()) for k, v in sorted(stats.items())])


def log_stats(sw, prefix, stats, step, flush=False):
  for key, value in stats.items():
    sw.scalar('{}/{}'.format(prefix, key), value, step)
  if flush:
    sw.flush()


def pipeline_metrics(
    step,
    train_loss,
    logs,
    sw,
    train_stats_accumulator,
):
  """Record stats for train step, and occasionally flush and print."""
  train_steps_per_print = 100
  train_steps_per_flush_log = 100

  should_flush = step > 0 and step % train_steps_per_flush_log == 0
  should_print = step > 0 and step % train_steps_per_print == 0
  train_loss, logs = jax.device_get((train_loss, logs))
  new_train_stats = {'loss': train_loss}
  new_train_stats.update(logs)

  if sw:
    log_stats(sw, 'train', new_train_stats, step, should_flush)

  for key, value in new_train_stats.items():
    if key not in train_stats_accumulator:
      train_stats_accumulator[key] = []
    train_stats_accumulator[key].append(value)

  if should_print:
    mean_train_stats = {
        key: np.nanmean(values)
        for key, values in train_stats_accumulator.items()
    }
    logging.info(get_logline(str(step) + ', train', mean_train_stats))
    train_stats_accumulator.clear()

  # check that loss is not nan or inf
  if not np.isfinite(new_train_stats['loss']):
    logging.warn('Non-finite loss: %s', new_train_stats['loss'])


def train_loop(
    hps,
    log_dir,
    preprocessor,
    dataset,
    params,
    apply_fn,
):
  """Run a training loop.

  Args:
    hps: Hyperparameter dict -- see config/ for examples.
    log_dir: Directory in which to store TensorBoard logs and checkpoints.
    preprocessor: Data input pipeline.
    dataset: training dataset.
    params: Model params, as returned from Module().init()["params"].
    apply_fn: Model apply function.
  """
  state, init_step, optimizer = get_initial_state(
      params, apply_fn, hps, log_dir
  )
  sw = None
  if jax.process_index() == 0:
    utils.save_config(hps, log_dir)
    summary_writer_dir = os.path.join(log_dir, 'train')
    tf.io.gfile.makedirs(summary_writer_dir, mode=0o775)
    sw = utils.SummaryWriterIgnoreNans(summary_writer_dir)

  train_steps_per_print = 100
  train_stats_accumulator = {}  # Accumulate per training step, then print means
  train_iterator = preprocessor.get_iterator_from_dataset(dataset)
  max_updates = hps.train_steps
  step = init_step + 1
  while step <= max_updates:
    start_time = time.time()
    state, train_loss, train_loss_per_target = train_step(
        hps, state, next(train_iterator), optimizer
    )
    train_step_time = time.time() - start_time
    logs = {'train_step_time': train_step_time}
    if isinstance(train_loss_per_target, dict):
      logs.update({key + '_loss': val
                   for key, val in train_loss_per_target.items()})
    if jax.process_index() == 0:
      pipeline_metrics(
          step, train_loss, logs, sw, train_stats_accumulator
      )
      if step % train_steps_per_print == 0:
        logging.info('Train step: %d, loss %0.2f', step, train_loss)
      if step % hps.get('checkpoint_frequency', 500) == 0:
        checkpoints.save_checkpoint_multiprocess(
            log_dir, jax.device_get(state), step, keep=5
        )
    step += 1


def run():
  """Run a single mode (train/eval/visualize)."""
  hps = config_lib.get_config()

  experiment_dir = _EXPERIMENT_DIR.value
  rng = jax.random.PRNGKey(0)

  host_id = jax.process_index()
  partitions = hps.get('partitions', None)
  if partitions is None:
    with hps.unlocked():
      hps['partitions'] = jax.local_device_count()
  partitions = hps.partitions
  num_total_devices = jax.device_count() // partitions
  num_local_devices = jax.local_device_count() // partitions

  full_batch_size = hps.batch_size
  local_batch_size = full_batch_size // num_local_devices
  device_batch_size = full_batch_size // num_total_devices

  logging.info('Host id: %i', host_id)
  logging.info('Total num devices: %i', num_total_devices)
  logging.info('Local num devices: %i', num_local_devices)
  logging.info('Total batch size: %i', full_batch_size)
  logging.info('Local batch size: %i', local_batch_size)
  logging.info('Device batch size: %i', device_batch_size)

  preprocessor = dataset_preprocessor.DatasetPreprocessor(hps)
  dataset = preprocessor.get_dataset('train')
  input_specs = jax.tree.map(
      functools.partial(utils.get_shape_and_dtype, hps=hps),
      dataset.element_spec['inputs'])

  num_output_channels = sum([head.num_output_channels
                             for head in hps.heads.values()])

  params, apply_fn = get_params_and_apply_fn(
      ml_collections.FrozenConfigDict(hps),
      num_output_channels,
      input_specs,
      rng)

  train_loop(
      hps=hps,
      log_dir=experiment_dir,
      preprocessor=preprocessor,
      dataset=dataset,
      params=params,
      apply_fn=apply_fn,
  )


def main_with_post_mortem(_):
  try:
    with chex.fake_pmap_and_jit(_DEBUG.value, _DEBUG.value):
      if _DEBUG.value:  # chex doesn't disable pjit so we disable it manually.
        @functools.wraps(pjit.pjit)
        def fake_pjit(fn, *unused_args, **unused_kwargs):
          return fn
        mock.patch('jax.experimental.pjit.pjit', fake_pjit).__enter__()
      run()
  except:  # pylint: disable=bare-except
    # Check the tty so that we don't hang waiting for input in an
    # non-interactive scenario.
    if _POST_MORTEM.value == 'ipdb' and sys.stdout.isatty():
      traceback.print_exc()
      print()
      print(' *** Entering post-mortem debugging ***')
      print()
      ipdb.post_mortem()
    raise


if __name__ == '__main__':
  # JAX uses a different flags library -- parse from absl so that the task
  # knows its task_id for pod training.
  config.config_with_absl()
  app.run(main_with_post_mortem)
