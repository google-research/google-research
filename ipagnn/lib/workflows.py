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

# Lint as: python3
"""JAX workflows used by the Learned Interpreters framework.

Contains runner functions for training, evaluation, inference, and analysis of
Learned Interpreter models.
"""

import itertools
import os
import time

from absl import logging
import flax
from flax import jax_utils
from flax.metrics import tensorboard
from flax.training import common_utils
import jax
import jax.numpy as jnp
import tensorflow as tf

from ipagnn.lib import checkpoint_utils


def run(run_configuration):
  """Runs the Learned Interpreter code with the specified configuration."""
  mode = run_configuration.mode
  method = run_configuration.method
  original_checkpoint_path = run_configuration.original_checkpoint_path

  if mode == 'train':
    run_train(run_configuration)
  elif mode == 'train-single':
    run_train_single_device(run_configuration)
  elif mode == 'eval':
    run_eval(run_configuration)
  elif mode == 'eval_all':
    eval_all(run_configuration)
  elif mode == 'eval_once':
    eval_once(run_configuration, original_checkpoint_path)
  elif mode in ('interact', 'predict'):
    predict_once(run_configuration)
  else:
    raise ValueError('Unexpected mode', mode)


def run_train(run_configuration):
  """Runs the training workflow."""
  config = run_configuration.config
  run_dir = run_configuration.run_dir
  adapter = run_configuration.adapter
  log_dir = os.path.join(run_dir, 'train')
  checkpoint_path = run_configuration.original_checkpoint_path

  dataset = run_configuration.dataset_info.dataset
  info = run_configuration.dataset_info.info

  random_seed = 0
  rng = jax.random.PRNGKey(random_seed)
  rng = jax.random.fold_in(rng, jax.host_id())
  rng, init_rng = jax.random.split(rng)
  dropout_rngs = jax.random.split(rng, jax.local_device_count())

  # Set up optimizer.
  optimizer = adapter.create_optimizer(run_configuration, rng=init_rng)

  # Set up train step.
  train_step = adapter.make_train_step()

  if jax.host_id() == 0:
    summary_writer = tensorboard.SummaryWriter(log_dir)

  # Set up checkpointing.
  # TODO(dbieber): Set up phoenix.
  checkpoint_dir = checkpoint_utils.build_checkpoint_dir(run_dir)
  if checkpoint_path is None:
    checkpoint_path = checkpoint_utils.latest_checkpoint(checkpoint_dir)
  optimizer = checkpoint_utils.handle_restart_behavior(
      checkpoint_path, optimizer, config)

  start_step = int(optimizer.state.step)
  num_train_steps = config.train.total_steps

  # Replicate optimizer.
  optimizer = flax.jax_utils.replicate(optimizer)

  # Begin training loop.
  dataset_iter_raw = iter(dataset)
  dataset_iter = adapter.preprocess(dataset_iter_raw)

  summary_freq = config.logging.summary_freq
  metrics_all = []
  tick = time.time()
  for step, example in zip(range(start_step, num_train_steps), dataset_iter):
    train_inputs = adapter.get_train_inputs(example)
    optimizer, metrics, dropout_rngs, logits, state = train_step(
        optimizer, train_inputs, dropout_rngs)
    metrics_all.append(metrics)

    # Save a Checkpoint
    if ((step % config.logging.save_freq == 0 and step > 0)
        or step == num_train_steps - 1):
      if jax.host_id() == 0 and config.logging.save_freq:
        # Save unreplicated optimizer + model state.
        checkpoint_utils.save_checkpoint(
            checkpoint_dir, jax_utils.unreplicate(optimizer), step)

    # Periodic metric handling.
    if summary_freq and step % summary_freq == 0 and step > 0:
      metrics_all = common_utils.get_metrics(metrics_all)
      lr = metrics_all.pop('learning_rate').mean()
      metrics_sums = jax.tree_map(jnp.sum, metrics_all)
      denominator = metrics_sums.pop('denominator')
      summary = jax.tree_map(lambda x: x / denominator, metrics_sums)  # pylint: disable=cell-var-from-loop
      summary['learning_rate'] = lr
      # Calculate (clipped) perplexity after averaging log-perplexities:
      summary['perplexity'] = jnp.clip(jnp.exp(summary['loss']), a_max=1.0e4)
      logging.info('train step: %d, loss: %.4f', step, summary['loss'])
      if jax.host_id() == 0:
        tock = time.time()
        steps_per_sec = summary_freq / (tock - tick)
        examples_per_sec = denominator / (tock - tick)
        tick = tock
        summary_writer.scalar('per-second/steps', steps_per_sec, step)
        summary_writer.scalar('per-second/examples', examples_per_sec, step)
        for key, val in summary.items():
          summary_writer.scalar(key, val, step)

        adapter.write_summaries(
            example, logits, summary_writer, info, step, state)

        summary_writer.flush()
      # Reset metric accumulation for next evaluation cycle.
      metrics_all = []


def run_train_single_device(run_configuration):
  """Runs the training workflow without pmap or jit."""
  config = run_configuration.config
  run_dir = run_configuration.run_dir
  adapter = run_configuration.adapter
  checkpoint_path = run_configuration.original_checkpoint_path
  dataset = run_configuration.dataset_info.dataset

  random_seed = 0
  rng = jax.random.PRNGKey(random_seed)
  rng = jax.random.fold_in(rng, jax.host_id())
  dropout_rng, init_rng = jax.random.split(rng)

  # Set up optimizer.
  optimizer = adapter.create_optimizer(run_configuration, rng=init_rng)

  # Set up train step.
  train_step = adapter.make_train_step(single_device=True)

  # Set up checkpointing.
  # TODO(dbieber): Set up phoenix.
  checkpoint_dir = checkpoint_utils.build_checkpoint_dir(run_dir)
  if checkpoint_path is None:
    checkpoint_path = checkpoint_utils.latest_checkpoint(checkpoint_dir)
  optimizer = checkpoint_utils.handle_restart_behavior(
      checkpoint_path, optimizer, config)

  start_step = int(optimizer.state.step)
  num_train_steps = config.train.total_steps

  # Begin training loop.
  dataset_iter_raw = iter(dataset)
  dataset_iter = adapter.preprocess(dataset_iter_raw, single_device=True)

  for step, example in zip(range(start_step, num_train_steps), dataset_iter):
    print(f'Step #{step}')
    train_inputs = adapter.get_train_inputs(example)
    optimizer, metrics, dropout_rng, logits, state = train_step(
        optimizer, train_inputs, dropout_rng)
    del metrics, logits, state  # Unused.

    # Save a Checkpoint.
    if ((step % config.logging.save_freq == 0 and step > 0) or
        step == num_train_steps - 1):
      if jax.host_id() == 0 and config.logging.save_freq:
        # Save unreplicated optimizer + model state.
        checkpoint_utils.save_checkpoint(checkpoint_dir, optimizer, step)


def run_eval(run_configuration):
  """Evaluates on checkpoints as they become available."""
  config = run_configuration.config
  run_dir = run_configuration.run_dir
  adapter = run_configuration.adapter
  optimizer = adapter.create_optimizer(run_configuration)

  last_checkpoint_path = None
  last_checkpoint_time = time.time()
  checkpoint_dir = checkpoint_utils.build_checkpoint_dir(run_dir)
  success_path = checkpoint_utils.build_success_path(run_dir)
  error_count = 0
  while True:
    success = tf.io.gfile.exists(success_path)
    checkpoint_path = checkpoint_utils.latest_checkpoint(checkpoint_dir)
    if checkpoint_path is not None and checkpoint_path != last_checkpoint_path:
      logging.info('Evaluating with checkpoint_path: %s', checkpoint_path)
      try:
        eval_once(run_configuration, checkpoint_path, optimizer)
      except:  # pylint: disable=bare-except
        logging.info('Could not evaluate %s', checkpoint_path)
        error_count += 1
        if error_count >= 10 or config.debug:
          raise
      last_checkpoint_path = checkpoint_path
      last_checkpoint_time = time.time()
    else:
      if success:
        logging.info('SUCCESS file found. Stopping.')
        break
      if time.time() - last_checkpoint_time > config.eval_timeout:
        logging.info('Timed out waiting for checkpoint. Stopping.')
        break
      logging.info('Waiting for checkpoint.')
      time.sleep(15)


def eval_all(run_configuration):
  """Evaluates on all available checkpoints."""
  run_dir = run_configuration.run_dir
  adapter = run_configuration.adapter
  optimizer = adapter.create_optimizer(run_configuration)

  checkpoint_dir = checkpoint_utils.build_checkpoint_dir(run_dir)
  checkpoint_paths = checkpoint_utils.get_all_checkpoint_paths(checkpoint_dir)
  error_count = 0
  logging.info('Found %d checkpoints to evaluate.', len(checkpoint_paths))
  for checkpoint_path in checkpoint_paths:
    logging.info('Evaluating with checkpoint_path: %s', checkpoint_path)
    if checkpoint_path is not None:
      try:
        eval_once(run_configuration, checkpoint_path, optimizer)
      except:  # pylint: disable=bare-except
        logging.info('Could not evaluate %s', checkpoint_path)
        error_count += 1
        if error_count >= 10 or run_configuration.config.debug:
          raise


def eval_once(run_configuration, checkpoint_path, optimizer=None):
  """Evaluates a single checkpoint on a single epoch of data."""
  config = run_configuration.config
  run_dir = run_configuration.run_dir
  adapter = run_configuration.adapter
  optimizer = optimizer or adapter.create_optimizer(run_configuration)
  dataset = run_configuration.dataset_info.dataset
  info = run_configuration.dataset_info.info

  eval_name = config.eval_name or 'eval'
  log_dir = os.path.join(run_dir, eval_name)

  if jax.host_id() == 0:
    summary_writer = tensorboard.SummaryWriter(log_dir)

  # Restore checkpoint
  optimizer = checkpoint_utils.restore_checkpoint(checkpoint_path, optimizer)
  step = int(optimizer.state.step)

  # Replicate optimizer.
  optimizer = flax.jax_utils.replicate(optimizer)
  eval_step = adapter.make_eval_step()
  eval_step_parallel = jax.pmap(eval_step, axis_name='batch')

  # Perform evaluation
  tick = time.time()
  metrics_all = []

  example = None
  dataset_iter_raw = iter(dataset)
  dataset_iter = adapter.preprocess(dataset_iter_raw)
  for unused_eval_step, example in zip(
      range(config.eval_steps), dataset_iter):
    train_inputs = adapter.get_train_inputs(example)
    metrics, logits, state = eval_step_parallel(
        optimizer.target, train_inputs)
    metrics_all.append(metrics)

  # Write results.
  metrics_all = common_utils.get_metrics(metrics_all)
  metrics_sums = jax.tree_map(jnp.sum, metrics_all)
  denominator = metrics_sums.pop('denominator')
  summary = jax.tree_map(lambda x: x / denominator, metrics_sums)  # pylint: disable=cell-var-from-loop
  summary['perplexity'] = jnp.clip(jnp.exp(summary['loss']), a_max=1.0e4)
  logging.info('eval @ train step: %d, loss: %.4f', step, summary['loss'])
  if jax.host_id() == 0:
    tock = time.time()
    steps_per_sec = len(metrics_all) / (tock - tick)
    examples_per_sec = denominator / (tock - tick)
    summary_writer.scalar('per-second/steps', steps_per_sec, step)
    summary_writer.scalar('per-second/examples', examples_per_sec, step)
    for key, val in summary.items():
      summary_writer.scalar(key, val, step)

    adapter.write_summaries(example, logits, summary_writer, info, step, state)
    summary_writer.flush()


def predict_once(run_configuration, optimizer=None):
  """Predict the result once for each element in the dataset."""
  adapter = run_configuration.adapter
  checkpoint_path = run_configuration.original_checkpoint_path
  optimizer = optimizer or adapter.create_optimizer(run_configuration)
  dataset = run_configuration.dataset_info.dataset

  # Restore checkpoint
  optimizer = checkpoint_utils.restore_checkpoint(checkpoint_path, optimizer)

  # Replicate optimizer.
  optimizer = flax.jax_utils.replicate(optimizer)
  predict_step = adapter.make_predict_step()
  predict_step_parallel = jax.pmap(predict_step, axis_name='batch')

  # Perform inference
  dataset_iter_raw = iter(dataset)
  dataset_iter = adapter.preprocess(dataset_iter_raw)
  metrics_all = []
  for example in itertools.islice(dataset_iter, 200):
    train_inputs = adapter.get_train_inputs(example)
    metrics, logits, state = predict_step_parallel(
        optimizer.target, train_inputs)
    adapter.handle_predict(metrics, logits, state)
    metrics_all.append(metrics)
  metrics_all = common_utils.get_metrics(metrics_all)
  metrics = jax.tree_map(jnp.sum, metrics_all)
  return metrics
