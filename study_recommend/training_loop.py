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

"""The main body of the training loop.

The main body of the training loop to train IndividualRecommender and
StudyRecommender. The only public facing interface of this file is the function
training_loop(...).
"""

import collections
from collections.abc import Callable, Sequence
import functools
import json
import math
import os
import queue
import threading
import time
from typing import Any, Optional, Union

from absl import logging
from clu import metric_writers
from flax import jax_utils
from flax import linen as nn
from flax.training import common_utils
from flax.training import train_state
import flax.training.orbax_utils
import jax
from jax import lax
import jax.dlpack
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import orbax.checkpoint
from study_recommend import datasource as datasource_lib
from study_recommend import models
from study_recommend import types
from study_recommend.utils import training_utils as utils

file_open = open
file_exists = os.path.exists
make_dirs = functools.partial(os.makedirs, exist_ok=True)
list_dir = os.listdir

FIELDS = types.ModelInputFields
CHECKPOINT_SUBDIR = 'checkpoints'


def _eval_step(
    params,
    inputs,
    config,
    model_class,
    oov_token,
    separator_token = None,
):
  """Run evaluations on a batch on inputs.

  Args:
    params: Flax model parameter dict
    inputs: A batch of inputs. Expected fields can be found in
      types.ModelInputFields.
    config: Configuration to use run model forward pass.
    model_class: The class of the model to use for evaluation.
    oov_token: The value assigned to out of vocabulary tokens.
    separator_token: The value assigned to separator tokens.

  Returns:
    A dictionary of computed evaluation metrics. Metrics returned are
    aggregate sums. A denominator to normalize is also provided for users to
    compute averages downstream.
  """
  titles = inputs[FIELDS.TITLES]
  weights = utils.compute_weight_matrix(titles, separator_token)
  logits = model_class(config).apply({'params': params}, inputs)
  return utils.compute_metrics(logits, titles, weights, oov_token)


def eval_on_reference_batches(
    params,
    batches,
    p_eval_step,
):
  """Run evaluations on a list of reference batches and return aggregate.

  Args:
    params: A PyTree of Flax model parameters.
    batches: A list of batches to eval
    p_eval_step: A pjit compiled parallel eval function.

  Returns:
    A dictionary of metrics. Metrics returned are  aggregate sums. A denominator
    to normalize is also provided for users to compute averages downstream.
  """
  overall_metrics = collections.defaultdict(lambda: 0)

  for batch in batches:
    metrics = p_eval_step(params, batch)
    for metric_type in metrics:
      overall_metrics[metric_type] += np.array(metrics[metric_type])
  return dict(overall_metrics)


def _train_step(
    state,
    inputs,
    compute_metrics_from_logits,
    config,
    model_class,
    learning_rate_fn,
    dropout_rng,
    oov_token,
    separator_token = None,
):
  """Perform a single training step.

  Args:
    state: a TrainState representing current model parameters and optimizer
      state.
    inputs: A batch of data to train on.
    compute_metrics_from_logits: Whether or not to compute metrics from the
      logits.
    config: A TransformerConfig to use for forward pass on model.
    model_class: The class of flax model to use for the forward side,
    learning_rate_fn: A learning rate schedule function.
    dropout_rng: A jax random state to supply randomness for dropout.
    oov_token: The value assigned to out of vocabulary tokens
    separator_token: The value assigned to separator tokens.

  Returns:
    new_state: An updated TrainState after applying a train step
    metrics: A dictionary of
  """

  titles = inputs[FIELDS.TITLES]
  weights = utils.compute_weight_matrix(titles, separator_token)
  dropout_rng = jax.random.fold_in(dropout_rng, state.step)
  rngs = {
      'dropout': dropout_rng,
  }

  def loss_fn(params):
    """loss function used for training."""
    logits = model_class(config).apply({'params': params}, inputs, rngs=rngs)

    loss, weight_sum = utils.compute_weighted_cross_entropy(
        logits, titles, weights
    )
    mean_loss = loss / weight_sum
    return mean_loss, logits

  step = state.step
  lr = learning_rate_fn(step)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grads = grad_fn(state.params)
  metrics = {}
  if compute_metrics_from_logits:
    metrics.update(utils.compute_metrics(logits, titles, weights, oov_token))

  grads = lax.pmean(grads, 'batch')
  new_state = state.apply_gradients(grads=grads)
  metrics['learning_rate'] = lr

  return new_state, metrics


def get_reference_batches(
    n_reference_data_points,
    batch_size,
    datasource,
):
  """Get a list fixed of reference batches from a DataLoader.

  Args:
    n_reference_data_points: The number of reference datapoints to return.
    batch_size: The number of samples in each batch.
    datasource: A datasource to get datapoints from.

  Returns:
    A list of batches. Each batch will be sharded across all available
    devices.
  """
  reference_batches = []
  n_samples = 0
  n_batches = math.ceil(n_reference_data_points / batch_size)
  for new_batch in datasource_lib.iterator_with_replacement(
      datasource, batch_size, num_batches=n_batches
  ):
    reference_batches.append(new_batch)
    n_samples += new_batch[FIELDS.TITLES].shape[0]

  reference_batches = jax.tree_util.tree_map(
      shard_and_discard, reference_batches
  )

  return reference_batches


def shard_and_discard(array):
  """A utility function to shard an array across all available local devices.

  If the size of the leading dimension is not divisible by the number of local
  devices then the remainder after sharding the divisble part will be discarded.
  Args:
    array: jax.Array to shard.

  Returns:
    A sharded array.
  """

  array = jnp.array(array)
  offset = array.shape[0] % jax.local_device_count()
  # discard samples so batch size is dividable by n_devices
  if offset:
    array = array[:-offset]
  array = common_utils.shard(array)
  return array


WriteScalarArgs = tuple[tuple[Any], dict[Any, Any]]


def worker_fn(
    args_queue, logdir
):
  """Infifnite loop worker thread body for ParallelSummaryWriter."""
  writer = metric_writers.SummaryWriter(logdir)
  while 1:
    message = args_queue.get()
    if message == 'exit':
      return
    else:
      args, kwargs = message
      writer.write_scalars(*args, **kwargs)


class ParallelSummaryWriter:
  """A multi-threaded wrapper around clu.metric_writers.SummaryWriter.

  clu.metric_writers.SummaryWriter is a class to log training metrics
  to tensorboard or other training logging solutions. In this wrapper
  calls to write_scalars are delegated to a separate thread that calls
  clu.metric_writers.SummaryWriter.write_scalars. This is because this call
  leads to a write to disk which can be slow. This wrapper makes such calls
  asychronous to the rest of the execution of the code.
  """

  def __init__(self, logdir):
    logging.info('Starting ParallelSummaryWriter to dir %s', logdir)

    self._queue = queue.Queue()
    self._worker = threading.Thread(
        target=worker_fn, args=(self._queue, logdir)
    )

    self._worker.start()
    logging.info('Launched ParallelSummaryWriter thread sucessfully.')

  def write_scalars(self, *args, **kwargs):
    """A thin wrapper around clu.metric_writers.SummaryWriter.write_scalars."""
    self._queue.put((args, kwargs))

  def close(self):
    """Terminate the worker thread."""
    self._queue.put('exit')
    self._worker.join()


def init_or_load(
    config,
    train_datasource,
    override_load_path = None,
):
  """Initalize or load a trainstate containing model and optimizer parameters.

  If a saved model is found in config.working_dir or the supplied load path
  then the most recent checkpoint is loaded. To avoid this set
  config.restore_checkpoints to False.

  Args:
    config: Configuration of the training job.
    train_datasource: A datasource to generate a sample batch from to use to
      initialize parameters.
    override_load_path: An optional path to load the checkpoints from instead of
      config.working_dir.

  Returns:
    train_state: Loaded or initialized TrainState with model and optimizer
      parameters
    train_config: TransformerConfig to use for training forward pass (i.e. with
      dropout enabled, ...)
    eval_config: TranformerConfig to use for evaluation forward pass (i.e. with
      dropout disables, ..)
    model_class: The class of model instantiated.
  """
  train_config = models.generate_model_config(config)

  if config.model_class == 'individual':
    model_class = models.IndividualRecommender
  elif config.model_class == 'study':
    model_class = models.StudyRecommender
  else:
    raise ValueError(f'Invalid model_class {config.model_class}')

  logging.info('Model class is %s', config.model_class)

  # Turn off dropout in eval config
  eval_config = train_config.replace(deterministic=True)

  model = model_class(eval_config)
  rng = jax.random.PRNGKey(config.seed)
  dropout_init_rng, init_params_rng = jax.random.split(rng, num=2)

  logging.info('Init model')
  # Get a sample batch to initalize model parameters
  sample_batch = datasource_lib.collate(train_datasource[[1, 2, 3, 4]])
  sample_batch = jax.tree_util.tree_map(jnp.array, sample_batch)
  # Initialize model parameters.
  initial_variables = model.init(
      rngs={
          'dropout': dropout_init_rng,
          'params': init_params_rng,
      },
      inputs=sample_batch,
  )

  learning_rate_fn = utils.create_learning_rate_schedule(
      learning_rate=config.learning_rate, warmup_steps=config.warmup_steps
  )

  logging.info('init optimizer')
  optimizer = optax.adamw(
      learning_rate_fn,
      b1=0.9,
      b2=0.98,
      eps=1e-9,
      weight_decay=config.weight_decay,
  )
  state = train_state.TrainState.create(
      apply_fn=model.apply, params=initial_variables['params'], tx=optimizer
  )

  if config.restore_checkpoints:
    # Restore unreplicated optimizer + model state from last checkpoint.
    load_path = override_load_path or config.working_dir
    checkpoints_subdir = os.path.join(load_path, CHECKPOINT_SUBDIR)
    if file_exists(checkpoints_subdir):
      checkpoints = list_dir(checkpoints_subdir)
    else:
      checkpoints = None
    # If checkpoints is not None or an empty list then we
    # proceed to restore from the checkpoint.
    if checkpoints:
      # restore the most recent checkpoint.
      checkpoints.sort()
      latest_checkpoint = checkpoints[-1]
      latest_checkpoint_step = int(latest_checkpoint.split('_')[-1])
      # Get the training step.
      latest_checkpoint_path = os.path.join(
          checkpoints_subdir, latest_checkpoint
      )

      ckptr = orbax.checkpoint.Checkpointer(
          orbax.checkpoint.PyTreeCheckpointHandler()
      )
      state = ckptr.restore(
          latest_checkpoint_path,
          state,
          restore_args=flax.training.orbax_utils.restore_args_from_target(
              state, mesh=None
          ),
      )
      state = state.replace(step=latest_checkpoint_step + 1)
      logging.info('Restored from %s', config.working_dir)
    else:
      logging.info('No checkpoints found in %s', config.working_dir)

  return state, train_config, eval_config, model_class


def training_loop(
    config,
    train_datasource,
    valid_datasource,
    preloaded = None,
):
  """Main training loop to train IndividualRecommender or StudyRecommender.

  Args:
    config: A ConfigDict with all requires configurations for training run. a
      config dict with default values can be found in config.py
    train_datasource: Datasource to retrieve training set data points from.
    valid_datasource: Datasource to retrieve validation set data points from.
    preloaded: The precomputed value of init_or_load(...)

  Returns:
    state: Final model and optimizer paramaters after training.
    eval_config: TransformerConfig for doing inference.
  """
  if preloaded is not None:
    logging.info('the Utilising loaded state and config.')
    state, train_config, eval_config, model_class = preloaded
  else:
    logging.info('Loading generating/loading initial state and config.')
    state, train_config, eval_config, model_class = init_or_load(
        config, train_datasource
    )

  batch_size = config.per_device_batch_size * jax.local_device_count()
  learning_rate_fn = utils.create_learning_rate_schedule(
      learning_rate=config.learning_rate, warmup_steps=config.warmup_steps
  )

  rng = jax.random.PRNGKey(config.seed)

  start_step = round(state.step)
  logging.info('Starting from step %d', start_step)

  logging.info('Instantiating training iterator')
  start_index = config.per_device_batch_size * len(jax.devices()) * start_step
  train_iterator = datasource_lib.iterator_with_replacement(
      datasource=train_datasource,
      batch_size=batch_size,
      num_shards=jax.process_count(),
      shard_index=jax.process_index(),
      start_index=start_index,
      num_batches=config.num_train_steps,
  )

  logging.info('Generating reference batches.')
  reference_train_batches = get_reference_batches(
      config.reference_train_batch_size,
      batch_size,
      train_datasource,
  )
  reference_valid_batches = get_reference_batches(
      config.reference_valid_batch_size,
      batch_size,
      valid_datasource,
  )

  writer = None
  try:
    if config.write_summary:
      # Get the tensorboard output directory if was supplied. Else write
      # tensorboard to a subdir of the working dir.
      logdir = config.tensorboard_dir or os.path.join(
          config.working_dir, 'logdir'
      )

      writer = ParallelSummaryWriter(logdir)
      if jax.process_index() == 0:
        writer.write_scalars(0, {'clock start': 0})

    # make different hosts use different RNG for dropout
    dropout_rng = jax.random.fold_in(rng, jax.process_index())

    dropout_rngs = jax.random.split(dropout_rng, jax.local_device_count())

    state = jax_utils.replicate(state)

    p_train_step = jax.pmap(
        functools.partial(
            _train_step,
            config=train_config,
            model_class=model_class,
            learning_rate_fn=learning_rate_fn,
            separator_token=config.separator_token,
            oov_token=config.oov_token,
        ),
        axis_name='batch',
        donate_argnums=(0,),
        static_broadcasted_argnums=2,
    )

    p_eval_step = jax.pmap(
        functools.partial(
            _eval_step,
            config=eval_config,
            separator_token=config.separator_token,
            model_class=model_class,
            oov_token=config.oov_token,
        ),
        axis_name='batch',
    )

    # Initialize variables for logging speed every n steps.
    time_fn = time.time
    ts = time_fn()
    last_logged = start_step

    start_time = time.time()
    logging.info('Starting training loop')

    for step, batch in enumerate(train_iterator, start=start_step):
      # Shard the batch across all local devices.
      batch = jax.tree_util.tree_map(shard_and_discard, batch)

      if step == 0:
        logging.info(
            'Time to first batch %.2f seconds', (time.time() - start_time)
        )

      is_last_step = step == (config.num_train_steps - 1)
      is_logging_metrics = step % config.log_every_steps == 0 or is_last_step
      is_logging_metrics = is_logging_metrics and (step != 0)
      state, metrics = p_train_step(
          state,
          batch,
          is_logging_metrics,
          dropout_rng=dropout_rngs,
      )

      is_host = jax.process_index() == 0

      # Do a light logging of speed and metrics computed on last training batch.
      if is_logging_metrics:
        learning_rate = metrics.pop('learning_rate').mean()
        metrics = utils.normalize_metrics(metrics)
        metrics['learning_rate'] = learning_rate

        logging.info('Batch %s / %s', step, config.num_train_steps)
        logging.info('Metrics: %s', metrics)

        # Log training speed
        new_ts = time_fn()
        num_steps = step - last_logged
        duration = new_ts - ts
        # num_steps can be zero on restoring with certain
        # hyperparam configs.
        if num_steps:
          duration = duration / num_steps
        logging.info('Training at %.3f batch/second', 1 / duration)

        ts = new_ts
        last_logged = step

        # Log metrics to tensorboard.
        if writer and is_host:
          writer.write_scalars(step, metrics)

      # Do an infrequent evaluation on the train and test reference batches.
      if step % config.eval_every_steps == 0 or is_last_step:
        logging.info('Doing reference evals .....')
        logging.info('On train data...')
        train_metrics = eval_on_reference_batches(
            state.params, reference_train_batches, p_eval_step
        )
        logging.info('On valid data...')
        valid_metrics = eval_on_reference_batches(
            state.params, reference_valid_batches, p_eval_step
        )
        logging.info('Post processing train reference data')
        train_metrics = utils.normalize_metrics(train_metrics)
        logging.info('Post processing train reference data')
        valid_metrics = utils.normalize_metrics(valid_metrics)

        logging.info('Eval train : %s', train_metrics)
        logging.info('Eval valid : %s', valid_metrics)
        summary = {
            'train_' + metric: value for metric, value in train_metrics.items()
        }
        summary.update(
            {
                'valid_' + metric: value
                for metric, value in valid_metrics.items()
            }
        )
        if writer and is_host:
          writer.write_scalars(step, summary)

      # Save a checkpoint
      is_checkpoint_step = (
          step and step % config.checkpoint_every_steps == 0
      ) or is_last_step

      if config.save_checkpoints and is_checkpoint_step:
        logging.info('Saving Checkpoint....')
        checkpoint_dir = os.path.join(config.working_dir, CHECKPOINT_SUBDIR)
        make_dirs(checkpoint_dir)
        this_checkpoint_dir = os.path.join(
            checkpoint_dir, f'checkpoint_{step:015}'
        )
        ckptr = orbax.checkpoint.Checkpointer(
            orbax.checkpoint.PyTreeCheckpointHandler()
        )
        payload = jax.tree.map(
            utils.convert_host_local_array_to_global_array,
            state,
        )
        ckptr.save(
            this_checkpoint_dir,
            payload,
            force=True,
        )

        # Now save the model config if not saved.
        config_path = os.path.join(config.working_dir, 'config.json')
        if not file_exists(config_path) and is_host:
          with file_open(config_path, 'w') as f:
            f.write(json.dumps(config.to_dict()))

  finally:
    if writer:
      writer.close()
  return jax_utils.unreplicate(state), eval_config
