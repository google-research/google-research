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

"""Train/eval utility functions."""
import functools
import json
import os
import time
from typing import Any, Callable, Dict, Generator, Iterable, List, Mapping, Optional, Tuple, Union

from absl import logging
from clu import parameter_overview
import flax
from flax import jax_utils
from flax import traverse_util
from flax.core import frozen_dict
from flax.core.scope import DenyList
from flax.training import checkpoints
import gin
import jax
import jax.example_libraries.optimizers as jax_opt
from jax.experimental.compilation_cache import compilation_cache as cc
import jax.numpy as jnp
from metrics import metrics
from modeling import base
import tensorflow as tf
from utils import checkpoint_utils
from utils import tensorboard
from utils import types


# register Jax and tf dtypes with gin.
gin.constant('jnp.bfloat16', jnp.bfloat16)
gin.constant('jnp.float32', jnp.float32)
gin.constant('jnp.int32', jnp.int32)
gin.constant('tf.int32', tf.int32)
gin.constant('tf.uint8', tf.uint8)
gin.constant('tf.float32', tf.float32)
gin.constant('tf.bfloat16', tf.bfloat16)

Array = jnp.ndarray
ArrayDict = Dict[str, Array]
LossArray = Union[Array, ArrayDict]
TrainState = checkpoint_utils.TrainState
Batch = Union[Tuple[Any, Any], List[Any]]
FilterVars = Tuple[Tuple[str, str], Ellipsis]

_EVAL_REFRESH_SECONDS = 60
EPSILON = 1e-6


def maybe_log_progress(step, loss,
                       summary_writer,
                       learning_rate, train_state,
                       grads_norm, log_every,
                       total_train_steps):
  """Logs training progress at `log_every` intervals.

  This method logs and records the training progress to tensorboard summaries at
  every `log_every` steps or at the last training step.

  Args:
    step: The current step.
    loss: An array or a dictionary of named loss arrays.
    summary_writer: A tensorboard summary writer.
    learning_rate: A scalar array of current learning rate.
    train_state: A TrainState object with potentially debugging collections in
      its model_state.
    grads_norm: A dictionary containing gradient norms for each parameter.
    log_every: The interval to log speed and loss.
    total_train_steps: The total number of training steps.
  """
  try:
    maybe_log_progress.time_keeper
  except AttributeError:
    maybe_log_progress.time_keeper = time.time()
    maybe_log_progress.last_step = step
  if step % log_every == 0 or step == total_train_steps - 1:
    time_delta = time.time() - maybe_log_progress.time_keeper
    steps_per_second = float(step - maybe_log_progress.last_step) / time_delta
    loss = jax_utils.unreplicate(loss)
    grads_norm = jax_utils.unreplicate(grads_norm)
    activation_norm = (
        train_state.model_state['activation_norm']
        if 'activation_norm' in train_state.model_state else {})
    activation_norm = jax_utils.unreplicate(activation_norm)
    model_loss = loss['model_loss'] if isinstance(loss, dict) else loss
    logging.info('loss at training step %i: %f', step, model_loss)
    logging.info('train steps per second: %f', steps_per_second)
    # Only write to tensorboard from one host.
    if checkpoint_utils.is_chief():
      summary_writer.scalar('train/global_steps_per_second', steps_per_second,
                            step)
      for param, norm in grads_norm.items():
        summary_writer.scalar(f'grad_norm/{param}', norm, step)
      for name, norm in activation_norm.items():
        summary_writer.scalar(f'activation_norm/{name}', norm, step)
      summary_writer.scalar('train/learning_rate', learning_rate, step)
      if isinstance(loss, jnp.ndarray):
        loss = {'model_loss': loss}

      for name, value in loss.items():
        if isinstance(value, dict):
          # Handle nested dict values.
          for name2, value2 in value.items():
            summary_writer.scalar(f'train/{name}/{name2}', value2, step)
        else:
          summary_writer.scalar(f'train/{name}', value, step)

      summary_writer.flush()
    maybe_log_progress.time_keeper = time.time()
    maybe_log_progress.last_step = step


def maybe_log_eval_metrics(step,
                           eval_step,
                           summary_writer,
                           train_state, batch, rng,
                           log_every, total_train_steps):
  """Computes and logs evaluation metrics at `log_every` intervals.

  This method computes, logs and records evaluation metrics to tensorboard
  summaries at every `log_every` steps or at the last training step.

  Args:
    step: The current step.
    eval_step: The function to calculate metrics in a distributed mode. It
      should return the following positional arguments: (1) a `TrainState`
        object (2) a batch of data to compute metrics on (3) an RNG. This
        function should return a `MetricsCollection` containing the names and
        values of eval metrics and a new RNG to be usable for the next call.
    summary_writer: A tensorboard `SummaryWriter`.
    train_state: A `TrainState` that contains the model state used for
      evaluation.
    batch: The batch of data to run evaluation on.
    rng: An RNG key.
    log_every: The interval to log speed and loss.
    total_train_steps: The total number of training steps.

  Returns:
    a new RNG key usable for the next evaluation call.
  """
  if step % log_every == 0 or step == total_train_steps - 1:
    eval_metrics, new_rng = eval_step(train_state, batch, rng)
    # Only write to tensorboard from one host.
    if checkpoint_utils.is_chief():
      eval_metrics = jax_utils.unreplicate(eval_metrics)
      metrics.write_metrics(
          step=step, eval_metrics=eval_metrics, summary_writer=summary_writer)
    return new_rng
  return rng


def log_model_size(params):
  """Logs the number of parameters.

  Args:
    params: A dictionary of string to parameter arrays.
  """
  parameter_overview.log_parameter_overview(params)
  params_size = jax.tree.map(lambda x: x.size, params)
  params_size = sum(jax.tree.flatten(params_size)[0])
  logging.info('Model params size: %d', params_size)


def log_model_flops(model_fn, batch):
  """Computes the number of GFLOPs in the model and logs it."""
  # features are of type dict
  features, _ = batch if isinstance(batch, tuple) else (batch, {})

  features = jax_utils.unreplicate(features)
  # batch size is taken after unreplication for
  # consistency when device count changes.
  inputs_key = next(iter(features))
  inputs = features[inputs_key]
  batch_size = inputs.shape[0]

  analysis = jax.jit(model_fn).lower(**features).cost_analysis()

  flops = analysis['flops']
  # Remove mutliply-add ops by dividing by 2 (standard flops counting).
  flops = flops / 2
  gflops = flops / (10**9)
  gflops = gflops / batch_size
  logging.info('#' * 30)
  logging.info('Model GFLOPs: %.2f', gflops)
  # One sample is not enough to show differences in Giga Flops for small models.
  # 256 samples is about a typical batch size for most jobs.
  logging.info('Model GFLOPs (Batch-Size, 256): %.2f', gflops * 256)
  logging.info(
      'For input shape %s',
      str({k: v.shape for k, v in features.items() if hasattr(v, 'shape')}))
  logging.info('#' * 30)
  return gflops


def save_config(output_dir,
                summary_writer):
  """Saves the gin config values to a text file and writes to tensorboard.

  Saves both the config and operative config to `output_dir`:
    config: the user supplied values through a gin file and any --gin_bindings.
    operative config: the set of parameter values affecting the current
      program's execution, regardless of whether the user explicitly set it.
      Only tracks functions that are gin configurable.

  Args:
    output_dir: The directory to save gin config.
    summary_writer: A tensorboard `SummaryWriter`.
  """
  tf.io.gfile.makedirs(output_dir)
  for suffix, config_str in (('user_specified_config',
                              gin.config_str()), ('all_hparams_config',
                                                  gin.operative_config_str())):
    output_path = os.path.join(output_dir, f'{suffix}.gin')
    if tf.io.gfile.exists(output_path):
      tf.io.gfile.remove(output_path)
    with tf.io.gfile.GFile(output_path, 'w') as f:
      f.write(config_str)

    summary_writer.text(
        tag='user_specified_config',
        textdata=''.join(
            f'\t{line}' for line in gin.config_str().splitlines(keepends=True)),
        step=0)
    summary_writer.text(
        tag='all_hparams_config',
        textdata=''.join(
            f'\t{line}'
            for line in gin.operative_config_str().splitlines(keepends=True)),
        step=0)


def create_input_generator(
    input_fn, device_buffer_size
    ):
  """Prefetches data to local devices and returns a wrapped data generator.

  Args:
    input_fn: The data loader function.
    device_buffer_size: The number of batches of data to prefetch to local
      devices.

  Returns:
    An iterable that behaves the same way as the output of `input_fn`. When
    using difficulty sampling, this also returns a variable for the mixing
    weights.
  """
  data_generator = input_fn()
  if isinstance(data_generator, types.DatasetReturnObject):
    data_generator.dataset = jax_utils.prefetch_to_device(
        data_generator.dataset, device_buffer_size)
    return data_generator
  if device_buffer_size == 0:
    return (d for d in data_generator)
  # prefetch data to device memory to have overlapping compute and data
  # transfers.
  return jax_utils.prefetch_to_device(data_generator, device_buffer_size)


def generate_rng_dict(base_rng):
  """Generates a dictionary of rngs to pass in to `nn.Module`s.

  Stochastic layers in Flax Modules use separate stream of random number
  generators (e.g. dropout requires an rng named 'dropout'). This function
  generates all rngs needed for stochastic layers.

  Args:
    base_rng: The base rng to split.

  Returns:
    A dictionary of rngs to be used in calling modules.
  """
  keys = ('dropout', 'stochastic_depth', 'rng')
  rngs = jax.random.split(base_rng, len(keys))
  return {key: rngs[i] for i, key in enumerate(keys)}


def init_model(rng, model_fn, input_sample,
               mode):
  """Initializes the model and returns the variables.

  Args:
    rng: A random number generator.
    model_fn: The model constructor that accepts a keyword bool input named
      `train`.
    input_sample: the input sample to initialize the model with. For supervised
      scenario it is a tuple of (features, labels) and for unsupervised case it
      is a list of features.
    mode: the execution mode.

  Returns:
    A mapping of initialized variables keyed by their names.
  """
  if mode == base.ExecutionMode.PREDICT:
    raise ValueError('Cannot initialize model in mode PREDICT.')
  if isinstance(input_sample, tuple):
    features, _ = input_sample
  else:
    features = input_sample
  features = jax_utils.unreplicate(features)
  rng_generator, rng_params = jax.random.split(rng)
  rng_dict = generate_rng_dict(rng_generator)
  # add params rng that is only used during initialization
  rng_dict['params'] = rng_params
  # put the initialized variables on cpu since we will replicate them to all
  # local devices later.
  init_fn = jax.jit(
      lambda rng_key: frozen_dict.freeze(model_fn(mode=mode).init(
          rng_key, _do_remap=True, **features)),
      backend='cpu')
  return init_fn(rng_dict)


def create_host_eval_step(
    model_fn,
    mode,
):
  """Creates an eval_step function.

  Args:
    model_fn: the model constructor.
    mode: the execution mode. Normally used to specify EVAL vs PREDICT
      (autoregressive decoding) modes during eval.

  Returns:
    A pmapped function that computes outputs for a single batch of data. It
    accepts a `TrainState` a batch of data and an RNG key and returns a
    `A dictionary of outputs` and a new RNG key usable for the next call.
  """
  @functools.partial(jax.pmap, axis_name='batch', donate_argnums=(1, 2))
  def eval_step(train_state, batch,
                rng):
    """A single evaluation step."""
    variables = {'params': train_state.optimizer.target}
    variables.update(train_state.model_state)

    features, _ = batch if isinstance(batch, tuple) else (batch, {})
    rng, new_rng = jax.random.split(rng)
    model_outputs = model_fn(mode=mode).apply(
        variables,
        **features,
        mutable=False,
        _do_remap=True,
        rngs=generate_rng_dict(rng))
    return model_outputs, new_rng

  return eval_step


def create_eval_step(
    model_fn,
    eval_metrics,
    mode,
):
  """Creates an eval_step function.

  Args:
    model_fn: the model constructor.
    eval_metrics: a dictionary of metric name to `Metric`.
    mode: the execution mode. Normally used to specify EVAL vs PREDICT
      (autoregressive decoding) modes during eval.

  Returns:
    A pmapped function that computes eval metrics for a single batch of data. It
    accepts a `TrainState` a batch of data and an RNG key and returns a
    `MetricsCollection` and a new RNG key usable for the next call.
  """
  if isinstance(eval_metrics, str):
    # Handle the case where the metrics are passed in from xm flags, e.g.,
    # hyper.sweep('evaluate.eval_metrics', ["{'vqa_metrics': '%vqa_metrics'}"])
    # In that case, the gin parser returns a string for the metrics instead
    # of a dict, which we need to parse into the actual metric functions.
    eval_metrics = json.loads(eval_metrics.replace("'", '"'))
    for k, v in eval_metrics.items():
      eval_metrics[k] = gin.query_parameter(v.replace('%', ''))

  @functools.partial(jax.pmap, axis_name='batch', donate_argnums=(1, 2))
  def eval_step(train_state, batch,
                rng):
    """A single evaluation step."""
    variables = {'params': train_state.optimizer.target}
    variables.update(train_state.model_state)

    features, labels = batch if isinstance(batch, tuple) else (batch, {})
    rng, new_rng = jax.random.split(rng)
    model_outputs = model_fn(mode=mode).apply(
        variables,
        **features,
        mutable=False,
        _do_remap=True,
        rngs=generate_rng_dict(rng))
    return {
        key: jax.lax.all_gather(
            value.from_model_output(model_outputs, _do_remap=True, **labels),
            axis_name='batch').reduce() for key, value in eval_metrics.items()
    }, new_rng

  return eval_step


def create_train_state(optimizer_def,
                       params,
                       model_state):
  """Creates and returns an optimizer and a populated `TrainState`.

  Train state contains the step, model_state (includes any state of the model
  that are not trainable parameters. e.g. batch norm moving averages) and a
  Flax.optim.Optimizer (that includes model parameters and the optimizer state
  like momentum).

  Args:
    optimizer_def: A flax `OptimizerDef`.
    params: A pytree to model parameters. It should contain all values used for
      gradient computation.
    model_state: A pytree of model states. e.g. batch norm moving averages.

  Returns:
    A `TrainState` object containing an snapshot of the training state.
  """
  # put the initialized variables on cpu since we will replicate them to all
  # local devices later.
  create = jax.jit(optimizer_def.create, backend='cpu')
  opt = create(params)
  return TrainState(model_state=model_state, optimizer=opt, step=0)


def has_nan(inputs):
  """returns a boolean if any NaN is found in the `inputs` pytree."""
  inputs = jax_utils.unreplicate(inputs)
  return any([jnp.any(jnp.isnan(x)) for x in jax.tree.leaves(inputs)])


def pop_non_savable_collections(
    train_state):
  """Removes non savable collections from model_state for a train state.

  Args:
    train_state: Training state.

  Returns:
    New train state without collections that are not meant to be saved in
    model_state.
    A mapping of collection names and their values that are popped so they can
    be re-added later.
  """
  model_state = frozen_dict.unfreeze(train_state.model_state)
  non_savable_collections = {'params_axes', 'activation_norm'}
  popped_collections = {}
  for col_name in non_savable_collections:
    if col_name in train_state.model_state:
      col = model_state.pop(col_name)
      popped_collections[col_name] = col
  return train_state.replace(
      model_state=frozen_dict.freeze(model_state)), popped_collections


def re_add_non_savable_collections(
    train_state, collections):
  """Adds non savable collections to model_state for a train state.

  Args:
    train_state: Training state.
    collections: A mapping of collection names and their values to re-add.

  Returns:
    New train state with the new collections added to model_state.
  """
  model_state = frozen_dict.unfreeze(train_state.model_state)
  model_state.update(collections)
  return train_state.replace(model_state=frozen_dict.freeze(model_state))


@gin.configurable(denylist=['output_dir'])
def train(output_dir,
          input_fn = gin.REQUIRED,
          model_fn = gin.REQUIRED,
          optimizer_def = gin.REQUIRED,
          loss_fn = gin.REQUIRED,
          learning_rate = gin.REQUIRED,
          total_train_steps = gin.REQUIRED,
          process_gradients_fn = None,
          checkpoint_every = 1000,
          keep_latest_n_checkpoints = 20,
          log_every = 100,
          device_buffer_size = 2,
          eval_metrics = flax.core.FrozenDict({}),
          eval_mode = base.ExecutionMode.EVAL,
          debug_grad_norm = False,
          rng_seed = 0,
          pretrain_dir = '',
          pretrain_target_filters = (),
          pretrain_source_filters = (),
          pretrained_t5_path = '',
          t5_target_filters = (),
          t5_pretrained_filters = (),
          pretrained_bert = '',
          bert_target_filters = (),
          bert_pretrained_filters = (),
          pretrained_clip = '',
          clip_target_filters = (),
          clip_pretrained_filters = (),
          frozen_clip_target_filters = (),
          log_flops = True,
          jax_cache_dir = '',
          save_config_file = True,
          pretrained_detector = '',):
  """Train the model.

  Args:
    output_dir: Path to the directory to write summaries and model checkpoints.
    input_fn: The input function to obtain a data generator.
    model_fn: A flax Module constructor that accept a keyword bool input named
      `train`.
    optimizer_def: A flax `OptimizerDef`.
    loss_fn: A loss function that accepts model outputs and input labels and
      returns a scalar loss or a dictionary of component losses with a key
      `model_loss` that represents the total loss to differentiate.
    learning_rate: A float learning rate or a learning rate scheduler.
    total_train_steps: The number of training steps to take.
    process_gradients_fn: A function that takes in the gradients pytree and
      returns a pytree of the same structure and shapes. For example, gradient
      clipping can be implemented with this function.
    checkpoint_every: The interval to save model weights in number of steps.
    keep_latest_n_checkpoints: The number of latest checkpoints to keep.
    log_every: The interval to log metrics in number of steps.
    device_buffer_size: The number of batches of data to prefetch in device
      memory. For TPU workloads, we normally don't need anything more than 2.
      This value should be optimized based on the tradeoff between consuming
      more device memory vs. input preprocessing time. If the workload is more
      input bound increasing this value could help.
    eval_metrics: a mapping of names to metrics that will be recorded during
      training. These metrics are reported only on a single batch of data (the
      same data used during training) and at `log_every` steps.
    eval_mode: the execution mode to use during evaluation. Can be used to
      specify EVAL or PREDICT (for autoregressive decoding).
    debug_grad_norm: A boolean to enable reporting gradient norms for debugging.
    rng_seed: the seed to initialize the PRNGKey used for model randomness
    pretrain_dir: Path to the directory to load pretrained checkpoint from.
    pretrain_target_filters: A list of tuples of (prefix_string, filter_string)
      to select the target state to restore. The prefix string is removed from
      variable names to match the pretrained variable names. The filter_string
      selects the variables to restore.
    pretrain_source_filters: A list of tuples of (prefix_string, filter_string)
      to select the pretrained state to restore. The prefix string is removed
      from variable names to match the target variable names. The filter_string
      selects the variables to restore.
    pretrained_t5_path: string indicating where to load a T5 model weights from.
      Use the empty string to not load any T5 weights.
    t5_target_filters: The filter of the T5 model within the main model to load.
      For example, if your model has self.language_model = t5_model, then this
      should be set to [('language_model', ('.+')]. This has the same format as
      the pretrained_target_filters.
    t5_pretrained_filters: The filter of the T5 model to load. Same format as
      above.
    pretrained_bert: string indicating which bert model to load. Use the empty
      string to not load any weights.
    bert_target_filters: The filter of the BERT model within the main model. For
     example, if your model has self.language_model = bert_model, then this
     should be set to [('language_model', ('.+')]. This has the same format as
     the pretrained_target_filters.
    bert_pretrained_filters: The filter of bert model to load. Same format as
      above.
    pretrained_clip: string indicating which CLIP model to load. Use the empty
      string to not load any weights.
    clip_target_filters: The filter of the CLIP model within the main model. For
     example, if your model has self.language_model = clip_model, then this
     should be set to [('language_model', ('.+')]. This has the same format as
     the pretrained_target_filters.
    clip_pretrained_filters: The filter of CLIP model to load. Same format as
      above.
    frozen_clip_target_filters: The filter of the CLIP model within the main
      model, especially for the frozen vision backbone ensemble. If the frozen
      backbone is used, this should be set to [('frozen_vision_model', ('.+'))].
    log_flops: Set to false to disable logging of FLOPs.
    jax_cache_dir: If set, caches the jax compilation in this directory with a
      30 day ttl.
    save_config_file: Set to True to save the config file, false does not save.
    pretrained_detector: string indicating a pretrained detector model to load.
      Use the empty string to not load any weights.
  Raises:
    RuntimeError: if loss becomes NaN.
  """
  if jax_cache_dir:
    if not tf.io.gfile.Exists(jax_cache_dir):
      tf.io.gfile.MkDir(f'{jax_cache_dir}%ttl=30')
    cc.initialize_cache(jax_cache_dir)
  if not output_dir:
    raise ValueError('output_dir should be a non-empty path.')
  logging.info('Starting train function.')
  log_dir = os.path.join(output_dir, 'train')
  summary_writer = tensorboard.SummaryWriter(
      log_dir) if checkpoint_utils.is_chief() else None

  logging.info('Creating input generator.')
  data_generator = create_input_generator(
      input_fn, device_buffer_size=device_buffer_size)
  if isinstance(data_generator, types.DatasetReturnObject):
    # In this case, the data generator is returning the weights as well.
    # This is needed so we can dynamically adjust the weights later, if using
    # sampling.
    data_generator = data_generator.dataset
  logging.info('Getting first batch.')
  batch = next(data_generator)
  logging.info('Building model.')
  rng = jax.random.PRNGKey(rng_seed)
  rng, init_rng = jax.random.split(rng)
  variables = init_model(
      init_rng, model_fn, batch, mode=base.ExecutionMode.TRAIN)

  # separate out params and model state. This is needed for only computing
  # gradients w.r.t. the params.
  model_state = dict(variables)
  params = model_state.pop('params')
  logging.info('Creating train state.')
  train_state = create_train_state(optimizer_def, params, model_state)
  train_state, non_savable_collections = pop_non_savable_collections(
      train_state)
  train_state = checkpoints.restore_checkpoint(output_dir, train_state)
  train_state = re_add_non_savable_collections(train_state,
                                               non_savable_collections)
  del variables  # do not store a copy of the initial variables.

  init_step = train_state.step
  if init_step == total_train_steps:
    # Quick check for when using xmflow to resume graph with finished workers.
    # This prevents waiting for compiling of model before continuning experiment
    return

  # Only load the pretrained weights at init.
  # The prevent overwriting the weights when restoring after a preemption.
  if init_step == 0 and pretrain_dir:
    pretrained_state = checkpoints.restore_checkpoint(pretrain_dir, None)
    if pretrained_state is None:
      raise ValueError(f'Checkpoint not found at: {pretrain_dir}')
    train_state = checkpoint_utils.restore_with_assignment_map(
        train_state, pretrained_state, pretrain_target_filters,
        pretrain_source_filters)
  if init_step == 0 and pretrained_t5_path:
    t5_weights = checkpoint_utils.load_pretrained_t5_weights(pretrained_t5_path)
    train_state = checkpoint_utils.restore_with_assignment_map(
        train_state, t5_weights, t5_target_filters, t5_pretrained_filters)
  if init_step == 0 and pretrained_bert:
    bert_weights = checkpoint_utils.load_bert_weights(pretrained_bert)
    train_state = checkpoint_utils.restore_with_assignment_map(
        train_state, bert_weights, bert_target_filters, bert_pretrained_filters)
  if init_step == 0 and pretrained_clip:
    if pretrained_detector:
      clip_weights = checkpoint_utils.load_pretrained_detector_ckpt_weights(
          pretrained_detector)
      clip_target_filters = ([('', ('.+')),])
    else:
      clip_weights = checkpoint_utils.load_pretrained_clip_or_vit_v2_weights(
          pretrained_clip)
    train_state = checkpoint_utils.restore_with_assignment_map(
        train_state, clip_weights, clip_target_filters, clip_pretrained_filters)
    if not pretrained_detector and frozen_clip_target_filters:
      train_state = checkpoint_utils.restore_with_assignment_map(
          train_state, clip_weights, frozen_clip_target_filters,
          clip_pretrained_filters)

  train_state = jax_utils.replicate(train_state)
  logging.info('Starting model training from step: %d', init_step)
  log_model_size(params)
  variables = {'params': params}
  variables.update(model_state)
  if log_flops:
    logging.info('Getting flops.')
    log_model_flops(
        functools.partial(
            model_fn(mode=base.ExecutionMode.EVAL).apply,
            variables,
            _do_remap=True,
            rngs=generate_rng_dict(rng)),
        batch=batch
        )
  del variables

  @functools.partial(jax.pmap, axis_name='batch', donate_argnums=(0, 2))
  def train_step(
      train_state, data,
      rng):
    """A single training step.

    Args:
      train_state: Current training state object.
      data: A tuple of feature and label or a list of features.
      rng: An array of Jax random key as a base key to generate other keys for
        this iteration.

    Returns:
      new_train_state: A new training state object.
      losses: An array of losses or a dictionary of named loss arrays.
      new_rng: A new base key for the next iteration.
      grads_norm: A dictionary of weight names and their gradient norms. Useful
      for debugging.
    """
    if isinstance(data, tuple):
      features, labels = data
    else:
      features, labels = data, {}
    step = train_state.step
    optimizer = train_state.optimizer
    model_state = train_state.model_state
    lr = learning_rate(step) if callable(learning_rate) else learning_rate
    rng, new_rng = jax.random.split(rng)

    def loss_step(params):
      """Loss step that is used to compute the gradients.

      Args:
        params: A dictionary of string to parameter arrays.

      Returns:
        loss: A scalar loss for differentiation.
        aux_output: A dictionary with key-value pairs:
          new_model_state: A dictionary of new variables.
          losses: An array or a dictionary of loss for logging.
      """
      variables = {'params': params}
      variables.update(model_state)
      outputs, new_variables = model_fn(mode=base.ExecutionMode.TRAIN).apply(
          variables,
          _do_remap=True,
          **features,
          mutable=DenyList('params'),
          rngs=generate_rng_dict(rng))

      losses = loss_fn(outputs, **labels, _do_remap=True, params=params)
      new_model_state = dict(new_variables)
      aux_output = {'new_model_state': new_model_state, 'losses': losses}
      if isinstance(losses, dict):
        if 'model_loss' not in losses:
          raise ValueError(
              'Losses must contain `model_loss` key as total model loss.')
        model_loss = losses['model_loss']
      elif isinstance(losses, jnp.ndarray):
        model_loss = losses
      else:
        raise ValueError('Encountered invalid loss type: ', type(losses))

      return model_loss, aux_output

    grad_fn = jax.value_and_grad(loss_step, has_aux=True)
    (_, aux_output), grads = grad_fn(optimizer.target)
    grads = jax.lax.pmean(grads, axis_name='batch')
    if debug_grad_norm:
      grads_norm = flax.core.unfreeze(grads)
      grads_norm = traverse_util.flatten_dict(grads_norm)
      grads_norm = {'.'.join(k): v for k, v in grads_norm.items()}
      grads_norm = jax.tree.map(jax_opt.l2_norm, grads_norm)
      grads_norm['global'] = jax_opt.l2_norm(grads)
    else:
      grads_norm = {}
    if process_gradients_fn is not None:
      grads = process_gradients_fn(grads)
    new_optimizer = optimizer.apply_gradient(grads, learning_rate=lr)
    new_model_state = jax.lax.pmean(
        aux_output['new_model_state'], axis_name='batch')
    losses = jax.lax.pmean(aux_output['losses'], axis_name='batch')
    new_train_state = train_state.replace(
        step=step + 1,
        optimizer=new_optimizer,
        model_state=new_model_state)
    return new_train_state, losses, new_rng, grads_norm

  logging.info('Creating eval step.')
  eval_step = create_eval_step(
      model_fn=model_fn, eval_metrics=eval_metrics, mode=eval_mode)
  train_rng_key, eval_rng_key = jax.random.split(rng)
  train_rng_key = jax.random.split(train_rng_key, jax.local_device_count())
  eval_rng_key = jax.random.split(eval_rng_key, jax.local_device_count())

  logging.info('Running train loop.')
  for step in range(init_step, total_train_steps):
    train_state, losses, train_rng_key, grads_norm = train_step(
        train_state, batch, train_rng_key)
    if has_nan(losses):
      logging.info(losses)
      raise RuntimeError('NaN encountered in losses.')
    lr = learning_rate(step) if callable(learning_rate) else learning_rate

    eval_rng_key = maybe_log_eval_metrics(
        step=step,
        rng=eval_rng_key,
        eval_step=eval_step,
        summary_writer=summary_writer,
        train_state=train_state,
        batch=batch,
        log_every=log_every,
        total_train_steps=total_train_steps)
    maybe_log_progress(
        step=step,
        loss=losses,
        learning_rate=lr,
        train_state=train_state,
        grads_norm=grads_norm,
        summary_writer=summary_writer,
        log_every=log_every,
        total_train_steps=total_train_steps)
    checkpoint_utils.maybe_save_checkpoint(
        step=step,
        train_state=train_state,
        output_dir=output_dir,
        checkpoint_every=checkpoint_every,
        keep_latest_n_checkpoints=keep_latest_n_checkpoints,
        total_train_steps=total_train_steps)
    if step == init_step and checkpoint_utils.is_chief() and save_config_file:
      save_config(log_dir, summary_writer)

    batch = next(data_generator) if step < total_train_steps - 1 else None


def _merge_eval_results(
    eval_results,
    updates):
  """Update the running evaluation results with the updates of one eval step."""
  if eval_results is None:
    return updates
  return {name: eval_results[name].merge(updates[name]) for name in updates}


@gin.configurable(denylist=['output_dir'])
def evaluate(output_dir,
             input_fn = gin.REQUIRED,
             model_fn = gin.REQUIRED,
             optimizer_def = gin.REQUIRED,
             eval_steps = gin.REQUIRED,
             total_train_steps = gin.REQUIRED,
             eval_metrics = flax.core.FrozenDict({}),
             eval_mode = base.ExecutionMode.EVAL,
             metrics_dir_name = 'eval',
             device_buffer_size = 2,
             pretrained_clip = '',
             clip_target_filters = (),
             clip_pretrained_filters = (),
             frozen_clip_target_filters = (),
             save_zeroshot_ckpt = False,
             host_evaluator = None,
             save_config_file = True,
             pretrained_detector = '',
             ):
  """Evaluate the model.

  Args:
    output_dir: Path to the directory to write summaries and model checkpoints.
    input_fn: The input function to obtain a data generator.
    model_fn: A flax Module constructor that accept a keyword bool input named
      `train`.
    optimizer_def: A flax `OptimizerDef`.
    eval_steps: How many evaluation steps to take.
    total_train_steps: Total number of training steps to evaluate.
    eval_metrics: a mapping of names to metrics that will be recorded.
    eval_mode: the execution mode to use during evaluation. Can be used to
      specify EVAL or PREDICT (for autoregressive decoding).
    metrics_dir_name: subdirectory under output_dir where the metrics are stored
    device_buffer_size: How many batches of data to prefetch in device memory.
    pretrained_clip: string indicating which CLIP model to load. Use the empty
      string to not load any weights.
    clip_target_filters: The filter of the CLIP model within the main model. For
     example, if your model has self.language_model = clip_model, then this
     should be set to [('language_model', ('.+')]. This has the same format as
     the pretrained_target_filters.
    clip_pretrained_filters: The filter of CLIP model to load. Same format as
      above.
    frozen_clip_target_filters: The filter of the CLIP model within the main
      model, especially for the frozen vision backbone ensemble. If the frozen
      backbone is used, this should be set to [('frozen_vision_model', ('.+'))].
    save_zeroshot_ckpt: Flag to save zero-shot checkpoint. Default 'False' to
      avoid duplicating other checkpoints.
    host_evaluator: Evaluator to perform host evaluation.
    save_config_file: Set to True to save the config file, false does not save.
    pretrained_detector: string indicating a pretrained detector model to load.
      Use the empty string to not load any weights.
  """
  if not output_dir:
    raise ValueError('output_dir should be a non-empty path.')

  log_dir = os.path.join(output_dir, metrics_dir_name)
  summary_writer = tensorboard.SummaryWriter(
      log_dir) if checkpoint_utils.is_chief() else None

  data_generator = create_input_generator(
      input_fn, device_buffer_size=device_buffer_size)
  if isinstance(data_generator, types.DatasetReturnObject):
    data_generator = data_generator.dataset

  batch = next(data_generator)
  rng = jax.random.PRNGKey(0)
  rng, init_rng = jax.random.split(rng)
  variables = init_model(
      init_rng, model_fn, batch, mode=base.ExecutionMode.EVAL)
  # separate out params and model state. This is needed for only computing
  # gradients w.r.t. the params.
  model_state = dict(variables)
  params = model_state.pop('params')
  train_state = create_train_state(optimizer_def, params, model_state)
  del variables  # do not store a copy of the initial variables.
  if host_evaluator is None:
    eval_step = create_eval_step(
        model_fn=model_fn, eval_metrics=eval_metrics, mode=eval_mode)
  else:
    eval_step = create_host_eval_step(model_fn=model_fn, mode=eval_mode)

  last_checkpoint = train_state.step
  ticker = time.time()
  rng = jax.random.split(rng, num=jax.local_device_count())
  while True:
    train_state, non_savable_collections = pop_non_savable_collections(
        train_state)
    train_state = checkpoints.restore_checkpoint(output_dir, train_state)
    train_state = re_add_non_savable_collections(train_state,
                                                 non_savable_collections)
    if pretrained_clip:
      if pretrained_detector:
        clip_weights = checkpoint_utils.load_pretrained_detector_ckpt_weights(
            pretrained_detector)
        clip_target_filters = ([('', ('.+')),])
      else:
        clip_weights = checkpoint_utils.load_pretrained_clip_or_vit_v2_weights(
            pretrained_clip)
      train_state = checkpoint_utils.restore_with_assignment_map(
          train_state, clip_weights, clip_target_filters,
          clip_pretrained_filters)
      if not pretrained_detector and frozen_clip_target_filters:
        train_state = checkpoint_utils.restore_with_assignment_map(
            train_state, clip_weights, frozen_clip_target_filters,
            clip_pretrained_filters)

    if save_zeroshot_ckpt:
      checkpoints.save_checkpoint(output_dir, train_state,
                                  step=train_state.step, overwrite=True)
    try:
      if last_checkpoint == train_state.step and train_state.step > 0:
        if time.time() - ticker < _EVAL_REFRESH_SECONDS:
          logging.info('Waiting for a new model checkpoint')
        time.sleep(_EVAL_REFRESH_SECONDS)  # sleep for 60 seconds
        continue
    except ValueError:
      time.sleep(_EVAL_REFRESH_SECONDS)
      continue

    current_step = train_state.step
    logging.info('Starting model evaluation at step: %d', current_step)
    train_state = jax_utils.replicate(train_state)
    eval_results = None
    for i in range(eval_steps):
      logging.info('Evaluation step: %d', i)
      if host_evaluator is None:
        updates, rng = eval_step(train_state, batch, rng)
        updates = jax_utils.unreplicate(updates)
        if checkpoint_utils.is_chief():
          eval_results = _merge_eval_results(eval_results, updates)
      else:
        outputs, rng = eval_step(train_state, batch, rng)
        if checkpoint_utils.is_chief():
          host_evaluator.task_update(outputs)

      if i < eval_steps - 1:
        batch = next(data_generator)
    logging.info('Evaluation for checkpoint %d Finished', current_step)

    if checkpoint_utils.is_chief():
      if host_evaluator is None:
        metrics.write_metrics(current_step, eval_results, summary_writer)
      else:
        output_metrics = host_evaluator.task_evaluate()
        metrics.write_dict_metrics(current_step, output_metrics, summary_writer)
        metric_path = os.path.join(log_dir, 'detection_metrics.json')
        logging.info('Writing to metric path %s', metric_path)
        with tf.io.gfile.GFile(metric_path, 'w') as fid:
          # Cast numpy scalar metric to float.
          json.dump({k: float(v) for k, v in output_metrics.items()}, fid)

      if save_config_file:
        save_config(log_dir, summary_writer)

    # Terminate when the total_steps have been reached.
    if current_step >= total_train_steps:
      logging.info('Evaluation finished after reaching total steps: %d >= %d',
                   current_step, total_train_steps)
      break
    last_checkpoint = current_step
    ticker = time.time()
    data_generator = create_input_generator(
        input_fn, device_buffer_size=device_buffer_size)
    if isinstance(data_generator, types.DatasetReturnObject):
      data_generator = data_generator.dataset
    batch = next(data_generator)
