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

"""Machine Translation example.

This script trains a Transformer on a WMT dataset.
"""
import concurrent.futures
import dataclasses
import functools
import json
import os
import os.path
import pickle
import re
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Type, TypeVar, Union

from absl import app
from absl import flags
from absl import logging
import flax
from flax import jax_utils
from flax import optim
from flax.metrics import tensorboard
from flax.training import checkpoints
from flax.training import common_utils
import jax
from jax import random
import jax.nn
import jax.numpy as jnp
from ml_collections import config_flags
import numpy as np
import sacrebleu
import tensorflow.compat.v2 as tf
import tensorflow_text as tftxt


from aqt.jax import compute_cost_utils
from aqt.jax import hlo_utils
from aqt.jax import quant_config
from aqt.jax import train_utils
from aqt.jax.wmt_mlperf import bleu
from aqt.jax.wmt_mlperf import input_pipeline
from aqt.jax.wmt_mlperf import models
from aqt.jax.wmt_mlperf import predict

# train_flags is imported for the side effect of flag definitions.
# These are used both in xm_launch.py and this training executable.
from aqt.jax.wmt_mlperf import train_flags  # pylint: disable=unused-import
from aqt.jax.wmt_mlperf import training_hparams
from aqt.utils import hparams_utils as os_hparams_utils
from aqt.utils import report_utils
from aqt.utils import summary_utils

T = TypeVar('T')

EOS_TOKEN = 2  # Default Sentencepiece EOS token.

COMPUTE_MEMORY_COST_FILENAME = 'compute_memory_cost.json'

FLAGS = flags.FLAGS


config_flags.DEFINE_config_file('hparams_config_dict', None,
                                'Path to file defining a config dict.')

flags.DEFINE_integer(
    'config_idx',
    default=None,
    help=(
        'Identifies which config within the sweep this training run should use.'
    ))

flags.DEFINE_string(
    'model_dir', default=None, help='Directory to store model data')

# More flags defined in train_flags.py.


def create_learning_rate_scheduler(
    hparams):
  """creates learning rate schedule.

  Args:
    hparams: A LearningRateSchedulerHParams instance that describes the desired
      learning rate schedule. See LearningRateSchedulerHParams documentation for
      details.

  Returns:
    a function learning_rate(step): float -> {'learning_rate': float}, the
    step-dependent lr.
  """
  factors = [n.strip() for n in hparams.factors.split('*')]

  def step_fn(step):
    """Step to learning rate function."""
    ret = 1.0
    for name in factors:
      if name == 'constant':
        ret *= hparams.base_learning_rate
      elif name == 'linear_warmup':
        ret *= jnp.minimum(1.0, step / hparams.warmup_steps)
      elif name == 'rsqrt_decay':
        ret /= jnp.sqrt(jnp.maximum(step, hparams.warmup_steps))
      elif name == 'rsqrt_normalized_decay':
        ret *= jnp.sqrt(hparams.warmup_steps)
        ret /= jnp.sqrt(jnp.maximum(step, hparams.warmup_steps))
      elif name == 'decay_every':
        # TODO(malmaud): Create an assert-style function to replace all these
        # if cond:
        #   raise ValueError
        # patterns.
        if hparams.decay_factor is None or hparams.steps_per_decay is None:
          raise ValueError('Missing parameters for "decay_every" factor.')
        ret *= (hparams.decay_factor**(step // hparams.steps_per_decay))
      elif name == 'cosine_decay':
        if hparams.steps_per_cycle is None:
          raise ValueError('Missing parameters for "cosine_decay" factor.')
        progress = jnp.maximum(0.0, (step - hparams.warmup_steps) /
                               float(hparams.steps_per_cycle))
        ret *= jnp.maximum(0.0,
                           0.5 * (1.0 + jnp.cos(jnp.pi * (progress % 1.0))))
      else:
        raise ValueError('Unknown factor %s.' % name)
    return jnp.asarray(ret, dtype=jnp.float32)

  return step_fn


@functools.partial(jax.jit, static_argnums=(1, 2, 3, 4))
def create_model(key, input_shape, target_shape,
                 transformer_kwargs,
                 hparams):
  """Instantiate transformer model."""
  model = models.Transformer(
      **transformer_kwargs,
      quant_context=quant_config.QuantContext(
          update_bounds=False, collect_acts_stats=FLAGS.collect_acts_stats),
      hparams=hparams,
      use_bfloat16=False,
      train=False,
      dropout_rate=0.1,
      attention_dropout_rate=0.1,
      should_decode=False)
  variables = model.init(
      key,
      jnp.zeros(input_shape, jnp.float32),
      jnp.zeros(target_shape, jnp.float32),
  )
  init_state, params = variables.pop('params')  # pytype: disable=attribute-error
  return params, init_state


# TODO(malmaud): Consider bundling the optimizer parameters into their own
# HParams class
def create_optimizer(params, learning_rate, weight_decay, beta1, beta2, eps):
  optimizer_def = optim.Adam(
      learning_rate,
      beta1=beta1,
      beta2=beta2,
      eps=eps,
      weight_decay=weight_decay)
  optimizer = optimizer_def.create(params)
  optimizer = optimizer.replicate()
  return optimizer


@jax.custom_gradient
def cross_entropy_with_logits(logits, targets):
  """Computes cross entropy loss with custom gradient support.

  Args:
    logits: [batch * length, num_classes] float array.
    targets: categorical targets [batch * length] float array.

  Returns:
    Tuple of scalar loss and custom gradient function.
  """
  shifted = logits - logits.max(axis=-1, keepdims=True)
  exp_shifted = jnp.exp(shifted)
  sum_exp = jnp.sum(exp_shifted, axis=-1, keepdims=True)
  log_softmax = shifted - jnp.log(sum_exp)
  loss = -jnp.sum(targets * log_softmax, axis=-1)

  def grad_fn(g):
    return jnp.expand_dims(g, axis=-1) * (exp_shifted / sum_exp - targets), g

  return loss, grad_fn


def compute_weighted_cross_entropy(logits,
                                   targets,
                                   weights=None,
                                   label_smoothing=0.1):
  """Compute weighted cross entropy and entropy for log probs and targets.

  Args:
    logits: [batch * length, num_classes] float array.
    targets: categorical targets [batch, length] int array.
    weights: None or array of shape [batch, length]
    label_smoothing: label smoothing constant, used to determine the on and off
      values.

  Returns:
    Tuple of scalar loss and batch normalizing factor.
  """
  targets = targets.reshape((-1))
  weights = weights.reshape((-1))
  if logits.ndim != targets.ndim + 1:
    raise ValueError('Incorrect shapes. Got shape %s logits and %s targets' %
                     (str(logits.shape), str(targets.shape)))
  vocab_size = logits.shape[-1]
  confidence = 1.0 - label_smoothing
  low_confidence = (1.0 - confidence) / (vocab_size - 1)
  normalizing_constant = -(
      confidence * jnp.log(confidence) +
      (vocab_size - 1) * low_confidence * jnp.log(low_confidence + 1e-20))
  soft_targets = common_utils.onehot(
      targets, vocab_size, on_value=confidence, off_value=low_confidence)

  loss = cross_entropy_with_logits(logits, soft_targets)

  loss = loss - normalizing_constant

  normalizing_factor = np.prod(targets.shape)
  if weights is not None:
    loss = loss * weights
    normalizing_factor = weights.sum()

  return loss.sum(), normalizing_factor


def compute_weighted_accuracy(logits, targets, weights=None):
  """Compute weighted accuracy for log probs and targets.

  Args:
   logits: [batch * length, num_classes] float array.
   targets: categorical targets [batch, length] int array.
   weights: None or array of shape [batch, length]

  Returns:
    Tuple of scalar loss and batch normalizing factor.
  """
  targets = targets.reshape((-1))
  weights = weights.reshape((-1))
  if logits.ndim != targets.ndim + 1:
    raise ValueError('Incorrect shapes. Got shape %s logits and %s targets' %
                     (str(logits.shape), str(targets.shape)))
  loss = jnp.equal(jnp.argmax(logits, axis=-1), targets)
  normalizing_factor = np.prod(logits.shape[:-1])
  if weights is not None:
    loss = loss * weights
    normalizing_factor = weights.sum()

  return loss.sum(), normalizing_factor


def compute_metrics(logits, labels, weights):
  """Compute summary metrics."""
  loss, weight_sum = compute_weighted_cross_entropy(logits, labels, weights)
  acc, _ = compute_weighted_accuracy(logits, labels, weights)
  metrics = {
      'loss': loss,
      'accuracy': acc,
      'denominator': weight_sum,
  }
  metrics = jax.lax.psum(metrics, axis_name='batch')
  return metrics


# TODO(shivaniagrawal): Make TrainingHparams hashable instead.
class WrapHashably:
  """Wrapper to turn np.ndarray into Hashable objects for pmap."""
  __slots__ = ['val']

  def __init__(self, val):
    self.val = val

  def __hash__(self):
    return id(self.val)

  def __eq__(self, other):
    return self.val is other.val


def train_step(optimizer,
               batch,
               quant_context,
               transformer_kwargs,
               hparams,
               state,
               dropout_rng=None):
  """Perform a single training step."""
  if isinstance(hparams, WrapHashably):
    hparams = hparams.val

  train_keys = [
      'inputs', 'targets', 'inputs_position', 'targets_position',
      'inputs_segmentation', 'targets_segmentation'
  ]
  (inputs, targets, inputs_positions, targets_positions, inputs_segmentation,
   targets_segmentation) = [batch.get(k, None) for k in train_keys]

  weights = jnp.where(targets > 0, 1, 0).astype(jnp.float32)

  # It's very important to handle PRNG splitting inside the top pmap, rather
  # than handling it outside in the training loop - doing the latter can add
  # bad stalls to the input data transfer.
  dropout_rng, new_dropout_rng = random.split(dropout_rng)

  def loss_fn(params):
    """loss function used for training."""
    model = models.Transformer(
        **transformer_kwargs,
        quant_context=quant_context,
        use_bfloat16=FLAGS.use_bfloat16,
        train=True,
        hparams=hparams.model_hparams,
        dropout_rate=0.1,
        attention_dropout_rate=0.1,
        should_decode=False)
    logits, new_variables = model.apply(
        {
            'params': params,
            **state
        },
        inputs,
        targets,
        inputs_positions=inputs_positions,
        targets_positions=targets_positions,
        inputs_segmentation=inputs_segmentation,
        targets_segmentation=targets_segmentation,
        mutable=True,
        rngs={'dropout': dropout_rng})
    new_state, _ = new_variables.pop('params')
    loss, weight_sum = compute_weighted_cross_entropy(logits, targets, weights)
    mean_loss = loss / weight_sum
    total_loss = mean_loss


    return total_loss, (logits, new_state)

  step = optimizer.state.step
  learning_rate_fn = create_learning_rate_scheduler(
      hparams.learning_rate_schedule)
  lr = learning_rate_fn(step)
  value_and_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, (logits, new_state)), loss_grad = value_and_grad_fn(optimizer.target)
  new_optimizer = optimizer.apply_gradient(loss_grad, learning_rate=lr)

  if FLAGS.compute_train_metrics:
    metrics = compute_metrics(logits, targets, weights)
  else:
    metrics = {}
  metrics['learning_rate'] = lr
  # Compute or_loss for logging
  # TODO(wanglisa): Is there a way to avoid computing it twice?
  or_loss = custom_losses.weight_outlier_regularization_loss(
      optimizer.target,
      weights_regex_pattern=hparams.weight_outlier_regularization_regex)
  metrics['or_loss'] = or_loss

  return new_state, new_optimizer, metrics, new_dropout_rng


# TODO(shivaniagrawal): parametrize the axis_name and use the same axis name
# throughout Transformer model.
p_train_step = jax.pmap(
    functools.partial(train_step),
    axis_name='batch',
    static_broadcasted_argnums=(3, 4))
# hparams, transformer_kwargs are static args.


def _write_beam_hlo(params, input_shape, cache, state,
                    transformer_kwargs,
                    hparams,
                    quant_context):
  """Writes HLO with beam search for given model."""
  if not FLAGS.output_beam_hlo_filename:
    return
  basename, _ = os.path.splitext(FLAGS.output_beam_hlo_filename)
  input_dummy = jnp.ones(input_shape)

  def _without_weights(inputs, params):
    """JAX computation with inputs and weights turned into parameters."""
    return predict_step(
        inputs,
        params,
        cache,
        state,
        EOS_TOKEN,
        transformer_kwargs=transformer_kwargs,
        hparams=hparams,
        quant_context=quant_context)

  def _with_weights(inputs):
    """JAX computation with only inputs turned into parameters."""
    return predict_step(
        inputs,
        params,
        cache,
        state,
        EOS_TOKEN,
        transformer_kwargs=transformer_kwargs,
        hparams=hparams,
        quant_context=quant_context)

  _write_hlo(basename, _without_weights, input_dummy, transformer_kwargs,
             params, cache, state)
  _write_hlo(basename + '_with_weights', _with_weights, input_dummy, state)


def get_jax_computation_of_model(transformer_kwargs,
                                 params, state,
                                 hparams,
                                 with_weights):
  """Creates jax computation functions of model with or without weights."""
  input_dummy = jnp.ones((hparams.per_host_batch_size, FLAGS.max_target_length))
  target_dummy = jnp.ones(
      (hparams.per_host_batch_size, FLAGS.max_target_length))
  model = models.Transformer(
      **transformer_kwargs,
      quant_context=quant_config.QuantContext(
          update_bounds=False, collect_acts_stats=False),
      train=False,
      hparams=hparams.model_hparams,
      use_bfloat16=False,
      dropout_rate=0.1,
      attention_dropout_rate=0.1,
      should_decode=False)

  def _without_weights(params, inputs, target, state):
    """JAX computation with inputs and weights turned into parameters."""
    logits = model.apply({
        'params': params,
        **state
    },
                         inputs,
                         target,
                         mutable=False)
    return logits

  def _with_weights(inputs, target, state):
    """JAX computation with only inputs turned into parameters."""
    logits = model.apply({'params': params, **state}, inputs, target)
    return logits

  if with_weights:
    return _with_weights, input_dummy, target_dummy, state, params
  else:
    return _without_weights, input_dummy, target_dummy, state, params


def estimate_compute_and_memory_cost(
    model_dir, params, transformer_kwargs,
    state, hparams):
  """Estimate compute and memory cost of model."""

  without_weights_fn, input_dummy, target_dummy, state, params = get_jax_computation_of_model(
      params=params,
      transformer_kwargs=transformer_kwargs,
      state=state,
      hparams=hparams,
      with_weights=False)
  cost_dict = compute_cost_utils.estimate_costs_of_dot_and_conv_ops_from_jax_fn(
      without_weights_fn, params, input_dummy, target_dummy, state)

  path = tf.io.gfile.join(model_dir, COMPUTE_MEMORY_COST_FILENAME)
  with tf.io.gfile.GFile(path, 'w') as file:
    json.dump(cost_dict, file, indent=2)
  logging.info('Estimated compute and memory costs and wrote to file')


def _write_train_hlo(transformer_kwargs,
                     params, state,
                     hparams):
  """Writes training HLO for given model."""
  if not FLAGS.output_hlo_filename:
    return

  basename, _ = os.path.splitext(FLAGS.output_hlo_filename)
  without_weights_fn, input_dummy, target_dummy, state, model = get_jax_computation_of_model(
      params=params,
      transformer_kwargs=transformer_kwargs,
      state=state,
      hparams=hparams,
      with_weights=False)
  _write_hlo(basename, without_weights_fn, model, input_dummy, target_dummy,
             state)
  with_weights_fn, input_dummy, target_dummy, state, model = get_jax_computation_of_model(
      params=params,
      transformer_kwargs=transformer_kwargs,
      state=state,
      hparams=hparams,
      with_weights=True)
  _write_hlo(basename + '_with_weights', with_weights_fn, input_dummy,
             target_dummy, state)


def _write_hlo(output, fn, *fn_args, **fn_kwargs):
  """Writes HLO for the given function.

  Writes the HLO of the provided function to destination
  FLAGS.model_dir/<output>{.txt,.pb}. Both the HLO-text and binary protobuf are
  written.

  Args:
    output: The basename of the output file (without extension)
    fn: the function for which the HLO is to be produced.
    *fn_args: the function's args.
    **fn_kwargs: the function's kwargs.

  Returns:
    None.
  """
  try:
    computation = jax.xla_computation(fn)(*fn_args, **fn_kwargs)
    logging.info('Generated XLA computation for HLO.')
    pb_path = tf.io.gfile.join(FLAGS.model_dir, f'{output}.pb')
    if not tf.io.gfile.exists(pb_path):
      hlo_utils.output_hlo(computation, pb_path)
      logging.info('Wrote serialized proto %s file.', pb_path)
    txt_path = tf.io.gfile.join(FLAGS.model_dir, f'{output}.txt')
    if not tf.io.gfile.exists(txt_path):
      hlo_utils.output_hlo(computation, txt_path)
      logging.info('Wrote HLO-text %s file.', txt_path)
  except Exception:  # pylint: disable=broad-except
    logging.exception('Exception in _write_hlo.')


def eval_step(params, batch, state,
              transformer_kwargs,
              hparams,
              quant_context):
  """Calculate evaluation metrics on a batch."""
  inputs, targets = batch['inputs'], batch['targets']
  weights = jnp.where(targets > 0, 1.0, 0.0)
  model = models.Transformer(
      **transformer_kwargs,
      quant_context=quant_context,
      use_bfloat16=FLAGS.use_bfloat16,
      train=False,
      hparams=hparams,
      dropout_rate=0.0,
      attention_dropout_rate=0.0,
      should_decode=False)
  logits = model.apply({
      'params': params,
      **state
  },
                       inputs,
                       targets,
                       mutable=False)
  return compute_metrics(logits, targets, weights)


p_eval_step = jax.pmap(
    eval_step, axis_name='batch', static_broadcasted_argnums=(3, 4))
# Argument 3 is transformer_kwargs. Argument 4 is hparams.


def predict_step(inputs, params, cache, state, eos_token,
                 transformer_kwargs,
                 hparams,
                 quant_context):
  return predict.step(
      inputs,
      params,
      cache,
      state,
      eos_token,
      FLAGS.max_predict_length,
      transformer_kwargs=transformer_kwargs,
      hparams=hparams,
      quant_context=quant_context)


p_pred_step = jax.pmap(
    predict_step, axis_name='batch', static_broadcasted_argnums=(4, 5, 6))
# argument 4: eos token is constant. argument 5: transformer_kwargs. argument 6:
# hparams.


def pad_examples(x, desired_batch_size):
  """Expand batch to desired size by repeating last slice."""
  batch_pad = desired_batch_size - x.shape[0]
  return np.concatenate([x, np.tile(x[-1], (batch_pad, 1))], axis=0)


def tohost(x):
  """Collect batches from all devices to host and flatten batch dimensions."""
  n_device, n_batch, *remaining_dims = x.shape
  return np.array(x).reshape((n_device * n_batch,) + tuple(remaining_dims))


def run_eval(*,
             ds,
             state,
             num_steps,
             params,
             summary_writer = None,
             step = None,
             transformer_kwargs,
             hparams):
  """Compute evaluation metrics for a given dataset."""
  eval_metrics = []
  eval_iter = iter(ds)
  quant_context = get_quant_context(hparams, step, train=False)
  for _, eval_batch in zip(range(num_steps), eval_iter):
    eval_batch = jax.tree_map(lambda x: x._numpy(), eval_batch)  # pylint: disable=protected-access
    eval_batch = common_utils.shard(eval_batch)
    metrics = p_eval_step(params, eval_batch, state, transformer_kwargs,
                          hparams.model_hparams, quant_context)
    eval_metrics.append(metrics)
  eval_metrics = common_utils.get_metrics(eval_metrics)
  eval_metrics_sums = jax.tree_map(jnp.sum, eval_metrics)
  eval_denominator = eval_metrics_sums.pop('denominator')
  eval_summary = jax.tree_map(
      lambda x: x / eval_denominator,  # pylint: disable=cell-var-from-loop
      eval_metrics_sums)
  eval_summary['perplexity'] = jnp.clip(
      jnp.exp(eval_summary['loss']), a_max=1.0e4)
  if summary_writer is not None:
    if jax.host_id() == 0:
      for key, val in eval_summary.items():
        summary_writer.scalar(key, val, step)
      summary_writer.flush()
  return eval_summary


# TODO(shivaniagrawal): This is set up to accommodate mocking of the encoder in
# train_test.py, which seems like it should be unnecessary.
def decode_tokens(toks, encoder):
  eos_tokens = (toks == EOS_TOKEN)
  if np.any(eos_tokens):
    valid_toks = toks[:np.argmax(eos_tokens) + 1].astype(np.int32)
  else:
    valid_toks = toks[:].astype(np.int32)
  return encoder.detokenize(valid_toks).numpy().decode('utf-8')


@functools.partial(jax.jit, static_argnums=(0, 1, 2))
def initialize_cache(batch_size, transformer_kwargs,
                     hparams):
  """Initialize a decoding code suitable for `run_inference`."""
  max_decode_len = transformer_kwargs['max_len']
  inputs_shape = (batch_size, 1)
  target_shape = (batch_size, max_decode_len)
  initial_variables = models.Transformer(
      **transformer_kwargs,
      hparams=hparams,
      should_decode=True,
      train=False,
      quant_context=quant_config.QuantContext(
          update_bounds=False, collect_acts_stats=False),
      dropout_rate=0.0,
      attention_dropout_rate=0.0,
      use_bfloat16=False).init(
          jax.random.PRNGKey(0), jnp.ones(inputs_shape, jnp.int32),
          jnp.ones(target_shape, jnp.int32))
  return initial_variables['cache']


def run_inference(*, ds, transformer_kwargs,
                  hparams, state, encoder,
                  params, write_beam_hlo, step):
  """Runs inference (ie, translates) over the given dataset."""

  predict_iter = iter(ds)
  sources, references, predictions = [], [], []
  quant_context = get_quant_context(hparams, step, train=False)
  for _, pred_batch in enumerate(predict_iter):
    pred_batch = jax.tree_map(lambda x: x._numpy(), pred_batch)  # pylint: disable=protected-access
    # Handle final odd-sized batch by padding instead of dropping it.
    cur_pred_batch_size = pred_batch['inputs'].shape[0]
    if cur_pred_batch_size != FLAGS.eval_batch_size:
      logging.info('Translation: uneven batch size %d.', cur_pred_batch_size)
      pred_batch = jax.tree_map(
          lambda x: pad_examples(x, FLAGS.eval_batch_size), pred_batch)
    pred_batch = common_utils.shard(pred_batch)
    per_device_batchsize = pred_batch['inputs'].shape[1]
    unreplicated_cache = initialize_cache(per_device_batchsize,
                                          flax.core.freeze(transformer_kwargs),
                                          hparams.model_hparams)

    if write_beam_hlo:
      logging.info('Writing inference HLO.')
      params = jax_utils.unreplicate(params)
      state = jax_utils.unreplicate(state)
      logging.info('Unreplicated optimizer and state.')
      _write_beam_hlo(
          params=params,
          input_shape=pred_batch['inputs'].shape[1:],
          cache=unreplicated_cache,
          state=state,
          transformer_kwargs=transformer_kwargs,
          hparams=hparams.model_hparams,
          quant_context=quant_context)
      logging.info('Finished writing inference HLO.')
      params = jax_utils.replicate(params)
      state = jax_utils.replicate(state)
      logging.info('Replicated optimizer.')

    cache = jax_utils.replicate(unreplicated_cache)
    predicted = p_pred_step(
        pred_batch['inputs'],
        params,
        cache,
        state,
        EOS_TOKEN,
        transformer_kwargs,
        hparams.model_hparams,
        quant_context=quant_context)
    predicted = tohost(predicted)
    inputs = tohost(pred_batch['inputs'])
    targets = tohost(pred_batch['targets'])
    # Iterate through non-padding examples of batch.
    for i, s in enumerate(predicted[:cur_pred_batch_size]):
      sources.append(decode_tokens(inputs[i], encoder))
      references.append(decode_tokens(targets[i], encoder))
      predictions.append(decode_tokens(s, encoder))
  return sources, references, predictions


# TODO(malmaud): Make these types more precise
# TODO(malmaud): Expand the usage of this class throughout the training loop
@dataclasses.dataclass
class TrainingState:
  """Represents the state of the training loop at any given step."""
  flax_state: Any
  optimizer: Any
  dropout_rngs: Any
  transformer_kwargs: Any

  @classmethod
  def initialize(
      cls,
      encoder,
      hparams,
      transformer_kwargs_overrides = None):
    """Initializes a training state in the form of a TrainingState instance."""
    random_seed = hparams.random_seed
    rng = random.PRNGKey(random_seed)
    rng, init_rng = random.split(rng)
    batch_size = hparams.per_host_batch_size
    max_target_length = FLAGS.max_target_length
    max_eval_target_length = FLAGS.max_eval_target_length
    max_length = max(max_target_length, max_eval_target_length)
    input_shape = (batch_size, max_target_length)
    target_shape = (batch_size, max_target_length)
    vocab_size = int(encoder.vocab_size())

    transformer_kwargs = {
        'vocab_size': vocab_size,
        'output_vocab_size': vocab_size,
        'max_len': max_length,
    }

    if transformer_kwargs_overrides is not None:
      transformer_kwargs.update(transformer_kwargs_overrides)

    transformer_kwargs = flax.core.freeze(transformer_kwargs)
    params, init_state = create_model(init_rng, tuple(input_shape),
                                      tuple(target_shape),
                                      transformer_kwargs,
                                      hparams.model_hparams)
    init_state = jax_utils.replicate(init_state)
    optimizer = create_optimizer(
        params,
        learning_rate=hparams.learning_rate_schedule.base_learning_rate,
        weight_decay=hparams.weight_decay,
        beta1=hparams.beta1,
        beta2=hparams.beta2,
        eps=hparams.eps)

    # We init the first set of dropout PRNG keys, but update it afterwards
    # inside the main pmap'd training update for performance.
    dropout_rngs = random.split(rng, jax.local_device_count())
    return cls(
        optimizer=optimizer,
        flax_state=init_state,
        dropout_rngs=dropout_rngs,
        transformer_kwargs=transformer_kwargs)

  def save_checkpoint(self,
                      *,
                      model_dir,
                      step,
                      prefix = 'checkpoint_'):
    """Saves TrainingState checkpoint to model_dir."""
    unreplicated_flax_state = jax_utils.unreplicate(self.flax_state)
    checkpoints.save_checkpoint(
        model_dir, (self.optimizer, unreplicated_flax_state, self.dropout_rngs,
                    self.transformer_kwargs),
        step,
        prefix=prefix)  # pytype: disable=wrong-arg-types

  def restore_checkpoint(self,
                         *,
                         model_dir,
                         prefix = 'checkpoint_'):
    """Restore TrainingState from checkpoint in model_dir."""
    unreplicated_flax_state = jax_utils.unreplicate(self.flax_state)
    optimizer, flax_state, dropout_rngs, transformer_kwargs = checkpoints.restore_checkpoint(
        model_dir, (self.optimizer, unreplicated_flax_state, self.dropout_rngs,
                    self.transformer_kwargs),
        prefix=prefix)
    flax_state = jax_utils.replicate(flax_state)
    return type(self)(
        flax_state=flax_state,
        optimizer=optimizer,
        dropout_rngs=dropout_rngs,
        transformer_kwargs=transformer_kwargs)


def get_quant_context(hparams, step,
                      train):
  """Returns quantization context for a given training step.

  Args:
    hparams: A TrainingHParams instance.
    step: Current training step.
    train: Boolean of whether this contexet will be used for a training or eval
      step.

  Returns:
    A QuantContext instance that has been replicated so it can be passed to
      a pmapped function.
  """
  if train:
    collect_acts_stats = FLAGS.collect_acts_stats
  else:
    collect_acts_stats = False
  quant_context = train_utils.get_quant_context_for_step(
      activation_bound_update_freq=hparams.activation_bound_update_freq,
      activation_bound_start_step=hparams.activation_bound_start_step,
      step=step,
      collect_acts_stats=collect_acts_stats,
      prefer_int8_to_int32_dot=hparams.prefer_int8_to_int32_dot)
  if not train:
    quant_context = dataclasses.replace(quant_context, update_bounds=False)
  return jax_utils.replicate(quant_context)


def run_train_step(*, training_state, step, batch,
                   hparams):
  """Run a single step of training."""
  quant_context = get_quant_context(hparams, step, train=True)
  # Shard data to devices and do a training step.
  batch = common_utils.shard(jax.tree_map(lambda x: x._numpy(), batch))  # pylint: disable=protected-access
  flax_state, optimizer, metrics, dropout_rngs = p_train_step(
      training_state.optimizer,
      batch,
      quant_context,
      training_state.transformer_kwargs,
      WrapHashably(hparams),
      state=training_state.flax_state,
      dropout_rng=training_state.dropout_rngs)
  new_training_state = TrainingState(
      flax_state=flax_state,
      optimizer=optimizer,
      dropout_rngs=dropout_rngs,
      transformer_kwargs=training_state.transformer_kwargs)
  return new_training_state, metrics


@dataclasses.dataclass
class Datasets:
  """Represents all the datasets used by the training loop.

  Datasets are assumed to share a vocabularity, which is encapsulated in the
  encoder attribute.
  """
  train_ds: tf.data.Dataset  # Training dataset
  # Evaluation dataset dictionary in the format {name: dataset} for calculating
  # perplexity.
  eval_ds_dict: Dict[str, tf.data.Dataset]
  train_eval_ds: tf.data.Dataset  # Sample of training dataset for evaluation
  # Prediction dataset dictionary in the format {name: dataset} for calculating
  # BLEU scores. Same as the eval dataset list, but preprocessed slightly
  # differently.
  predict_ds_dict: Dict[str, tf.data.Dataset]
  # Encoder for the vocabulary shared all the preceding datasets.
  encoder: tftxt.SentencepieceTokenizer

  @classmethod
  def load(cls, random_seed,
           eval_dataset_list, batch_size):
    """Loads all datasets used by the training loop."""
    n_devices = jax.local_device_count()
    vocab_path = FLAGS.vocab_path
    if vocab_path is None:
      vocab_path = os.path.join(FLAGS.model_dir, 'sentencepiece_model')
    train_ds, eval_ds_dict, train_eval_ds, predict_ds_dict, encoder = input_pipeline.get_wmt_datasets(
        n_devices=n_devices,
        dataset_name=FLAGS.dataset_name,
        eval_dataset_list=eval_dataset_list,
        shard_idx=jax.host_id(),
        shard_count=jax.host_count(),
        data_dir=FLAGS.data_dir,
        vocab_path=vocab_path,
        batch_size=batch_size,
        max_length=FLAGS.max_target_length,
        max_eval_length=FLAGS.max_eval_target_length,
        seed=random_seed)
    return cls(
        train_ds=train_ds,
        eval_ds_dict=eval_ds_dict,
        train_eval_ds=train_eval_ds,
        predict_ds_dict=predict_ds_dict,
        encoder=encoder)




def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  # This seems to be necessary even when importing TF2?
  tf.enable_v2_behavior()
  if FLAGS.hparams_config_dict:
    # In this case, there are multiple training configs defined in the config
    # dict, so we pull out the one this training run should use.
    if 'configs' in FLAGS.hparams_config_dict:
      hparams_config_dict = FLAGS.hparams_config_dict.configs[FLAGS.config_idx]
    else:
      hparams_config_dict = FLAGS.hparams_config_dict
    hparams = os_hparams_utils.load_dataclass_from_config_dict(
        training_hparams.TrainingHParams, hparams_config_dict)


  # Number of local devices for this host.
  n_devices = jax.local_device_count()

  if hparams.per_host_batch_size % n_devices:
    raise ValueError(
        f'Batch size must be divisible by the number of devices. Got batch '
        f'size {hparams.per_host_batch_size} for {n_devices}.')

  logging.info('Initializing dataset.')
  eval_dataset_names_list = FLAGS.additional_eval_datasets
  if FLAGS.eval_dataset_name not in eval_dataset_names_list:
    eval_dataset_names_list.append(FLAGS.eval_dataset_name)

  datasets = Datasets.load(
      hparams.random_seed,
      eval_dataset_list=eval_dataset_names_list,
      batch_size=hparams.per_host_batch_size)

  with concurrent.futures.ThreadPoolExecutor(max_workers=1) as io_executor:
    run_training(datasets=datasets, hparams=hparams, io_executor=io_executor)



def save_best_checkpoint(*, model_dir, training_state,
                         loss):
  """Saves a checkpoint if 'loss' is smaller than the loss of the previous best-model checkpoint."""

  prefix = 'best_checkpoint_eval_loss_'

  # If the loss is NaN or inf, then this isn't the best checkpoint. We
  # explicitly return early since 'checkpoints.save_checkpoint' would overwrite
  # the best checkpoint with a bad checkpoint.
  if not jnp.isfinite(loss):
    return

  # Scan the existing checkpoints and see if any correspond to a lower loss. If
  # so, this isn't the best checkpoint so return. Otherwise, we have a new best
  # checkpoint so delete the current one and write a new one.

  for dir_name, _, file_names in tf.io.gfile.walk(model_dir):
    for filename in file_names:
      # This regex will match a filename like 'best_checkpoint_eval_loss_1.5'
      # and will extract '1.5' into 'current_best_loss'.
      checkpoint_match = re.match(rf'.*{prefix}(.*)$', str(filename))
      if checkpoint_match is not None:
        current_best_loss = float(checkpoint_match.group(1))
        if loss >= current_best_loss:
          return
        else:
          file_path = tf.io.gfile.join(dir_name, filename)
          tf.io.gfile.remove(file_path)

  training_state.save_checkpoint(model_dir=model_dir, step=loss, prefix=prefix)


def get_state_dict_keys_from_flags():
  """Returns key suffixes to look up in flax state dict."""
  state_dict_keys = []
  if FLAGS.visualize_acts_bound:
    state_dict_keys.append('bounds')
  if FLAGS.collect_acts_stats:
    # State names from googlex.positron.tensorflow.jax.aqt.stats_tag.StatsTag
    state_dict_keys.extend([
        'min_per_ch', 'max_per_ch', 'mean_per_ch', 'stddev_per_ch',
        'absdev_per_ch', 'stddev_per_ch_uncentered', 'absdev_per_ch_uncentered'
    ])
  return state_dict_keys




def eval_ds_name_to_summary_dir(eval_dataset_name):
  return 'eval_' + str('--'.join(os.path.abspath(eval_dataset_name).parts))


def does_checkpoint_exist(model_dir):
  """Determine if a checkpoint for this training run has already been saved."""
  # The model directory where checkpoints are saved doesn't exist yet.
  if not tf.io.gfile.exists(model_dir):
    return False
  for _, _, file_names in tf.io.gfile.walk(model_dir):
    for file in file_names:
      # This will match a checkpoint with a filename matching checkpoint_{step
      # number}, indicating a checkpoint already exists.
      if re.match(r'checkpoint_(\d+)', file):
        return True
  return False


def run_training(
    datasets,
    hparams,
    io_executor,
    transformer_kwargs_overrides = None,
):
  """Run the main training loop until completion."""
  os_hparams_utils.write_hparams_to_file_with_host_id_check(
      hparams, FLAGS.model_dir)

  # Use hardware RNG for bernoulli randoms in dropout mask creation.
  if hparams.hardware_rng:
    models.set_hardware_bernoulli()

  batch_size = hparams.per_host_batch_size
  num_train_steps = hparams.num_train_steps
  logging.info('num train steps: %d', num_train_steps)
  num_eval_steps = FLAGS.num_eval_steps
  eval_freq = FLAGS.eval_frequency
  stats_serialization_freq = FLAGS.stats_serialization_freq

  train_summary_writer = None
  eval_summary_writer = None
  additional_eval_summary_writers = {}
  train_eval_summary_writer = None
  if jax.host_id() == 0:
    model_dir = FLAGS.model_dir
    train_summary_writer = tensorboard.SummaryWriter(model_dir + '/train')
    eval_summary_writer = tensorboard.SummaryWriter(model_dir + '/eval')
    train_eval_summary_writer = tensorboard.SummaryWriter(model_dir +
                                                          '/train_eval')
    for eval_dataset_name in datasets.eval_ds_dict:
      if eval_dataset_name == FLAGS.eval_dataset_name:
        continue
      additional_eval_summary_writers[
          eval_dataset_name] = tensorboard.SummaryWriter(
              model_dir + '/' + eval_ds_name_to_summary_dir(eval_dataset_name))

  local_device_count = jax.local_device_count()
  if (batch_size % local_device_count or
      FLAGS.eval_batch_size % local_device_count):
    raise ValueError(
        f'Batch size must be divisible by the number of devices. Training '
        f'batch_size is {batch_size} and eval_batch_size is '
        f'{FLAGS.eval_batch_size} for number of devices {local_device_count}.'
    )

  train_iter = iter(datasets.train_ds)
  start_step = 0
  training_state = TrainingState.initialize(
      encoder=datasets.encoder,
      hparams=hparams,
      transformer_kwargs_overrides=transformer_kwargs_overrides)

  # Load model state from a checkpoint. Summary of model restoration flowchart:
  # 1) If the training run is just starting up and so hasn't made any progress
  #    yet (as determined by the absence of a checkpoint in model_dir),
  #    we check if the user specified an initial checkpoint to start training
  #    from via the `restore_checkpoint_model_dir` flag.
  #
  #   1a) If so, we load that checkpoint.
  #
  #   1b) If not, we start the training from scratch.
  #
  # 2) If the training run was already in progress but was pre-empted by
  #    XManager and is now resuming (as determined by the presence of a saved
  #    checkpoint in the model directory), we resume training based on the
  #    latest saved checkpoint.
  #
  # In a typical usecase, 'restore_checkpoint_model_dir' might point to a
  # bfloat16 model directory trained to convergence and this training run will
  # perform quantization-aware finetuning of that model.
  checkpoint_model_dir = None
  if FLAGS.restore_checkpoint_model_dir is not None and not does_checkpoint_exist(
      FLAGS.model_dir):
    checkpoint_model_dir = FLAGS.restore_checkpoint_model_dir
  elif FLAGS.restore_checkpoints:
    if not FLAGS.save_checkpoints:
      raise ValueError(
          'If restore_checkpoints is enabled, then save_checkpoints must be enabled as well.'
      )
    checkpoint_model_dir = FLAGS.model_dir
  if checkpoint_model_dir is not None:
    logging.info('Restoring checkpoint from: %s', checkpoint_model_dir)
    training_state = training_state.restore_checkpoint(
        model_dir=checkpoint_model_dir)
    # Grab last step from the first of the optimizer replicas.
    start_step = int(training_state.optimizer.state.step[0])

  transformer_kwargs = training_state.transformer_kwargs
  metrics_all = []
  state_dict_summary_all = []
  state_dict_keys = get_state_dict_keys_from_flags()

  wrote_beam_hlo = False

  # Save the HLO and estimate compute / memory costs
  if jax.host_id() == 0:
    logging.info('Writing training HLO and estimating compute/memory costs.')
    state = training_state.flax_state
    optimizer = training_state.optimizer
    optimizer = optimizer.unreplicate()
    state = jax_utils.unreplicate(state)
    logging.info('Unreplicated optimizer and state.')
    _write_train_hlo(
        params=optimizer.target,
        state=state,
        hparams=hparams,
        transformer_kwargs=transformer_kwargs)
    logging.info('Wrote training HLO.')
    if FLAGS.estimate_compute_and_memory_cost:
      estimate_compute_and_memory_cost(
          FLAGS.model_dir,
          params=optimizer.target,
          state=state,
          hparams=hparams,
          transformer_kwargs=transformer_kwargs)
    state = jax_utils.replicate(state)
    optimizer = optimizer.replicate()
    logging.info('Replicated optimizer.')

  t_loop_start = time.time()
  t_train = timer.MultiIntervalTimer()
  t_train.Start()
  for step, batch in zip(range(start_step, num_train_steps), train_iter):
    # Shard data to devices and do a training step.
    training_state, metrics = run_train_step(
        training_state=training_state, batch=batch, step=step, hparams=hparams)
    state = training_state.flax_state
    optimizer = training_state.optimizer
    metrics_all.append(metrics)
    state_dict_summary = summary_utils.get_state_dict_summary(
        state, state_dict_keys)
    state_dict_summary_all.append(state_dict_summary)

    # Save a Checkpoint
    if step % FLAGS.checkpoint_freq == 0 and step > 0:
      if jax.host_id() == 0 and FLAGS.save_checkpoints:
        training_state.save_checkpoint(model_dir=FLAGS.model_dir, step=step)


    # Periodic metric handling.
    if step % eval_freq == 0:
      t_train.Stop()

      # Training Metrics
      if FLAGS.compute_train_metrics:
        metrics_all = common_utils.get_metrics(metrics_all)
        state_dict_summary_all = common_utils.get_metrics(
            state_dict_summary_all)
        lr = metrics_all.pop('learning_rate').mean()
        metrics_sums = jax.tree_map(jnp.sum, metrics_all)
        denominator = metrics_sums.pop('denominator')
        or_loss = metrics_sums.pop('or_loss').mean()
        summary = jax.tree_map(lambda x: x / denominator, metrics_sums)  # pylint: disable=cell-var-from-loop
        summary['learning_rate'] = lr
        summary['or_loss'] = or_loss
        summary['perplexity'] = jnp.clip(jnp.exp(summary['loss']), a_max=1.0e4)
        logging.info('train in step: %d, loss: %.4f', step, summary['loss'])
        steps_per_eval = eval_freq if step != 0 else 1
        steps_per_sec = steps_per_eval / (time.time() - t_loop_start)
        train_steps_per_sec = steps_per_eval / t_train.GetDuration()
        t_loop_start = time.time()

        if jax.host_id() == 0:
          assert train_summary_writer is not None, ('train_summary_writer was '
                                                    'not initialized on host 0')
          train_summary_writer.scalar('steps per second', steps_per_sec, step)
          train_summary_writer.scalar('training steps per second',
                                      train_steps_per_sec, step)
          for key, val in summary.items():
            train_summary_writer.scalar(key, val, step)

          summary_utils.write_state_dict_summaries_to_tb(
              state_dict_summary_all, train_summary_writer,
              FLAGS.state_dict_summary_freq, step)
          train_summary_writer.flush()

      state_dict_summary_all = []
      metrics_all = []
      logging.info('train time: %.4f s step %d', t_train.GetDuration(), step)

      # TODO(shivaniagrawal): Add bleu score summaries for additional eval
      # datasets?
      # Additional eval metrics
      for eval_dataset_name, eval_ds in datasets.eval_ds_dict.items():
        summary_writer = None
        t_add_eval_start = time.time()
        if eval_dataset_name == FLAGS.eval_dataset_name:
          summary_writer = eval_summary_writer
        elif eval_dataset_name in additional_eval_summary_writers:
          summary_writer = additional_eval_summary_writers[eval_dataset_name]
        eval_summary = run_eval(
            ds=eval_ds,
            state=state,
            num_steps=num_eval_steps,
            params=optimizer.target,
            summary_writer=summary_writer,
            step=step,
            hparams=hparams,
            transformer_kwargs=transformer_kwargs)
        logging.info('eval for %s in step: %d, loss: %.4f', eval_dataset_name,
                     step, eval_summary['loss'])
        logging.info('eval time for %s: %.4f s step %d', eval_dataset_name,
                     time.time() - t_add_eval_start, step)

        # Save a checkpoint corresponding to when loss on the eval set is
        # minimized.
        if jax.host_id() == 0 and (eval_dataset_name == FLAGS.eval_dataset_name
                                   and FLAGS.save_minimum_loss_checkpoint):
          save_best_checkpoint(
              model_dir=FLAGS.model_dir,
              training_state=training_state,
              loss=eval_summary['loss'])

      if FLAGS.run_train_eval:
        # Eval-Train Metrics (Eval on the train set)
        # TODO(b/156127248): Might want to add odd-sized remainders to this
        # evaluation as well.
        t_train_eval_start = time.time()
        train_eval_summary = run_eval(
            ds=datasets.train_eval_ds,
            state=state,
            num_steps=num_eval_steps,
            params=optimizer.target,
            summary_writer=train_eval_summary_writer,
            step=step,
            hparams=hparams,
            transformer_kwargs=transformer_kwargs)
        logging.info('train eval in step: %d, loss: %.4f', step,
                     train_eval_summary['loss'])
        logging.info('train eval time: %.4f s step %d',
                     time.time() - t_train_eval_start, step)

      # Translation and BLEU Score.
      t_inference_start = time.time()
      should_write_beam_hlo = jax.host_id() == 0 and (not wrote_beam_hlo)
      sources, references, predictions = run_inference(
          ds=datasets.predict_ds_dict[FLAGS.eval_dataset_name],
          hparams=hparams,
          state=state,
          encoder=datasets.encoder,
          params=optimizer.target,
          write_beam_hlo=should_write_beam_hlo,
          transformer_kwargs=transformer_kwargs,
          step=step)
      if should_write_beam_hlo:
        wrote_beam_hlo = True
      logging.info('inference time: %.4f s step %d.',
                   time.time() - t_inference_start, step)
      logging.info('Translation: %d predictions %d references %d sources.',
                   len(predictions), len(references), len(sources))

      # Calculate BLEU score for translated eval corpus against reference.
      t_bleu_start = time.time()
      bleu_score = bleu.bleu_local(references, predictions)
      logging.info('bleu time: %.4f s step %d',
                   time.time() - t_bleu_start, step)
      t_bleu_start = time.time()
      sacrebleu_score = sacrebleu.corpus_bleu(predictions, [references]).score
      logging.info('sacrebleu time: %.4f s step %d',
                   time.time() - t_bleu_start, step)
      # Save translation samples for tensorboard.
      exemplars = ''
      for n in np.random.choice(np.arange(len(predictions)), 8):
        exemplars += f'{sources[n]}\n\n{references[n]}\n\n{predictions[n]}\n\n'
      if jax.host_id() == 0:
        assert eval_summary_writer is not None, ('eval_summary_writer was not '
                                                 'initialized on host 0')
        eval_summary_writer.scalar('bleu', bleu_score, step)
        eval_summary_writer.scalar('sacrebleu', sacrebleu_score, step)
        eval_summary_writer.text('samples', exemplars, step)
        eval_summary_writer.flush()

      # restart training-only timer
      t_train.Reset()
      t_train.Start()


if __name__ == '__main__':
  app.run(main)
