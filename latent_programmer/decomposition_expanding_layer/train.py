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

# python3
"""Train seq-to-seq model on random supervised training tasks."""

# pylint: disable=g-bare-generic
# pytype: disable=wrong-arg-count
# pytype: disable=attribute-error

import collections
import functools
import json
import os
import queue
import random
import time
from typing import Callable

from absl import app
from absl import flags
from absl import logging
from flax import jax_utils
from flax import linen as nn
from flax import optim
from flax import traverse_util
from flax.metrics import tensorboard
from flax.training import checkpoints
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow.compat.v2 as tf

from latent_programmer import decode
from latent_programmer import models as base_models
from latent_programmer.decomposition_expanding_layer import decomposition_models as models
from latent_programmer.decomposition_expanding_layer import input_pipeline
from latent_programmer.tasks.robust_fill import dsl
from latent_programmer.tasks.robust_fill import tokens as dsl_tokens

gfile = tf.io.gfile

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 0, 'Fixed random seed for training.')
flags.DEFINE_float('lr', 1e-3, 'Learning rate.')
flags.DEFINE_float('weight_decay', 1e-1,
                   'Decay factor for AdamW-style weight decay.')
flags.DEFINE_integer('embedding_dim', 128, 'Embedding dimension.')
flags.DEFINE_integer('hidden_dim', 512, 'Hidden dimension.')
flags.DEFINE_integer('num_heads', 4, 'Number of layers.')
flags.DEFINE_integer('num_layers', 3, 'Number of Transformer heads.')
flags.DEFINE_integer('num_partial_programs', 1,
                     'Number of partial programs to decompose into.')
flags.DEFINE_boolean('best_first_search', False,
                     'Use best-first search (True) or beam search (False).')
flags.DEFINE_boolean('slow_decode', True, 'Use slow decoding for predction?')
flags.DEFINE_boolean('use_expanding_layer', True, 'Use expanding layer?')

flags.DEFINE_string('dataset_filepattern', None,
                    'Filepattern for TFRecord dataset.')
flags.DEFINE_integer('per_device_batch_size', 16,
                     'Number of program tasks in a batch.')
flags.DEFINE_integer('num_strings_per_task', 4,
                     'Number of input/output strings per task.')
flags.DEFINE_integer('max_expressions', 6,
                     'Maximum number of expressions in program.')
flags.DEFINE_integer('max_program_length', 50,
                     'Maximum number of tokens in program.')
flags.DEFINE_integer('max_characters', 100,
                     'Maximum number of characters in input/output strings.')

flags.DEFINE_string('save_dir', None, 'Directory to save results to.')
flags.DEFINE_integer('num_train_steps', 1500000, 'Number of training steps.')
flags.DEFINE_integer('num_eval_steps', 100, 'Number of evaluation steps.')
flags.DEFINE_integer('log_freq', 10000, 'Number of steps between logs.')
flags.DEFINE_integer('checkpoint_freq', 10000,
                     'Number of steps between checkpoint saves.')
flags.DEFINE_bool('restore_checkpoints', True,
                  'Whether to restore from existing model checkpoints.')

flags.DEFINE_integer('num_pretrain_steps', 0, 'Number of pretraining steps.')
flags.DEFINE_string('pretrain_checkpoint_format', None,
                    'Pretrained model checkpoint format (with a {} to be '
                    'replaced with the number of expressions)')
flags.DEFINE_bool('restore_encoder', True,
                  'Restore the encoder from the pretrained checkpoint?')
flags.DEFINE_bool('restore_decoder', True,
                  'Restore the decoder from the pretrained checkpoint?')
flags.DEFINE_bool('freeze_encoder', False, 'Freeze encoder during training.')
flags.DEFINE_bool('freeze_decoder', False, 'Freeze decoder during training.')
flags.DEFINE_bool('match_split_encoding', False,
                  'Train the encoder to match the split ones?')
flags.DEFINE_float('alpha_encoding', 1,
                   'Scaling hyperparameter for the encoding loss.')

flags.DEFINE_bool('use_relative_attention', True,
                  'Whether to use relative positonal embeddings.')
flags.DEFINE_integer('num_relative_position_buckets', 32,
                     'Number of buckets when computing relative positions.')


def create_learning_rate_scheduler(
    base_learning_rate=0.5,
    factors='constant * linear_warmup * rsqrt_normalized_decay',
    warmup_steps=16000,
    decay_factor=0.5,
    steps_per_decay=50000,
    steps_per_cycle=100000):
  """Creates learning rate schedule.

  Interprets factors in the factors string which can consist of:
  * constant: interpreted as the constant value,
  * linear_warmup: interpreted as linear warmup until warmup_steps,
  * rsqrt_decay: divide by square root of max(step, warmup_steps)
  * decay_every: Every k steps decay the learning rate by decay_factor.
  * cosine_decay: Cyclic cosine decay, uses steps_per_cycle parameter.

  Args:
    base_learning_rate: float, the starting constant for the lr schedule.
    factors: a string with factors separated by '*' that defines the schedule.
    warmup_steps: how many steps to warm up for in the warmup schedule.
    decay_factor: The amount to decay the learning rate by.
    steps_per_decay: How often to decay the learning rate.
    steps_per_cycle: Steps per cycle when using cosine decay.

  Returns:
    A function learning_rate(step): float -> {'learning_rate': float}, the
    step-dependent lr.
  """
  factors = [n.strip() for n in factors.split('*')]

  def step_fn(step):
    """Step to learning rate function."""
    ret = 1.0
    for name in factors:
      if name == 'constant':
        ret *= base_learning_rate
      elif name == 'linear_warmup':
        ret *= jnp.minimum(1.0, step / warmup_steps)
      elif name == 'rsqrt_decay':
        ret /= jnp.sqrt(jnp.maximum(1.0, step - warmup_steps))
      elif name == 'rsqrt_normalized_decay':
        ret *= jnp.sqrt(warmup_steps)
        ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
      elif name == 'decay_every':
        ret *= (decay_factor**(step // steps_per_decay))
      elif name == 'cosine_decay':
        progress = jnp.maximum(0.0,
                               (step - warmup_steps) / float(steps_per_cycle))
        ret *= jnp.maximum(0.0,
                           0.5 * (1.0 + jnp.cos(jnp.pi * (progress % 1.0))))
      else:
        raise ValueError('Unknown factor %s.' % name)
    return jnp.asarray(ret, dtype=jnp.float32)

  return step_fn


def compute_weighted_cross_entropy(logits, targets, weights=None):
  """Compute weighted cross entropy and entropy for log probs and targets.

  Args:
   logits: `[batch, length, num_classes]` float array.
   targets: categorical targets `[batch, length]` int array.
   weights: None or array of shape [batch, length]

  Returns:
    Tuple of scalar loss and batch normalizing factor.
  """
  if logits.ndim != targets.ndim + 1:
    raise ValueError('Incorrect shapes. Got shape %s logits and %s targets' %
                     (str(logits.shape), str(targets.shape)))

  onehot_targets = common_utils.onehot(targets, logits.shape[-1])
  loss = -jnp.sum(onehot_targets * nn.log_softmax(logits), axis=-1)
  normalizing_factor = jnp.prod(jnp.asarray(targets.shape))
  if weights is not None:
    loss = loss * weights
    normalizing_factor = weights.sum()

  return loss.sum(), normalizing_factor


def compute_weighted_squared_error(predicted, targets, weights=None):
  """Compute weighted squared error for predicted and targets.

  Args:
   predicted: `[batch, length, dim]` float array.
   targets: `[batch, length, dim]` float array.
   weights: None or array of shape [batch, length]

  Returns:
    Scalar loss.
  """
  if predicted.ndim != targets.ndim:
    raise ValueError('Incorrect shapes. Got shape %s logits and %s targets' %
                     (str(predicted.shape), str(targets.shape)))

  loss = jnp.sum(jnp.square(predicted - targets), axis=-1)
  normalizing_factor = jnp.prod(jnp.asarray(targets.shape[:1]))
  if weights is not None:
    loss = loss * weights
    normalizing_factor = weights.sum()

  return loss.sum(), normalizing_factor


def compute_weighted_accuracy(logits, targets, weights=None):
  """Compute weighted accuracy for log probs and targets.

  Args:
   logits: `[batch, length, num_classes]` float array.
   targets: categorical targets `[batch, length]` int array.
   weights: None or array of shape [batch, length, 1]

  Returns:
    Tuple of scalar accuracy and batch normalizing factor.
  """
  if logits.ndim != targets.ndim + 1:
    raise ValueError('Incorrect shapes. Got shape %s logits and %s targets' %
                     (str(logits.shape), str(targets.shape)))
  acc = jnp.equal(jnp.argmax(logits, axis=-1), targets)
  normalizing_factor = jnp.prod(jnp.asarray(targets.shape))
  if weights is not None:
    acc = acc * weights
    normalizing_factor = weights.sum()

  return acc.sum(), normalizing_factor


def compute_metrics(logits, targets, weights):
  """Compute summary metrics."""
  loss, weight_sum = compute_weighted_cross_entropy(logits, targets, weights)
  acc, _ = compute_weighted_accuracy(logits, targets, weights)
  metrics = {
      'loss': loss,
      'accuracy': acc,
      'denominator': weight_sum,
  }
  metrics = jax.lax.psum(metrics, 'batch')
  return metrics


# Pretrain / train / eval / decode step functions.
# -----------------------------------------------------------------------------


def pretrain_step(optimizer,
                  inputs,
                  outputs,
                  programs,
                  num_partial_programs,
                  learning_rate_fn,
                  config,
                  use_expanding_layer,
                  split_params=None,  # Pretrained split model parameters.
                  split_outputs=None,  # Outputs split by partial program.
                  pretrain_rng=None):
  """Pretrain on batch of program tasks."""
  pretrain_rng, new_pretrain_rng = jax.random.split(pretrain_rng)

  weights = jnp.where(programs > 0, 1, 0).astype(jnp.float32)

  match_split_encoding = ((split_params is not None) and
                          (split_outputs is not None))
  if match_split_encoding:
    # Get the i/o encodings of pretrained split model.
    split_inputs = models.add_and_tile_dim(inputs, num_partial_programs, axis=1)
    # split_outputs shape == [batch_size, num_partial, num_io, length]
    split_outputs = jnp.swapaxes(split_outputs, 1, 2)
    split_encoded = models.DecomposeExpandingLayerTransformer(
        config=config.replace(deterministic=True), num_partial_programs=1,
        use_expanding_layer=False).apply(
            {'params': split_params},
            split_inputs,
            split_outputs,
            method=models.DecomposeExpandingLayerTransformer.encode)
    split_encoded_weights = jnp.where(
        split_outputs > 0, 1, 0).astype(jnp.float32)

  def loss_fn(params):
    """Loss function used for training."""
    logits = models.DecomposeExpandingLayerTransformer(
        config=config, num_partial_programs=num_partial_programs,
        use_expanding_layer=use_expanding_layer).apply(
            {'params': params},
            inputs,
            outputs,
            programs,
            rngs={'dropout': pretrain_rng})
    ce_loss, weight_sum = compute_weighted_cross_entropy(
        logits, programs, weights)
    mean_ce_loss = ce_loss / weight_sum

    mean_encoded_loss = 0
    if match_split_encoding:
      encoded = models.DecomposeExpandingLayerTransformer(
          config=config, num_partial_programs=num_partial_programs,
          use_expanding_layer=True).apply(
              {'params': params},
              inputs,
              outputs,
              rngs={'dropout': pretrain_rng},
              method=models.DecomposeExpandingLayerTransformer.encode)
      encoded = models.DecomposeExpandingLayerTransformer(
          config=config, num_partial_programs=num_partial_programs,
          use_expanding_layer=True).apply(
              {'params': params},
              encoded,
              rngs={'dropout': pretrain_rng},
              method=models.DecomposeExpandingLayerTransformer.decompose)
      encoded_loss, weight_sum = compute_weighted_squared_error(
          encoded, split_encoded, split_encoded_weights)
      mean_encoded_loss = encoded_loss / weight_sum

    mean_loss = mean_ce_loss + FLAGS.alpha_encoding * mean_encoded_loss
    return mean_loss, (logits, mean_ce_loss, mean_encoded_loss)

  step = optimizer.state.step
  lr = learning_rate_fn(step)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, (logits, _, encoded_loss)), grad = grad_fn(optimizer.target)
  grad = jax.lax.pmean(grad, 'batch')
  new_optimizer = optimizer.apply_gradient(grad, learning_rate=lr)

  # Get metrics.
  metrics = compute_metrics(logits, programs, weights)
  metrics['learning_rate'] = lr
  if match_split_encoding:
    metrics['encoded_loss'] = encoded_loss

  return new_optimizer, metrics, new_pretrain_rng


def train_step(optimizer,
               inputs,
               outputs,
               programs,
               num_partial_programs,
               learning_rate_fn,
               config,
               use_expanding_layer,
               train_rng=None):
  """Train on batch of program tasks."""
  # We handle PRNG splitting inside the top pmap, rather
  # than handling it outside in the training loop - doing the
  # latter can add some stalls to the devices.
  train_rng, new_train_rng = jax.random.split(train_rng)

  weights = jnp.where(programs > 0, 1, 0).astype(jnp.float32)

  def loss_fn(params):
    """Loss function used for training."""
    logits = models.DecomposeExpandingLayerTransformer(
        config=config, num_partial_programs=num_partial_programs,
        use_expanding_layer=use_expanding_layer).apply(
            {'params': params},
            inputs,
            outputs,
            programs,
            rngs={'dropout': train_rng})
    loss, weight_sum = compute_weighted_cross_entropy(logits, programs, weights)
    mean_loss = loss / weight_sum
    return mean_loss, logits

  step = optimizer.state.step
  lr = learning_rate_fn(step)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grad = grad_fn(optimizer.target)
  grad = jax.lax.pmean(grad, 'batch')
  new_optimizer = optimizer.apply_gradient(grad, learning_rate=lr)

  # Get metrics.
  metrics = compute_metrics(logits, programs, weights)
  metrics['learning_rate'] = lr

  return new_optimizer, metrics, new_train_rng


def eval_step(params, inputs, outputs, programs, num_partial_programs,
              eos_token, config, use_expanding_layer):
  """Evaluate on batch of program tasks."""
  weights = jnp.where(
      jnp.logical_and(programs > 0,
                      jnp.logical_and(programs != config.bos_token,
                                      programs != eos_token)),
      1, 0).astype(jnp.float32)

  m = models.DecomposeExpandingLayerTransformer(
      config=config, num_partial_programs=num_partial_programs,
      use_expanding_layer=use_expanding_layer)
  logits = m.apply({'params': params}, inputs, outputs, programs)

  return compute_metrics(logits, programs, weights)


def initialize_cache(inputs, outputs, programs, num_partial_programs,
                     max_decode_len, config, use_expanding_layer):
  """Initialize a cache for a given input shape and max decode length."""
  target_shape = programs.shape[:-1] + (max_decode_len,)

  m = models.DecomposeExpandingLayerTransformer(
      config=config, num_partial_programs=num_partial_programs,
      use_expanding_layer=use_expanding_layer)
  initial_variables = m.init(
      jax.random.PRNGKey(0),
      jnp.ones(inputs.shape, config.dtype),
      jnp.ones(outputs.shape, config.dtype),
      jnp.ones(target_shape, config.dtype))
  return initial_variables['cache']


def predict_step(params,
                 inputs,
                 outputs,
                 cache,
                 beam_size,
                 num_partial_programs,
                 max_decode_len,
                 eos_token,
                 config,
                 use_expanding_layer,
                 slow_decode=False,
                 use_split_encoding=False,
                 split_params=None,
                 split_outputs=None):
  """Predict translation with fast decoding beam search on a batch."""
  per_partial_beam_size = max(beam_size // num_partial_programs, 1)

  m = models.DecomposeExpandingLayerTransformer(
      config=config, num_partial_programs=num_partial_programs,
      use_expanding_layer=use_expanding_layer)
  # Prepare transformer fast-decoder call for beam search: for beam search, we
  # need to set up our decoder model to handle a batch size equal to
  # batch_size * beam_size, where each batch item's data is expanded in-place
  # rather than tiled.
  if use_split_encoding:
    # Use pretrained split model to get encodings
    assert (split_params is not None) and (split_outputs is not None)

    split_inputs = models.add_and_tile_dim(inputs, num_partial_programs, axis=1)
    # split_outputs shape == [batch_size, num_partial, num_io, length]
    split_outputs = jnp.swapaxes(split_outputs, 1, 2)
    encoded = models.DecomposeExpandingLayerTransformer(
        config=config, num_partial_programs=1,
        use_expanding_layer=False).apply(
            {'params': split_params},
            split_inputs,
            split_outputs,
            method=models.DecomposeExpandingLayerTransformer.encode)
    flat_encoded = decode.flat_batch_beam_expand(encoded, per_partial_beam_size)

    encoded_padding_mask = jnp.where(
        split_outputs > 0, 1, 0).astype(jnp.float32)
    flat_encoded_padding_mask = decode.flat_batch_beam_expand(
        encoded_padding_mask, per_partial_beam_size)
  else:
    flat_encoded = decode.flat_batch_beam_expand(
        m.apply(
            {'params': params},
            inputs,
            outputs,
            method=models.DecomposeExpandingLayerTransformer.encode),
        per_partial_beam_size)
    flat_encoded = m.apply(
        {'params': params},
        flat_encoded,
        method=models.DecomposeExpandingLayerTransformer.decompose)

    encoded_padding_mask = jnp.where(outputs > 0, 1, 0).astype(jnp.float32)
    flat_encoded_padding_mask = decode.flat_batch_beam_expand(
        encoded_padding_mask, per_partial_beam_size)
    flat_encoded_padding_mask = models.add_and_tile_dim(
        flat_encoded_padding_mask, num_partial_programs, axis=1)

  if slow_decode:
    def tokens_ids_to_logits(flat_ids, i):
      """Token slice to logits from decoder model."""
      # --> [batch * beam, 1, vocab]
      flat_logits = models.DecomposeExpandingLayerTransformer(
          config=config, num_partial_programs=num_partial_programs,
          use_expanding_layer=use_expanding_layer).apply(
              {'params': params},
              flat_ids,
              flat_encoded[:, i],
              flat_encoded_padding_mask[:, i],
              method=models.DecomposeExpandingLayerTransformer.decode)
      return flat_logits
  else:
    def tokens_ids_to_logits(flat_ids, flat_cache, i):
      """Token slice to logits from decoder model."""
      # --> [batch * beam, 1, vocab]
      flat_logits, new_vars = models.DecomposeExpandingLayerTransformer(
          config=config, num_partial_programs=num_partial_programs,
          use_expanding_layer=use_expanding_layer).apply(
              {'params': params, 'cache': flat_cache},
              flat_ids,
              flat_encoded[:, i],
              flat_encoded_padding_mask[:, i],
              mutable=['cache'],
              method=models.DecomposeExpandingLayerTransformer.decode)
      new_flat_cache = new_vars['cache']
      # Remove singleton sequence-length dimension:
      # [batch * beam, 1, vocab] --> [batch * beam, vocab]
      flat_logits = flat_logits.squeeze(axis=1)
      return flat_logits, new_flat_cache

  # Perform beam search independently for each partial program.
  all_beam_seqs = []
  all_beam_log_probs = []
  for i in range(num_partial_programs):
    beam_seqs, beam_log_probs = decode.beam_search(
        inputs,
        cache,
        functools.partial(tokens_ids_to_logits, i=i),
        beam_size=per_partial_beam_size,
        alpha=0.6,
        bos_token=config.bos_token,
        eos_token=eos_token,
        max_decode_len=max_decode_len,
        slow_decode=slow_decode)
    all_beam_seqs.append(beam_seqs)
    all_beam_log_probs.append(beam_log_probs)

  all_beam_seqs = jnp.stack(all_beam_seqs, axis=1)
  all_beam_log_probs = jnp.stack(all_beam_log_probs, axis=1)

  # all_beam_seqs shape == [batch, n_partial, n_beam_per_partial, length]
  assert len(all_beam_seqs.shape) == 4
  # all_beam_log_probs shape == [batch, n_partial, n_beam_per_partial]
  assert len(all_beam_log_probs.shape) == 3

  # Sort beams in order of decreasing probability.
  order = jnp.argsort(all_beam_log_probs, axis=2)[:, :, ::-1]
  all_beam_log_probs = jnp.take_along_axis(all_beam_log_probs, order, axis=2)
  all_beam_seqs = jnp.take_along_axis(all_beam_seqs, order[Ellipsis, jnp.newaxis],
                                      axis=2)

  return all_beam_seqs, all_beam_log_probs


# Util functions for evaluation / prediction
# -----------------------------------------------------------------------------


def evaluate(*, p_eval_step, target, eval_ds):
  """Evaluate the target an return a dictionary with the metrics."""
  eval_metrics = []
  for batches in eval_ds.as_numpy_iterator():
    inputs, outputs, programs, _ = common_utils.shard(batches)

    metrics = p_eval_step(target, inputs, outputs, programs)
    eval_metrics.append(metrics)

  eval_metrics = common_utils.get_metrics(eval_metrics)
  eval_metrics_sums = jax.tree_map(jnp.sum, eval_metrics)
  eval_denominator = eval_metrics_sums.pop('denominator')
  eval_summary = jax.tree_map(
      lambda x: x / eval_denominator,  # pylint: disable=cell-var-from-loop
      eval_metrics_sums)
  return eval_summary


def pad_examples(x, desired_batch_size):
  """Expand batch to desired size by repeating last slice."""
  batch_pad = desired_batch_size - x.shape[0]
  tile_dims = [1] * len(x.shape)
  tile_dims[0] = batch_pad
  return np.concatenate([x, np.tile(x[-1], tile_dims)], axis=0)


def tohost(x):
  """Collect batches from all devices to host and flatten batch dimensions."""
  n_device, n_batch, *remaining_dims = x.shape
  return x.reshape((n_device * n_batch,) + tuple(remaining_dims))


def per_host_sum_pmap(in_tree):
  """Execute psum on in_tree's leaves over one device per host."""
  host2devices = collections.defaultdict(list)
  for d in jax.devices():
    host2devices[d.host_id].append(d)
  devices = [host2devices[k][0] for k in host2devices]
  host_psum = jax.pmap(lambda x: jax.lax.psum(x, 'i'), 'i', devices=devices)
  def pre_pmap(xs):
    return jax.tree_map(lambda x: jnp.broadcast_to(x, (1,) + x.shape), xs)
  def post_pmap(xs):
    return jax.tree_map(lambda x: x[0], xs)
  return post_pmap(host_psum(pre_pmap(in_tree)))


def best_first_search(log_probs, beam_size):
  """Finds sequences of partial programs with the highest joint probability."""
  decode_len, num_values = log_probs.shape
  candidates = []
  pq = queue.PriorityQueue()
  pq.put((-np.sum(log_probs[:, 0]), [0] * decode_len))
  seen = set()
  while not pq.empty() and len(candidates) < beam_size:
    unused_top_score, top_candidate = pq.get()
    candidates.append(top_candidate)
    for i in range(decode_len):
      new_candidate = list(top_candidate)
      new_candidate[i] += 1
      new_candidate_tuple = tuple(new_candidate)
      if new_candidate[i] >= num_values or new_candidate_tuple in seen:
        continue
      seen.add(new_candidate_tuple)
      new_score = -np.sum(log_probs[np.arange(decode_len), new_candidate])
      pq.put((new_score, new_candidate))
  return candidates


def beam_decoder(log_probs, beam_size):
  """Finds top sequences from log-probability tensor using beam search."""
  decode_len, num_values = log_probs.shape

  candidates = np.zeros((1, 1), dtype=np.int32)
  candidate_log_probs = np.zeros((1,), dtype=np.int32)

  for i in range(decode_len):
    candidates = np.repeat(candidates, num_values, axis=0)
    next_partial = np.tile(
        np.arange(num_values), (min(beam_size, candidate_log_probs.shape[-1])))
    candidates = np.concatenate([candidates, next_partial[Ellipsis, None]], axis=-1)
    candidate_log_probs = (
        np.repeat(candidate_log_probs, num_values, axis=0) +
        np.tile(
            log_probs[i], (min(beam_size, candidate_log_probs.shape[-1])))
    )
    top_beams = (-candidate_log_probs).argsort(axis=-1)[:beam_size]
    candidates = candidates[top_beams]
    candidate_log_probs = candidate_log_probs[top_beams]

  # return shape == [beam_size, decode_len]
  return candidates[:, 1:]


def compute_score(predicted, inputs, outputs, decode_program):
  """Evaluate predicted program beams."""
  best_p, best_score = None, -1

  # predicted shape [beam_size, length]
  for beam in predicted:
    try:
      p = decode_program(beam)
      p_outs = [p(inp) for inp in inputs]
      score = np.sum([p_out == out for p_out, out in zip(p_outs, outputs)])
      if score > best_score:
        best_p, best_score = p, score
    except:  # pylint: disable=bare-except
      pass
    if best_score >= len(inputs):  # Found solution.
      break
  return best_p, best_score


def predict_and_compute_score(*, p_pred_step, p_init_cache, target,
                              predict_ds,
                              decode_io,
                              decode_program,
                              beam_size,
                              num_partial_programs,
                              use_best_first_search = False,
                              slow_decode = False):
  """Generates program and computes score."""
  n_devices = jax.local_device_count()

  pred_acc = 0
  pred_denominator = 0

  ios, targets, predictions = [], [], []
  for batches in predict_ds.as_numpy_iterator():
    pred_batch = batches
    # Handle final odd-sized batch by padding instead of dropping it.
    cur_pred_batch_size = pred_batch[0].shape[0]
    if cur_pred_batch_size % n_devices:
      padded_size = int(
          np.ceil(cur_pred_batch_size / n_devices) * n_devices)
      # pylint: disable=cell-var-from-loop
      pred_batch = jax.tree_map(
          lambda x: pad_examples(x, padded_size), pred_batch)
    inputs, outputs, programs, split_outputs = common_utils.shard(pred_batch)

    cache = (p_init_cache(inputs, outputs, programs[:, :, 0])
             if not slow_decode else None)
    predicted, log_probs = p_pred_step(target,
                                       inputs,
                                       outputs,
                                       cache,
                                       beam_size,
                                       split_outputs=split_outputs)
    predicted, log_probs = map(tohost, (predicted, log_probs))
    inputs, outputs, programs = map(tohost, (inputs, outputs, programs))

    pred_denominator += programs.shape[0]
    for i, partial_beams in enumerate(predicted):
      inps, outs = decode_io(inputs[i], outputs[i])

      # Find the best orderings of partial programs.
      # partial_seqs shape == [n_beam, n_partial]
      if use_best_first_search:
        partial_seqs = best_first_search(log_probs[i], beam_size)
      else:
        partial_seqs = beam_decoder(log_probs[i], beam_size)
      # beams shape == [n_beam, n_partial, length]
      beams = partial_beams[np.arange(num_partial_programs), partial_seqs]

      # Execute predicted programs on i/o examples.
      p, p_score = compute_score(beams, inps, outs, decode_program)
      if p_score >= len(inps):
        pred_acc += 1
      ios.append(' ; '.join(map(str, zip(inps, outs))))
      targets.append(decode_program(programs[i]).to_string())
      try:
        predictions.append(p.to_string())
      except:  # pylint: disable=bare-except
        predictions.append('')
      logging.info('ios: %s', ios[-1])
      logging.info('target: %s', targets[-1])
      beams_log = []
      for beam in beams:
        try:
          beams_log.append(decode_program(beam).to_string())
        except:  # pylint: disable=bare-except
          beams_log.append('None')
      logging.info('predicted beam: %s', '\n'.join(beams_log))

  all_pred_acc, all_pred_denominator = per_host_sum_pmap(
      jax.tree_map(np.array, (pred_acc, pred_denominator)))

  # Record beam search results as text summaries.
  message = []
  for n in np.random.choice(np.arange(len(predictions)), 8):
    text = (f'ios: {ios[n]}\n\ntarget: {targets[n]}\n\n'
            f'predicted: {predictions[n]}\n\n')
    message.append(text)

  return all_pred_acc / all_pred_denominator, message


# Utility functions for pretraining.
# -----------------------------------------------------------------------------


def flatten_dict(d):
  return {'/'.join(k): v for k, v in traverse_util.flatten_dict(d).items()}


def unflatten_dict(d):
  return traverse_util.unflatten_dict(
      {tuple(k.split('/')): v for k, v in d.items()})


def restore_selected_paths(optimizer, checkpoint_dir, restore_paths):
  """Restores only selected paths from a checkpoint."""
  checkpoint = checkpoints.restore_checkpoint(checkpoint_dir, target=None)
  flat_state_dict = flatten_dict(optimizer.state_dict())
  flat_checkpoint = flatten_dict(checkpoint)

  for key in flat_state_dict:
    for path in restore_paths:
      if key.startswith(path):
        flat_state_dict[key] = flat_checkpoint[key]
        break
  restored_state_dict = unflatten_dict(flat_state_dict)

  return optimizer.restore_state(restored_state_dict)


def maybe_copy_model_from_pretraining(optimizer, pretrain_optimizer, step,
                                      adam_opt_def):
  """Copy model parameters from pretraining."""
  if step < FLAGS.num_pretrain_steps:
    optimizer = jax_utils.unreplicate(optimizer)
    state_dict = adam_opt_def.state_dict(
        target=jax_utils.unreplicate(pretrain_optimizer).target,
        state=optim.OptimizerState(jnp.asarray(step, dtype=jnp.int32),
                                   optimizer.state.param_states))
    optimizer = optimizer.restore_state(state_dict)
    optimizer = jax_utils.replicate(optimizer)
  return optimizer


def shorten(key):
  splits = key.split('_')
  return ''.join(s[0] for s in splits)


# Main training code
def main(_):
  tf.enable_v2_behavior()

  tf.random.set_seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)
  random.seed(FLAGS.seed)

  if not gfile.isdir(FLAGS.save_dir):
    gfile.mkdir(FLAGS.save_dir)

  hparam_str_dict = dict(seed=FLAGS.seed, lr=FLAGS.lr)
  # Get hyperparmaters
  if FLAGS.xm_parameters:
    for key, value in json.loads(FLAGS.xm_parameters).items():
      if key not in hparam_str_dict:
        hparam_str_dict[key] = value

  hparam_str = ','.join(['%s=%s' % (shorten(k), str(hparam_str_dict[k]))
                         for k in sorted(hparam_str_dict.keys())])

  # Number of local devices for this host.
  n_devices = jax.local_device_count()

  if jax.host_id() == 0:
    summary_writer = tensorboard.SummaryWriter(
        os.path.join(FLAGS.save_dir, 'tb', hparam_str))

  batch_size = FLAGS.per_device_batch_size * n_devices
  io_shape = (FLAGS.per_device_batch_size,
              FLAGS.num_strings_per_task,
              FLAGS.max_characters)
  program_shape = (FLAGS.per_device_batch_size,
                   FLAGS.num_partial_programs,
                   FLAGS.max_program_length)
  split_io_shape = (FLAGS.per_device_batch_size,
                    FLAGS.num_strings_per_task,
                    FLAGS.num_partial_programs,
                    FLAGS.max_characters)

  # Setup DSL
  # ---------------------------------------------------------------------------

  # Build token tables.
  id_char_table = {i+1: char for (i, char) in enumerate(dsl.CHARACTER)}
  char_id_table = {char: id for id, char in id_char_table.items()}
  id_token_table, token_id_table = dsl_tokens.build_token_tables()
  io_vocab_size = len(char_id_table) + 1  # For padding.
  program_vocab_size = len(token_id_table) + 1

  bos_token = token_id_table[dsl.BOS]
  eos_token = token_id_table[dsl.EOS]

  # Parse io and program token sequences (for eval).
  def decode_io(inputs, outputs):
    """Decode io examples tokens."""
    def decode_str(s):
      """Decode string tokens."""
      return ''.join([id_char_table[c_id] for c_id in s if c_id > 0])

    inps, outs = [], []
    for inp, out in zip(inputs, outputs):
      inps.append(decode_str(inp))
      outs.append(decode_str(out))
    return inps, outs

  def decode_program(program):
    """Decode program tokens."""
    # Concatenate all partial programs.
    full_program = []
    for p in program:
      full_program.extend(p[:np.argmax(p == eos_token)].astype(np.int32))
    full_program = np.concatenate([full_program, [eos_token]], axis=0)

    try:
      return dsl.decode_program(full_program, id_token_table)
    except:  # pylint: disable=bare-except
      return None  # Program does not compile.

  # Load Dataset
  # ---------------------------------------------------------------------------
  logging.info('Initializing dataset.')
  if not FLAGS.dataset_filepattern:
    raise ValueError('Must specify filepattern to dataset.')

  # Training dataset.
  dataset = input_pipeline.create_dataset_from_tf_record(
      FLAGS.dataset_filepattern,
      token_id_table,
      char_id_table,
      num_partial_programs=FLAGS.num_partial_programs)
  dataset = dataset.padded_batch(
      batch_size,
      padded_shapes=(io_shape[1:], io_shape[1:], program_shape[1:],
                     split_io_shape[1:]),
      drop_remainder=True)
  # Split evaluation and training.
  eval_ds = dataset.take(FLAGS.num_eval_steps)
  # Decrease batch of predict dataset to handle beam search.
  predict_ds = eval_ds.unbatch().padded_batch(
      int(np.ceil(batch_size / 10)),
      padded_shapes=(io_shape[1:], io_shape[1:], program_shape[1:],
                     split_io_shape[1:]))
  train_ds = dataset.skip(FLAGS.num_eval_steps).repeat().prefetch(5)
  train_iter = train_ds.as_numpy_iterator()

  # Build Model and Optimizer
  # ---------------------------------------------------------------------------
  train_config = base_models.TransformerConfig(
      vocab_size=io_vocab_size,
      output_vocab_size=program_vocab_size,
      shift=True,
      emb_dim=FLAGS.embedding_dim,
      num_heads=FLAGS.num_heads,
      num_layers=FLAGS.num_layers,
      qkv_dim=FLAGS.embedding_dim,
      mlp_dim=FLAGS.hidden_dim,
      max_len=max(FLAGS.max_characters, FLAGS.max_program_length),
      use_relative_attention=FLAGS.use_relative_attention,
      num_relative_position_buckets=FLAGS.num_relative_position_buckets,
      deterministic=False,
      decode=False,
      bos_token=bos_token)
  eval_config = train_config.replace(deterministic=True)
  predict_config = train_config.replace(
      shift=False, deterministic=True, decode=not FLAGS.slow_decode)

  rng = jax.random.PRNGKey(FLAGS.seed)
  rng = jax.random.fold_in(rng, jax.host_id())
  rng, init_rng = jax.random.split(rng)

  m = models.DecomposeExpandingLayerTransformer(
      config=eval_config, num_partial_programs=FLAGS.num_partial_programs,
      use_expanding_layer=FLAGS.use_expanding_layer)
  initial_variables = jax.jit(m.init)(
      init_rng,
      jnp.ones(io_shape, jnp.float32),
      jnp.ones(io_shape, jnp.float32),
      jnp.ones(program_shape, jnp.float32))

  adam_opt_def = optim.Adam(
      FLAGS.lr,
      beta1=0.9,
      beta2=0.98,
      eps=1e-9,
      weight_decay=FLAGS.weight_decay)
  optimizer = adam_opt_def.create(initial_variables['params'])

  del initial_variables  # Don't keep a copy of the initial model.

  start_step = 0
  if FLAGS.restore_checkpoints:
    # Restore unreplicated optimizer + model state from last checkpoint.
    optimizer = checkpoints.restore_checkpoint(
        os.path.join(FLAGS.save_dir, 'checkpoints', hparam_str), optimizer)
    # Grab last step.
    start_step = int(optimizer.state.step)
    logging.info('Found model checkpointed at step %d.', start_step)
    if start_step > 0:
      start_step += 1

  # Build Pretraining Model and Optimizer (if specified)
  # ---------------------------------------------------------------------------
  pretrain_optimizer = None  # Optimizer used for pretrainined
  split_target = None  # Split pretrained model on partial programs.
  if start_step < FLAGS.num_pretrain_steps:
    # Load in pretraining optimizer.
    def filter_fn(path, value):
      del value
      if FLAGS.freeze_encoder and path.startswith('/encoder'):
        return False
      if FLAGS.freeze_decoder and path.startswith('/decoder'):
        return False
      return True
    trainable_weights = optim.ModelParamTraversal(filter_fn)
    pretrain_opt_def = optim.MultiOptimizer((trainable_weights, adam_opt_def))
    pretrain_optimizer = pretrain_opt_def.create(optimizer.target)

    if FLAGS.pretrain_checkpoint_format:
      pretrain_exprs = FLAGS.max_expressions // FLAGS.num_partial_programs
      checkpoint_dir = FLAGS.pretrain_checkpoint_format.format(pretrain_exprs)

      if gfile.isdir(checkpoint_dir):
        # Use the pretrained parameters if no training has occurred yet.
        if start_step == 0:
          restore_paths = []
          if FLAGS.restore_encoder:
            restore_paths.append('target/encoder')
          if FLAGS.restore_decoder:
            restore_paths.append('target/decoder')

          pretrain_optimizer = restore_selected_paths(
              pretrain_optimizer,
              checkpoint_dir=checkpoint_dir,
              restore_paths=restore_paths)
          logging.info('Found model pretrained at %s.', checkpoint_dir)

        if FLAGS.match_split_encoding:
          split_model = models.DecomposeExpandingLayerTransformer(
              config=eval_config, num_partial_programs=1,
              use_expanding_layer=False)
          split_program_shape = (FLAGS.per_device_batch_size,
                                 1,
                                 FLAGS.max_program_length)
          split_initial_variables = jax.jit(split_model.init)(
              init_rng,
              jnp.ones(io_shape, jnp.float32),
              jnp.ones(io_shape, jnp.float32),
              jnp.ones(split_program_shape, jnp.float32))
          split_optimizer = adam_opt_def.create(
              split_initial_variables['params'])
          split_optimizer = checkpoints.restore_checkpoint(
              checkpoint_dir, split_optimizer)
          split_target = split_optimizer.target
      else:
        logging.warn('Could not find model at %s.', checkpoint_dir)

    if FLAGS.match_split_encoding and (split_target is None):
      raise RuntimeError('We could not load the pretrained checkpoint, '
                         'which is needed to match split embeddings.')

  learning_rate_fn = create_learning_rate_scheduler(base_learning_rate=FLAGS.lr)
  p_pretrain_step = jax.pmap(
      functools.partial(
          pretrain_step,
          num_partial_programs=FLAGS.num_partial_programs,
          learning_rate_fn=learning_rate_fn,
          config=train_config,
          use_expanding_layer=FLAGS.use_expanding_layer,
          split_params=split_target),
      axis_name='batch')
  p_train_step = jax.pmap(
      functools.partial(
          train_step,
          num_partial_programs=FLAGS.num_partial_programs,
          learning_rate_fn=learning_rate_fn,
          config=train_config,
          use_expanding_layer=FLAGS.use_expanding_layer),
      axis_name='batch')
  p_eval_step = jax.pmap(
      functools.partial(
          eval_step,
          num_partial_programs=FLAGS.num_partial_programs,
          eos_token=eos_token,
          config=eval_config,
          use_expanding_layer=FLAGS.use_expanding_layer),
      axis_name='batch')
  p_init_cache = jax.pmap(
      functools.partial(
          initialize_cache,
          num_partial_programs=FLAGS.num_partial_programs,
          max_decode_len=FLAGS.max_program_length,
          config=predict_config,
          use_expanding_layer=FLAGS.use_expanding_layer),
      axis_name='batch')
  p_pred_step = jax.pmap(
      functools.partial(
          predict_step,
          num_partial_programs=FLAGS.num_partial_programs,
          max_decode_len=FLAGS.max_program_length,
          eos_token=eos_token,
          config=predict_config,
          slow_decode=FLAGS.slow_decode,
          use_expanding_layer=FLAGS.use_expanding_layer),
      axis_name='batch',
      static_broadcasted_argnums=(4,))
  p_split_pred_step = jax.pmap(
      functools.partial(
          predict_step,
          num_partial_programs=FLAGS.num_partial_programs,
          max_decode_len=FLAGS.max_program_length,
          eos_token=eos_token,
          config=predict_config,
          slow_decode=FLAGS.slow_decode,
          use_expanding_layer=False,
          use_split_encoding=True,
          split_params=split_target),
      axis_name='batch',
      static_broadcasted_argnums=(4,))

  # Main Train Loop
  # ---------------------------------------------------------------------------
  train_rngs = jax.random.split(rng, jax.local_device_count())
  del rng

  # Replicate optimizer.
  if pretrain_optimizer:
    pretrain_optimizer = jax_utils.replicate(pretrain_optimizer)

  optimizer = jax_utils.replicate(optimizer)

  metrics_all = []
  tick = time.time()
  for step in range(start_step, FLAGS.num_train_steps):
    inputs, outputs, programs, split_outputs = (
        common_utils.shard(next(train_iter)))

    if step < FLAGS.num_pretrain_steps:
      pretrain_optimizer, metrics, train_rngs = p_pretrain_step(
          pretrain_optimizer, inputs, outputs, programs,
          split_outputs=split_outputs,
          pretrain_rng=train_rngs)
    else:
      optimizer, metrics, train_rngs = p_train_step(
          optimizer, inputs, outputs, programs,
          train_rng=train_rngs)

    metrics_all.append(metrics)
    is_last_pretrain_step = step == FLAGS.num_pretrain_steps - 1
    is_last_step = step == FLAGS.num_train_steps - 1

    if is_last_pretrain_step:
      optimizer = maybe_copy_model_from_pretraining(
          optimizer, pretrain_optimizer, step, adam_opt_def)

    # Save a Checkpoint
    if (step % FLAGS.checkpoint_freq == 0 and step > 0) or is_last_step:
      optimizer = maybe_copy_model_from_pretraining(
          optimizer, pretrain_optimizer, step, adam_opt_def)
      if jax.host_id() == 0:
        # Save unreplicated optimizer + model state.
        checkpoints.save_checkpoint(
            os.path.join(FLAGS.save_dir, 'checkpoints', hparam_str),
            jax_utils.unreplicate(optimizer),
            step)

    # Periodic metric handling.
    if not step or (step % FLAGS.log_freq != 0 and not is_last_step and
                    not is_last_pretrain_step):
      continue

    optimizer = maybe_copy_model_from_pretraining(
        optimizer, pretrain_optimizer, step, adam_opt_def)

    logging.info('Gathering training metrics.')
    # Training Metrics
    metrics_all = common_utils.get_metrics(metrics_all)
    lr = metrics_all.pop('learning_rate').mean()
    metrics_sums = jax.tree_map(jnp.sum, metrics_all)
    denominator = metrics_sums.pop('denominator')
    summary = jax.tree_map(
        lambda x: x / denominator,  # pylint: disable=cell-var-from-loop
        metrics_sums)
    summary['learning_rate'] = lr
    # Calculate (clipped) perplexity after averaging log-perplexities:
    summary['perplexity'] = jnp.clip(jnp.exp(summary['loss']), a_max=1.0e4)

    if jax.host_id() == 0:
      logging.info('Train in step: %d, loss: %.4f', step, summary['loss'])
      tock = time.time()
      steps_per_sec = FLAGS.log_freq / (tock - tick)
      tick = tock
      summary_writer.scalar('train/steps per second', steps_per_sec, step)
      for key, val in summary.items():
        summary_writer.scalar('train/' + key, val, step)
      summary_writer.flush()
    # Reset metric accumulation for next evaluation cycle.
    metrics_all = []

    # Evaluation Metrics
    logging.info('Gathering evaluation metrics.')
    t_evaluation_start = time.time()

    eval_summary = evaluate(
        p_eval_step=p_eval_step,
        target=optimizer.target,
        eval_ds=eval_ds)
    if jax.host_id() == 0:
      logging.info('Evaluation time: %.4f s step %d, loss: %.4f.',
                   time.time()-t_evaluation_start, step, eval_summary['loss'])
      for key, val in eval_summary.items():
        summary_writer.scalar('eval/' + key, val, step)
      summary_writer.flush()

    # Beam search metrics.
    logging.info('Gathering beam search metrics.')
    for beam_size in [1, 10, 12, 24, 48, 96]:
      t_inference_start = time.time()

      pred_acc, message = predict_and_compute_score(
          p_pred_step=p_pred_step,
          p_init_cache=p_init_cache,
          target=optimizer.target,
          predict_ds=predict_ds,
          decode_io=decode_io,
          decode_program=decode_program,
          beam_size=beam_size,
          num_partial_programs=FLAGS.num_partial_programs,
          use_best_first_search=FLAGS.best_first_search,
          slow_decode=FLAGS.slow_decode)

      # Write to tensorboard.
      if jax.host_id() == 0:
        slow_or_fast = 'slow' if FLAGS.slow_decode else 'fast'
        logging.info(
            'Prediction time, %s (beam %d): %.4f s, step %d, score %.4f',
            slow_or_fast, beam_size, time.time() - t_inference_start, step,
            pred_acc)
        beam_search_or_bfs = 'bfs' if FLAGS.best_first_search else 'beam-search'
        summary_writer.scalar(
            'predict-{}/score-{}-{}'.format(slow_or_fast,
                                            beam_search_or_bfs,
                                            beam_size),
            pred_acc, step)
        summary_writer.text('samples-{}'.format(beam_size),
                            '\n------\n'.join(message), step)
        summary_writer.flush()

      if step < FLAGS.num_pretrain_steps and FLAGS.match_split_encoding:
        pred_acc, message = predict_and_compute_score(
            p_pred_step=p_split_pred_step,
            p_init_cache=p_init_cache,
            target=optimizer.target,
            predict_ds=predict_ds,
            decode_io=decode_io,
            decode_program=decode_program,
            beam_size=beam_size,
            num_partial_programs=FLAGS.num_partial_programs,
            use_best_first_search=FLAGS.best_first_search,
            slow_decode=FLAGS.slow_decode)

        # Write to tensorboard.
        if jax.host_id() == 0:
          slow_or_fast = 'slow' if FLAGS.slow_decode else 'fast'
          beam_search_or_bfs = ('bfs' if FLAGS.best_first_search
                                else 'beam-search')
          summary_writer.scalar(
              'predict-split-{}/score-{}-{}'.format(slow_or_fast,
                                                    beam_search_or_bfs,
                                                    beam_size),
              pred_acc, step)
          summary_writer.text('samples-split-{}'.format(beam_size),
                              '\n------\n'.join(message), step)
          summary_writer.flush()

if __name__ == '__main__':
  app.run(main)
