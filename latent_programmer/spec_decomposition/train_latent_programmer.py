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

# coding=utf-8
# Copyright 2023 The Google Research Authors.
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

"""Train seq-to-seq model on random supervised training tasks."""

import collections
import dataclasses
import functools
import os
import random
import sys
import time
import typing

from absl import app
from absl import flags
from absl import logging
import flax  # pylint: disable=unused-import
from flax import jax_utils
from flax import linen as nn
from flax import optim
from flax.metrics import tensorboard
from flax.training import checkpoints
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from latent_programmer import decode
from latent_programmer import models
from latent_programmer.spec_decomposition import input_pipeline
from latent_programmer.tasks.deepcoder import deepcoder_dsl
from latent_programmer.tasks.robust_fill import dsl as robust_fill_dsl
from latent_programmer.tasks.robust_fill import tokens as dsl_tokens

sys.path.append('../../')
gfile = tf.io.gfile

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 0, 'Fixed random seed for training.')
flags.DEFINE_float('lr', 1e-3, 'Learning rate.')
flags.DEFINE_float('weight_decay', 1e-1,
                   'Decay factor for AdamW-style weight decay.')
flags.DEFINE_integer('embedding_dim', 256, 'Embedding dimension.')
flags.DEFINE_integer('hidden_dim', 512, 'Hidden dimension.')
flags.DEFINE_integer('num_heads', 4, 'Number of layers.')
flags.DEFINE_integer('num_layers', 3, 'Number of Transformer heads.')
flags.DEFINE_boolean('slow_decode', True, 'Use slow decoding for prediction?')
flags.DEFINE_float('dropout_rate', 0.1, 'Dropout rate')
flags.DEFINE_float('attention_dropout_rate', 0.1, 'Attention dropout rate')
flags.DEFINE_integer('latent_vocab_size', 40, 'Number of latent tokens.')
flags.DEFINE_integer('c', 2, 'Latent length reduced by factor of 2^c.')
flags.DEFINE_float('commitment_cost_vq', 0.25, 'Weight for VQ-VAE loss.')
flags.DEFINE_integer('beam_size', 10, 'Beam size.')
flags.DEFINE_integer('latent_beam_size', 3, 'Number of latent beams.')

flags.DEFINE_string('dataset_dir', None,
                    'Directory to find TFRecord datasets for train and test.')
flags.DEFINE_string('experiment', 'NONE',
                    'Which compositional generalization experiment to use.')
flags.DEFINE_integer('per_device_batch_size', 16,
                     'Number of program tasks in a batch.')
flags.DEFINE_integer('num_examples', 4,
                     'Number of input/output strings per task.')
flags.DEFINE_integer('max_input_length', 120,
                     'Maximum number of characters in input/output strings.')
flags.DEFINE_integer('predict_max_input_length', 200,
                     'Maximum number of characters in input/output strings for '
                     'prediction.')
flags.DEFINE_integer('max_target_length', 200,
                     'Maximum number of characters in the target.')

flags.DEFINE_string('save_dir', None, 'Directory to save results to.')
flags.DEFINE_integer('num_train_steps', 1000000, 'Number of training steps.')
flags.DEFINE_integer('num_pretrain_steps', 10000, 'Number of pretraining steps')
flags.DEFINE_integer('num_eval_steps', 10, 'Number of evaluation steps.')
flags.DEFINE_integer('num_quick_test_steps', 10,
                     'Number of test steps during training.')
flags.DEFINE_integer('num_final_test_steps', 100,
                     'Number of test steps after training is finished.')
flags.DEFINE_integer('log_freq', 2000, 'Number of steps between training logs.')
flags.DEFINE_integer('eval_freq', 10000, 'Number of steps between eval.')
flags.DEFINE_integer('predict_freq', 50000,
                     'Number of steps between prediction (beam search).')
flags.DEFINE_integer('checkpoint_freq', 50000,
                     'Number of steps between checkpoint saves.')
flags.DEFINE_bool('restore_checkpoints', True,
                  'Whether to restore from existing model checkpoints.')

flags.DEFINE_bool('use_relative_attention', True,
                  'Whether to use relative positonal embeddings.')
flags.DEFINE_integer('num_position_buckets', 32,
                     'Number of relative attention position buckets.')
flags.DEFINE_integer('max_distance', 128,
                     'Max distance for relative attention positions.')
flags.DEFINE_integer('max_program_cross_embed_distance', 128,
                     'Max distance for relative attention positions.')
flags.DEFINE_bool('aligned_relative_attention', True,
                  'Whether to align relative attention positions between '
                  'targets and encoded I/O examples.')

flags.DEFINE_enum('dataset_type', 'deepcoder',
                  ['robustfill', 'deepcoder'],
                  'The kind of dataset to use.')
flags.DEFINE_enum('model_type', 'baseline_model',
                  ['baseline_model'],
                  'Which model to train.')

flags.DEFINE_bool('do_training', True,
                  'Whether to do training.')
flags.DEFINE_bool('do_evaluation', True,
                  'Whether to do evaluation.')
flags.DEFINE_bool('do_prediction', True,
                  'Whether to do beam search prediction.')


_internal = False
if not _internal:
  flags.DEFINE_string('xm_parameters', None,
                      'String specifying hyperparamter search.')


# pytype has hardcoded special-case support for dataclasses.dataclass
flax_dataclass = (
    flax.struct.dataclass
    if not typing.TYPE_CHECKING else dataclasses.dataclass)


@flax_dataclass
class TrainState:
  step: int
  optimizer: optim.Optimizer
  model_state: typing.Any
  lp_optimizer: optim.Optimizer


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
  """Computes weighted cross entropy and entropy for log probs and targets.

  Args:
   logits: `[batch, length, num_classes]` float array.
   targets: categorical targets `[batch, length]` int array.
   weights: None or array of shape [batch, length, 1]

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


def compute_weighted_accuracy(logits, targets, weights=None):
  """Computes weighted accuracy for log probs and targets.

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


def compute_metrics(logits, targets, weights, prefix=''):
  """Computes summary metrics."""
  loss, weight_sum = compute_weighted_cross_entropy(logits, targets, weights)
  acc, _ = compute_weighted_accuracy(logits, targets, weights)
  metrics = {
      prefix + 'loss': loss,
      prefix + 'accuracy': acc,
      prefix + 'denominator': weight_sum,
  }
  metrics = jax.lax.psum(metrics, 'batch')
  return metrics


def add_eos_token(indices, eos_token):
  """Add EOS token to sequence."""
  batch_size = indices.shape[0]
  lengths = jnp.count_nonzero(indices, axis=1)

  indices = jnp.pad(
      indices, pad_width=[(0, 0), (0, 1)], mode='constant')
  # Add EOS token.
  indices = indices.at[jnp.arange(batch_size), lengths].set(eos_token)
  return indices.astype(jnp.int32)


# Train / eval / decode step functions.
# -----------------------------------------------------------------------------


def train_step(state,
               inputs,
               outputs,
               targets,
               pretrain,
               learning_rate_fn,
               bos_token,
               eos_token,
               config,
               lp_config,
               dropout_rng):
  """Train on batch of program tasks."""
  # We handle PRNG splitting inside the top pmap, rather
  # than handling it outside in the training loop - doing the
  # latter can add some stalls to the devices.
  dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

  weights = jnp.where(targets > 0, 1, 0).astype(jnp.float32)

  # Embedding mask for autoencoding.
  emb_mask = jnp.ones((1, FLAGS.latent_vocab_size),
                      jnp.float32).at[:, [0, bos_token, eos_token]].set(0)

  def ae_loss_fn(params):
    """Loss function used for training autoencoder."""
    (logits, vq), new_variables = models.LatentProgramTransformer(config).apply(
        {'params': params, 'vqvae': state.model_state},
        inputs,
        outputs,
        targets,
        emb_mask,
        pretrain=pretrain,
        mutable=['vqvae'],
        rngs={'dropout': dropout_rng})
    loss, weight_sum = compute_weighted_cross_entropy(logits, targets, weights)

    # Add EOS token for latent predictor loss.
    vq_weight_sum = jnp.sum(
        jnp.where(vq['latent_indices'] > 0, 1, 0).astype(jnp.float32))
    latent_indices = add_eos_token(vq['latent_indices'], eos_token)

    mean_loss = loss / weight_sum + vq['loss'] / vq_weight_sum
    return mean_loss, (new_variables['vqvae'], logits, latent_indices)

  step = state.step
  optimizer = state.optimizer
  lp_optimizer = state.lp_optimizer
  lr = learning_rate_fn(step)
  grad_fn = jax.value_and_grad(ae_loss_fn, has_aux=True)
  (_, (new_model_state, ae_logits, latent_indices)), ae_grad = grad_fn(
      optimizer.target)
  ae_grad = jax.lax.pmean(ae_grad, 'batch')

  latent_weights = jnp.where(latent_indices > 0, 1, 0).astype(jnp.float32)

  encoded_mask = jnp.where(outputs > 0, 1, 0).astype(jnp.float32)
  # Additionally mask out eos token in latents.
  latents_mask = jnp.where(
      jnp.logical_and(latent_indices > 0, latent_indices != eos_token),
      1, 0).astype(jnp.float32)

  def loss_fn(params, lp_params):
    """Loss function used for training."""
    latent_logits = models.ProgramTransformer(lp_config).apply(
        {'params': lp_params},
        inputs,
        outputs,
        latent_indices,
        rngs={'dropout': dropout_rng})
    latent_loss, latent_weight_sum = compute_weighted_cross_entropy(
        latent_logits, latent_indices, latent_weights)

    # End-to-end prediction.
    encoded = models.LatentProgramTransformer(config).apply(
        {'params': params, 'vqvae': state.model_state},
        inputs,
        outputs,
        mutable=False,
        rngs={'dropout': dropout_rng},
        method=models.LatentProgramTransformer.encode)
    latents = models.LatentProgramTransformer(config).apply(
        {'params': params, 'vqvae': state.model_state},
        latent_logits,
        mutable=False,
        rngs={'dropout': dropout_rng},
        method=models.LatentProgramTransformer.quantize)
    logits = models.LatentProgramTransformer(config).apply(
        {'params': params, 'vqvae': state.model_state},
        targets,
        latents,
        encoded,
        latents_mask,
        encoded_mask,
        mutable=False,
        rngs={'dropout': dropout_rng},
        method=models.LatentProgramTransformer.decode)
    loss, weight_sum = compute_weighted_cross_entropy(logits, targets, weights)

    mean_loss = latent_loss / latent_weight_sum
    if not pretrain:
      mean_loss += loss / weight_sum
    return mean_loss, (logits, latent_logits)

  grad_fn = jax.value_and_grad(loss_fn, argnums=[0, 1], has_aux=True)
  (_, (logits, latent_logits)), grads = grad_fn(
      optimizer.target, lp_optimizer.target)
  grads = jax.lax.pmean(grads, 'batch')
  new_optimizer = optimizer.apply_gradient(
      jax.tree.map(jnp.add, grads[0], ae_grad), learning_rate=lr)
  new_lp_optimizer = lp_optimizer.apply_gradient(grads[1], learning_rate=lr)

  metrics = compute_metrics(logits, targets, weights)
  metrics['learning_rate'] = lr
  metrics.update(compute_metrics(ae_logits, targets, weights, prefix='ae_'))
  latent_metrics = compute_metrics(
      latent_logits, latent_indices, latent_weights, prefix='latent_')

  new_state = state.replace(
      step=step + 1,
      optimizer=new_optimizer,
      model_state=jax.lax.pmean(new_model_state, 'batch'),
      lp_optimizer=new_lp_optimizer)
  return new_state, metrics, latent_metrics, new_dropout_rng


def eval_step(state,
              inputs,
              outputs,
              targets,
              bos_token,
              eos_token,
              config,
              lp_config):
  """Collect metrics for evaluation during training."""
  params = state.optimizer.target
  lp_params = state.lp_optimizer.target

  weights = jnp.where(
      jnp.logical_and(targets > 0,
                      jnp.logical_and(targets != bos_token,
                                      targets != eos_token)),
      1, 0).astype(jnp.float32)
  emb_mask = jnp.ones((1, FLAGS.latent_vocab_size),
                      jnp.float32).at[:, [0, bos_token, eos_token]].set(0)

  ae_logits, vq = models.LatentProgramTransformer(config).apply(
      {'params': params, 'vqvae': state.model_state},
      inputs,
      outputs,
      targets,
      emb_mask,
      mutable=False)

  # Postprocess latent indices.
  latent_indices = add_eos_token(vq['latent_indices'], eos_token)
  latent_weights = jnp.where(latent_indices > 0, 1, 0).astype(jnp.float32)

  encoded_mask = jnp.where(outputs > 0, 1, 0).astype(jnp.float32)
  # Additionally mask out eos token in latents.
  latents_mask = jnp.where(
      jnp.logical_and(latent_indices > 0, latent_indices != eos_token),
      1, 0).astype(jnp.float32)

  latent_logits = models.ProgramTransformer(lp_config).apply(
      {'params': lp_params}, inputs, outputs, latent_indices)

  encoded = models.LatentProgramTransformer(config).apply(
      {'params': params, 'vqvae': state.model_state},
      inputs,
      outputs,
      mutable=False,
      method=models.LatentProgramTransformer.encode)
  latents = models.LatentProgramTransformer(config).apply(
      {'params': params, 'vqvae': state.model_state},
      latent_logits,
      mutable=False,
      method=models.LatentProgramTransformer.quantize)
  logits = models.LatentProgramTransformer(config).apply(
      {'params': params, 'vqvae': state.model_state},
      targets,
      latents,
      encoded,
      latents_mask,
      encoded_mask,
      mutable=False,
      method=models.LatentProgramTransformer.decode)

  metrics = compute_metrics(logits, targets, weights)
  metrics.update(compute_metrics(ae_logits, targets, weights, prefix='ae_'))
  latent_metrics = compute_metrics(
      latent_logits, latent_indices, latent_weights, prefix='latent_')
  return metrics, latent_metrics


def initialize_cache(
    inputs, outputs, targets, max_decode_len, config, lp_config):
  """Initializes a cache for a given input shape and max decode length."""
  target_shape = (targets.shape[0], max_decode_len)
  initial_variables = models.LatentProgramTransformer(config).init(
      jax.random.PRNGKey(0),
      jnp.ones(inputs.shape, config.dtype),
      jnp.ones(outputs.shape, config.dtype),
      jnp.ones(target_shape, config.dtype))
  lp_initial_variables = models.ProgramTransformer(lp_config).init(
      jax.random.PRNGKey(0),
      jnp.ones(inputs.shape, lp_config.dtype),
      jnp.ones(outputs.shape, lp_config.dtype),
      jnp.ones(target_shape, lp_config.dtype))

  return initial_variables['cache'], lp_initial_variables['cache']


def predict_step(state,
                 inputs,
                 outputs,
                 cache,
                 lp_cache,
                 beam_size,
                 bos_token,
                 eos_token,
                 max_decode_len,
                 config,
                 lp_config,
                 slow_decode=True):
  """Predict translation with fast decoding beam search on a batch."""
  params = state.optimizer.target
  lp_params = state.lp_optimizer.target

  latent_beam_size = FLAGS.latent_beam_size
  if beam_size == 1:
    latent_beam_size = 1

  # Split beam over latent sequences and programs.
  per_latent_beam_size = beam_size // latent_beam_size
  beam_size = latent_beam_size * per_latent_beam_size

  # Prepare transformer fast-decoder call for beam search: for beam search, we
  # need to set up our decoder model to handle a batch size equal to
  # batch_size * beam_size, where each batch item's data is expanded in-place
  # rather than tiled.
  flat_lp_encoded = decode.flat_batch_beam_expand(
      models.ProgramTransformer(lp_config).apply(
          {'params': lp_params},
          inputs,
          outputs,
          method=models.ProgramTransformer.encode),
      latent_beam_size)

  encoded_padding_mask = jnp.where(outputs > 0, 1, 0).astype(jnp.float32)
  flat_encoded_padding_mask = decode.flat_batch_beam_expand(
      encoded_padding_mask, latent_beam_size)

  if slow_decode:
    def tokens_ids_to_latent_logits(flat_ids):
      """Token slice to logits from decoder model."""
      # --> [batch * beam, 1, vocab]
      flat_logits = models.ProgramTransformer(lp_config).apply(
          {'params': lp_params},
          flat_ids,
          flat_lp_encoded,
          flat_encoded_padding_mask,
          method=models.ProgramTransformer.decode)
      return flat_logits

  else:
    raise NotImplementedError()

  # Step 1: Beam-search over latent tokens.
  latent_beam_seqs, _ = decode.beam_search(
      inputs,
      lp_cache,
      tokens_ids_to_latent_logits,
      beam_size=latent_beam_size,
      alpha=0.6,
      bos_token=bos_token,
      eos_token=eos_token,
      max_decode_len=np.ceil(max_decode_len / 2**FLAGS.c).astype(np.int32),
      slow_decode=slow_decode)

  flat_latent_seqs = decode.flat_batch_beam_expand(
      decode.flatten_beam_dim(latent_beam_seqs), per_latent_beam_size)
  # Quantize the predicted latent codes.
  flat_latents = models.LatentProgramTransformer(config).apply(
      {'params': params, 'vqvae': state.model_state},
      flat_latent_seqs,
      mutable=False,
      method=models.LatentProgramTransformer.quantize)

  flat_encoded = decode.flat_batch_beam_expand(
      models.LatentProgramTransformer(config).apply(
          {'params': params},
          inputs,
          outputs,
          method=models.LatentProgramTransformer.encode),
      beam_size)

  flat_latents_mask = jnp.where(
      jnp.logical_and(flat_latent_seqs > 0, flat_latent_seqs != eos_token),
      1, 0).astype(jnp.float32)
  encoded_padding_mask = jnp.where(outputs > 0, 1, 0).astype(jnp.float32)
  flat_encoded_padding_mask = decode.flat_batch_beam_expand(
      encoded_padding_mask, beam_size)

  if slow_decode:
    def tokens_ids_to_logits(flat_ids):
      """Token slice to logits from decoder model."""
      # --> [batch * beam, 1, vocab]
      flat_logits = models.LatentProgramTransformer(config=config).apply(
          {'params': params},
          flat_ids,
          flat_latents,
          flat_encoded,
          flat_latents_mask,
          flat_encoded_padding_mask,
          method=models.LatentProgramTransformer.decode)
      return flat_logits
  else:
    raise NotImplementedError()

  # Step 2: Beam-search over program tokens.
  per_latent_inputs = decode.flat_batch_beam_expand(
      inputs, latent_beam_size)
  per_latent_cache = jax.tree.map(
      lambda x: decode.flat_batch_beam_expand(x, latent_beam_size), cache)
  beam_seqs, _ = decode.beam_search(
      per_latent_inputs,
      per_latent_cache,
      tokens_ids_to_logits,
      beam_size=per_latent_beam_size,
      alpha=0.6,
      bos_token=bos_token,
      eos_token=eos_token,
      max_decode_len=max_decode_len,
      slow_decode=slow_decode)
  # Collapse both beam dimensions into one.
  beam_seqs = beam_seqs.reshape(
      (inputs.shape[0], beam_size) + beam_seqs.shape[2:])
  latent_beam_seqs = jnp.repeat(
      latent_beam_seqs, per_latent_beam_size, axis=1)

  # Beam search returns [n_batch, n_beam, n_length] with beam dimension
  # sorted in increasing order of log-probability.
  return beam_seqs, latent_beam_seqs


# Util functions for prediction
# -----------------------------------------------------------------------------


def run_program(program, inputs):
  """Returns a list of outputs from running a program on a list of inputs.

  Args:
    program: A program returned from `decode_program()`.
    inputs: A list of inputs as returned by `decode_io`.
  """
  if FLAGS.dataset_type == 'robustfill':
    return [program(i) for i in inputs]
  elif FLAGS.dataset_type == 'deepcoder':
    # `program` is a deepcoder_dsl.Statement or deepcoder_dsl.Program.
    if program is None:
      return [None] * len(inputs)
    initial_states = [deepcoder_dsl.ProgramState.from_str(i) for i in inputs]
    if FLAGS.model_type == 'baseline_model':
      result_states = [program.run(state.state) for state in initial_states]  # pytype: disable=attribute-error
    else:
      result_states = [program.run(state) for state in initial_states]  # pytype: disable=attribute-error
    outputs = [deepcoder_dsl.result_to_str(result_state.get_output())
               if result_state else None
               for result_state in result_states]
    return outputs
  else:
    raise ValueError('Unhandled dataset_type {}'.format(FLAGS.dataset_type))


def eval_predicted_spec_decomposer_model(predicted, ground_truth, decode_spec):
  """Evaluate predicted program beams."""
  beams_target = [decode_spec(beam) for beam in predicted[::-1]]
  success = ground_truth in beams_target
  if success:
    return ground_truth, 1
  else:
    return beams_target[0], 0


def eval_predicted_synthesizer_model(predicted, inputs, outputs,
                                     decode_program):
  """Evaluate predicted program beams."""
  best_program_str, best_score = None, -99

  # predicted shape [beam_size, length]
  for beam in predicted[::-1]:
    if FLAGS.dataset_type == 'robustfill':
      program = decode_program(beam)
      try:
        p_outs = run_program(program, inputs)
        score = (np.sum([p_out == out for p_out, out in zip(p_outs, outputs)])
                 / len(inputs))
        program_str = program.to_string()
      except:  # pylint: disable=bare-except
        score = -1
        program_str = 'did not compile'

    elif FLAGS.dataset_type == 'deepcoder':
      statement = decode_program(beam)
      if statement is None:
        score = -1
        program_str = 'did not compile'
      else:
        try:
          p_outs = run_program(statement, inputs)
          score = (np.sum([p_out == out for p_out, out in zip(p_outs, outputs)])
                   / len(inputs))
          program_str = str(statement)
        except deepcoder_dsl.RunError:
          score = -0.5
          program_str = 'encountered RunError'

    else:
      raise ValueError('Unhandled dataset_type {}'.format(FLAGS.dataset_type))

    if score > best_score:
      best_program_str, best_score = program_str, score

    if best_score >= 1:  # Found solution.
      break

  # best_program_str could be None if no RobustFill program compiles.
  return best_program_str, best_score


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
  """Executes psum on in_tree's leaves over one device per host."""
  host2devices = collections.defaultdict(list)
  for d in jax.devices():
    host2devices[d.host_id].append(d)
  devices = [host2devices[k][0] for k in host2devices]
  host_psum = jax.pmap(lambda x: jax.lax.psum(x, 'i'), 'i', devices=devices)
  def pre_pmap(xs):
    return jax.tree.map(lambda x: jnp.broadcast_to(x, (1,) + x.shape), xs)
  def post_pmap(xs):
    return jax.tree.map(lambda x: x[0], xs)
  return post_pmap(host_psum(pre_pmap(in_tree)))


def shorten(key):
  splits = key.split('_')
  return ''.join(s[0] for s in splits)


def load_data(batches, rng):
  """Returns info from batches in dictionaries."""
  data_dict = common_utils.shard(batches)
  if FLAGS.model_type == 'synthesizer_model':
    rng, new_rng = jax.random.split(rng)
    # shape: [num_devices, per_device_batch_size]
    is_corrupted = (
        jax.random.uniform(rng, shape=data_dict['outputs'].shape[:2])
        < FLAGS.synthesizer_corrupted_next_part_rate)
    outputs = np.where(is_corrupted[Ellipsis, None, None],
                       data_dict['corrupted_outputs'],
                       data_dict['outputs'])
  else:
    outputs = data_dict['outputs']
    new_rng = rng

  return (data_dict['inputs'],
          outputs,
          data_dict['target'],
          new_rng)


def main(_):
  tf.random.set_seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)
  random.seed(FLAGS.seed)

  if not gfile.isdir(FLAGS.save_dir):
    gfile.makedirs(FLAGS.save_dir)

  xm_client = xmanager_api.XManagerApi(xm_deployment_env='alphabet')
  work_unit = xm_client.get_current_work_unit()
  hparam_dict = work_unit.parameters['args']
  hparam_str = 'hparams-' + ','.join(sorted(['%s=%s' % (shorten(k), str(v))
                                             for k, v in hparam_dict.items()]))

  # Number of local devices for this host.
  n_devices = jax.local_device_count()
  logging.info('host %d says n_devices=%d', jax.host_id(), n_devices)
  logging.info('There are %d hosts', jax.host_count())

  if jax.host_count() * n_devices * FLAGS.per_device_batch_size != 128:
    raise ValueError('The per_device_batch_size might be wrong')

  if jax.host_id() == 0:
    summary_writer = tensorboard.SummaryWriter(
        os.path.join(FLAGS.save_dir, 'tb', hparam_str))
  else:
    # Only to appease pytype.
    summary_writer = tensorboard.SummaryWriter('')

  batch_size = FLAGS.per_device_batch_size * n_devices
  io_shape = (FLAGS.per_device_batch_size,
              FLAGS.num_examples,
              FLAGS.max_input_length)
  predict_io_shape = (FLAGS.per_device_batch_size,
                      FLAGS.num_examples,
                      FLAGS.predict_max_input_length)
  target_shape = (FLAGS.per_device_batch_size, FLAGS.max_target_length)

  # Setup DSL
  # ---------------------------------------------------------------------------

  # Build token tables.
  if FLAGS.dataset_type == 'robustfill':
    spec_vocab = robust_fill_dsl.CHARACTER + input_pipeline.SEPARATOR_TOKEN
    spec_id_token_table = {i+3: token
                           for i, token in enumerate(spec_vocab)}
    bos_id = 1
    eos_id = 2
    spec_id_token_table[bos_id] = robust_fill_dsl.BOS
    spec_id_token_table[eos_id] = robust_fill_dsl.EOS
    spec_token_id_table = {token: id
                           for id, token in spec_id_token_table.items()}
    spec_vocab_size = len(spec_token_id_table) + 1  # For padding.
    program_id_token_table, _ = dsl_tokens.build_token_tables()
    program_vocab_size = len(program_id_token_table) + 1

  elif FLAGS.dataset_type == 'deepcoder':
    id_to_token, token_to_id = deepcoder_dsl.vocab_tables()
    bos_id, eos_id = deepcoder_dsl.BOS_ID, deepcoder_dsl.EOS_ID
    vocab_size = len(id_to_token)  # Already includes padding.

    spec_vocab_size = program_vocab_size = vocab_size
    program_id_token_table = spec_id_token_table = id_to_token
    spec_token_id_table = token_to_id

  else:
    raise ValueError('Unhandled dataset_type: {}'.format(FLAGS.dataset_type))

  # Parse io and program token sequences (for eval).
  def decode_io(inputs, outputs):
    """Converts from int tensors to strings."""
    if FLAGS.dataset_type == 'robustfill':
      def decode_str(s):
        """Decode string tokens."""
        return ''.join([spec_id_token_table[t_id] for t_id in s if t_id > 0])

      inps, outs = [], []
      for inp, out in zip(inputs, outputs):
        inps.append(decode_str(inp))
        outs.append(decode_str(out))
      return inps, outs

    elif FLAGS.dataset_type == 'deepcoder':
      def decode_str(s):
        return ' '.join(spec_id_token_table[i] for i in s if i > 0)
      inps = [decode_str(inp) for inp in inputs]
      outs = [decode_str(out) for out in outputs]
      return inps, outs

    else:
      raise ValueError('Unhandled dataset_type: {}'.format(FLAGS.dataset_type))

  def decode_spec(target):
    """Converts from int tensor to a string."""
    target = target[:np.argmax(target == eos_id)].astype(np.int32)

    if FLAGS.dataset_type == 'robustfill':
      target = target[target != bos_id].tolist()
      return ''.join([spec_id_token_table[t_id] for t_id in target if t_id > 0])
    elif FLAGS.dataset_type == 'deepcoder':
      target = target[target != bos_id].tolist()
      return ' '.join([spec_id_token_table[t_id]
                       for t_id in target if t_id > 0])
    else:
      raise ValueError('Unhandled dataset_type: {}'.format(FLAGS.dataset_type))

  def decode_program(program):
    """Decode program tokens into a program object."""
    program = program[:np.argmax(program == eos_id) + 1].astype(np.int32)

    if FLAGS.dataset_type == 'robustfill':
      # Returns either a Concat program object, or None.
      program = program[program != bos_id].tolist()
      try:
        return robust_fill_dsl.decode_program(program, program_id_token_table)
      except:  # pylint: disable=bare-except
        return None  # Program does not compile.

    if FLAGS.dataset_type == 'deepcoder':
      tokens = [program_id_token_table[t_id] for t_id in program.tolist()
                if t_id > 0 and t_id != eos_id and t_id != bos_id]
      try:
        if FLAGS.model_type == 'baseline_model':
          # Parse the entire program.
          return deepcoder_dsl.Program.from_tokens(tokens)
        else:
          # For DeepCoder, the model only predicts the RHS of the next
          # statement. Note that `output` is not a valid variable name token.
          # That should not matter if we only run this statement on a program
          # state, without constructing a full Program using this statement.
          statement_str = 'output = ' + ' '.join(tokens)
          return deepcoder_dsl.Statement.from_str(statement_str,
                                                  check_variable_name=False)
      except (deepcoder_dsl.ParseError, deepcoder_dsl.RunError):
        return None  # Program does not compile.

    else:
      raise ValueError('Unhandled dataset_type: {}'.format(FLAGS.dataset_type))

  def decode_program_str(program):  # pylint: disable=unused-variable
    """Decode program tokens into a string."""
    if FLAGS.dataset_type == 'robustfill':
      try:
        return decode_program(program).to_string()  # pytype: disable=attribute-error
      except:  # pylint: disable=bare-except
        return 'did not compile'
    elif FLAGS.dataset_type == 'deepcoder':
      # This does not check if the program actually compiles.
      return ' '.join([spec_id_token_table[t_id] for t_id in program.tolist()
                       if t_id > 0 and t_id != eos_id and t_id != bos_id])
    else:
      raise ValueError(f'Unhandled dataset_type: {FLAGS.dataset_type}')

  # Load Dataset
  # ---------------------------------------------------------------------------
  logging.info('Initializing dataset.')
  if not FLAGS.dataset_dir:
    raise ValueError('Must specify dataset_dir.')
  decomposition_or_entire_programs = (
      'entire_programs' if FLAGS.model_type == 'baseline_model'
      else 'decomposition_data')
  train_dataset_path = os.path.join(
      FLAGS.dataset_dir, f'{FLAGS.experiment}_data',
      f'{decomposition_or_entire_programs}_train.tf_records-*')
  test_dataset_path = os.path.join(
      FLAGS.dataset_dir, f'{FLAGS.experiment}_data',
      f'{decomposition_or_entire_programs}_test.tf_records-*')

  # Training dataset.
  logging.info('Loading dataset from %s', train_dataset_path)
  padded_shapes = {
      'inputs': io_shape[1:],
      'outputs': io_shape[1:],
      'target': target_shape[1:],
  }
  logging.info('padded_shapes: %s', padded_shapes)

  if FLAGS.dataset_type in ['robustfill', 'deepcoder']:
    if FLAGS.dataset_type == 'robustfill':
      input_pipeline_fn = input_pipeline.create_robust_fill_dataset
      program_part_key = 'program_part'
    else:
      assert FLAGS.dataset_type == 'deepcoder'
      input_pipeline_fn = input_pipeline.create_deepcoder_dataset
      program_part_key = 'program_part_rhs'

    if FLAGS.model_type == 'spec_decomposer_model':
      create_dataset_fn = functools.partial(
          input_pipeline_fn,
          renaming_dict={
              'inputs': 'inputs',
              'outputs': 'outputs',
              'target': 'joined_next_part',
          })
    elif FLAGS.model_type == 'synthesizer_model':
      create_dataset_fn = functools.partial(
          input_pipeline_fn,
          renaming_dict={
              'inputs': 'inputs',
              'outputs': 'next_part',
              'corrupted_outputs': 'corrupted_next_part',
              'target': program_part_key,
          })
      padded_shapes['corrupted_outputs'] = io_shape[1:]
    elif FLAGS.model_type == 'joint_model':
      create_dataset_fn = functools.partial(
          input_pipeline_fn,
          renaming_dict={
              'inputs': 'inputs',
              'outputs': 'outputs',
              'target': program_part_key,
          })
    elif FLAGS.model_type == 'baseline_model':
      create_dataset_fn = functools.partial(
          input_pipeline_fn,
          renaming_dict={
              'inputs': 'inputs',
              'outputs': 'outputs',
              'target': 'program',
          })
    else:
      raise ValueError(f'Unhandled model_type: {FLAGS.model_type}')

  else:
    raise ValueError('Unhandled dataset_type: {}'.format(FLAGS.dataset_type))

  dataset = create_dataset_fn(
      train_dataset_path, spec_token_id_table, FLAGS.num_examples,
      entire_programs=(FLAGS.model_type == 'baseline_model'))
  dataset = dataset.padded_batch(
      batch_size,
      padded_shapes=padded_shapes,
      drop_remainder=True)
  # Split evaluation and training.
  eval_ds = dataset.take(FLAGS.num_eval_steps)
  # Decrease batch of predict dataset to handle beam search.
  predict_padded_shapes = padded_shapes.copy()
  predict_padded_shapes['inputs'] = predict_io_shape[1:]
  predict_padded_shapes['outputs'] = predict_io_shape[1:]
  if FLAGS.model_type == 'synthesizer':
    predict_padded_shapes['corrupted_outputs'] = predict_io_shape[1:]

  logging.info('predict_padded_shapes: %s', predict_padded_shapes)
  predict_ds = eval_ds.unbatch().padded_batch(
      n_devices, padded_shapes=predict_padded_shapes)
  train_ds = dataset.skip(FLAGS.num_eval_steps)
  train_ds = train_ds.repeat()

  test_dataset = create_dataset_fn(
      test_dataset_path, spec_token_id_table, FLAGS.num_examples,
      entire_programs=(FLAGS.model_type == 'baseline_model'))
  if FLAGS.model_type == 'baseline_model':
    test_dataset = test_dataset.padded_batch(
        1,
        padded_shapes=predict_padded_shapes,
        drop_remainder=False)
    test_batch_size = 8
    if test_batch_size % n_devices:
      raise ValueError(f'Test batch size {test_batch_size} should be divisible '
                       f'by n_devices {n_devices}')
    quick_test_dataset = (test_dataset
                          # In end-to-end predict, we used 1000 programs
                          # (not batches!).
                          .take(1000)
                          .unbatch()
                          .padded_batch(test_batch_size,
                                        padded_shapes=predict_padded_shapes,
                                        drop_remainder=False))
    final_test_dataset = quick_test_dataset
    logging.info('baseline_model test dataset has %d batches of size %d',
                 quick_test_dataset.cardinality(), test_batch_size)
  else:
    test_dataset = test_dataset.padded_batch(
        batch_size,
        padded_shapes=predict_padded_shapes,
        drop_remainder=False)
    quick_test_dataset = (test_dataset
                          .take(FLAGS.num_quick_test_steps)
                          .unbatch()
                          .padded_batch(int(np.ceil(batch_size / 10)),
                                        padded_shapes=predict_padded_shapes))
    final_test_dataset = (test_dataset
                          .take(FLAGS.num_final_test_steps)
                          .unbatch()
                          .padded_batch(int(np.ceil(batch_size / 10)),
                                        padded_shapes=predict_padded_shapes))

  # Build Model and Optimizer
  # ---------------------------------------------------------------------------
  if FLAGS.model_type == 'spec_decomposer_model':
    output_vocab_size = spec_vocab_size
  elif FLAGS.model_type in ['synthesizer_model', 'joint_model',
                            'baseline_model']:
    output_vocab_size = program_vocab_size
  else:
    raise ValueError(f'Unhandled model_type: {FLAGS.model_type}')

  base_config = models.TransformerConfig(
      vocab_size=spec_vocab_size,
      output_vocab_size=output_vocab_size,
      shift=True,
      emb_dim=FLAGS.embedding_dim,
      num_heads=FLAGS.num_heads,
      num_layers=FLAGS.num_layers,
      qkv_dim=FLAGS.embedding_dim,
      mlp_dim=FLAGS.hidden_dim,
      max_len=max(FLAGS.max_input_length, FLAGS.max_target_length),
      dropout_rate=FLAGS.dropout_rate,
      attention_dropout_rate=FLAGS.attention_dropout_rate,
      use_relative_attention=FLAGS.use_relative_attention,
      deterministic=False,
      decode=False,
      bos_token=bos_id,
      num_input_relative_position_buckets=FLAGS.num_position_buckets,
      max_input_distance=FLAGS.max_distance,
      num_output_relative_position_buckets=FLAGS.num_position_buckets,
      max_output_distance=FLAGS.max_distance,
      num_input_cross_output_relative_position_buckets=(
          FLAGS.num_position_buckets),
      max_input_cross_output_distance=FLAGS.max_distance,
      num_program_relative_position_buckets=FLAGS.num_position_buckets,
      max_program_distance=FLAGS.max_distance,
      num_program_cross_embed_relative_position_buckets=(
          FLAGS.num_position_buckets),
      max_program_cross_embed_distance=FLAGS.max_program_cross_embed_distance)
  train_config = models.LatentTransformerConfig(
      base_cfg=base_config,
      latent_vocab_size=FLAGS.latent_vocab_size,
      c=FLAGS.c,
      train_vq=True,
      commitment_cost_vq=FLAGS.commitment_cost_vq)
  eval_config = train_config.replace(
      base_cfg=base_config.replace(deterministic=True),
      train_vq=False)
  predict_config = train_config.replace(
      base_cfg=base_config.replace(
          shift=False, deterministic=True,
          decode=not FLAGS.slow_decode,
          max_len=max(FLAGS.predict_max_input_length, FLAGS.max_target_length)),
      train_vq=False)

  lp_train_config = base_config.replace(
      vocab_size=spec_vocab_size,
      output_vocab_size=FLAGS.latent_vocab_size)
  lp_eval_config = lp_train_config.replace(deterministic=True)
  lp_predict_config = lp_train_config.replace(
      shift=False, deterministic=True, decode=not FLAGS.slow_decode)

  rng = jax.random.PRNGKey(FLAGS.seed)
  rng = jax.random.fold_in(rng, jax.host_id())
  rng, init_rng = jax.random.split(rng)

  dropout_rng = jax.random.split(rng, jax.local_device_count())

  m = models.LatentProgramTransformer(eval_config)
  initial_variables = jax.jit(m.init)(
      init_rng,
      jnp.ones(io_shape, jnp.float32),
      jnp.ones(io_shape, jnp.float32),
      jnp.ones(target_shape, jnp.float32))
  lp_m = models.ProgramTransformer(lp_eval_config)
  lp_initial_variables = jax.jit(lp_m.init)(
      init_rng,
      jnp.ones(io_shape, jnp.float32),
      jnp.ones(io_shape, jnp.float32),
      jnp.ones(target_shape, jnp.float32))

  optimizer_def = optim.Adam(
      FLAGS.lr,
      beta1=0.9,
      beta2=0.98,
      eps=1e-9,
      weight_decay=FLAGS.weight_decay)
  optimizer = optimizer_def.create(initial_variables['params'])
  lp_optimizer = optimizer_def.create(lp_initial_variables['params'])

  state = TrainState(step=0,
                     optimizer=optimizer,
                     model_state=initial_variables['vqvae'],
                     lp_optimizer=lp_optimizer)
  # Don't keep a copy of the initial model.
  del initial_variables, lp_initial_variables

  start_step = 0
  if FLAGS.restore_checkpoints:
    # Restore unreplicated optimizer + model state from last checkpoint.
    state = checkpoints.restore_checkpoint(
        os.path.join(FLAGS.save_dir, 'checkpoints', hparam_str), state)
    # Grab last step.
    start_step = int(state.step)
    logging.info('Found model checkpointed at step %d.', start_step)

    if FLAGS.do_training:
      # TODO(kshi): It is likely that this code can lead to the job stalling for
      # 10+ hours when restarting from a checkpoint that had been trained a long
      # time, possibly because dataset skipping is slow.
      logging.info('Skipping %s steps...', start_step)
      train_ds = train_ds.skip(start_step)
      dummy_p_train_step = jax.pmap(
          lambda dropout_rng: jax.random.split(dropout_rng)[1])
      for _ in range(start_step):
        dropout_rng = dummy_p_train_step(dropout_rng)
      logging.info('Finished skipping steps')
      logging.info('Host %s has dropout_rng = %s', jax.host_id(), dropout_rng)

  # Replicate optimizer.
  state = jax_utils.replicate(state)

  # TODO(jxihong): Implement fast decoding.
  assert FLAGS.slow_decode, 'Fast decoding is not implemented yet.'

  learning_rate_fn = create_learning_rate_scheduler(
      base_learning_rate=FLAGS.lr)
  if FLAGS.do_training:
    p_train_step = jax.pmap(
        functools.partial(
            train_step,
            bos_token=bos_id,
            eos_token=eos_id,
            learning_rate_fn=learning_rate_fn,
            config=train_config,
            lp_config=lp_train_config),
        axis_name='batch',
        static_broadcasted_argnums=(4,))

  if FLAGS.do_evaluation:
    p_eval_step = jax.pmap(
        functools.partial(eval_step,
                          bos_token=bos_id,
                          eos_token=eos_id,
                          config=eval_config,
                          lp_config=lp_eval_config),
        axis_name='batch')

  if FLAGS.do_prediction:
    p_init_cache = jax.pmap(
        functools.partial(
            initialize_cache,
            max_decode_len=FLAGS.max_target_length,
            config=predict_config,
            lp_config=lp_predict_config),
        axis_name='batch')
    p_pred_step = jax.pmap(
        functools.partial(
            predict_step,
            bos_token=bos_id,
            eos_token=eos_id,
            max_decode_len=FLAGS.max_target_length,
            config=predict_config,
            lp_config=lp_predict_config,
            slow_decode=FLAGS.slow_decode),
        axis_name='batch',
        static_broadcasted_argnums=(5,))

  # Main Train Loop
  # ---------------------------------------------------------------------------

  logging.info('Starting training!')
  metrics_all = []
  latent_metrics_all = []
  tick = time.time()
  train_iter = train_ds.as_numpy_iterator()

  if FLAGS.do_prediction and start_step == FLAGS.num_train_steps:
    start_step -= 1

  for step in range(start_step, FLAGS.num_train_steps):
    is_last_step = step == FLAGS.num_train_steps - 1

    if FLAGS.do_training:
      inputs, outputs, targets, rng = load_data(next(train_iter), rng)

      state, metrics, latent_metrics, dropout_rng = p_train_step(  # pylint: disable=undefined-variable
          state, inputs, outputs, targets, step <= FLAGS.num_pretrain_steps,
          dropout_rng=dropout_rng)
      metrics_all.append(metrics)
      latent_metrics_all.append(latent_metrics)

      # Periodic metric handling.

      # Training Metrics
      if (step and step % FLAGS.log_freq == 0) or is_last_step:
        logging.info('Gathering training metrics.')
        metrics_all = common_utils.get_metrics(metrics_all)
        lr = metrics_all.pop('learning_rate').mean()
        metrics_sums = jax.tree.map(jnp.sum, metrics_all)
        denominator = metrics_sums.pop('denominator')
        summary = jax.tree.map(
            lambda x: x / denominator,  # pylint: disable=cell-var-from-loop
            metrics_sums)
        summary['learning_rate'] = lr
        # Calculate (clipped) perplexity after averaging log-perplexities:
        summary['perplexity'] = jnp.clip(jnp.exp(summary['loss']), max=1.0e4)

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
      if FLAGS.do_evaluation and (
          (step and step % FLAGS.eval_freq == 0) or is_last_step):
        logging.info('Gathering evaluation metrics.')
        t_evaluation_start = time.time()
        eval_metrics = []
        latent_eval_metrics = []
        for batches in eval_ds.as_numpy_iterator():
          inputs, outputs, targets, rng = load_data(batches, rng)

          metrics, latent_metrics = p_eval_step(state, inputs, outputs, targets)  # pylint: disable=undefined-variable
          eval_metrics.append(metrics)
          latent_eval_metrics.append(latent_metrics)

        eval_metrics = common_utils.get_metrics(eval_metrics)
        eval_metrics_sums = jax.tree.map(jnp.sum, eval_metrics)
        eval_denominator = eval_metrics_sums.pop('denominator')
        eval_summary = jax.tree.map(
            lambda x: x / eval_denominator,  # pylint: disable=cell-var-from-loop
            eval_metrics_sums)

        if jax.host_id() == 0:
          logging.info('Evaluation time: %.4f s step %d, loss: %.4f.',
                       time.time()-t_evaluation_start, step,
                       eval_summary['loss'])
          for key, val in eval_summary.items():
            summary_writer.scalar('eval/' + key, val, step)
          summary_writer.flush()

    # Beam search metrics.
    if FLAGS.do_prediction and (
        (step and step % FLAGS.predict_freq == 0) or is_last_step):
      logging.info('Gathering beam search metrics.')
      test_ds = final_test_dataset if is_last_step else quick_test_dataset

      for dataset, predict_or_test in [(predict_ds, 'predict'),
                                       (test_ds, 'test')]:

        for beam_size in [FLAGS.beam_size]:
          t_inference_start = time.time()
          total_successes = 0
          total_denominator = 0

          ios, targets_list, predictions, _, top_of_beams, scores = (
              [], [], [], [], [], [])
          logging.info('host %d: starting %s',
                       jax.host_id(), predict_or_test)
          for batch_i, batches in enumerate(dataset.as_numpy_iterator()):
            pred_batch = batches
            # Handle final odd-sized batch by padding instead of dropping it.
            cur_pred_batch_size = pred_batch['inputs'].shape[0]
            logging.info('host %d %s: got batch %d, cur_pred_batch_size = %d, '
                         'n_devices = %d',
                         jax.host_id(), predict_or_test, batch_i,
                         cur_pred_batch_size, n_devices)
            if cur_pred_batch_size % n_devices:
              padded_size = int(
                  np.ceil(cur_pred_batch_size / n_devices) * n_devices)
              # pylint: disable=cell-var-from-loop
              pred_batch = jax.tree.map(
                  lambda x: pad_examples(x, padded_size), pred_batch)
              logging.info('host %d %s: padded batch to padded_size=%d',
                           jax.host_id(), predict_or_test, padded_size)
            inputs, outputs, targets, rng = load_data(pred_batch, rng)

            cache, lp_cache = (p_init_cache(inputs, outputs, targets)  # pylint: disable=undefined-variable
                               if not FLAGS.slow_decode else (None, None))
            predicted, _ = p_pred_step(state, inputs, outputs, cache, lp_cache,  # pylint: disable=undefined-variable
                                       beam_size)
            predicted = tohost(predicted)
            inputs, outputs, targets = map(tohost, (inputs, outputs, targets))

            for i, beams in enumerate(predicted):
              inps, outs = decode_io(inputs[i], outputs[i])

              if FLAGS.model_type == 'spec_decomposer_model':
                ground_truth = decode_spec(targets[i])
                best_prediction, score = eval_predicted_spec_decomposer_model(
                    beams, ground_truth, decode_spec)
                decode_to_str_fn = decode_spec
              elif FLAGS.model_type == 'synthesizer_model':
                ground_truth = decode_program_str(targets[i])
                best_prediction, score = eval_predicted_synthesizer_model(
                    beams, inps, outs, decode_program)
                decode_to_str_fn = decode_program_str
              elif FLAGS.model_type in ['joint_model', 'baseline_model']:
                ground_truth = decode_program_str(targets[i])
                ground_truth_program = decode_program(targets[i])
                ground_truth_outs = run_program(ground_truth_program, inps)
                best_prediction, score = eval_predicted_synthesizer_model(
                    beams, inps, ground_truth_outs, decode_program)
                decode_to_str_fn = decode_program_str
              else:
                raise ValueError(f'Unknown model type {FLAGS.model_type}')

              if score == 1:
                total_successes += 1
              total_denominator += 1

              beams_target = [decode_to_str_fn(beam) for beam in beams]

              ios.append(' ; '.join(map(str, zip(inps, outs))))
              targets_list.append(ground_truth)
              predictions.append(best_prediction)
              scores.append(score)
              logging.info('')
              logging.info('ios: %s', ios[-1])
              logging.info('targets[%s]: %s', i, targets[i])
              logging.info('ground_truth: %s', ground_truth)
              logging.info('predicted beam: %s', '\n'.join(beams_target))
              logging.info('best_prediction: %s', best_prediction)
              logging.info('score: %s', score)
              logging.info('beams: %s', beams)

              if not ground_truth:
                logging.warn('ground_truth is empty!')

              top_of_beam = []
              for index, beam in enumerate(beams[:-5:-1]):
                top_of_beam.append('index: {}, decoded: {}, tokens: {}'.format(
                    index, decode_to_str_fn(beam), beam))
              top_of_beams.append('\n\n'.join(top_of_beam))

          logging.info('host %d %s: total_success=%d, total_denominator=%d',
                       jax.host_id(), predict_or_test, total_successes,
                       total_denominator)
          all_total_successes, all_total_denominator = per_host_sum_pmap(
              jax.tree.map(np.array, (total_successes, total_denominator)))
          logging.info('host %d %s: all_total_successes=%d, '
                       'all_total_denominator=%d',
                       jax.host_id(), predict_or_test, all_total_successes,
                       all_total_denominator)

          # Record beam search results as text summaries.
          message = []
          for n in np.random.choice(np.arange(len(predictions)), 8):
            text = (f'ios: {ios[n]}\n\ntarget: {targets_list[n]}\n\n'
                    f'predicted: {predictions[n]}\n\n'
                    f'score: {scores[n]}\n\n'
                    f'top of beam:\n\n{top_of_beams[n]}\n\n')
            message.append(text)

          # Write to tensorboard.
          if jax.host_id() == 0:
            accuracy = 100 * all_total_successes / all_total_denominator
            logging.info(
                '%s results, step %d, beam size %d: %s / %s = %.2f%% (%.2f s)',
                predict_or_test, step, beam_size,
                all_total_successes, all_total_denominator, accuracy,
                time.time() - t_inference_start)
            summary_writer.scalar(
                '{}/beam-size-{}'.format(predict_or_test, beam_size),
                accuracy, step)

            summary_writer.text('{}-samples-beam-{}'.format(predict_or_test,
                                                            beam_size),
                                '\n------\n'.join(message), step)
            summary_writer.flush()

      if not FLAGS.do_training:
        # If only prediction, then don't do it multiple times.
        break

    # Save a Checkpoint. Do this at the end of the training loop, so that if a
    # worker is descheduled during a round of prediction (which takes a while),
    # we will redo prediction upon restarting (to avoid losing data).
    if FLAGS.do_training and (
        (step % FLAGS.checkpoint_freq == 0 and step > 0) or is_last_step):
      # Save unreplicated optimizer + model state.
      checkpoints.save_checkpoint_multiprocess(
          os.path.join(FLAGS.save_dir, 'checkpoints', hparam_str),
          jax_utils.unreplicate(state),
          step,
          keep_every_n_steps=100_000)

if __name__ == '__main__':
  app.run(main)
