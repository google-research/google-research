# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

# pytype: disable=wrong-arg-count
# pytype: disable=attribute-error

import collections
import functools
import os
import random
import sys
import time

from absl import app
from absl import flags
from absl import logging
from flax import jax_utils
from flax import linen as nn
from flax import optim
from flax.metrics import tensorboard
from flax.training import checkpoints
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow.compat.v2 as tf

from latent_programmer import decode
from latent_programmer import models as base_models
from latent_programmer.decomposition_transformer_attention import decomposition_models as models
from latent_programmer.decomposition_transformer_attention import input_pipeline
from latent_programmer.tasks.robust_fill import dsl as robust_fill_dsl
from latent_programmer.tasks.robust_fill import tokens as dsl_tokens
from latent_programmer.tasks.scan import scan_vocab
from latent_programmer.tasks.scan import translate_scan

sys.path.append('../../')
gfile = tf.io.gfile

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 0, 'Fixed random seed for training.')
flags.DEFINE_integer('repeat', 0, 'An ID for this repetition of the same seed.')
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

flags.DEFINE_string('dataset_filepattern', None,
                    'Filepattern for TFRecord dataset.')
flags.DEFINE_string('test_dataset_filepattern', None,
                    'Filepattern for TFRecord test dataset.')
flags.DEFINE_integer('per_device_batch_size', 16,
                     'Number of program tasks in a batch.')
flags.DEFINE_integer('num_strings_per_task', 4,
                     'Number of input/output strings per task.')
flags.DEFINE_integer('max_program_length', 100,
                     'Maximum number of tokens in program.')
flags.DEFINE_integer('max_characters', 120,
                     'Maximum number of characters in input/output strings.')
flags.DEFINE_integer('predict_max_characters', 200,
                     'Maximum number of characters in input/output strings for '
                     'prediction.')

flags.DEFINE_string('save_dir', None, 'Directory to save results to.')
flags.DEFINE_integer('num_train_steps', 2000000, 'Number of training steps.')
flags.DEFINE_integer('num_eval_steps', 10, 'Number of evaluation steps.')
flags.DEFINE_integer('num_quick_test_steps', 10,
                     'Number of test steps during training.')
flags.DEFINE_integer('num_final_test_steps', 100,
                     'Number of test steps after training is finished.')
flags.DEFINE_integer('train_set_batches', -1,
                     'Number of batches for a smaller training set, or -1 to '
                     'use the original training set size.')
flags.DEFINE_integer('log_freq', 2000, 'Number of steps between training logs.')
flags.DEFINE_integer('eval_freq', 10000, 'Number of steps between eval.')
flags.DEFINE_integer('predict_freq', 50000,
                     'Number of steps between prediction (beam search).')
flags.DEFINE_integer('checkpoint_freq', 50000,
                     'Number of steps between checkpoint saves.')
flags.DEFINE_integer('finetune_start_step', -1,
                     'Step the initial checkpoint should start at for '
                     'finetuning, or -1 if not finetuning.')
flags.DEFINE_bool('restore_checkpoints', True,
                  'Whether to restore from existing model checkpoints.')

flags.DEFINE_bool('use_bos_separators', True,
                  'Whether to use BOS tokens between partial programs.')
flags.DEFINE_string('attention_mask_type', 'bos_full_attention',
                    'The kind of attention mask to use. Options are: baseline, '
                    'bos_to_last, bos_to_bos_and_last, bos_full_attention')

flags.DEFINE_bool('use_relative_attention', True,
                  'Whether to use relative positonal embeddings.')
flags.DEFINE_bool('bos_special_attention', False,
                  'Whether to use special relative attention computation for '
                  'BOS tokens.')
flags.DEFINE_integer('num_position_buckets', 32,
                     'Number of relative attention position buckets.')
flags.DEFINE_integer('max_distance', 128,
                     'Max distance for relative attention positions.')

flags.DEFINE_enum('dataset_type', 'robust_fill',
                  ['robust_fill', 'robust_fill_base', 'scan'],
                  'The kind of dataset to use.')


_internal = False
if not _internal:
  flags.DEFINE_string('xm_parameters', None,
                      'String specifying hyperparamter search.')


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


# Train / eval / decode step functions.
# -----------------------------------------------------------------------------


def train_step(optimizer,
               inputs,
               outputs,
               programs,
               learning_rate_fn,
               config,
               dropout_rng):
  """Train on batch of program tasks."""
  # We handle PRNG splitting inside the top pmap, rather
  # than handling it outside in the training loop - doing the
  # latter can add some stalls to the devices.
  dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

  weights = jnp.where(programs > 0, 1, 0).astype(jnp.float32)

  def loss_fn(params):
    """Loss function used for training."""
    logits = models.DecomposeAttentionTransformer(config).apply(
        {'params': params},
        inputs,
        outputs,
        programs,
        rngs={'dropout': dropout_rng})
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
  return new_optimizer, metrics, new_dropout_rng


def eval_step(params, inputs, outputs, programs, eos_token, config):
  """Collect metrics for evaluation during training."""
  weights = jnp.where(
      jnp.logical_and(programs > 0,
                      jnp.logical_and(programs != config.base_config.bos_token,
                                      programs != eos_token)),
      1, 0).astype(jnp.float32)
  logits = models.DecomposeAttentionTransformer(config).apply(
      {'params': params}, inputs, outputs, programs)

  return compute_metrics(logits, programs, weights)


def initialize_cache(inputs, outputs, programs, max_decode_len, config):
  """Initialize a cache for a given input shape and max decode length."""
  target_shape = (programs.shape[0], max_decode_len)
  dtype = config.base_config.dtype
  initial_variables = models.DecomposeAttentionTransformer(config).init(
      jax.random.PRNGKey(0),
      jnp.ones(inputs.shape, dtype),
      jnp.ones(outputs.shape, dtype),
      jnp.ones(target_shape, dtype))
  return initial_variables['cache']


def predict_step(params,
                 inputs,
                 outputs,
                 cache,
                 beam_size,
                 eos_token,
                 max_decode_len,
                 config,
                 slow_decode=True):
  """Predict translation with fast decoding beam search on a batch."""
  # Prepare transformer fast-decoder call for beam search: for beam search, we
  # need to set up our decoder model to handle a batch size equal to
  # batch_size * beam_size, where each batch item's data is expanded in-place
  # rather than tiled.
  flat_encoded = decode.flat_batch_beam_expand(
      models.DecomposeAttentionTransformer(config).apply(
          {'params': params},
          inputs,
          outputs,
          method=models.DecomposeAttentionTransformer.encode),
      beam_size)

  encoded_padding_mask = jnp.where(outputs > 0, 1, 0).astype(jnp.float32)
  flat_encoded_padding_mask = decode.flat_batch_beam_expand(
      encoded_padding_mask, beam_size)

  if slow_decode:
    def tokens_ids_to_logits(flat_ids):
      """Token slice to logits from decoder model."""
      # --> [batch * beam, 1, vocab]
      flat_logits = models.DecomposeAttentionTransformer(config=config).apply(
          {'params': params},
          flat_ids,
          flat_encoded,
          flat_encoded_padding_mask,
          method=models.DecomposeAttentionTransformer.decode)
      return flat_logits
  else:
    def tokens_ids_to_logits(flat_ids, flat_cache):
      """Token slice to logits from decoder model."""
      # --> [batch * beam, 1, vocab]
      flat_logits, new_vars = models.DecomposeAttentionTransformer(
          config=config).apply(
              {'params': params, 'cache': flat_cache},
              flat_ids,
              flat_encoded,
              flat_encoded_padding_mask,
              mutable=['cache'],
              method=models.DecomposeAttentionTransformer.decode)
      new_flat_cache = new_vars['cache']
      # Remove singleton sequence-length dimension:
      # [batch * beam, 1, vocab] --> [batch * beam, vocab]
      flat_logits = flat_logits.squeeze(axis=1)
      return flat_logits, new_flat_cache

  # Using the above-defined single-step decoder function, run a
  # beam search over possible sequences given input encoding.
  beam_seqs, _ = decode.beam_search(
      inputs,
      cache,
      tokens_ids_to_logits,
      beam_size=beam_size,
      alpha=0.6,
      bos_token=config.base_config.bos_token,
      eos_token=eos_token,
      max_decode_len=max_decode_len,
      slow_decode=slow_decode)

  # Beam search returns [n_batch, n_beam, n_length] with beam dimension
  # sorted in increasing order of log-probability.
  return beam_seqs


# Util functions for prediction
# -----------------------------------------------------------------------------


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
    return jax.tree.map(lambda x: jnp.broadcast_to(x, (1,) + x.shape), xs)
  def post_pmap(xs):
    return jax.tree.map(lambda x: x[0], xs)
  return post_pmap(host_psum(pre_pmap(in_tree)))


def eval_predicted(predicted, inputs, outputs, parse_beam_fn):
  """Evaluate predicted program beams."""
  best_program_str, best_score = None, -1

  # predicted shape [beam_size, length]
  for beam in predicted[::-1]:
    program = parse_beam_fn(beam)
    if FLAGS.dataset_type in ['robust_fill', 'robust_fill_base']:
      try:
        p_outs = [program(inp) for inp in inputs]
        score = np.sum([p_out == out for p_out, out in zip(p_outs, outputs)])
        program_str = program.to_string()
      except:  # pylint: disable=bare-except
        score = -1
        program_str = 'did not compile'
    elif FLAGS.dataset_type == 'scan':
      assert len(inputs) == 1, 'inputs should have length 1: {}'.format(inputs)
      assert len(outputs) == 1
      input_tokens = inputs[0].split()
      expected_tokens = translate_scan.translate(input_tokens,
                                                 add_separators=False)
      expected_str = ' '.join(expected_tokens)
      score = int(program == expected_str)
      program_str = program
    else:
      raise ValueError('Unhandled dataset_type {}'.format(FLAGS.dataset_type))

    if score > best_score:
      best_program_str, best_score = program_str, score

    if best_score >= len(inputs):  # Found solution.
      break

  # best_program_str could be None if no RobustFill program compiles.
  return best_program_str, best_score


def shorten(key):
  splits = key.split('_')
  return ''.join(s[0] for s in splits)


def main(_):
  tf.enable_v2_behavior()

  tf.random.set_seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)
  random.seed(FLAGS.seed)

  # BOS special attention only makes sense if we are using relative attention
  # and it's not the baseline.
  if FLAGS.bos_special_attention and (not FLAGS.use_relative_attention or
                                      FLAGS.attention_mask_type == 'baseline'):
    raise ValueError(
        "bos_special_attention doesn't work when use_relative_attention={} and "
        'attention_mask_type={}'.format(FLAGS.use_relative_attention,
                                        FLAGS.attention_mask_type))

  if not gfile.isdir(FLAGS.save_dir):
    gfile.makedirs(FLAGS.save_dir)

  xm_client = xmanager_api.XManagerApi(xm_deployment_env='alphabet')
  work_unit = xm_client.get_current_work_unit()
  hparam_dict = work_unit.parameters['args']
  hparam_str = ','.join(['%s=%s' % (shorten(k), str(v))
                         for k, v in hparam_dict.items()])

  # Number of local devices for this host.
  n_devices = jax.local_device_count()

  if jax.host_id() == 0:
    summary_writer = tensorboard.SummaryWriter(
        os.path.join(FLAGS.save_dir, 'tb', hparam_str))

  batch_size = FLAGS.per_device_batch_size * n_devices
  io_shape = (FLAGS.per_device_batch_size,
              FLAGS.num_strings_per_task,
              FLAGS.max_characters)
  predict_io_shape = (FLAGS.per_device_batch_size,
                      FLAGS.num_strings_per_task,
                      FLAGS.predict_max_characters)
  program_shape = (FLAGS.per_device_batch_size, FLAGS.max_program_length)

  # Setup DSL
  # ---------------------------------------------------------------------------

  # Build token tables.
  if FLAGS.dataset_type in ['robust_fill', 'robust_fill_base']:
    spec_vocab = robust_fill_dsl.CHARACTER + '|'
    spec_id_token_table = {i + 3: token for i, token in enumerate(spec_vocab)}
    bos_id = 1
    eos_id = 2
    spec_id_token_table[bos_id] = robust_fill_dsl.BOS
    spec_id_token_table[eos_id] = robust_fill_dsl.EOS
    spec_token_id_table = {
        token: id for id, token in spec_id_token_table.items()
    }
    spec_vocab_size = len(spec_token_id_table) + 1  # For padding.
    program_id_token_table, _ = dsl_tokens.build_token_tables()
    program_vocab_size = len(program_id_token_table) + 1
  elif FLAGS.dataset_type == 'scan':
    raise ValueError('Not implemented yet')
    # id_char_table = {}
    # char_id_table = {}
    # id_token_table, token_id_table = scan_vocab.build_token_tables()
    # bos_token = token_id_table[scan_vocab.BOS]
    # eos_token = token_id_table[scan_vocab.EOS]
    # io_vocab_size = len(id_token_table) + 1  # For padding.
    # program_vocab_size = io_vocab_size
  else:
    raise ValueError('Unhandled dataset_type: {}'.format(FLAGS.dataset_type))

  # Parse io and program token sequences (for eval).
  def decode_io(inputs, outputs):
    """Decode io examples tokens."""
    if FLAGS.dataset_type == 'robust_fill':
      def decode_str(s):
        """Decode string tokens."""
        return ''.join([spec_id_token_table[c_id] for c_id in s if c_id > 0])

      inps, outs = [], []
      for inp, out in zip(inputs, outputs):
        inps.append(decode_str(inp))
        outs.append(decode_str(out))
      return inps, outs

    elif FLAGS.dataset_type == 'robust_fill_base':
      def decode_str(s):
        """Decode string tokens."""
        return ''.join([spec_id_token_table[c_id] for c_id in s if c_id > 0])

      inps = [decode_str(inp) for inp in inputs]
      assert len(inps) == 1
      combined_io_examples = inps[0]
      io_pairs = combined_io_examples.split('>')
      inps, outs = [], []
      for pair in io_pairs:
        inp, output = pair.split('<')
        inps.append(inp)
        outs.append(output)
      return inps, outs

    elif FLAGS.dataset_type == 'scan':
      def decode_str(s):
        """Decode string tokens."""
        return ' '.join([program_id_token_table[t_id]
                         for t_id in s if t_id > 0])

      inps = [decode_str(inp) for inp in inputs]
      dummy_outs = [''] * len(inps)
      return inps, dummy_outs

    else:
      raise ValueError('Unhandled dataset_type: {}'.format(FLAGS.dataset_type))

  def decode_program(program):
    """Decode program tokens."""
    program = program[:np.argmax(program == eos_id) + 1].astype(np.int32)

    if FLAGS.dataset_type in ['robust_fill', 'robust_fill_base']:
      # Returns either a Concat program object, or None.
      program = program[program != bos_id].tolist()
      try:
        return robust_fill_dsl.decode_program(program, program_id_token_table)
      except:  # pylint: disable=bare-except
        return None  # Program does not compile.
    elif FLAGS.dataset_type == 'scan':
      # Returns a string.
      program = program[jnp.logical_and(program != bos_id,
                                        program != eos_id)].tolist()
      return ' '.join(scan_vocab.decode(program, program_id_token_table))
    else:
      raise ValueError('Unhandled dataset_type: {}'.format(FLAGS.dataset_type))

  def decode_program_str(program):
    """Decode program tokens into a string."""
    decoded = decode_program(program)
    if FLAGS.dataset_type in ['robust_fill', 'robust_fill_base']:
      try:
        return decoded.to_string()
      except:  # pylint: disable=bare-except
        return 'did not compile'
    else:
      assert isinstance(decoded, str), '{} should be string'.format(decoded)
      return decoded

  # Load Dataset
  # ---------------------------------------------------------------------------
  logging.info('Initializing dataset.')
  if not FLAGS.dataset_filepattern:
    raise ValueError('Must specify filepattern to dataset.')

  # Training dataset.
  logging.info('Loading dataset from %s', FLAGS.dataset_filepattern)
  padded_shapes = (io_shape[1:], io_shape[1:], program_shape[1:])
  logging.info('padded_shapes: %s', padded_shapes)

  if FLAGS.dataset_type == 'robust_fill':
    create_dataset_fn = functools.partial(
        input_pipeline.create_robust_fill_dataset,
        num_strings_per_task=FLAGS.num_strings_per_task)
  elif FLAGS.dataset_type == 'robust_fill_base':
    create_dataset_fn = functools.partial(
        input_pipeline.create_robust_fill_dataset_from_old_tf_record,
        split_ios=False)
  elif FLAGS.dataset_type == 'scan':
    create_dataset_fn = input_pipeline.create_scan_dataset_from_tf_record
  else:
    raise ValueError('Unhandled dataset_type: {}'.format(FLAGS.dataset_type))

  dataset = create_dataset_fn(
      FLAGS.dataset_filepattern, spec_token_id_table,
      use_bos_separators=FLAGS.use_bos_separators)
  dataset = dataset.padded_batch(
      batch_size,
      padded_shapes=padded_shapes,
      drop_remainder=True)
  # Split evaluation and training.
  eval_ds = dataset.take(FLAGS.num_eval_steps)
  # Decrease batch of predict dataset to handle beam search.
  predict_padded_shapes = (predict_io_shape[1:],
                           predict_io_shape[1:],
                           program_shape[1:])
  logging.info('predict_padded_shapes: %s', predict_padded_shapes)
  predict_ds = eval_ds.unbatch().padded_batch(
      int(np.ceil(batch_size / 10)),
      padded_shapes=predict_padded_shapes)
  train_ds = dataset.skip(FLAGS.num_eval_steps)
  if FLAGS.train_set_batches > 0:
    train_ds = train_ds.take(FLAGS.train_set_batches)
  train_ds = train_ds.repeat()

  test_dataset = create_dataset_fn(
      FLAGS.test_dataset_filepattern, spec_token_id_table,
      use_bos_separators=FLAGS.use_bos_separators)
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
  default_config = base_models.TransformerConfig(
      vocab_size=spec_vocab_size, output_vocab_size=program_vocab_size)
  base_config = base_models.TransformerConfig(
      vocab_size=spec_vocab_size,
      output_vocab_size=program_vocab_size,
      shift=True,
      emb_dim=FLAGS.embedding_dim,
      num_heads=FLAGS.num_heads,
      num_layers=FLAGS.num_layers,
      qkv_dim=FLAGS.embedding_dim,
      mlp_dim=FLAGS.hidden_dim,
      max_len=max(FLAGS.max_characters, FLAGS.max_program_length),
      dropout_rate=FLAGS.dropout_rate,
      attention_dropout_rate=FLAGS.attention_dropout_rate,
      use_relative_attention=FLAGS.use_relative_attention,
      deterministic=False,
      decode=False,
      bos_token=bos_id,
      num_input_relative_position_buckets=FLAGS.num_position_buckets,
      max_input_distance=min(
          FLAGS.max_distance, default_config.max_input_distance),
      num_output_relative_position_buckets=FLAGS.num_position_buckets,
      max_output_distance=min(
          FLAGS.max_distance, default_config.max_output_distance),
      num_input_cross_output_relative_position_buckets=(
          FLAGS.num_position_buckets),
      max_input_cross_output_distance=min(
          FLAGS.max_distance, default_config.max_input_cross_output_distance),
      num_program_relative_position_buckets=FLAGS.num_position_buckets,
      max_program_distance=min(
          FLAGS.max_distance, default_config.max_program_distance),
      num_program_cross_embed_relative_position_buckets=(
          FLAGS.num_position_buckets),
      max_program_cross_embed_distance=min(
          FLAGS.max_distance, default_config.max_program_cross_embed_distance))
  train_config = models.DecomposeAttentionTransformerConfig(
      base_config=base_config,
      attention_mask_type=FLAGS.attention_mask_type,
      bos_special_attention=FLAGS.bos_special_attention,
      dataset_type=FLAGS.dataset_type)
  eval_config = models.DecomposeAttentionTransformerConfig(
      base_config=base_config.replace(deterministic=True),
      attention_mask_type=FLAGS.attention_mask_type,
      bos_special_attention=FLAGS.bos_special_attention,
      dataset_type=FLAGS.dataset_type)
  predict_config = models.DecomposeAttentionTransformerConfig(
      base_config=base_config.replace(
          shift=False, deterministic=True,
          decode=not FLAGS.slow_decode,
          max_len=max(FLAGS.max_characters, FLAGS.max_program_length,
                      FLAGS.predict_max_characters)),
      attention_mask_type=FLAGS.attention_mask_type,
      bos_special_attention=FLAGS.bos_special_attention,
      dataset_type=FLAGS.dataset_type)

  rng = jax.random.PRNGKey(FLAGS.seed)
  rng = jax.random.fold_in(rng, jax.host_id())
  rng, init_rng = jax.random.split(rng)

  dropout_rng = jax.random.split(rng, jax.local_device_count())
  del rng

  m = models.DecomposeAttentionTransformer(eval_config)
  initial_variables = jax.jit(m.init)(
      init_rng,
      jnp.ones(io_shape, jnp.float32),
      jnp.ones(io_shape, jnp.float32),
      jnp.ones(program_shape, jnp.float32))

  optimizer_def = optim.Adam(
      FLAGS.lr,
      beta1=0.9,
      beta2=0.98,
      eps=1e-9,
      weight_decay=FLAGS.weight_decay)
  optimizer = optimizer_def.create(initial_variables['params'])

  del initial_variables  # Don't keep a copy of the initial model.

  start_step = 0
  if FLAGS.restore_checkpoints:
    # Restore unreplicated optimizer + model state from last checkpoint.
    optimizer = checkpoints.restore_checkpoint(
        os.path.join(FLAGS.save_dir, 'checkpoints', hparam_str), optimizer)
    # Grab last step.
    start_step = int(optimizer.state.step)
    logging.info('Found model checkpointed at step %d.', start_step)
    if FLAGS.finetune_start_step > 0:
      logging.info('Checking that start_step (%s) == finetune_start_step (%s)',
                   start_step, FLAGS.finetune_start_step)
      assert start_step >= FLAGS.finetune_start_step
      steps_to_skip = start_step - FLAGS.finetune_start_step
    else:
      steps_to_skip = start_step

    # TODO(kshi): It is likely that this code can lead to the job stalling for
    # 10+ hours when restarting from a checkpoint that had been trained a long
    # time, possibly because dataset skipping is slow.
    logging.info('Skipping %s steps...', steps_to_skip)
    train_ds = train_ds.skip(steps_to_skip)
    dummy_p_train_step = jax.pmap(
        lambda dropout_rng: jax.random.split(dropout_rng)[1])
    for _ in range(steps_to_skip):
      dropout_rng = dummy_p_train_step(dropout_rng)
    logging.info('Finished skipping steps')
    logging.info('Host %s has dropout_rng = %s', jax.host_id(), dropout_rng)

  # Replicate optimizer.
  optimizer = jax_utils.replicate(optimizer)

  # TODO(jxihong): Implement fast decoding.
  assert FLAGS.slow_decode, 'Fast decoding is not implemented yet.'

  if FLAGS.finetune_start_step <= 0:
    learning_rate_fn = create_learning_rate_scheduler(
        base_learning_rate=FLAGS.lr)
  else:
    # Constant LR for finetuning.
    learning_rate_fn = create_learning_rate_scheduler(
        base_learning_rate=FLAGS.lr,
        factors='constant')
  p_train_step = jax.pmap(
      functools.partial(
          train_step,
          learning_rate_fn=learning_rate_fn,
          config=train_config),
      axis_name='batch')
  p_eval_step = jax.pmap(
      functools.partial(eval_step,
                        eos_token=eos_id,
                        config=eval_config),
      axis_name='batch')
  p_init_cache = jax.pmap(
      functools.partial(
          initialize_cache,
          max_decode_len=FLAGS.max_program_length,
          config=predict_config),
      axis_name='batch')
  p_pred_step = jax.pmap(
      functools.partial(
          predict_step,
          eos_token=eos_id,
          max_decode_len=FLAGS.max_program_length,
          config=predict_config,
          slow_decode=FLAGS.slow_decode),
      axis_name='batch',
      static_broadcasted_argnums=(4,))

  # Main Train Loop
  # ---------------------------------------------------------------------------

  logging.info('Starting training!')
  metrics_all = []
  tick = time.time()
  train_iter = train_ds.as_numpy_iterator()
  for step in range(start_step, FLAGS.num_train_steps):
    inputs, outputs, programs = common_utils.shard(next(train_iter))

    optimizer, metrics, dropout_rng = p_train_step(
        optimizer, inputs, outputs, programs, dropout_rng=dropout_rng)
    metrics_all.append(metrics)
    is_last_step = step == FLAGS.num_train_steps - 1

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
    if (step and step % FLAGS.eval_freq == 0) or is_last_step:
      logging.info('Gathering evaluation metrics.')
      t_evaluation_start = time.time()
      eval_metrics = []
      for batches in eval_ds.as_numpy_iterator():
        inputs, outputs, programs = common_utils.shard(batches)

        metrics = p_eval_step(optimizer.target, inputs, outputs, programs)
        eval_metrics.append(metrics)

      eval_metrics = common_utils.get_metrics(eval_metrics)
      eval_metrics_sums = jax.tree.map(jnp.sum, eval_metrics)
      eval_denominator = eval_metrics_sums.pop('denominator')
      eval_summary = jax.tree.map(
          lambda x: x / eval_denominator,  # pylint: disable=cell-var-from-loop
          eval_metrics_sums)

      if jax.host_id() == 0:
        logging.info('Evaluation time: %.4f s step %d, loss: %.4f.',
                     time.time()-t_evaluation_start, step, eval_summary['loss'])
        for key, val in eval_summary.items():
          summary_writer.scalar('eval/' + key, val, step)
        summary_writer.flush()

    # Beam search metrics.
    if (step and step % FLAGS.predict_freq == 0) or is_last_step:
      logging.info('Gathering beam search metrics.')
      test_ds = final_test_dataset if is_last_step else quick_test_dataset

      for dataset, predict_or_test in [(predict_ds, 'predict'),
                                       (test_ds, 'test')]:

        for beam_size in [1, 10]:
          t_inference_start = time.time()
          total_successes = 0
          total_denominator = 0
          pred_successes = collections.defaultdict(int)
          pred_denominators = collections.defaultdict(int)

          ios, targets, predictions, top_of_beams = [], [], [], []
          for batches in dataset.as_numpy_iterator():
            pred_batch = batches
            # Handle final odd-sized batch by padding instead of dropping it.
            cur_pred_batch_size = pred_batch[0].shape[0]
            if cur_pred_batch_size % n_devices:
              padded_size = int(
                  np.ceil(cur_pred_batch_size / n_devices) * n_devices)
              # pylint: disable=cell-var-from-loop
              pred_batch = jax.tree.map(
                  lambda x: pad_examples(x, padded_size), pred_batch)
            inputs, outputs, programs = common_utils.shard(pred_batch)

            cache = (p_init_cache(inputs, outputs, programs)
                     if not FLAGS.slow_decode else None)
            predicted = p_pred_step(optimizer.target, inputs, outputs, cache,
                                    beam_size)
            predicted = tohost(predicted)
            inputs, outputs, programs = map(tohost, (inputs, outputs, programs))

            for i, beams in enumerate(predicted):
              inps, outs = decode_io(inputs[i], outputs[i])
              predicted_program_str, p_score = eval_predicted(
                  beams, inps, outs, parse_beam_fn=decode_program)

              ground_truth_tokens = programs[i]
              ground_truth_str = decode_program_str(ground_truth_tokens)

              # Split by length of program.
              if FLAGS.use_bos_separators:
                # There is a BOS token between parts and at the end.
                num_expressions = int(jnp.sum(ground_truth_tokens == bos_id))
              else:
                # TODO(kshi): Fix this if we care about it enough.
                num_expressions = -1
              pred_denominators[num_expressions] += 1
              total_denominator += 1
              if p_score >= len(inps):
                pred_successes[num_expressions] += 1
                total_successes += 1

              ios.append(' ; '.join(map(str, zip(inps, outs))))
              targets.append(ground_truth_str)
              predictions.append(predicted_program_str)
              logging.info('ios: %s', ios[-1])
              logging.info('target: %s', targets[-1])
              beams_log = [decode_program_str(beam) for beam in beams]
              logging.info('predicted beam: %s', '\n'.join(beams_log))

              top_of_beam = []
              for index, beam in enumerate(beams[:-5:-1]):
                top_of_beam.append('index: {}, decoded: {}, tokens: {}'.format(
                    index, decode_program_str(beam), beam))
              top_of_beams.append('\n\n'.join(top_of_beam))

          all_total_successes, all_total_denominator = per_host_sum_pmap(
              jax.tree.map(np.array, (total_successes, total_denominator)))
          all_pred_successes, all_pred_denominators = per_host_sum_pmap(
              jax.tree.map(np.array, (pred_successes, pred_denominators)))

          # Record beam search results as text summaries.
          message = []
          for n in np.random.choice(np.arange(len(predictions)), 8):
            text = (f'ios: {ios[n]}\n\ntarget: {targets[n]}\n\n'
                    f'predicted: {predictions[n]}\n\n'
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

            for length in sorted(all_pred_successes.keys()):
              this_length_accuracy = (100 * all_pred_successes[length]
                                      / all_pred_denominators[length])
              logging.info('  accuracy for length %s: %s / %s = %.2f%%',
                           length, all_pred_successes[length],
                           all_pred_denominators[length], this_length_accuracy)
              summary_writer.scalar(
                  '{}-by-length/beam-size-{}-length-{}'.format(
                      predict_or_test, beam_size, length),
                  this_length_accuracy, step)

            summary_writer.text('{}-samples-beam-{}'.format(predict_or_test,
                                                            beam_size),
                                '\n------\n'.join(message), step)
            summary_writer.flush()

    # Save a Checkpoint. Do this at the end of the training loop, so that if a
    # worker is descheduled during a round of prediction (which takes a while),
    # we will redo prediction upon restarting (to avoid losing data).
    if (step % FLAGS.checkpoint_freq == 0 and step > 0) or is_last_step:
      if jax.host_id() == 0:
        # Save unreplicated optimizer + model state.
        checkpoints.save_checkpoint(
            os.path.join(FLAGS.save_dir, 'checkpoints', hparam_str),
            jax_utils.unreplicate(optimizer),
            step)

if __name__ == '__main__':
  app.run(main)
