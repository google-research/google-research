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

"""Machine Translation example.

This script trains a Transformer on a WMT dataset.
"""

import csv
import functools
import os
import time

from absl import app
from absl import flags
from absl import logging
from flax import jax_utils
from flax import linen as nn
from flax import optim
from flax.training import checkpoints
from flax.training import common_utils
import jax
import jax.nn
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from data_selection.wmt import common
from data_selection.wmt import decode
from data_selection.wmt import input_pipeline
from data_selection.wmt import models

LAYERNORM_ADAPTER = 'LayerNorm'
ENDCODE_DECODE_B5 = 'encoderdecoderblock_5'
PASSTHRU = ''
NONE = 'None'
FLAGS = flags.FLAGS

flags.DEFINE_string(
    'model_dir', default=None,
    help='Directory to store model data.')

flags.DEFINE_string(
    'is_save_path', default=None,
    help='Path to save is scores to.')

flags.DEFINE_string(
    'is_score_filename', default=None,
    help='Filename to save is scores to.')

flags.DEFINE_string(
    'is_diff_name', default=None,
    help='Filename to save is diff scores to.')

flags.DEFINE_string(
    'base_log_loss_file', default=None,
    help='Filename of log loss from base run.')

flags.DEFINE_string(
    'pretrained_model_dir', default=None,
    help='Directory of pretrained model data.')

flags.DEFINE_string(
    'data_dir', default=None,
    help='Tensorflow datasets directory.')

flags.DEFINE_string(
    'vocab_path', default=None,
    help='Path to load or store sentencepiece vocab file.')

flags.DEFINE_integer(
    'vocab_size', default=32000,
    help='Vocabulary size if `vocab_path` is not given.')

flags.DEFINE_string(
    'dataset_name', default='wmt17_translate/de-en',
    help='Name of TFDS translation dataset to use.')

flags.DEFINE_integer(
    'batch_size', default=256,
    help='Per host batch size for training.')

flags.DEFINE_float(
    'learning_rate', default=0.0625,
    help='Base learning rate.')

flags.DEFINE_float(
    'label_smoothing', default=0.1,
    help='Cross entropy loss label smoothing.')

flags.DEFINE_float(
    'weight_decay', default=0.0,
    help='Decay factor for AdamW style weight decay.')

flags.DEFINE_integer(
    'max_target_length', default=256,
    help='Maximum length cutoff for training examples.')

flags.DEFINE_integer(
    'max_eval_target_length', default=256,
    help='Maximum length cutoff for eval examples.')

flags.DEFINE_bool(
    'share_embeddings', default=True,
    help='Inputs and targets share embedding.')

flags.DEFINE_bool(
    'logits_via_embedding', default=True,
    help='Final logit transform uses embedding matrix transpose.')

flags.DEFINE_integer(
    'num_layers', default=6,
    help='Number of transformer layers.')

flags.DEFINE_integer(
    'qkv_dim', default=1024,
    help='Size of query/key/value for attention.')

flags.DEFINE_integer(
    'emb_dim', default=1024,
    help='Size of embeddings.')

flags.DEFINE_integer(
    'mlp_dim', default=4096,
    help='Size of the MLP.')

flags.DEFINE_integer(
    'num_heads', default=16,
    help='Number of attention heads.')

flags.DEFINE_float(
    'dropout_rate', default=0.1,
    help='Dropout rate.')

flags.DEFINE_float(
    'attention_dropout_rate', default=0.1,
    help='Attention dropout rate.')

flags.DEFINE_integer(
    'random_seed', default=0,
    help='Integer for PRNG random seed.')


flags.DEFINE_bool(
    'save_checkpoints', default=True,
    help='Whether to save model checkpoints.')

flags.DEFINE_bool(
    'restore_checkpoints', default=True,
    help='Whether to restore from existing model checkpoints.')

flags.DEFINE_bool(
    'use_bfloat16', default=True,
    help=('Use bfloat16 mixed precision training instead of float32.'))

flags.DEFINE_string(
    'jax_backend_target', default=None,
    help=('TPU grpc target for use with cloud TPUs.'
          ' e.g. grpc://192.168.0.2:8470'))

flags.DEFINE_integer(
    'paracrawl_size', default=1200000,
    help='Number of examples to sample from paracrawl.')

flags.DEFINE_enum(
    'adapter', default=NONE, enum_values=[LAYERNORM_ADAPTER,
                                          ENDCODE_DECODE_B5,
                                          PASSTHRU,
                                          NONE],
    help='Whether to finetune only some parameters.')


def compute_per_example_loss(logits,
                             targets,
                             weights=None,
                             label_smoothing=0.0):
  """Compute weighted cross entropy and entropy for log probs and targets.

  Args:
   logits: [batch, length, num_classes] float array.
   targets: categorical targets [batch, length] int array.
   weights: None or array of shape [batch, length].
   label_smoothing: label smoothing constant, used to determine the on and
     off values.

  Returns:
    Tuple of scalar loss and batch normalizing factor.
  """
  if logits.ndim != targets.ndim + 1:
    raise ValueError('Incorrect shapes. Got shape %s logits and %s targets' %
                     (str(logits.shape), str(targets.shape)))
  vocab_size = logits.shape[-1]
  confidence = 1.0 - label_smoothing
  low_confidence = (1.0 - confidence) / (vocab_size - 1)
  normalizing_constant = -(
      confidence * jnp.log(confidence) + (vocab_size - 1) *
      low_confidence * jnp.log(low_confidence + 1e-20))
  soft_targets = common_utils.onehot(
      targets, vocab_size, on_value=confidence, off_value=low_confidence)

  loss = -jnp.sum(soft_targets * nn.log_softmax(logits), axis=-1)
  loss = loss - normalizing_constant

  if weights is not None:
    loss = loss * weights

  return loss.sum(axis=-1)/ weights.sum(axis=-1)


def eval_for_is_step(params, batch, config, label_smoothing=0.0):
  """Calculate evaluation metrics on a batch."""
  inputs, targets = batch['inputs'], batch['targets']
  weights = jnp.where(targets > 0, 1.0, 0.0)
  logits = models.Transformer(config).apply({'params': params}, inputs, targets)
  losses = compute_per_example_loss(logits,
                                    targets,
                                    weights,
                                    label_smoothing)
  length = weights.sum(axis=-1)
  return losses, length


def compute_is_scores(filename):
  """Compute IS scores for training data."""

  # Make sure tf does not allocate gpu memory.
  tf.config.experimental.set_visible_devices([], 'GPU')

  if FLAGS.jax_backend_target:
    jax.config.FLAGS.jax_xla_backend = 'tpu_driver'
    jax.config.FLAGS.jax_backend_target = FLAGS.jax_backend_target

  # Number of local devices for this host.
  n_devices = jax.local_device_count()

  if jax.host_id() == 0:
    tf.io.gfile.makedirs(FLAGS.model_dir)

  if FLAGS.batch_size % n_devices:
    raise ValueError('Batch size must be divisible by the number of devices')

  vocab_path = FLAGS.vocab_path
  if vocab_path is None:
    vocab_path = os.path.join(FLAGS.model_dir, 'sentencepiece_model')
  tf.io.gfile.makedirs(os.path.split(vocab_path)[0])

  # Load Dataset
  print('Loading data')
  logging.info('Initializing dataset.')
  train_ds, encoder = input_pipeline.get_wmt_is_datasets(
      n_devices=n_devices,
      dataset_name=FLAGS.dataset_name,
      shard_idx=jax.host_id(),
      shard_count=jax.host_count(),
      data_dir=FLAGS.data_dir,
      vocab_path=vocab_path,
      target_vocab_size=FLAGS.vocab_size,
      batch_size=FLAGS.batch_size,
      max_length=FLAGS.max_target_length,
      paracrawl_size=FLAGS.paracrawl_size)
  print('Datasets created')

  train_iter = iter(train_ds)
  vocab_size = int(encoder.vocab_size())
  eos_id = decode.EOS_ID  # Default Sentencepiece EOS token.
  print('data iterators created')

  logging.info('Initializing model, optimizer, and step functions.')
  # Build Model and Optimizer
  # ---------------------------------------------------------------------------
  eval_config = models.TransformerConfig(
      vocab_size=vocab_size,
      output_vocab_size=vocab_size,
      share_embeddings=FLAGS.share_embeddings,
      logits_via_embedding=FLAGS.logits_via_embedding,
      dtype=jnp.bfloat16 if FLAGS.use_bfloat16 else jnp.float32,
      emb_dim=FLAGS.emb_dim,
      num_heads=FLAGS.num_heads,
      num_layers=FLAGS.num_layers,
      qkv_dim=FLAGS.qkv_dim,
      mlp_dim=FLAGS.mlp_dim,
      max_len=max(FLAGS.max_target_length, FLAGS.max_eval_target_length),
      dropout_rate=FLAGS.dropout_rate,
      attention_dropout_rate=FLAGS.attention_dropout_rate,
      deterministic=True,
      decode=False,
      kernel_init=nn.initializers.xavier_uniform(),
      bias_init=nn.initializers.normal(stddev=1e-6))

  start_step = 0
  rng = jax.random.PRNGKey(FLAGS.random_seed)
  rng, init_rng = jax.random.split(rng)
  # It's possible that is supposed to be per device batch size
  input_shape = (FLAGS.batch_size, FLAGS.max_target_length)
  target_shape = (FLAGS.batch_size, FLAGS.max_target_length)

  m = models.Transformer(eval_config)
  initial_variables = jax.jit(m.init)(init_rng,
                                      jnp.ones(input_shape, jnp.float32),
                                      jnp.ones(target_shape, jnp.float32))

  # apply an optimizer to this tree
  optimizer_def = optim.Adam(
      FLAGS.learning_rate,
      beta1=0.9,
      beta2=0.98,
      eps=1e-9,
      weight_decay=FLAGS.weight_decay)
  optimizer = optimizer_def.create(initial_variables['params'])

  # We access model params only from optimizer below via optimizer.target.
  del initial_variables

  if FLAGS.restore_checkpoints:
    logging.info('Restoring checkpoint.')
    # If we have a pretrained model, use that. Else, just continue where leftoff
    model_path = FLAGS.pretrained_model_dir if FLAGS.pretrained_model_dir else FLAGS.model_dir
    # When loading a checkpoint trained with adapters (ie. frozen weights)
    # restoring from the base optimizer fails. We catch this error and create
    # the optimizer with frozen weights.
    try:
      optimizer = checkpoints.restore_checkpoint(model_path, optimizer)
      # Grab last step.
      start_step = int(optimizer.state.step)
    except ValueError:
      adapter = optim.ModelParamTraversal(lambda path, _: FLAGS.adapter in path)
      optimizer = optimizer_def.create(optimizer.target, focus=adapter)
      optimizer = checkpoints.restore_checkpoint(model_path, optimizer)
      start_step = optimizer.state[0].step

  else:
    raise RuntimeError('Must restore checkpoint for IS')

  if FLAGS.adapter != NONE and not isinstance(optimizer, optim.MultiOptimizer):
    adapter = optim.ModelParamTraversal(lambda path, _: FLAGS.adapter in path)
    optimizer = optimizer_def.create(optimizer.target, focus=adapter)
  # Replicate optimizer.
  optimizer = jax_utils.replicate(optimizer)

  p_eval_step = jax.pmap(
      functools.partial(
          eval_for_is_step,
          config=eval_config),
      axis_name='batch')

  logging.info('Start scoring loop.')
  metrics_all = []
  t_loop_start = time.time()

  # Eval Metrics
  logging.info('Gathering evaluation metrics.')
  t_eval_start = time.time()
  save_file = FLAGS.is_save_path + '/' + filename + '-lengths.txt'
  length_fp = tf.io.gfile.GFile(save_file, 'w')
  lengths_writer = csv.writer(length_fp)

  save_file = FLAGS.is_save_path + '/' + filename + '.txt'
  with tf.io.gfile.GFile(save_file, 'w') as fp:
    writer = csv.writer(fp)

    for batch_idx, eval_batch in enumerate(train_iter):
      eval_batch = jax.tree_map(lambda x: x._numpy(), eval_batch)  # pylint: disable=protected-access
      cur_pred_batch_size = eval_batch['inputs'].shape[0]
      if cur_pred_batch_size % n_devices:
        padded_size = int(
            np.ceil(cur_pred_batch_size / n_devices) * n_devices)
        eval_batch = jax.tree_map(
            lambda x: common.pad_examples(x, padded_size), eval_batch)  # pylint: disable=cell-var-from-loop
      eval_batch = common_utils.shard(eval_batch)
      losses, lengths = p_eval_step(optimizer.target, eval_batch)
      if jax.host_id() == 0:
        losses = common.tohost(losses)
        lengths = common.tohost(lengths)
        if cur_pred_batch_size % n_devices:
          writer.writerow(losses[:cur_pred_batch_size])
          lengths_writer.writerow(lengths[:cur_pred_batch_size])
        else:
          writer.writerow(losses)
          lengths_writer.writerow(lengths)

      if batch_idx % 500 == 0:
        print('Batch', batch_idx)
        print(time.time() - t_loop_start)
  length_fp.close()


def main(_):
  compute_is_scores(FLAGS.is_score_filename)

  if FLAGS.base_log_loss_file:
    beforefile = FLAGS.base_log_loss_file
    afterfile = FLAGS.is_save_path + '/' + FLAGS.is_score_filename + '.txt'
    before_scores = []
    after_scores = []
    with tf.io.gfile.GFile(beforefile, 'r') as f:
      reader = csv.reader(f)
      for row in reader:
        before_scores.extend(row)
    with tf.io.gfile.GFile(afterfile, 'r') as f:
      reader = csv.reader(f)
      for row in reader:
        after_scores.extend(row)

    beforefile = beforefile.replace('.txt', '-lengths.txt')
    afterfile = afterfile.replace('.txt', '-lengths.txt')
    before_length = []
    after_length = []
    with tf.io.gfile.GFile(beforefile, 'r') as f:
      reader = csv.reader(f)
      for row in reader:
        before_length.extend(row)
    with tf.io.gfile.GFile(afterfile, 'r') as f:
      reader = csv.reader(f)
      for row in reader:
        after_length.extend(row)

    diff = [float(a)-float(b) for (a, b) in zip(after_scores, before_scores)]
    after_scores = [float(a) for a in after_scores]
    before_scores = [float(a) for a in before_scores]
    after_length = [float(a) for a in after_length]
    before_length = [float(b) for b in before_length]

    for a, b in zip(before_length, after_length):
      assert a == b

    is_diff_name = FLAGS.is_save_path + '/' + FLAGS.is_diff_name
    with tf.io.gfile.GFile(is_diff_name, 'w') as f:
      writer = csv.writer(f)
      for val in diff:
        writer.writerow([val])

    with tf.io.gfile.GFile(
        is_diff_name.replace('.csv', '_length.csv'), 'w') as f:
      writer = csv.writer(f)
      for val in after_length:
        writer.writerow([int(val)])


if __name__ == '__main__':
  app.run(main)
