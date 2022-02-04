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

import csv
import functools
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

NONE = 'None'
FLAGS = flags.FLAGS

flags.DEFINE_string(
    'model_dir', default=None,
    help='Directory to store model data.')

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

flags.DEFINE_bool(
    'split_tokenizer', default=False,
    help='Separate tokenizer for each language.')

flags.DEFINE_string(
    'save_path', default='',
    help='Path for saving the losses.')

flags.DEFINE_string(
    'model_template', default='',
    help='Name of model with %s instead of cell and %d instead of cluster id.')

flags.DEFINE_string(
    'model_range', default='',
    help='Start and stop index for model template, comma separated.')

flags.DEFINE_string(
    'eval_clusters', default='',
    help='Comma separated list of clusters to evaluate.')

flags.DEFINE_string(
    'aux_models', default='',
    help='Comma separated list of models to evaluate.')

flags.DEFINE_string(
    'data_dir_template', default='',
    help='Template for cluster data dir.')

flags.DEFINE_integer(
    'limit', default=40,
    help='Template for cluster data dir.')

flags.DEFINE_bool(
    'save_decodes', default=False,
    help='Whether to save decodes instead of losses.')


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


def setup():
  """Compute IS scores for training data."""

  # Make sure tf does not allocate gpu memory.
  tf.config.experimental.set_visible_devices([], 'GPU')

  if FLAGS.jax_backend_target:
    jax.config.FLAGS.jax_xla_backend = 'tpu_driver'
    jax.config.FLAGS.jax_backend_target = FLAGS.jax_backend_target

  # Number of local devices for this host.
  n_devices = jax.local_device_count()

  if FLAGS.batch_size % n_devices:
    raise ValueError('Batch size must be divisible by the number of devices')

  vocab_path = FLAGS.vocab_path
  if vocab_path is None:
    raise RuntimeError('Vocab path must be provided')

  # Load Dataset
  print('Loading data')
  logging.info('Initializing dataset.')
  _, (_, encoder_tgt) = input_pipeline.get_wmt_is_datasets(
      n_devices=n_devices,
      dataset_name=FLAGS.dataset_name,
      shard_idx=jax.process_index(),
      shard_count=jax.process_count(),
      data_dir=FLAGS.data_dir,
      vocab_path=vocab_path,
      target_vocab_size=FLAGS.vocab_size,
      batch_size=FLAGS.batch_size,
      max_length=FLAGS.max_target_length,
      paracrawl_size=FLAGS.paracrawl_size,
      split_tokenizer=FLAGS.split_tokenizer)
  print('Datasets created')

  encoder = encoder_tgt
  vocab_size = int(encoder.vocab_size())

  def decode_tokens(toks):
    valid_toks = toks[:np.argmax(toks == decode.EOS_ID) + 1].astype(np.int32)
    return encoder.detokenize(valid_toks).numpy().decode('utf-8')

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
  predict_config = eval_config.replace(deterministic=True, decode=True)
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

  p_eval_step = jax.pmap(
      functools.partial(
          eval_for_is_step,
          config=eval_config),
      axis_name='batch')
  p_init_cache = jax.pmap(
      functools.partial(
          initialize_cache,
          max_decode_len=256,
          config=predict_config),
      axis_name='batch')
  p_pred_step = jax.pmap(
      functools.partial(
          predict_step, config=predict_config, beam_size=4),
      axis_name='batch',
      static_broadcasted_argnums=(3, 4))  # eos token, max_length are constant

  return p_eval_step, optimizer, p_init_cache, p_pred_step, decode_tokens


def predict_step_full(inputs, params, cache, eos_id, max_decode_len, config,
                      beam_size=4):
  """Predict translation with fast decoding beam search on a batch."""
  # Prepare transformer fast-decoder call for beam search: for beam search, we
  # need to set up our decoder model to handle a batch size equal to
  # batch_size * beam_size, where each batch item"s data is expanded in-place
  # rather than tiled.
  # i.e. if we denote each batch element subtensor as el[n]:
  # [el0, el1, el2] --> beamsize=2 --> [el0,el0,el1,el1,el2,el2]
  encoded_inputs = decode.flat_batch_beam_expand(
      models.Transformer(config).apply({'params': params},
                                       inputs,
                                       method=models.Transformer.encode),
      beam_size)
  raw_inputs = decode.flat_batch_beam_expand(inputs, beam_size)

  def tokens_ids_to_logits(flat_ids, flat_cache):
    """Token slice to logits from decoder model."""
    # --> [batch * beam, 1, vocab]
    flat_logits, new_vars = models.Transformer(config).apply(
        {
            'params': params,
            'cache': flat_cache
        },
        encoded_inputs,
        raw_inputs,  # only needed for input padding mask
        flat_ids,
        mutable=['cache'],
        method=models.Transformer.decode)
    new_flat_cache = new_vars['cache']
    # Remove singleton sequence-length dimension:
    # [batch * beam, 1, vocab] --> [batch * beam, vocab]
    flat_logits = flat_logits.squeeze(axis=1)

    return flat_logits, new_flat_cache

  # Using the above-defined single-step decoder function, run a
  # beam search over possible sequences given input encoding.
  beam_seqs, scores = decode.beam_search(
      inputs,
      cache,
      tokens_ids_to_logits,
      beam_size=beam_size,
      alpha=0.6,
      eos_id=eos_id,
      max_decode_len=max_decode_len)

  # Beam search returns [n_batch, n_beam, n_length + 1] with beam dimension
  # sorted in increasing order of log-probability.
  # Return the highest scoring beam sequence, drop first dummy 0 token.
  return beam_seqs, scores


def predict_step(inputs, params, cache, eos_id, max_decode_len, config,
                 beam_size=4):
  beam_seqs, scores = predict_step_full(inputs, params, cache, eos_id,
                                        max_decode_len, config, beam_size)
  return beam_seqs[:, -1, 1:], scores[:, -1]


def initialize_cache(inputs, max_decode_len, config):
  """Initialize a cache for a given input shape and max decode length."""
  target_shape = (inputs.shape[0], max_decode_len) + inputs.shape[2:]
  initial_variables = models.Transformer(config).init(
      jax.random.PRNGKey(0), jnp.ones(inputs.shape, config.dtype),
      jnp.ones(target_shape, config.dtype))
  return initial_variables['cache']


def pad_examples(x, desired_batch_size):
  """Expand batch to desired size by repeating last slice."""
  batch_pad = desired_batch_size - x.shape[0]
  return np.concatenate([x, np.tile(x[-1], (batch_pad, 1))], axis=0)


def get_losses(ds_iter, optimizer, p_eval_step, model_id, test_cluster_id):
  """Given optimizer and dataset, compute losses and write to file."""
  logging.info('Start scoring loop.')
  n_devices = jax.local_device_count()
  t_loop_start = time.time()

  filename = '/losses_testcluster{test_cluster_id}_ftid{model_id}.csv'
  save_file = filename.format(test_cluster_id=test_cluster_id,
                              model_id=model_id)
  save_file = FLAGS.save_path + save_file
  with tf.io.gfile.GFile(save_file, 'w') as fp:
    writer = csv.writer(fp)

    for batch_idx, eval_batch in enumerate(ds_iter):
      eval_batch = jax.tree_map(lambda x: x._numpy(), eval_batch)  # pylint: disable=protected-access
      cur_pred_batch_size = eval_batch['inputs'].shape[0]
      if cur_pred_batch_size % n_devices:
        padded_size = int(
            np.ceil(cur_pred_batch_size / n_devices) * n_devices)
        eval_batch = jax.tree_map(
            lambda x: common.pad_examples(x, padded_size), eval_batch)  # pylint: disable=cell-var-from-loop
      eval_batch = common_utils.shard(eval_batch)
      losses, lengths = p_eval_step(optimizer.target, eval_batch)
      if jax.process_index() == 0:
        losses = common.tohost(losses)
        lengths = common.tohost(lengths)
        if cur_pred_batch_size % n_devices:
          writer.writerow(losses[:cur_pred_batch_size])
        else:
          writer.writerow(losses)

      if batch_idx % 500 == 0:
        print('Batch', batch_idx)
        print(time.time() - t_loop_start)

      if batch_idx >= FLAGS.limit:
        break


def get_decodes(ds_iter, optimizer, p_init_cache, p_pred_step, model_id,
                test_cluster_id, decode_tokens):
  """Given optimizer and dataset, compute losses and write to file."""
  logging.info('Start scoring loop.')
  n_devices = jax.local_device_count()
  predictions = []
  max_predict_length = 256

  filename = '/decodes_testcluster{test_cluster_id}_ftid{model_id}.csv'
  save_file = filename.format(test_cluster_id=test_cluster_id,
                              model_id=model_id)
  save_file = FLAGS.save_path + save_file
  with tf.io.gfile.GFile(save_file, 'w') as fp:
    writer = csv.writer(fp)

    for batch_idx, pred_batch in enumerate(ds_iter):
      pred_batch = jax.tree_map(lambda x: x._numpy(), pred_batch)  # pylint: disable=protected-access
      # Handle final odd-sized batch by padding instead of dropping it.
      cur_pred_batch_size = pred_batch['inputs'].shape[0]
      if cur_pred_batch_size % n_devices:
        padded_size = int(np.ceil(cur_pred_batch_size / n_devices) * n_devices)
        pred_batch = jax.tree_map(
            lambda x: pad_examples(x, padded_size),  # pylint: disable=cell-var-from-loop
            pred_batch)
      pred_batch = common_utils.shard(pred_batch)

      cache = p_init_cache(pred_batch['inputs'])
      predicted, _ = p_pred_step(pred_batch['inputs'], optimizer.target, cache,
                                 decode.EOS_ID, max_predict_length)
      if jax.process_index() == 0:
        predicted = common.tohost(predicted)
        # Iterate through non-padding examples of batch.
        for s in predicted[:cur_pred_batch_size]:
          predictions.append(decode_tokens(s))

      if batch_idx >= FLAGS.limit:
        break
    writer.writerow(predictions)


def reload_opt(optimizer, model_path):
  optimizer = checkpoints.restore_checkpoint(model_path, optimizer)
  # Replicate optimizer.
  optimizer = jax_utils.replicate(optimizer)
  return optimizer


def get_data(cl):
  """Get dataset iterator."""
  n_devices = jax.local_device_count()
  data_dir = FLAGS.data_dir_template
  data_dir = data_dir.format(cl)

  eval_ds, _ = input_pipeline.get_wmt_is_datasets(
      n_devices=n_devices,
      dataset_name=FLAGS.dataset_name,
      shard_idx=jax.process_index(),
      shard_count=jax.process_count(),
      data_dir=data_dir,
      vocab_path=FLAGS.vocab_path,
      target_vocab_size=FLAGS.vocab_size,
      batch_size=FLAGS.batch_size,
      max_length=FLAGS.max_target_length,
      paracrawl_size=FLAGS.paracrawl_size,
      split_tokenizer=FLAGS.split_tokenizer,
      use_eval_data=True,
      truncate=True)

  ds_iter = iter(eval_ds)
  return ds_iter


def main(_):

  # Given a list of cluster ids, list of model ids
  # for model:
  #   load model
  #   for clustered data
  #     Compute per example losses and write to file
  #      file is cluster id + model id and to save file, in txt
  aux_models = FLAGS.aux_models.split(',')
  model_range_start, model_range_end = FLAGS.model_range.split(',')
  model_range = range(int(model_range_start), int(model_range_end))
  if FLAGS.eval_clusters == 'all':
    eval_clusters = list(range(100))
  else:
    eval_clusters = [int(cl) for cl in FLAGS.eval_clusters.split(',')]
  model_dict = {}
  for i, model in enumerate(aux_models):
    if model:
      model_dict['aux'+str(i)] = model

  for model_id in model_range:
    for cell in ['tp', 'pw', 'el']:
      model_name = FLAGS.model_template.format(cell, model_id)
      if tf.io.gfile.exists(model_name):
        model_dict[model_id] = model_name
        break

  p_eval_step, optimizer, p_init_cache, p_pred_step, decode_tokens = setup()

  for model_id, model in model_dict.items():
    optimizer = reload_opt(optimizer, model)
    for cl in eval_clusters:
      ds_iter = get_data(cl)
      if FLAGS.save_decodes:
        get_decodes(
            ds_iter=ds_iter,
            optimizer=optimizer,
            p_init_cache=p_init_cache,
            p_pred_step=p_pred_step,
            model_id=model_id,
            test_cluster_id=cl,
            decode_tokens=decode_tokens)
      else:
        get_losses(
            ds_iter=ds_iter,
            optimizer=optimizer,
            p_eval_step=p_eval_step,
            model_id=model_id,
            test_cluster_id=cl)


if __name__ == '__main__':
  app.run(main)
