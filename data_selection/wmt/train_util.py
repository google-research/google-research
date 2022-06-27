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

"""Util functions for various training configs."""

import collections
import csv
import time

from absl import flags
from absl import logging
from flax import linen as nn
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from data_selection.wmt import bleu
from data_selection.wmt import common
from data_selection.wmt import decode
from data_selection.wmt import models

ORTH = 'ORTH'
GREEDY = 'GREEDY'
LAYERNORM_ADAPTER = 'LayerNorm'
ENDCODE_DECODE_B5 = 'encoderdecoderblock_5'
PASSTHRU = ''
NONE = 'None'

flags.DEFINE_string(
    'model_dir', default=None,
    help='Directory to store model data.')

flags.DEFINE_string(
    'is_save_path', default=None,
    help='Path to save is scores to.')

flags.DEFINE_string(
    'pseudo_path', default=None,
    help='Path to pseudo references.')

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

flags.DEFINE_integer(
    'data_selection_size', default=-1,
    help='Numer of examples to keep when doing data selection.')

flags.DEFINE_integer(
    'chkpts_to_keep', default=4,
    help='Numer of checkpoints to keep.')

flags.DEFINE_string(
    'dataset_name', default='wmt17_translate/de-en',
    help='Name of TFDS translation dataset to use.')

flags.DEFINE_string(
    'eval_dataset_name', default='wmt14_translate/de-en:test',
    help='Optional name of TFDS translation dataset to use for evaluation.')

flags.DEFINE_bool(
    'reverse_translation', default=False,
    help='Reverse the direction of translation.')

flags.DEFINE_integer(
    'batch_size', default=256,
    help='Per host batch size for training.')

flags.DEFINE_integer(
    'beam_size', default=4,
    help='Beam size for inference.')

flags.DEFINE_integer(
    'eval_frequency', default=1000,
    help='Frequency of eval during training, e.g. every 1000 steps.')

flags.DEFINE_integer(
    'num_train_steps', default=500000,
    help='Number of train steps.')

flags.DEFINE_integer(
    'num_eval_steps', default=20,
    help='Number of steps to take during evaluation.')

flags.DEFINE_float(
    'learning_rate', default=0.0625,
    help='Base learning rate.')

flags.DEFINE_integer(
    'warmup_steps', default=1000,
    help='Linear learning rate warmup.')

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

flags.DEFINE_integer(
    'max_predict_length', default=256,
    help='Maximum length cutoff for predicted tokens.')

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

flags.DEFINE_string(
    'is_scores_path', default=None,
    help='Path to IS scores file.')

flags.DEFINE_integer(
    'num_data_buckets', default=100,
    help='Number of data buckets to create.')

flags.DEFINE_bool(
    'save_checkpoints', default=True,
    help='Whether to save model checkpoints.')

flags.DEFINE_bool(
    'restore_checkpoints', default=True,
    help='Whether to restore from existing model checkpoints.')

flags.DEFINE_integer(
    'checkpoint_freq', default=10000,
    help='Save a checkpoint every these number of steps.')

flags.DEFINE_bool(
    'use_bfloat16', default=True,
    help=('Use bfloat16 mixed precision training instead of float32.'))

flags.DEFINE_string(
    'jax_backend_target', default=None,
    help=('TPU grpc target for use with cloud TPUs.'
          ' e.g. grpc://192.168.0.2:8470'))

flags.DEFINE_enum(
    'strategy', default=ORTH, enum_values=[ORTH, GREEDY],
    help='Bucket selection strategy.')

flags.DEFINE_integer(
    'paracrawl_size', default=1200000,
    help='Number of examples to sample from paracrawl.')

flags.DEFINE_integer(
    'newscommentary_size', default=None,
    help='Number of examples to sample from News commentary.')

flags.DEFINE_integer(
    'repeat_count', default=-1,
    help='Number of examples to sample from paracrawl.')

flags.DEFINE_enum(
    'adapter', default=NONE, enum_values=[LAYERNORM_ADAPTER,
                                          ENDCODE_DECODE_B5,
                                          PASSTHRU,
                                          NONE],
    help='Whether to finetune only some parameters.')

flags.DEFINE_bool(
    'compute_bleu', default=True,
    help='Whether to compute bleu scores.')

flags.DEFINE_bool(
    'finetune_lr', default=False,
    help='Special flag for modifying learning rate for finetuning.')

flags.DEFINE_integer(
    'steps_per_cycle', default=4500,
    help='Number of steps in an epoch.')

flags.DEFINE_string(
    'aux_eval_dataset', default='',
    help='Name of TFDS translation dataset also eval.')

flags.DEFINE_bool(
    'dynamic', default=False,
    help='Dynamically sample dataset.')

flags.DEFINE_integer(
    'resample_freq', default=100,
    help='How often re resample for dynamic selection.')

flags.DEFINE_string(
    'static', default=None,
    help='Static distribution from which to sample training data.')

flags.DEFINE_bool(
    'macro', default=False,
    help='Type of DDS.')

flags.DEFINE_bool(
    'split_tokenizer', default=False,
    help='Separate tokenizer for each language.')

flags.DEFINE_bool(
    'eval_only', default=False,
    help='Skip training.')


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


def compute_per_pos_loss(logits,
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

  return loss


def compute_weighted_cross_entropy(logits,
                                   targets,
                                   weights=None,
                                   label_smoothing=0.0):
  """Compute weighted cross entropy and entropy for log probs and targets.

  Args:
   logits: [batch, length, num_classes] float array.
   targets: categorical targets [batch, length] int array.
   weights: None or array of shape [batch, length].
   label_smoothing: label smoothing constant, used to determine the on and off
     values.

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
      confidence * jnp.log(confidence) +
      (vocab_size - 1) * low_confidence * jnp.log(low_confidence + 1e-20))
  soft_targets = common_utils.onehot(
      targets, vocab_size, on_value=confidence, off_value=low_confidence)

  loss = -jnp.sum(soft_targets * nn.log_softmax(logits), axis=-1)
  loss = loss - normalizing_constant

  normalizing_factor = np.prod(targets.shape)
  if weights is not None:
    loss = loss * weights
    normalizing_factor = weights.sum()

  return loss.sum(), normalizing_factor


def compute_weighted_accuracy(logits, targets, weights=None):
  """Compute weighted accuracy for log probs and targets.

  Args:
   logits: [batch, length, num_classes] float array.
   targets: categorical targets [batch, length] int array.
   weights: None or array of shape [batch, length]

  Returns:
    Tuple of scalar loss and batch normalizing factor.
  """
  if logits.ndim != targets.ndim + 1:
    raise ValueError('Incorrect shapes. Got shape %s logits and %s targets' %
                     (str(logits.shape), str(targets.shape)))
  loss = jnp.equal(jnp.argmax(logits, axis=-1), targets)
  normalizing_factor = np.prod(logits.shape[:-1])
  if weights is not None:
    loss = loss * weights
    normalizing_factor = weights.sum()

  return loss.sum(), normalizing_factor


def compute_metrics(logits, labels, weights, label_smoothing=0.0):
  """Compute summary metrics."""
  loss, weight_sum = compute_weighted_cross_entropy(logits, labels, weights,
                                                    label_smoothing)
  acc, _ = compute_weighted_accuracy(logits, labels, weights)
  metrics = {
      'loss': loss,
      'accuracy': acc,
      'denominator': weight_sum,
  }
  metrics = jax.lax.psum(metrics, axis_name='batch')
  return metrics


# Primary training / eval / decode step functions.
# -----------------------------------------------------------------------------


def get_diag_grads(optimizer,
                   batch,
                   config):
  """Perform a single training step."""
  # X_position and X_segmentation are needed only when using 'packed examples'
  # where multiple sequences are packed into the same example with this
  # metadata.
  # if such features are not present they are ignored and the example is treated
  # like a normal, unpacked sequence example.
  train_keys = ['inputs', 'targets',
                'inputs_position', 'targets_position',
                'inputs_segmentation', 'targets_segmentation']

  (inputs, targets,
   inputs_positions, targets_positions,
   inputs_segmentation, targets_segmentation) = [
       batch.get(k, None) for k in train_keys]

  weights = jnp.where(targets > 0, 1, 0).astype(jnp.float32)

  def loss_fn(params):
    """loss function used for training."""
    logits = models.Transformer(config).apply(
        {'params': params},
        inputs,
        targets,
        inputs_positions=inputs_positions,
        targets_positions=targets_positions,
        inputs_segmentation=inputs_segmentation,
        targets_segmentation=targets_segmentation)

    loss, weight_sum = compute_weighted_cross_entropy(logits, targets, weights)
    mean_loss = loss / weight_sum
    return mean_loss

  diag_grad_fn = jax.grad(loss_fn, has_aux=False)
  diag_grad = diag_grad_fn(optimizer.target)
  diag_grad = jax.lax.pmean(diag_grad, 'batch')

  return diag_grad


def train_step(optimizer,
               batch,
               config,
               learning_rate_fn,
               label_smoothing=0.0,
               dropout_rng=None):
  """Perform a single training step."""
  # X_position and X_segmentation are needed only when using "packed examples"
  # where multiple sequences are packed into the same example with this
  # metadata.
  # if such features are not present they are ignored and the example is treated
  # like a normal, unpacked sequence example.
  train_keys = [
      'inputs', 'targets', 'inputs_position', 'targets_position',
      'inputs_segmentation', 'targets_segmentation'
  ]
  (inputs, targets, inputs_positions, targets_positions, inputs_segmentation,
   targets_segmentation) = [batch.get(k, None) for k in train_keys]

  weights = jnp.where(targets > 0, 1, 0).astype(jnp.float32)

  if isinstance(optimizer.state, tuple):
    step = optimizer.state[0].step
  else:
    step = optimizer.state.step
  dropout_rng = jax.random.fold_in(dropout_rng, step)

  def loss_fn(params):
    """loss function used for training."""
    logits = models.Transformer(config).apply(
        {'params': params},
        inputs,
        targets,
        inputs_positions=inputs_positions,
        targets_positions=targets_positions,
        inputs_segmentation=inputs_segmentation,
        targets_segmentation=targets_segmentation,
        rngs={'dropout': dropout_rng})

    loss, weight_sum = compute_weighted_cross_entropy(logits, targets, weights,
                                                      label_smoothing)
    mean_loss = loss / weight_sum
    return mean_loss, logits

  lr = learning_rate_fn(step)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grad = grad_fn(optimizer.target)
  grad = jax.lax.pmean(grad, 'batch')
  new_optimizer = optimizer.apply_gradient(grad, learning_rate=lr)
  metrics = compute_metrics(logits, targets, weights)
  metrics['learning_rate'] = lr

  return new_optimizer, metrics


def eval_step(params, batch, config, label_smoothing=0.0):
  """Calculate evaluation metrics on a batch."""
  inputs, targets = batch['inputs'], batch['targets']
  weights = jnp.where(targets > 0, 1.0, 0.0)
  logits = models.Transformer(config).apply({'params': params}, inputs, targets)

  return compute_metrics(logits, targets, weights, label_smoothing)


def initialize_cache(inputs, max_decode_len, config):
  """Initialize a cache for a given input shape and max decode length."""
  target_shape = (inputs.shape[0], max_decode_len) + inputs.shape[2:]
  initial_variables = models.Transformer(config).init(
      jax.random.PRNGKey(0), jnp.ones(inputs.shape, config.dtype),
      jnp.ones(target_shape, config.dtype))
  return initial_variables['cache']


def eval_per_pos_step(params, batch, config, label_smoothing=0.0):
  """Calculate evaluation metrics on a batch."""
  inputs, targets = batch['inputs'], batch['targets']
  weights = jnp.where(targets > 0, 1.0, 0.0)
  logits = models.Transformer(config).apply({'params': params}, inputs, targets)
  losses = compute_per_pos_loss(logits,
                                targets,
                                weights,
                                label_smoothing)
  length = weights.sum(axis=-1)
  return losses, length


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


# Utils for prediction and BLEU calculation
# -----------------------------------------------------------------------------


def pad_examples(x, desired_batch_size):
  """Expand batch to desired size by repeating last slice."""
  batch_pad = desired_batch_size - x.shape[0]
  return np.concatenate([x, np.tile(x[-1], (batch_pad, 1))], axis=0)


def per_host_sum_pmap(in_tree):
  """Execute psum on in_tree"s leaves over one device per host."""
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


def tohost(x):
  """Collect batches from all devices to host and flatten batch dimensions."""
  n_device, n_batch, *remaining_dims = x.shape
  return np.array(x).reshape((n_device * n_batch,) + tuple(remaining_dims))


def evaluate(*, p_eval_step, target, eval_ds,
             num_eval_steps):
  """Evaluate the target an return a dictionary with the metrics."""
  logging.info('Gathering evaluation metrics.')
  eval_metrics = []
  eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
  for _, eval_batch in zip(range(num_eval_steps), eval_iter):
    eval_batch = jax.tree_map(lambda x: x._numpy(), eval_batch)  # pylint: disable=protected-access
    eval_batch = common_utils.shard(eval_batch)
    metrics = p_eval_step(target, eval_batch)
    eval_metrics.append(metrics)
  eval_metrics = common_utils.get_metrics(eval_metrics)
  eval_metrics_sums = jax.tree_map(jnp.sum, eval_metrics)
  eval_denominator = eval_metrics_sums.pop('denominator')
  eval_summary = jax.tree_map(
      lambda x: x / eval_denominator,  # pylint: disable=cell-var-from-loop
      eval_metrics_sums)
  return eval_summary


def write_per_example_losses(*, p_eval_step, target, eval_ds,
                             num_eval_steps, loss_filename):
  """Evaluate the target an return a dictionary with the metrics."""
  logging.info('Gathering evaluation metrics.')
  losses = []
  lengths = []
  eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
  for _, eval_batch in zip(range(num_eval_steps), eval_iter):
    eval_batch = jax.tree_map(lambda x: x._numpy(), eval_batch)  # pylint: disable=protected-access
    eval_batch = common_utils.shard(eval_batch)
    loss, length = p_eval_step(target, eval_batch)
    losses.append(common.tohost(loss))
    lengths.append(common.tohost(length))
  # Write losses and lengths
  if jax.process_index() == 0:
    with tf.io.gfile.GFile(loss_filename, 'w') as f:
      writer = csv.writer(f)
      for pos_losses in losses:
        for val in pos_losses:
          writer.writerow(list(val))
    with tf.io.gfile.GFile(loss_filename.replace('.csv', '_length.csv'),
                           'w') as f:
      writer = csv.writer(f)
      for val in lengths:
        writer.writerow([int(v) for v in list(val)])
  return


def translate_and_calculate_bleu(*, p_pred_step, p_init_cache, target,
                                 predict_ds, decode_tokens,
                                 max_predict_length,
                                 num_eval_steps = 1000,
                                 decode_file = ''):
  """Translates the `predict_ds` and calculates the BLEU score."""
  n_devices = jax.local_device_count()
  logging.info('Translating evaluation dataset.')
  sources, references, predictions = [], [], []
  for counter, pred_batch in zip(range(num_eval_steps), predict_ds):
    print(counter)
    start_batch = time.time()
    pred_batch = jax.tree_map(lambda x: x._numpy(), pred_batch)  # pylint: disable=protected-access
    # Handle final odd-sized batch by padding instead of dropping it.
    cur_pred_batch_size = pred_batch['inputs'].shape[0]
    if cur_pred_batch_size % n_devices:
      padded_size = int(np.ceil(cur_pred_batch_size / n_devices) * n_devices)
      pred_batch = jax.tree_map(
          lambda x: pad_examples(x, padded_size),  # pylint: disable=cell-var-from-loop
          pred_batch)
    pred_batch = common_utils.shard(pred_batch)
    end_batch_proc = time.time()
    cache = p_init_cache(pred_batch['inputs'])
    predicted, _ = p_pred_step(pred_batch['inputs'], target, cache,
                               decode.EOS_ID, max_predict_length)
    end_p_step = time.time()
    predicted = tohost(predicted)
    inputs = tohost(pred_batch['inputs'])
    targets = tohost(pred_batch['targets'])
    # Iterate through non-padding examples of batch.
    for i, s in enumerate(predicted[:cur_pred_batch_size]):
      sources.append(decode_tokens(inputs[i]))
      references.append(decode_tokens(targets[i]))
      predictions.append(decode_tokens(s))
    end = time.time()
    print('data proc', end_batch_proc - start_batch)
    print('pstep', end_p_step - end_batch_proc)
    print('last part', end - end_p_step)
  logging.info('Translation: %d predictions %d references %d sources.',
               len(predictions), len(references), len(sources))

  # Calculate BLEU score for translated eval corpus against reference.
  bleu_matches = bleu.bleu_partial(references, predictions)
  all_bleu_matches = per_host_sum_pmap(bleu_matches)
  bleu_score = bleu.complete_bleu(*all_bleu_matches)
  # Save translation samples for tensorboard.
  exemplars = ''
  for n in np.random.choice(np.arange(len(predictions)), 8):
    exemplars += f'{sources[n]}\n\n{references[n]}\n\n{predictions[n]}\n\n'

  if decode_file:
    with tf.io.gfile.GFile(decode_file, 'w') as f:
      writer = csv.writer(f)
      for val in zip(sources, references, predictions):
        writer.writerow(val)

  return exemplars, bleu_score

