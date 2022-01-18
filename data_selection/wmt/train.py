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

# pytype: disable=wrong-arg-count
# pytype: disable=attribute-error

import collections
import functools
import os

from absl import app
from absl import flags
from absl import logging
from clu import metric_writers
from clu import periodic_actions
from flax import jax_utils
from flax import linen as nn
from flax import optim
from flax.training import checkpoints
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from data_selection.wmt import bleu
from data_selection.wmt import common
from data_selection.wmt import decode
from data_selection.wmt import input_pipeline
from data_selection.wmt import models
from data_selection.wmt.gradient_utils import tree_diff
from data_selection.wmt.gradient_utils import tree_div
from data_selection.wmt.gradient_utils import tree_dot
from data_selection.wmt.gradient_utils import tree_mult
from data_selection.wmt.gradient_utils import tree_norm

ORTH = 'ORTH'
GREEDY = 'GREEDY'
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


def encode_step(model, batch, use_bfloat16=False):
  """Calculate evaluation metrics on a batch."""
  inputs, targets = batch['inputs'], batch['targets']
  weights = jnp.where(targets > 0, 1.0, 0.0)
  weights = jnp.expand_dims(weights, axis=2)
  encoded_output = model.encode(
      inputs, use_bfloat16=use_bfloat16, train=False, cache=None)
  encoded_output = encoded_output * weights
  encoded_output = jnp.mean(encoded_output, axis=1)
  return encoded_output


def eval_per_pos_step(model, batch, label_smoothing=0.0, use_bfloat16=False):
  """Calculate evaluation metrics on a batch."""
  inputs, targets = batch['inputs'], batch['targets']
  weights = jnp.where(targets > 0, 1.0, 0.0)
  logits = model(inputs, targets, use_bfloat16=use_bfloat16, train=False,
                 cache=None)
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


def translate_and_calculate_bleu(*, p_pred_step, p_init_cache, target,
                                 predict_ds, decode_tokens,
                                 max_predict_length):
  """Translates the `predict_ds` and calculates the BLEU score."""
  n_devices = jax.local_device_count()
  logging.info('Translating evaluation dataset.')
  sources, references, predictions = [], [], []
  for pred_batch in predict_ds:
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
    predicted, _ = p_pred_step(pred_batch['inputs'], target, cache,
                               decode.EOS_ID, max_predict_length)
    predicted = tohost(predicted)
    inputs = tohost(pred_batch['inputs'])
    targets = tohost(pred_batch['targets'])
    # Iterate through non-padding examples of batch.
    for i, s in enumerate(predicted[:cur_pred_batch_size]):
      sources.append(decode_tokens(inputs[i]))
      references.append(decode_tokens(targets[i]))
      predictions.append(decode_tokens(s))
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
  return exemplars, bleu_score


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

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
  # ---------------------------------------------------------------------------
  logging.info('Initializing dataset.')
  if FLAGS.dynamic:
    train_ds_mgr, eval_ds, predict_ds, encoder = input_pipeline.get_dynamic_datasets(
        dataset_name=FLAGS.dataset_name,
        eval_dataset_name=FLAGS.eval_dataset_name,
        shard_idx=jax.host_id(),
        shard_count=jax.host_count(),
        data_dir=FLAGS.data_dir,
        vocab_path=FLAGS.vocab_path,
        target_vocab_size=FLAGS.vocab_size,
        batch_size=FLAGS.batch_size,
        max_length=FLAGS.max_target_length,
        max_eval_length=FLAGS.max_eval_target_length,
        paracrawl_size=FLAGS.paracrawl_size,
        is_scores_path=FLAGS.is_scores_path,
        num_buckets=FLAGS.num_data_buckets)
    if FLAGS.static:
      weights = np.array([float(w) for w in FLAGS.static.split(',')])
      assert len(weights) == FLAGS.num_data_buckets
      train_ds = train_ds_mgr.sampled_dataset(weights)
      FLAGS.dynamic = False
    else:
      init_dist = np.zeros(FLAGS.num_data_buckets)
      if FLAGS.data_selection_size < FLAGS.num_data_buckets:
        init_dist[range(FLAGS.data_selection_size)] = 1.0
        train_ds = train_ds_mgr.sampled_dataset(init_dist)
      else:
        train_ds = build_split(train_ds_mgr, 1.0)

  else:
    train_ds, eval_ds, predict_ds, encoder = input_pipeline.get_wmt_datasets(
        dataset_name=FLAGS.dataset_name,
        eval_dataset_name=FLAGS.eval_dataset_name,
        shard_idx=jax.host_id(),
        shard_count=jax.host_count(),
        data_dir=FLAGS.data_dir,
        vocab_path=vocab_path,
        target_vocab_size=FLAGS.vocab_size,
        batch_size=FLAGS.batch_size,
        max_length=FLAGS.max_target_length,
        max_eval_length=FLAGS.max_eval_target_length,
        paracrawl_size=FLAGS.paracrawl_size,
        is_scores_path=FLAGS.is_scores_path,
        num_to_keep=FLAGS.data_selection_size,
        pseudo_path=FLAGS.pseudo_path,
        repeat_count=FLAGS.repeat_count,
        newscommentary_size=FLAGS.newscommentary_size)

  if FLAGS.aux_eval_dataset:
    aux_datasets = []
    aux_names = FLAGS.aux_eval_dataset.split(',')
    for name in aux_names:
      _, aux_eval_ds, _, _ = input_pipeline.get_wmt_datasets(
          dataset_name=name,
          eval_dataset_name=None,
          shard_idx=jax.host_id(),
          shard_count=jax.host_count(),
          data_dir=FLAGS.data_dir,
          vocab_path=vocab_path,
          target_vocab_size=FLAGS.vocab_size,
          batch_size=FLAGS.batch_size,
          max_length=FLAGS.max_target_length,
          max_eval_length=FLAGS.max_eval_target_length,
          paracrawl_size=FLAGS.paracrawl_size,
          is_scores_path=FLAGS.is_scores_path,
          num_to_keep=FLAGS.data_selection_size,
          pseudo_path=FLAGS.pseudo_path,
          repeat_count=FLAGS.repeat_count,
          newscommentary_size=FLAGS.newscommentary_size)
      aux_datasets.append(aux_eval_ds)

  train_iter = iter(train_ds)
  vocab_size = int(encoder.vocab_size())
  eos_id = decode.EOS_ID  # Default Sentencepiece EOS token.

  def decode_tokens(toks):
    valid_toks = toks[:np.argmax(toks == eos_id) + 1].astype(np.int32)
    return encoder.detokenize(valid_toks).numpy().decode('utf-8')

  logging.info('Initializing model, optimizer, and step functions.')

  # Build Model and Optimizer
  # ---------------------------------------------------------------------------
  train_config = models.TransformerConfig(
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
      deterministic=False,
      decode=False,
      kernel_init=nn.initializers.xavier_uniform(),
      bias_init=nn.initializers.normal(stddev=1e-6))
  eval_config = train_config.replace(deterministic=True)
  predict_config = train_config.replace(deterministic=True, decode=True)

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
    optimizer = checkpoints.restore_checkpoint(model_path, optimizer)
    # Grab last step.
    start_step = int(optimizer.state.step)

  if FLAGS.adapter != NONE:
    adapter = optim.ModelParamTraversal(lambda path, _: FLAGS.adapter in path)
    optimizer = optimizer_def.create(optimizer.target, focus=adapter)

  writer = metric_writers.create_default_writer(
      FLAGS.model_dir, just_logging=jax.process_index() > 0)

  flag_key = [k for k in FLAGS.flags_by_module_dict().keys() if 'wmt.par' in k
             ]
  if flag_key:
    flag_key = flag_key[0]
    local_flags = {
        f.name: f.value for f in FLAGS.flags_by_module_dict()[flag_key]
    }
    writer.write_hparams(local_flags)

  # Replicate optimizer.
  optimizer = jax_utils.replicate(optimizer)

  if FLAGS.adapter != NONE:
    learning_rate_fn = common.create_learning_rate_scheduler(
        factors='constant',
        base_learning_rate=FLAGS.learning_rate,
        warmup_steps=FLAGS.warmup_steps)
  else:
    learning_rate_fn = common.create_learning_rate_scheduler(
        base_learning_rate=FLAGS.learning_rate, warmup_steps=FLAGS.warmup_steps,
        steps_per_cycle=FLAGS.steps_per_cycle, init_step=start_step,
        finetune_lr=FLAGS.finetune_lr)

  # compile multidevice versions of train/eval/predict step and cache init fn.
  p_train_step = jax.pmap(
      functools.partial(
          train_step,
          config=train_config,
          learning_rate_fn=learning_rate_fn,
          label_smoothing=FLAGS.label_smoothing),
      axis_name='batch',
      donate_argnums=(0,))  # pytype: disable=wrong-arg-types
  p_eval_step = jax.pmap(
      functools.partial(eval_step, config=eval_config), axis_name='batch')
  p_init_cache = jax.pmap(
      functools.partial(
          initialize_cache,
          max_decode_len=FLAGS.max_predict_length,
          config=predict_config),
      axis_name='batch')
  p_pred_step = jax.pmap(
      functools.partial(
          predict_step, config=predict_config, beam_size=FLAGS.beam_size),
      axis_name='batch',
      static_broadcasted_argnums=(3, 4))  # eos token, max_length are constant

  p_get_diag_grads = jax.pmap(
      functools.partial(
          get_diag_grads,
          config=eval_config),
      axis_name='batch')

  p_get_bucket_score = jax.pmap(
      functools.partial(
          get_diag_score,
          strategy=FLAGS.strategy),
      axis_name='batch')

  # Main Train Loop
  # ---------------------------------------------------------------------------

  # We init the first set of dropout PRNG keys, but update it afterwards inside
  # the main pmap"d training update for performance.
  dropout_rngs = jax.random.split(rng, jax.local_device_count())
  del rng

  logging.info('Starting training loop.')
  hooks = []
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=FLAGS.num_train_steps, writer=writer)
  if jax.process_index() == 0:
    hooks += [
        report_progress,
        periodic_actions.Profile(logdir=FLAGS.model_dir, num_profile_steps=5)
    ]
  train_metrics = []
  total_steps = start_step + FLAGS.num_train_steps
  best_eval_loss = 1000
  curr_eval_loss = 1000
  with metric_writers.ensure_flushes(writer):
    for step in range(start_step, total_steps):
      is_last_step = step == total_steps - 1

      if FLAGS.dynamic and ((step - start_step) % FLAGS.resample_freq == 0):
        # Dynamic macro: use gradient alignment to score different ratios
        # of top k vs bottom N-k bins
        if FLAGS.macro:
          train_iter = get_macro_distribution(p_get_diag_grads,
                                              p_get_bucket_score, aux_eval_ds,
                                              train_ds_mgr, optimizer, eval_ds)
        else:
          # Use gradient alignment to score bins
          # take the top k bins and sample uniformly from them.
          raw_distribution = get_new_distribution(p_get_diag_grads,
                                                  p_get_bucket_score,
                                                  aux_eval_ds, train_ds_mgr,
                                                  optimizer,
                                                  eval_ds)
          logging.info(raw_distribution)
          selected = np.argsort(
              raw_distribution)[::-1][:FLAGS.data_selection_size]
          new_distribution = np.zeros(100)
          new_distribution[selected] = 1.0
          logging.info(new_distribution)
          train_ds = train_ds_mgr.sampled_dataset(new_distribution)
          train_iter = iter(train_ds)

      # Shard data to devices and do a training step.
      with jax.profiler.StepTraceAnnotation('train', step_num=step):
        try:
          batch = common_utils.shard(jax.tree_map(np.asarray, next(train_iter)))
          optimizer, metrics = p_train_step(
              optimizer, batch, dropout_rng=dropout_rngs)
          train_metrics.append(metrics)
        except StopIteration:
          is_last_step = True

      # Quick indication that training is happening.
      logging.log_first_n(logging.INFO, 'Finished training step %d.', 5, step)
      for h in hooks:
        h(step)

      # Periodic metric handling.
      if (step - start_step) % FLAGS.eval_frequency == 0 or is_last_step:
        with report_progress.timed('training_metrics'):
          logging.info('Gathering training metrics.')
          train_metrics = common_utils.get_metrics(train_metrics)
          lr = train_metrics.pop('learning_rate').mean()
          metrics_sums = jax.tree_map(jnp.sum, train_metrics)
          denominator = metrics_sums.pop('denominator')
          summary = jax.tree_map(lambda x: x / denominator, metrics_sums)  # pylint: disable=cell-var-from-loop
          summary['learning_rate'] = lr
          summary = {'train_' + k: v for k, v in summary.items()}
          writer.write_scalars(step, summary)
          train_metrics = []

        with report_progress.timed('eval'):
          eval_results = evaluate(
              p_eval_step=p_eval_step,
              target=optimizer.target,
              eval_ds=eval_ds,
              num_eval_steps=FLAGS.num_eval_steps)
          curr_eval_loss = eval_results['loss']
          writer.write_scalars(
              step, {'eval_' + k: v for k, v in eval_results.items()})

        if FLAGS.aux_eval_dataset:
          for aux_i, aux_eval_ds in enumerate(aux_datasets):
            with report_progress.timed('aux_eval'):
              eval_results = evaluate(
                  p_eval_step=p_eval_step,
                  target=optimizer.target,
                  eval_ds=aux_eval_ds,
                  num_eval_steps=FLAGS.num_eval_steps)
              writer.write_scalars(
                  step, {
                      'aux' + str(aux_i) + '_eval_' + k: v
                      for k, v in eval_results.items()
                  })

        if FLAGS.compute_bleu:
          with report_progress.timed('translate_and_bleu'):
            exemplars, bleu_score = translate_and_calculate_bleu(
                p_pred_step=p_pred_step,
                p_init_cache=p_init_cache,
                target=optimizer.target,
                predict_ds=predict_ds,
                decode_tokens=decode_tokens,
                max_predict_length=FLAGS.max_predict_length)
            writer.write_scalars(step, {'bleu': bleu_score})
            writer.write_texts(step, {'samples': exemplars})

      # Save a checkpoint on one host after every checkpoint_freq steps.
      save_checkpoint = ((step - start_step) % FLAGS.checkpoint_freq == 0 or
                         is_last_step)
      if FLAGS.save_checkpoints and save_checkpoint and jax.host_id() == 0:
        if curr_eval_loss < best_eval_loss:  # only save better checkpoints
          best_eval_loss = curr_eval_loss
          with report_progress.timed('checkpoint'):
            checkpoints.save_checkpoint(
                FLAGS.model_dir, jax_utils.unreplicate(optimizer),
                step, keep=FLAGS.chkpts_to_keep, overwrite=True)

      if is_last_step:
        break


def get_new_distribution(p_get_diag_grads, p_get_bucket_score, eval_ds,
                         train_ds_mgr, optimizer, ft_ds):
  """Compute the new training distribution."""
  new_distribution = [1./FLAGS.num_data_buckets]*FLAGS.num_data_buckets
  val_grad, curr_grad, ft_grad = None, None, None
  for bucket_id in range(-2, FLAGS.num_data_buckets):
    print('Running bucket', bucket_id)
    if curr_grad is not None:
      del curr_grad
      curr_grad = None

    if bucket_id == -2:
      diag_iter = iter(eval_ds)
    elif bucket_id == -1:
      diag_iter = iter(ft_ds)
    else:
      diag_iter = train_ds_mgr.get_bucket(bucket_id)

    diag_batch = next(diag_iter)
    diag_batch = common_utils.shard(jax.tree_map(
        lambda x: x._numpy(), diag_batch))  # pylint: disable=protected-access

    if bucket_id == -2:
      val_grad = p_get_diag_grads(optimizer, diag_batch)
    elif bucket_id == -1:
      ft_grad = p_get_diag_grads(optimizer, diag_batch)
    else:  # get diag grad
      curr_grad = p_get_diag_grads(optimizer, diag_batch)

    # compute bucket score
    if bucket_id == -2:
      print('Val grad mean')
      val_grad = jax.pmap(
          functools.partial(tree_div, val_y=FLAGS.num_train_steps),
          axis_name='batch')(val_grad)
    if bucket_id == -1:
      print('Val grad mean')
      ft_grad = jax.pmap(
          functools.partial(tree_div, val_y=FLAGS.num_train_steps),
          axis_name='batch')(ft_grad)
    if bucket_id >= 0:
      print('cur grad mean')
      curr_grad = jax.pmap(
          functools.partial(tree_div, val_y=FLAGS.num_train_steps),
          axis_name='batch')(curr_grad)

      print('compute bucket scores')
      score = p_get_bucket_score(val_grad, ft_grad, curr_grad)
      device_score = jax.tree_map(lambda x: x[0], score)
      score_np = jax.device_get(device_score)
      new_distribution[bucket_id] = score_np
  logging.info(new_distribution)
  new_distribution = np.array(new_distribution).ravel()

  return new_distribution


def build_split(train_ds_mgr, split):
  bucket_size = 4500000. / FLAGS.num_data_buckets
  in_domain = int(np.round(FLAGS.data_selection_size / bucket_size))
  deadbins = 20
  new_distribution = np.zeros(FLAGS.num_data_buckets)
  new_distribution[:in_domain] = split / in_domain
  out_of_domain = FLAGS.num_data_buckets - in_domain - deadbins
  new_distribution[in_domain:out_of_domain] = (1-split) / out_of_domain
  train_ds = train_ds_mgr.sampled_dataset(new_distribution)
  return iter(train_ds)


def get_macro_distribution(p_get_diag_grads, p_get_bucket_score, eval_ds,
                           train_ds_mgr, optimizer, ft_ds):
  """Compute the new training distribution."""
  options = [0] * 10
  val_grad, curr_grad, ft_grad = None, None, None
  splits = [1.0, .95, .9, .85, .8, .75, .6, .5, .4, .25]
  for bucket_id in range(-2, 10):
    print('Running bucket', bucket_id)
    if curr_grad is not None:
      del curr_grad
      curr_grad = None

    if bucket_id == -2:
      diag_iter = iter(eval_ds)
    elif bucket_id == -1:
      diag_iter = iter(ft_ds)
    else:
      diag_iter = build_split(train_ds_mgr, splits[bucket_id])

    diag_batch = next(diag_iter)
    diag_batch = common_utils.shard(jax.tree_map(
        lambda x: x._numpy(), diag_batch))  # pylint: disable=protected-access

    if bucket_id == -2:
      val_grad = p_get_diag_grads(optimizer, diag_batch)
    elif bucket_id == -1:
      ft_grad = p_get_diag_grads(optimizer, diag_batch)
    else:  # get diag grad
      curr_grad = p_get_diag_grads(optimizer, diag_batch)

    # compute bucket score
    if bucket_id == -2:
      print('Val grad mean')
      val_grad = jax.pmap(
          functools.partial(tree_div, val_y=FLAGS.num_train_steps),
          axis_name='batch')(val_grad)
    if bucket_id == -1:
      print('Val grad mean')
      ft_grad = jax.pmap(
          functools.partial(tree_div, val_y=FLAGS.num_train_steps),
          axis_name='batch')(ft_grad)
    if bucket_id >= 0:
      print('cur grad mean')
      curr_grad = jax.pmap(
          functools.partial(tree_div, val_y=FLAGS.num_train_steps),
          axis_name='batch')(curr_grad)

      print('compute bucket scores')
      score = p_get_bucket_score(val_grad, ft_grad, curr_grad)
      device_score = jax.tree_map(lambda x: x[0], score)
      score_np = jax.device_get(device_score)
      options[bucket_id] = score_np
  logging.info(options)
  logging.info(splits[np.argmax(options)])
  return build_split(train_ds_mgr, splits[np.argmax(options)])


def get_diag_score(g_val, g_ft, g_curr, strategy=ORTH):
  """Compute diagnostic - cosine similarity between gradients."""
  if strategy == ORTH:
    g_curr_proj_on_train = tree_div(
        tree_mult(g_val, tree_dot(g_curr, g_val)),
        tree_dot(g_val, g_val))
    g_curr_orth = tree_diff(g_curr, g_curr_proj_on_train)
    orth_dot_valid = tree_dot(g_curr_orth, g_ft)
    orth_score = orth_dot_valid / tree_norm(g_curr_orth) * tree_norm(g_ft)
    return orth_score

  else:
    obw_dot_valid = tree_dot(g_curr, g_val)
    greedy_score = obw_dot_valid / tree_norm(g_curr) * tree_norm(g_val)
    return greedy_score


if __name__ == '__main__':
  app.run(main)
