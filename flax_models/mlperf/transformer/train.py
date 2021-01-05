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

# Lint as: python3
"""Machine Translation example.

This script trains a Transformer on a WMT dataset.
"""

from concurrent.futures import thread
import functools
import gc
import math
import os
import time
from absl import app
from absl import flags
from absl import logging
import flax
from flax import jax_utils
from flax import nn
from flax import optim
from flax import serialization
from flax.metrics import tensorboard
from flax.training import common_utils
import jax
from jax import config
from jax import lax
from jax import random
from jax.interpreters.sharded_jit import sharded_jit
import jax.nn
import jax.numpy as jnp
import msgpack
import numpy as np
import tensorflow.compat.v2 as tf

from flax_models.mlperf.transformer import bleu
from flax_models.mlperf.transformer import decode
from flax_models.mlperf.transformer import mllog
from flax_models.mlperf.transformer import mlperf_encoder
from flax_models.mlperf.transformer import mlperf_input_pipeline as input_pipeline
from flax_models.mlperf.transformer import models
from flax_models.mlperf.transformer import partitions

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'model_dir', default=None,
    help='Directory to store model data')

flags.DEFINE_integer(
    'batch_size', default=None,
    help='Global batch size for training.')

flags.DEFINE_integer(
    'num_eval_steps', default=20,
    help='Number of eval steps, if extra_eval_metrics=True.')

flags.DEFINE_float(
    'learning_rate', default=None,
    help='Base learning rate.')

flags.DEFINE_float(
    'weight_decay', default=0.0,
    help='Decay factor for AdamW style weight decay.')

flags.DEFINE_integer(
    'max_target_length', default=256,
    help='Maximum length of training examples.')

flags.DEFINE_integer(
    'max_eval_target_length', default=97,
    help='Maximum length of eval examples.')

flags.DEFINE_integer(
    'max_predict_length', default=147,
    help='Maximum length for predicted tokens.')

flags.DEFINE_bool(
    'share_embeddings', default=True,
    help='Inputs and targets share embedding.')

flags.DEFINE_bool(
    'logits_via_embedding', default=True,
    help='Final logit transform uses embedding matrix transpose.')

flags.DEFINE_integer(
    'random_seed', default=None,
    help='Integer for PRNG random seed.')

flags.DEFINE_bool(
    'use_bfloat16', default=True,
    help=('Use bfloat16 mixed precision training instead of float32.'))

flags.DEFINE_bool(
    'hardware_rng', default=True,
    help='Whether to use hardware rng for dropout.')

flags.DEFINE_bool(
    'compute_train_metrics', default=False,
    help='Whether to compute metrics during training.')

flags.DEFINE_bool(
    'precompile', default=True,
    help='Perform all XLA compilation before touching data.')

flags.DEFINE_bool(
    'infeed', default=True,
    help='Use infeed in training loop.')

flags.DEFINE_integer(
    'num_epochs', default=7,
    help='Number of epochs to train for.')

flags.DEFINE_bool(
    'extra_eval_metrics', default=False,
    help='Whether to calculate non-BLEU eval metrics.')


flags.DEFINE_integer(
    'num_partitions', default=1,
    help='Number of SPMD partitions to use.')

flags.DEFINE_bool(
    'mlperf_logs', default=True,
    help='Use official MLPerf format for logs.')

config.parse_flags_with_absl()

mllogger = None
# Global Stopping Condition set by BLEU Evaluation Thread
BLEU_THRESHOLD_REACHED = False
START_TIME = None


def _unbroadcast(x):
  """Assuming `x` is replicated along its leading axis, remove that axis."""
  # Unbroadcast is a hack to take the output of a pmap with out_axes=0 and turn
  # it into the input of a pmap with in_axes=None. This is necessary because we
  # don't have out_axes=None in pmap, so the output arrays of the training step
  # function all still end up with an extra leading logical axis of size
  # `num_local_devices`.
  sharding_spec = x.sharding_spec
  # The leading logical axis should be sharded like the result of a pmap with
  # out_axes=0.
  assert sharding_spec.sharding[0] == jax.pxla.Unstacked(x.shape[0])
  # Remove that leading logical axis and its corresponding sharding.
  aval = jax.abstract_arrays.ShapedArray(x.shape[1:], x.dtype)
  sharding = sharding_spec.sharding[1:]
  # Replace the mesh mapping entry that pointed to that axis with Replicated,
  # and decrement the other entries.
  def replace_mesh_mapping(mm):
    if isinstance(mm, jax.pxla.ShardedAxis):
      if mm.axis == 0:
        return jax.pxla.Replicated(x.shape[0])
      return jax.pxla.ShardedAxis(mm.axis - 1)
    return mm
  mesh_mapping = map(replace_mesh_mapping, sharding_spec.mesh_mapping)
  sharding_spec = jax.pxla.ShardingSpec(sharding, mesh_mapping)
  return jax.pxla.ShardedDeviceArray(aval, sharding_spec, x.device_buffers)


def unbroadcast(tree):
  """Assuming `tree` is replicated along its leading axis, remove that axis."""
  return jax.tree_map(_unbroadcast, tree)


def _broadcast(tree, num_replicas, num_partitions):
  """Broadcast `tree` according to `num_replicas` and `num_partitions`.

  Args:
    tree: pytree of arrays
    num_replicas: number of replicas (i.e. pmap dimension size)
    num_partitions: number of partitions

  Returns:
    A tree of ShardedDeviceArrays with leading sharded axis of size
    `num_replicas`, each of which contains a copy of the tree element, and is
    further replicated `num_partitions` times. This is suitable for passing to
    pmap(sharded_jit) if the data should be replicated on every device.
  """
  assert num_replicas * num_partitions == jax.local_device_count()
  # Replicate across all devices
  replicated = jax_utils.replicate(tree)

  # Rewrite the sharding specs to include replicated partitioning
  def redo_sharding_spec(x):
    assert isinstance(x, jax.pxla.ShardedDeviceArray)
    sharding_spec = x.sharding_spec
    aval = jax.abstract_arrays.ShapedArray((num_replicas,) + x.shape[1:],
                                           x.dtype)
    ntrailing = x.ndim - 1
    sharding_spec = jax.pxla.ShardingSpec(
        shards_per_axis=(num_replicas,) + (1,) * ntrailing,
        is_axis_materialized=(False,) + (True,) * ntrailing,
        replication_factors=[(num_partitions, 1)])
    return jax.pxla.ShardedDeviceArray(aval, sharding_spec, x.device_buffers)

  if num_partitions > 1:
    return jax.tree_map(redo_sharding_spec, replicated)
  else:
    return replicated


def _sync_devices(x):
  return jax.lax.psum(x, 'i')


def sync_devices():
  """Creates a barrier across all hosts/devices."""
  jax.pmap(_sync_devices, 'i')(
      np.ones(jax.local_device_count())).block_until_ready()


def create_learning_rate_scheduler(
    factors='constant * linear_warmup * rsqrt_decay * rsqrt_hidden_size',
    base_learning_rate=0.5,
    warmup_steps=1000,
    decay_factor=0.5,
    steps_per_decay=20000,
    steps_per_cycle=100000,
    hidden_size=1024):
  """creates learning rate schedule.

  Interprets factors in the factors string which can consist of:
  * constant: interpreted as the constant value,
  * linear_warmup: interpreted as linear warmup until warmup_steps,
  * rsqrt_decay: divide by square root of max(step, warmup_steps)
  * decay_every: Every k steps decay the learning rate by decay_factor.
  * cosine_decay: Cyclic cosine decay, uses steps_per_cycle parameter.

  Args:
    factors: a string with factors separated by '*' that defines the schedule.
    base_learning_rate: float, the starting constant for the lr schedule.
    warmup_steps: how many steps to warm up for in the warmup schedule.
    decay_factor: The amount to decay the learning rate by.
    steps_per_decay: How often to decay the learning rate.
    steps_per_cycle: Steps per cycle when using cosine decay.
    hidden_size: size of feature dimension in attention layers.

  Returns:
    a function learning_rate(step): float -> {'learning_rate': float}, the
    step-dependent lr.
  """
  factors = [n.strip() for n in factors.split('*')]

  mllogger.event('opt_base_learning_rate', base_learning_rate)
  mllogger.event('opt_learning_rate_warmup_steps', warmup_steps)

  def step_fn(step):
    """Step to learning rate function."""
    ret = 1.0
    for name in factors:
      if name == 'constant':
        ret *= base_learning_rate
      elif name == 'linear_warmup':
        ret *= jnp.minimum(1.0, step / warmup_steps)
      elif name == 'rsqrt_decay':
        ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
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
      elif name == 'rsqrt_hidden_size':
        ret /= jnp.sqrt(1.0 * hidden_size)
      else:
        raise ValueError('Unknown factor %s.' % name)
    return jnp.asarray(ret, dtype=jnp.float32)

  return step_fn


@functools.partial(jax.jit, static_argnums=(1, 2, 3))
def create_model(key, input_shape, target_shape, model_kwargs):
  """Instantiate transformer model and associated autoregressive cache def."""
  model_def = models.Transformer.partial(**model_kwargs)
  with models.Cache().mutate() as cache_def:
    _, model = model_def.create_by_shape(key,
                                         [(input_shape, jnp.float32),
                                          (target_shape, jnp.float32)],
                                         cache=cache_def)
  return model, cache_def


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
    g_logits = jnp.expand_dims(g, axis=-1) * (exp_shifted / sum_exp - targets)
    return jnp.asarray(g_logits, logits.dtype), jnp.asarray(g, targets.dtype)

  return loss, grad_fn


def compute_weighted_cross_entropy(logits, targets, weights=None,
                                   label_smoothing=0.1):
  """Compute weighted cross entropy and entropy for log probs and targets.

  Args:
    logits: [batch * length, num_classes] float array.
    targets: categorical targets [batch, length] int array.
    weights: None or array of shape [batch, length]
    label_smoothing: label smoothing constant, used to determine the on and
      off values.

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
      confidence * jnp.log(confidence) + (vocab_size - 1) *
      low_confidence * jnp.log(low_confidence + 1e-20))
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
  return metrics


def train_step(optimizer, batch, prev_metrics, learning_rate_fn,
               dropout_rng=None):
  """Perform a single training step."""
  train_keys = ['inputs', 'targets',
                'inputs_position', 'targets_position',
                'inputs_segmentation', 'targets_segmentation']
  (inputs, targets,
   inputs_positions, targets_positions,
   inputs_segmentation, targets_segmentation) = [
       batch.get(k, None) for k in train_keys]

  weights = jnp.where(targets > 0, 1, 0).astype(jnp.float32)

  # We handle PRNG splitting inside the top pmap to prevent stalls to
  # device compute scheduling and data transfer.
  dropout_rng, new_dropout_rng = random.split(dropout_rng)

  def loss_fn(model):
    """loss function used for training."""
    with nn.stochastic(dropout_rng):
      logits = model(
          inputs,
          targets,
          use_bfloat16=FLAGS.use_bfloat16,
          inputs_positions=inputs_positions,
          targets_positions=targets_positions,
          inputs_segmentation=inputs_segmentation,
          targets_segmentation=targets_segmentation,
          train=True,
          cache=None)

    loss, weight_sum = compute_weighted_cross_entropy(logits, targets, weights)
    mean_loss = loss / weight_sum
    return mean_loss, logits

  step = optimizer.state.step
  lr = learning_rate_fn(step)
  (_, logits), grad = jax.value_and_grad(
      loss_fn, has_aux=True)(optimizer.target)
  if FLAGS.use_bfloat16:
    grad = jax.tree_map(lambda x: x.astype(jnp.bfloat16), grad)
  grad = lax.pmean(grad, 'batch')
  grad = jax.tree_map(lambda x: x.astype(jnp.float32), grad)
  new_optimizer = optimizer.apply_gradient(grad, learning_rate=lr)

  if FLAGS.compute_train_metrics:
    metrics = compute_metrics(logits, targets, weights)
    metrics = jax.tree_multimap(jnp.add, prev_metrics, metrics)
  else:
    metrics = prev_metrics

  return new_optimizer, metrics, new_dropout_rng


def eval_step(model, batch):
  """Calculate evaluation metrics on a batch."""
  inputs, targets = batch['inputs'], batch['targets']
  weights = jnp.where(targets > 0, 1.0, 0.0)
  logits = model(inputs, targets, use_bfloat16=FLAGS.use_bfloat16, train=False,
                 cache=None)
  return compute_metrics(logits, targets, weights)


def predict_step(inputs, model, cache):
  """Predict translation with fast decoding beam search on a batch."""
  batch_size = inputs.shape[0]
  beam_size = 4

  # Prepare transformer fast-decoder call for beam search:
  # for beam search, we need to set up our decoder model
  # to handle a batch size equal to batch_size * beam_size,
  # where each batch item's data is expanded in-place rather
  # than tiled.
  # i.e. if we denote each batch element subtensor as el[n]:
  # [el0, el1, el2] --> beamsize=2 --> [el0,el0,el1,el1,el2,el2]
  src_padding_mask = decode.flat_batch_beam_expand(
      (inputs > 0)[Ellipsis, None], beam_size)
  tgt_padding_mask = decode.flat_batch_beam_expand(
      jnp.ones((batch_size, 1, 1)), beam_size)
  encoded_inputs = decode.flat_batch_beam_expand(
      model.encode(inputs, use_bfloat16=FLAGS.use_bfloat16,
                   train=False, cache=None),
      beam_size)

  def tokens_ids_to_logits(flat_ids, flat_cache):
    """Token slice to logits from decoder model."""
    # --> [batch * beam, 1, vocab]
    with flat_cache.mutate() as new_flat_cache:
      flat_logits = model.decode(encoded_inputs,
                                 src_padding_mask,
                                 flat_ids,
                                 use_bfloat16=FLAGS.use_bfloat16,
                                 cache=new_flat_cache,
                                 shift=False,
                                 train=False,
                                 tgt_padding_mask=tgt_padding_mask)
    return flat_logits, new_flat_cache

  # using the above-defined single-step decoder function, run a
  # beam search over possible sequences given input encoding.
  beam_seqs, _ = decode.beam_search(
      inputs,
      cache,
      tokens_ids_to_logits,
      beam_size=beam_size,
      alpha=0.6,
      eos_token=1,
      max_decode_len=FLAGS.max_predict_length)

  # beam search returns [n_batch, n_beam, n_length + 1] with beam dimension
  # sorted in increasing order of log-probability
  # return the highest scoring beam sequence, drop first dummy 0 token.
  return beam_seqs[:, -1, 1:]


def pad_examples(x, desired_batch_size):
  """Expand batch to desired size by repeating last slice."""
  if x.shape[0] == 0:
    return np.ones((desired_batch_size,
                    input_pipeline.MAX_EVAL_LEN), dtype=np.int32)
  batch_pad = desired_batch_size - x.shape[0]
  return np.concatenate([x, np.tile(x[-1], (batch_pad, 1))], axis=0)


def tohost(x):
  """Collect batches from all devices to host and flatten batch dimensions."""
  n_device, n_batch, *remaining_dims = x.shape
  return np.array(x).reshape((n_device * n_batch,) + tuple(remaining_dims))


def write_train_summary(metrics, train_summary_writer, step):
  """Training summary task for summary thread."""
  metrics = jax.tree_map(lambda x: jax.device_get(x[0]), metrics)
  denominator = metrics.pop('denominator')
  summary = jax.tree_map(lambda x: x / denominator, metrics)  # pylint: disable=cell-var-from-loop
  logging.info('train in step: %d, loss: %.4f', step, summary['loss'])
  if jax.host_id() == 0:
    for key, val in summary.items():
      train_summary_writer.scalar(key, val, step)
    train_summary_writer.flush()


def write_eval_summary(metrics, eval_summary_writer, step):
  """Eval summary task for summary thread."""
  metrics = jax.tree_map(lambda x: jax.device_get(x[0]), metrics)
  eval_denominator = metrics.pop('denominator')
  eval_summary = jax.tree_map(
      lambda x: x / eval_denominator,  # pylint: disable=cell-var-from-loop
      metrics)
  logging.info('eval in step: %d, loss: %.4f', step, eval_summary['loss'])
  if jax.host_id() == 0:
    for key, val in eval_summary.items():
      eval_summary_writer.scalar(key, val, step)
    eval_summary_writer.flush()


def per_host_sum(x):
  return jax.lax.psum(x, 'hosts')


def per_host_sum_pmap(in_tree):
  """Execute sum on in_tree's leaves over ICI."""
  ldc = jax.local_device_count()
  host_psum = jax.pmap(per_host_sum, axis_name='hosts')
  def pre_pmap(x):
    y = np.zeros((ldc, *x.shape), dtype=x.dtype)
    y[0] = x
    return y
  def post_pmap(x):
    return jax.device_get(x)[0]
  return jax.tree_map(post_pmap, host_psum(jax.tree_map(pre_pmap, in_tree)))


def per_host_sum_fs(in_tree, step):
  """Execute sum on in_tree's leaves across each host.

  Data is shared via the filesystem.

  Args:
    in_tree: pytree w. array leaves.
    step: int: step number for marking temporary files.

  Returns:
    out_tree w. same shape as in_tree, result of sum across in_trees
    from each host.
  """
  def fname(step, host_id):
    return os.path.join(FLAGS.model_dir, f'partial_bleu_{step}_{host_id}')
  # Write this host's data to filesystem.
  logging.info('saving partial bleu stats: %s', fname(step, jax.host_id()))
  with tf.io.gfile.GFile(fname(step, jax.host_id()), 'wb') as fp:
    fp.write(serialization.msgpack_serialize(list(in_tree)))
  # Load other hosts' data by polling filesystem for known files.
  results = {k: None for k in jax.host_ids()}
  results[jax.host_id()] = tuple(in_tree)
  while not all(results.values()):
    unfinished = [k for k in results if results[k] is None]
    for host_id in unfinished:
      # If file exists, read contents.
      if tf.io.gfile.exists(fname(step, host_id)):
        with tf.io.gfile.GFile(fname(step, host_id), 'rb') as fp:
          data = fp.read()
        try:
          res = serialization.msgpack_restore(data)
          results[host_id] = tuple(res)
        # Catch incomplete written file edgecase and continue looping.
        except msgpack.exceptions.UnpackValueError:
          pass
    time.sleep(1)
  # Return sum-aggregated partial bleu statistics.
  return functools.reduce(
      lambda x, y: jax.tree_multimap(np.add, x, y),
      results.values())


def write_predict_summary(all_predicted, all_targets, all_bs, target_encoder,
                          eval_summary_writer, epoch, step, summary_thread):
  """Prediction (BLEU) summary task for summary thread."""
  global BLEU_THRESHOLD_REACHED
  references, predictions = [], []
  for predicted, targets, bsize in zip(all_predicted, all_targets, all_bs):
    predicted = tohost(predicted)
    targets = tohost(targets)
    # Iterate through non-padding examples of batch.
    for i, s in enumerate(predicted[:bsize]):
      references.append(target_encoder.decode(targets[i]))
      # TODO(levskaya): debug very rare initial 0-token predictions.
      try:
        predictions.append(target_encoder.decode(s))
      except ValueError:
        logging.error('bad predicted tokens: %s', s)
        predictions.append('Wir haben technische Schwierigkeiten.')
  # Calculate BLEU score for translated eval corpus against reference.
  bleu_matches = bleu.bleu_partial(references, predictions)
  all_bleu_matches = per_host_sum_pmap(bleu_matches)
  if jax.host_id() == 0:
    bleu_score = bleu.complete_bleu(*all_bleu_matches)
    def _write_predict_summary():
      eval_summary_writer.scalar('bleu', bleu_score, step)
      eval_summary_writer.flush()
    if not BLEU_THRESHOLD_REACHED:
      mllogger.event('eval_accuracy', bleu_score / 100,
                     metadata={'epoch_num': epoch + 1})
      mllogger.end('block_stop',
                   metadata={'first_epoch_num': epoch + 1, 'epoch_count': 1})
      if bleu_score >= 25:
        mllogger.end('run_stop', metadata={'status': 'success'})
        logging.info('training time: %.2f seconds', time.time() - START_TIME)
        BLEU_THRESHOLD_REACHED = True
    summary_thread.submit(_write_predict_summary)


def init_mllogger():
  global mllogger
  mllogger = mllog.MLLogger(
      logging.get_absl_logger(), jax.host_id(), full=FLAGS.mlperf_logs)


def main(argv):
  global BLEU_THRESHOLD_REACHED, START_TIME
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  init_mllogger()
  mllogger.event('cache_clear')
  mllogger.start('init_start')
  mllogger.event('submission_org', 'Google')
  mllogger.event('submission_platform', 'TPUv3-{}'.format(jax.device_count()))
  mllogger.event('submission_division', 'closed')
  mllogger.event('submission_status', 'research')
  mllogger.event('submission_benchmark', 'transformer')
  mllogger.event('train_samples', input_pipeline.N_TRAIN)
  mllogger.event('eval_samples', input_pipeline.N_EVAL)

  tf.enable_v2_behavior()

  # Use hardware RNG for bernoulli randoms in dropout mask creation.
  if FLAGS.hardware_rng:
    models.set_hardware_bernoulli()

  num_partitions = FLAGS.num_partitions
  batch_size = FLAGS.batch_size
  if batch_size is None:
    batch_size = min(16 * jax.device_count() // num_partitions, 2048)
  mllogger.event('global_batch_size', batch_size)

  num_eval_steps = FLAGS.num_eval_steps
  max_target_length = FLAGS.max_target_length
  max_eval_target_length = FLAGS.max_eval_target_length
  max_length = max(max_target_length, max_eval_target_length)
  mllogger.event(
      'max_sequence_length', max_length, metadata={'method': 'discard'})
  if FLAGS.random_seed is not None:
    seed = FLAGS.random_seed
  else:
    seed = np.uint32(time.time() if jax.host_id() == 0 else 0)
    seed = per_host_sum_pmap(seed)
  mllogger.event('seed', int(seed))
  steps_per_epoch = int(math.ceil(input_pipeline.N_TRAIN / batch_size))
  logging.info('steps per epoch: %d', steps_per_epoch)
  num_replicas = jax.local_device_count() // num_partitions
  device_train_input_shape = (batch_size // (num_replicas * jax.host_count()),
                              max_target_length)
  # This is per-host; in principle 64/replica or more should fit
  eval_batch_size = min(
      32 * num_replicas,
      int(math.ceil(input_pipeline.N_EVAL / (num_replicas * jax.host_count())))
      * num_replicas)
  logging.info('eval batch size: %d', eval_batch_size)
  pred_batches = int(math.ceil(input_pipeline.N_EVAL / (
      jax.host_count() * eval_batch_size)))
  logging.info('pred batches: %d', pred_batches)
  broadcast = functools.partial(_broadcast, num_replicas=num_replicas,
                                num_partitions=num_partitions)

  if jax.host_id() == 0:
    train_summary_writer = tensorboard.SummaryWriter(
        os.path.join(FLAGS.model_dir, 'train'))
    eval_summary_writer = tensorboard.SummaryWriter(
        os.path.join(FLAGS.model_dir, 'eval'))
  else:
    train_summary_writer = None
    eval_summary_writer = None
  # Write summaries in background thread to avoid blocking on device sync
  summary_thread = thread.ThreadPoolExecutor(1, 'summary')
  if FLAGS.infeed:
    # Infeed is currently synchronous, so do it in a background thread too
    infeed_pool = thread.ThreadPoolExecutor(jax.local_device_count(), 'infeed')


  # MLPerf 2020 WMT en-de dataset uses a custom T2T dataset:
  #   Shared 32K subword tokenization
  #   256-length packed training examples from WMT17
  #   97-length unpacked evaluation examples from WMT14
  train_keys = ['inputs', 'targets',
                'inputs_position', 'targets_position',
                'inputs_segmentation', 'targets_segmentation']
  encoder = mlperf_encoder.SubwordTextEncoder(filename=FLAGS.vocab_path)
  input_encoder = encoder
  target_encoder = encoder
  vocab_size = input_encoder.vocab_size
  output_vocab_size = target_encoder.vocab_size

  input_shape = (batch_size, max_target_length)
  target_shape = (batch_size, max_target_length)

  transformer_kwargs = flax.core.FrozenDict({
      'vocab_size': vocab_size,
      'output_vocab_size': output_vocab_size,
      'emb_dim': 1024,
      'num_heads': 16,
      'num_layers': 6,
      'qkv_dim': 1024,
      'mlp_dim': 4096,
      'max_len': max_length,
      'share_embeddings': FLAGS.share_embeddings,
      'logits_via_embedding': FLAGS.logits_via_embedding,
      'num_partitions': num_partitions,
  })

  rng = random.PRNGKey(seed)
  rng, init_rng = random.split(rng)
  logging.info('initializing model')
  model, cache_def = create_model(init_rng,
                                  tuple(input_shape),
                                  tuple(target_shape),
                                  transformer_kwargs)
  mllogger.event('opt_name', 'adam')
  if batch_size < 1024:
    learning_rate = 4.0  # 0.0625
    warmup_steps = 1000
    beta1 = 0.9
    beta2 = 0.98
  if batch_size < 2048:
    learning_rate = 2.0
    warmup_steps = 500  # ??
    beta1 = 0.9  # ??
    beta2 = 0.98  # ??
  else:
    learning_rate = 3.3092157691415953
    warmup_steps = 664
    beta1 = 0.9086575725261137
    beta2 = 0.9198719118104947
  epsilon = 1e-9
  if FLAGS.learning_rate is not None:
    learning_rate = FLAGS.learning_rate
  mllogger.event('opt_adam_beta_1', beta1)
  mllogger.event('opt_adam_beta_2', beta2)
  mllogger.event('opt_adam_epsilon', epsilon)
  logging.info('initializing optimizer')
  optimizer_def = optim.Adam(
      learning_rate,
      beta1=beta1,
      beta2=beta2,
      eps=epsilon,
      weight_decay=FLAGS.weight_decay)
  optimizer = optimizer_def.create(model)
  del model  # don't keep a copy of the initial model

  # Build parameter partition annotations for preserving partitions from train
  # to eval.
  partition_rules = [
      (('encoder', 'posembed_input'), partitions.empty_dict),
      (('decoder', 'posembed_targets'), partitions.empty_dict),
      (('embedding',), partitions.spec(num_partitions, 1)),
      ((r'LayerNorm_\d+', '(bias|scale)'), None),
      ((r'encoder(decoder)?_norm', '(bias|scale)'), None),
      ((r'MultiHeadDotProductAttention_\d+', '(query|key|value)', 'kernel'),
       partitions.spec(1, num_partitions, 1)),
      ((r'MultiHeadDotProductAttention_\d+', 'out', 'kernel'),
       partitions.spec(num_partitions, 1, 1)),
      ((r'MlpBlock_\d+', r'Dense_\d+', 'bias'), None),
      ((r'MlpBlock_\d+', 'Dense_0', 'kernel'),
       partitions.spec(1, num_partitions)),
      ((r'MlpBlock_\d+', 'Dense_1', 'kernel'),
       partitions.spec(num_partitions, 1)),
      (('state', 'step'), None),
  ]
  optimizer_partitions = optimizer.restore_state(
      partitions.set_partitions(partition_rules, optimizer.state_dict()))

  optimizer = broadcast(optimizer)
  empty_metrics = broadcast({'loss': 0.0, 'accuracy': 0, 'denominator': 0})

  learning_rate_fn = create_learning_rate_scheduler(
      base_learning_rate=learning_rate,
      warmup_steps=warmup_steps,
      hidden_size=transformer_kwargs['qkv_dim'])

  p_train_step = jax.pmap(
      functools.partial(train_step, learning_rate_fn=learning_rate_fn),
      axis_name='batch',
      in_axes=(None, 0, 0, 0))
  if num_partitions > 1:
    sharded_predict_step = sharded_jit(
        predict_step,
        in_parts=(None, optimizer_partitions.target, None),
        out_parts=None)
  else:
    sharded_predict_step = predict_step
  if FLAGS.extra_eval_metrics:
    p_eval_step = jax.pmap(eval_step, axis_name='batch', in_axes=(None, 0))
  p_pred_step = jax.pmap(sharded_predict_step, axis_name='batch',
                         in_axes=(0, None, None))
  p_allreduce_metrics = jax.pmap(
      functools.partial(lax.psum, axis_name='batch'),
      axis_name='batch')

  def device_train_loop_cond(args):
    _, _, _, _, step, epoch = args
    return step // steps_per_epoch == epoch
  def device_train_loop_body(args):
    optimizer, dropout_rngs, metrics, token, step, epoch = args
    input_data, token = lax.infeed(token, shape=tuple(
        [jax.ShapedArray(device_train_input_shape, jnp.int32)
         for _ in train_keys]))
    batch = {k: v for k, v in zip(train_keys, input_data)}
    optimizer, metrics, dropout_rngs = train_step(
        optimizer, batch, metrics, learning_rate_fn, dropout_rng=dropout_rngs)
    step += 1
    return optimizer, dropout_rngs, metrics, token, step, epoch
  def device_train_loop(optimizer, dropout_rngs, metrics, step, epoch):
    token = lax.create_token(step)
    optimizer, dropout_rngs, metrics, _, step, _ = lax.while_loop(
        device_train_loop_cond,
        device_train_loop_body,
        (optimizer, dropout_rngs, metrics, token, step, epoch))
    return optimizer, dropout_rngs, metrics, step
  if num_partitions > 1:
    device_train_loop = sharded_jit(
        device_train_loop,
        in_parts=(optimizer_partitions, None, None, None, None),
        out_parts=(optimizer_partitions, None, None, None))
  p_train_epoch = jax.pmap(
      device_train_loop,
      axis_name='batch',
      in_axes=(None, 0, 0, None, None))

  p_allreduce_metrics_train = functools.partial(lax.psum, axis_name='batch')
  if num_partitions > 1:
    p_allreduce_metrics_train = sharded_jit(
        p_allreduce_metrics_train,
        in_parts=None,
        out_parts=None,
        num_partitions=num_partitions)
  p_allreduce_metrics_train = jax.pmap(
      p_allreduce_metrics_train, axis_name='batch')

  # Precompile all needed computations with fake data so as not to include
  # compilation time in MLPerf metrics.
  if FLAGS.precompile:
    logging.info('precompiling step/epoch functions')
    if FLAGS.infeed:
      # the device training loop condition will immediately be false, but
      # the optimizer tree will be resharded here
      optimizer, *_ = p_train_epoch(
          unbroadcast(optimizer), random.split(rng, num_replicas),
          empty_metrics, jnp.array(0, dtype=jnp.int32), 1)
    else:
      metrics = empty_metrics
      train_input_shape = (num_replicas,
                           batch_size // num_replicas,
                           input_pipeline.MAX_TRAIN_LEN)
      fake_batch = {k: jnp.ones(train_input_shape, jnp.int32)
                    for k in train_keys}
      p_train_step(unbroadcast(optimizer), fake_batch, metrics,
                   dropout_rng=random.split(rng, num_replicas))
    eval_input_shape = (num_replicas,
                        eval_batch_size // num_replicas,
                        input_pipeline.MAX_EVAL_LEN)
    fake_eval_batch = {
        'inputs': jnp.ones(eval_input_shape, jnp.int32),
        'targets': jnp.ones(eval_input_shape, jnp.int32),
    }
    if FLAGS.extra_eval_metrics:
      p_eval_step(unbroadcast(optimizer.target), fake_eval_batch)
    fake_cache = cache_def.initialize_cache(
        (eval_input_shape[1], FLAGS.max_predict_length))
    p_pred_step(
        fake_eval_batch['inputs'], unbroadcast(optimizer.target), fake_cache)
    time.sleep(20)
    sync_devices()
    fake_bleu_1 = np.zeros((4,), dtype=np.int32)
    fake_bleu_2 = np.zeros((), dtype=np.int32)
    per_host_sum_pmap((fake_bleu_1, fake_bleu_1, fake_bleu_2, fake_bleu_2))
    sync_devices()
    p_allreduce_metrics_train(empty_metrics)
    sync_devices()
    logging.info('finished precompiling step/epoch functions')

  # We init the first set of dropout PRNG keys, but update it afterwards inside
  # the main pmap'd training update for performance.
  dropout_rngs = random.split(rng, num_replicas)

  # Record time-0 metrics for proper tensorboard plot x-axis scaling.
  if jax.host_id() == 0:
    if FLAGS.compute_train_metrics:
      train_summary_writer.scalar('loss', 9.999, 0)
      train_summary_writer.scalar('accuracy', 0.0, 0)
      train_summary_writer.flush()
    eval_summary_writer.scalar('bleu', 0.0, 0)
    eval_summary_writer.flush()

  train_ds = input_pipeline.get_wmt_dataset(
      batch_size=batch_size // jax.host_count(), train=True)
  eval_ds = input_pipeline.get_wmt_dataset(
      batch_size=eval_batch_size, train=False)
  train_iter = iter(train_ds)
  eval_iter = iter(eval_ds)
  local_devices = jax.local_devices()
  host_step, device_step = 0, broadcast(0)
  gc.disable()
  mllogger.end('init_stop')
  if jax.host_id() == 0:
    mllogger.start('run_start')
    START_TIME = time.time()
  for epoch in range(FLAGS.num_epochs):
    if jax.host_id() == 0 and not BLEU_THRESHOLD_REACHED:
      mllogger.start('block_start',
                     metadata={'first_epoch_num': epoch + 1,
                               'epoch_count': 1})
    metrics = empty_metrics
    if FLAGS.infeed:
      optimizer, dropout_rngs, metrics, device_step = p_train_epoch(
          unbroadcast(optimizer), dropout_rngs, metrics,
          unbroadcast(device_step), epoch)
    while int(host_step // steps_per_epoch) == epoch:
      # pylint: disable=protected-access
      batch = jax.tree_map(lambda x: x._numpy(), next(train_iter))
      # Shard data to devices and do a training step.
      batch = jax.tree_map(
          lambda x: x.reshape((num_replicas, -1) + x.shape[1:]), batch)
      if FLAGS.infeed:
        for i, device in enumerate(local_devices):
          replica_id = i // num_partitions
          input_tuple = tuple([batch[k][replica_id] for k in train_keys])
          assert input_tuple[0].shape == device_train_input_shape, (
              'infeed shape error %s != %s' % (
                  input_tuple[0].shape, device_train_input_shape))
          assert input_tuple[0].dtype == jnp.int32, (
              'infeed dtype error %s != %s' % (
                  input_tuple[0].dtype, jnp.int32))
          infeed_pool.submit(functools.partial(device.transfer_to_infeed,
                                               input_tuple))
      else:
        optimizer, metrics, dropout_rngs = p_train_step(
            unbroadcast(optimizer), batch, metrics, dropout_rng=dropout_rngs)
      host_step += 1

    if FLAGS.compute_train_metrics:
      metrics = p_allreduce_metrics_train(metrics)
      # Schedule training metric handling.
      summary_thread.submit(functools.partial(
          write_train_summary, metrics, train_summary_writer, host_step))

    # Optional, extra evaluation metrics.
    if FLAGS.extra_eval_metrics:
      eval_metrics = []
      eval_iter = iter(eval_ds)
      for _, eval_batch in zip(range(num_eval_steps), eval_iter):
        eval_batch = common_utils.shard(eval_batch)
        metrics = p_eval_step(unbroadcast(optimizer.target), eval_batch)
        eval_metrics.append(metrics)
      eval_metrics = p_allreduce_metrics(eval_metrics)
      # Schedule metric summarization/logging.
      summary_thread.submit(functools.partial(
          write_eval_summary, eval_metrics, eval_summary_writer, host_step))

    # Translation and BLEU Score.
    all_predicted, all_targets, all_bs = [], [], []
    for i in range(pred_batches):
      # pylint: disable=protected-access
      pred_batch = jax.tree_map(lambda x: x._numpy(), next(eval_iter))
      # Handle final odd-sized batch by padding instead of dropping it.
      cur_pred_batch_size = pred_batch['inputs'].shape[0]
      if cur_pred_batch_size != eval_batch_size:
        pred_batch = jax.tree_map(
            lambda x: pad_examples(x, eval_batch_size), pred_batch)
      pred_batch = jax.tree_map(
          lambda x: x.reshape((num_replicas, -1) + x.shape[1:]), pred_batch)
      per_device_batchsize = pred_batch['inputs'].shape[1]
      cache = cache_def.initialize_cache((per_device_batchsize,
                                          FLAGS.max_predict_length))
      all_predicted.append(p_pred_step(
          pred_batch['inputs'], unbroadcast(optimizer.target), cache))
      all_targets.append(pred_batch['targets'])
      all_bs.append(cur_pred_batch_size)
    # Schedule BLEU calculation and summarization/logging.
    # We use the ICI as part of BLEU score computation, so we call this from the
    # main thread so the BLEU pmap runs before the next train epoch pmap
    write_predict_summary(all_predicted, all_targets, all_bs, target_encoder,
                          eval_summary_writer, epoch, host_step, summary_thread)

  # Wait until computations are done before exiting
  sync_devices()
  if jax.host_id() == 0:
    summary_thread.shutdown()
    if not BLEU_THRESHOLD_REACHED:
      mllogger.end('run_stop', metadata={'status': 'aborted'})


if __name__ == '__main__':
  app.run(main)
