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

# pytype: disable=wrong-arg-count
# pytype: disable=attribute-error

import functools
import json
import os
import random
import sys
import time

from absl import app
from absl import flags
from absl import logging
from flax import jax_utils
from flax import optim
from flax.metrics import tensorboard
from flax.training import checkpoints
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow.compat.v2 as tf

from latent_programmer import models
from latent_programmer import train_lib
from latent_programmer.tasks.scan import tokens
from latent_programmer.tasks.scan.dataset import input_pipeline

gfile = tf.io.gfile
sys.path.append('..')

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 0, 'Fixed random seed for training.')
flags.DEFINE_float('lr', 1e-3, 'Learning rate.')
flags.DEFINE_float('weight_decay', 1e-1,
                   'Decay factor for AdamW-style weight decay.')
flags.DEFINE_integer('embedding_dim', 128, 'Embedding dimension.')
flags.DEFINE_integer('hidden_dim', 512, 'Hidden dimension.')
flags.DEFINE_integer('num_heads', 4, 'Number of layers.')
flags.DEFINE_integer('num_layers', 3, 'Number of Transformer heads.')

flags.DEFINE_string('dataset_filepattern', None,
                    'Filepattern for TFRecord dataset.')
flags.DEFINE_integer('per_device_batch_size', 16,
                     'Number of program tasks in a batch.')
flags.DEFINE_integer('num_strings_per_task', 4,
                     'Number of input/output strings per task.')
flags.DEFINE_integer('max_expressions', 10,
                     'Maximum number of expressions in program.')
flags.DEFINE_integer('max_program_length', 50,
                     'Maximum number of tokens in program.')
flags.DEFINE_integer('max_characters', 100,
                     'Maximum number of characters in input/output strings.')

flags.DEFINE_string('save_dir', None, 'Directory to save results to.')
flags.DEFINE_integer('num_train_steps', 1500000, 'Number of training steps.')
flags.DEFINE_integer('num_eval_steps', 10, 'Number of evaluation steps.')
flags.DEFINE_integer('log_freq', 1000, 'Number of steps between logs.')
flags.DEFINE_integer('checkpoint_freq', 1000,
                     'Number of steps between checkpoint saves.')
flags.DEFINE_bool('restore_checkpoints', True,
                  'Whether to restore from existing model checkpoints.')

flags.DEFINE_bool('use_relative_attention', False,
                  'Whether to use relative positonal embeddings.')
flags.DEFINE_integer('num_relative_position_buckets', 32,
                     'Number of buckets when computing relative positions.')

flags.DEFINE_string('xm_parameters', None,
                    'String specifying hyperparamter search.')


def train_step(optimizer,
               inputs,
               programs,
               learning_rate_fn,
               config,
               train_rng=None):
  """Train on batch of program tasks."""
  # We handle PRNG splitting inside the top pmap, rather
  # than handling it outside in the training loop - doing the
  # latter can add some stalls to the devices.
  train_rng, new_train_rng = jax.random.split(train_rng)

  weights = jnp.where(programs > 0, 1, 0).astype(jnp.float32)

  def loss_fn(params):
    """Loss function used for training."""
    logits = models.BaseTransformer(config).apply(
        {'params': params},
        inputs,
        programs,
        rngs={'dropout': train_rng})
    loss, weight_sum = train_lib.compute_weighted_cross_entropy(logits, programs, weights)
    mean_loss = loss / weight_sum
    return mean_loss, logits

  step = optimizer.state.step
  lr = learning_rate_fn(step)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grad = grad_fn(optimizer.target)
  grad = jax.lax.pmean(grad, 'batch')
  new_optimizer = optimizer.apply_gradient(grad, learning_rate=lr)

  # Get metrics.
  metrics = train_lib.compute_metrics(logits, programs, weights)
  metrics['learning_rate'] = lr
  return new_optimizer, metrics, new_train_rng


def eval_step(params, inputs, programs, config):
  weights = jnp.where(programs > 0, 1, 0).astype(jnp.float32)
  logits = models.BaseTransformer(config).apply(
      {'params': params}, inputs, outputs, programs)

  return train_lib.compute_metrics(logits, programs, weights)


def initialize_cache(inputs, programs, max_decode_len, config):
  """Initialize a cache for a given input shape and max decode length."""
  target_shape = (programs.shape[0], max_decode_len)
  initial_variables = models.BaseTransformer(config).init(
      jax.random.PRNGKey(0),
      jnp.ones(inputs.shape, config.dtype),
      jnp.ones(target_shape, config.dtype))
  return initial_variables['cache']


def predict_step(params,
                 inputs,
                 cache,
                 eos_token,
                 max_decode_len,
                 beam_size,
                 config):
  """Predict translation with fast decoding beam search on a batch."""
  # Prepare transformer fast-decoder call for beam search: for beam search, we
  # need to set up our decoder model to handle a batch size equal to
  # batch_size * beam_size, where each batch item's data is expanded in-place
  # rather than tiled.
  flat_encoded = decode.flat_batch_beam_expand(
      models.BaseTransformer(config).apply(
          {'params': params},
          inputs,
          method=models.BaseTransformer.encode),
      beam_size)

  encoded_padding_mask = jnp.where(outputs > 0, 1, 0).astype(jnp.float32)
  flat_encoded_padding_mask = decode.flat_batch_beam_expand(
      encoded_padding_mask, beam_size)

  def tokens_ids_to_logits(flat_ids, flat_cache):
    """Token slice to logits from decoder model."""
    # --> [batch * beam, 1, vocab]
    flat_logits, new_vars = models.BaseTransformer(config).apply(
        {'params': params, 'cache': flat_cache},
        flat_ids,
        flat_encoded,
        flat_encoded_padding_mask,
        mutable=['cache'],
        method=models.BaseTransformer.decode)
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
      bos_token=config.bos_token,
      eos_token=eos_token,
      max_decode_len=max_decode_len)

  # Beam search returns [n_batch, n_beam, n_length] with beam dimension
  # sorted in increasing order of log-probability.
  return beam_seqs


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

  hparam_str = ','.join(['%s=%s' % (k, str(hparam_str_dict[k])) for k in
                         sorted(hparam_str_dict.keys())])

  # Number of local devices for this host.
  n_devices = jax.local_device_count()

  if jax.host_id() == 0:
    summary_writer = tensorboard.SummaryWriter(
        os.path.join(FLAGS.save_dir, 'tb', hparam_str))

  batch_size = FLAGS.per_device_batch_size * n_devices
  input_shape = (FLAGS.per_device_batch_size, FLAGS.max_characters)
  program_shape = (FLAGS.per_device_batch_size, FLAGS.max_program_length)

  # Setup DSL
  # ---------------------------------------------------------------------------

  # Build token tables.
  id_input_table, input_id_table = tokens.build_input_token_tables()
  id_program_table, program_id_table = tokens.build_program_token_tables()
  input_vocab_size = len(input_id_table) + 1  # For padding.
  program_vocab_size = len(program_id_table) + 1

  bos_token = program_id_table[tokens.BOS]
  eos_token = program_id_table[tokens.EOS]

  def decode(tokens, id_token_table):
    """Decode io examples tokens."""
    return ' '.join([id_token_table[token] for token in tokens if token > 0])

  # Load Dataset
  # ---------------------------------------------------------------------------
  logging.info('Initializing dataset.')
  if not FLAGS.dataset_filepattern:
    raise ValueError('Must specify filepattern to dataset.')

  # Training dataset.
  dataset = input_pipeline.create_dataset_from_tf_record(
      FLAGS.dataset_filepattern, input_id_table, program_id_table)
  dataset = dataset.padded_batch(
      batch_size,
      padded_shapes=(input_shape[1:], program_shape[1:]),
      drop_remainder=True)
  # Split evaluation and training.
  eval_ds = dataset.take(FLAGS.num_eval_steps)
  # Decrease batch of predict dataset to handle beam search.
  predict_ds = eval_ds.unbatch().padded_batch(
      int(np.ceil(batch_size / 10)),
      padded_shapes=(input_shape[1:], program_shape[1:]))
  train_ds = dataset.skip(FLAGS.num_eval_steps).repeat()
  train_iter = train_ds.as_numpy_iterator()

  # Build Model and Optimizer
  # ---------------------------------------------------------------------------
  train_config = models.TransformerConfig(
      vocab_size=input_vocab_size,
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
      shift=False, deterministic=True, decode=True)

  rng = jax.random.PRNGKey(FLAGS.seed)
  rng = jax.random.fold_in(rng, jax.host_id())
  rng, init_rng = jax.random.split(rng)

  m = models.BaseTransformer(eval_config)
  initial_variables = jax.jit(m.init)(
      init_rng,
      jnp.ones(input_shape, jnp.float32),
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

  # Replicate optimizer.
  optimizer = jax_utils.replicate(optimizer)

  learning_rate_fn = train_lib.create_learning_rate_scheduler(
      base_learning_rate=FLAGS.lr)
  p_train_step = jax.pmap(
      functools.partial(
          train_step,
          learning_rate_fn=learning_rate_fn,
          config=train_config),
      axis_name='batch')
  p_eval_step = jax.pmap(
      functools.partial(eval_step, config=eval_config),
      axis_name='batch')
  p_init_cache = jax.pmap(
      functools.partial(
          initialize_cache,
          max_decode_len=FLAGS.max_program_length,
          config=predict_config),
      axis_name='batch')
  p_pred_step = jax.pmap(
      functools.partial(predict_step, config=predict_config),
      axis_name='batch',
      static_broadcasted_argnums=(3, 4, 5))

  # Main Train Loop
  # ---------------------------------------------------------------------------
  train_rngs = jax.random.split(rng, jax.local_device_count())
  del rng

  metrics_all = []
  tick = time.time()
  for step in range(start_step, FLAGS.num_train_steps):
    inputs, programs = common_utils.shard(next(train_iter))

    optimizer, metrics, train_rngs = p_train_step(
        optimizer, inputs, programs, train_rng=train_rngs)
    metrics_all.append(metrics)

    # Save a Checkpoint
    if ((step % FLAGS.checkpoint_freq == 0 and step > 0) or
        step == FLAGS.num_train_steps - 1):
      if jax.host_id() == 0:
        # Save unreplicated optimizer + model state.
        checkpoints.save_checkpoint(
            os.path.join(FLAGS.save_dir, 'checkpoints', hparam_str),
            jax_utils.unreplicate(optimizer),
            step)

    # Periodic metric handling.
    if not step or step % FLAGS.log_freq != 0:
      continue

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
    eval_metrics = []
    for batches in eval_ds.as_numpy_iterator():
      inputs, programs = common_utils.shard(batches)

      metrics = p_eval_step(optimizer.target, inputs, programs)
      eval_metrics.append(metrics)

    eval_metrics = common_utils.get_metrics(eval_metrics)
    eval_metrics_sums = jax.tree_map(jnp.sum, eval_metrics)
    eval_denominator = eval_metrics_sums.pop('denominator')
    eval_summary = jax.tree_map(
        lambda x: x / eval_denominator,  # pylint: disable=cell-var-from-loop
        eval_metrics_sums)

    if jax.host_id() == 0:
      logging.info('Evaluation time: %.4f s step %d, loss: %.4f.',
                   time.time()-t_evaluation_start, step, eval_summary['loss'])
      for key, val in eval_summary.items():
        summary_writer.scalar('eval/' + key, val, step)
      summary_writer.flush()

    # Beam search metrics.
    logging.info('Gathering beam search metrics.')
    for beam_size in [10, 100]:
      t_inference_start = time.time()
      pred_acc = 0
      pred_denominator = 0

      specs, targets, predictions = [], [], []
      for batches in predict_ds.as_numpy_iterator():
        pred_batch = batches
        # Handle final odd-sized batch by padding instead of dropping it.
        cur_pred_batch_size = pred_batch[0].shape[0]
        if cur_pred_batch_size % n_devices:
          padded_size = int(
              np.ceil(cur_pred_batch_size / n_devices) * n_devices)
          # pylint: disable=cell-var-from-loop
          pred_batch = jax.tree_map(
              lambda x: train_lib.pad_examples(x, padded_size), pred_batch)
        inputs, programs = common_utils.shard(pred_batch)

        cache = p_init_cache(inputs, programs)
        predicted = p_pred_step(optimizer.target,
                                inputs,
                                cache,
                                eos_token,
                                programs.shape[-1],
                                beam_size)
        predicted = train_lib.tohost(predicted)
        inputs, programs = map(train_lib.tohost, (inputs, programs))

        pred_denominator += programs.shape[0]
        for i, beams in enumerate(predicted):
          spec = decode(inputs[i], id_input_table)
          target = decode(programs[i], id_program_table)
          for beam in beams:
            predicted = decode(beam, id_program_table)
            if predicted == target:
              pred_acc += 1
              break
          specs.append(spec)
          targets.append(target)
          predictions.append(predicted)

      all_pred_acc, all_pred_denominator = train_lib.per_host_sum_pmap(
          jax.tree_map(np.array, (pred_acc, pred_denominator)))

      # Record beam search results as text summaries.
      message = []
      for n in np.random.choice(np.arange(len(predictions)), 8):
        text = (f'specs: {specs[n]}\n\ntarget: {targets[n]}\n\n'
                f'predicted: {predictions[n]}\n\n')
        message.append(text)

      # Write to tensorboard.
      if jax.host_id() == 0:
        logging.info('Prediction time (beam %d): %.4f s step %d, score %.4f.',
                     beam_size, time.time() - t_inference_start, step,
                     all_pred_acc / all_pred_denominator)
        summary_writer.scalar('predict/score-{}'.format(beam_size),
                              all_pred_acc / all_pred_denominator, step)
        summary_writer.text('samples-{}'.format(beam_size),
                            '\n------\n'.join(message), step)
        summary_writer.flush()


if __name__ == '__main__':
  app.run(main)
