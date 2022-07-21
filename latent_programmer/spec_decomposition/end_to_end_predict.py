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

"""Train seq-to-seq model on random supervised training tasks."""

# pytype: disable=wrong-arg-count
# pytype: disable=attribute-error

import collections
import functools
import os
import sys

from absl import app
from absl import flags
from absl import logging
from flax import optim
from flax.metrics import tensorboard
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow.compat.v2 as tf

from latent_programmer import decode
from latent_programmer import models as base_models
from latent_programmer.spec_decomposition import decomposition_models as models
from latent_programmer.tasks.robust_fill import dsl as robust_fill_dsl
from latent_programmer.tasks.robust_fill import tokens as dsl_tokens

sys.path.append('../../')
gfile = tf.io.gfile

FLAGS = flags.FLAGS

flags.DEFINE_integer('embedding_dim', 256, 'Embedding dimension.')
flags.DEFINE_integer('hidden_dim', 512, 'Hidden dimension.')
flags.DEFINE_integer('num_heads', 4, 'Number of layers.')
flags.DEFINE_integer('num_layers', 3, 'Number of Transformer heads.')
flags.DEFINE_boolean('slow_decode', True, 'Use slow decoding for prediction?')
flags.DEFINE_float('dropout_rate', 0.1, 'Dropout rate')
flags.DEFINE_float('attention_dropout_rate', 0.1, 'Attention dropout rate')

flags.DEFINE_string('save_dir', None, 'Directory to save results to.')
flags.DEFINE_string('test_dataset_filepattern', None,
                    'Filepattern for TFRecord test dataset.')
flags.DEFINE_integer('num_strings_per_task', 4,
                     'Number of input/output strings per task.')
flags.DEFINE_integer('max_characters', 120,
                     'Maximum number of characters in input/output strings.')
flags.DEFINE_integer('max_program_length', 100,
                     'Maximum number of tokens in program.')
flags.DEFINE_integer('max_spec_part_length', 200,
                     'Maximum number of characters in spec part.')
flags.DEFINE_bool('use_relative_attention', True,
                  'Whether to use relative positonal embeddings.')

flags.DEFINE_string('spec_decomposer_path', None,
                    'Directory with saved weights for SpecDecomposer.')
flags.DEFINE_string('synthesizer_path', None,
                    'Directory with saved weights for Synthesizer.')
flags.DEFINE_integer(
    'spec_decomposer_num_position_buckets', 32,
    'Number of relative attention position buckets in SpecDecomposer.')
flags.DEFINE_integer(
    'synthesizer_num_position_buckets', 16,
    'Number of relative attention position buckets in Synthesizer.')
flags.DEFINE_integer(
    'spec_decomposer_max_distance', 128,
    'Max distance for relative attention positions in SpecDecomposer.')
flags.DEFINE_integer(
    'synthesizer_max_distance', 20,
    'Max distance for relative attention positions in Synthesizer.')
flags.DEFINE_integer(
    'spec_decomposer_max_program_cross_embed_distance', 800,
    'Max distance for relative attention positions in SpecDecomposer.')
flags.DEFINE_integer(
    'synthesizer_max_program_cross_embed_distance', 100,
    'Max distance for relative attention positions in Synthesizer.')
flags.DEFINE_bool(
    'spec_decomposer_encoded_self_attention', True,
    'Whether to apply self-attention to encoded I/O in SpecDecomposer.')
flags.DEFINE_bool(
    'synthesizer_encoded_self_attention', False,
    'Whether to apply self-attention to encoded I/O in Synthesizer.')

flags.DEFINE_enum('dataset_type', 'robust_fill',
                  ['robust_fill', 'robust_fill_base', 'scan'],
                  'The kind of dataset to use.')


_internal = False
if not _internal:
  flags.DEFINE_string('xm_parameters', None,
                      'String specifying hyperparamter search.')

# Test dataset input pipeline.
# -----------------------------------------------------------------------------


def create_robust_fill_dataset(file_pattern, spec_token_id_table,
                               num_strings_per_task):
  """Returns an instance of tf.data.Dataset."""
  filenames = tf.io.gfile.glob(file_pattern)
  raw_dataset = tf.data.TFRecordDataset(filenames)

  spec_vocab_table = tf.lookup.StaticVocabularyTable(
      tf.lookup.KeyValueTensorInitializer(
          # Add padding.
          [''] + list(spec_token_id_table.keys()),
          [0] + list(spec_token_id_table.values()),
          key_dtype=tf.string,
          value_dtype=tf.int64),
      len(spec_token_id_table) + 1)
  eos_id = spec_token_id_table[robust_fill_dsl.EOS]

  def _parse_fn(record):
    """Parses a record into a feature_dict."""
    empty_default = [''] * num_strings_per_task
    feature_values = tf.io.parse_single_example(
        serialized=record,
        features={
            'inputs':
                tf.io.FixedLenFeature([num_strings_per_task],
                                      tf.string,
                                      default_value=empty_default),
            'outputs':
                tf.io.FixedLenFeature([num_strings_per_task],
                                      tf.string,
                                      default_value=empty_default),
            'program_encoding':
                tf.io.FixedLenFeature([], tf.string, default_value=''),
        })

    # Map characters to tokens.
    inputs = tf.strings.unicode_split(feature_values['inputs'],
                                      'UTF-8').to_tensor()
    inputs = spec_vocab_table.lookup(inputs)

    outputs = tf.strings.unicode_split(feature_values['outputs'],
                                       'UTF-8').to_tensor()
    outputs = spec_vocab_table.lookup(outputs)

    program = tf.strings.split(
        tf.strings.split(feature_values['program_encoding'], sep='|'), sep=' ')
    program = program.merge_dims(0, -1)
    program = tf.strings.to_number(program, out_type=tf.int32)
    program = tf.concat([program, [eos_id]], axis=-1)

    # inputs: [num_strings, max_length_of_input]
    # outputs: [num_strings, max_length_of_output]
    return {
        'inputs': inputs,
        'outputs': outputs,
        'target': program,
    }

  dataset = raw_dataset.map(_parse_fn)
  return dataset


# Decode step functions.
# -----------------------------------------------------------------------------


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
  return beam_seqs[:, -1, :]


def main(_):
  tf.enable_v2_behavior()

  if not gfile.isdir(FLAGS.save_dir):
    gfile.makedirs(FLAGS.save_dir)

  if jax.host_id() == 0:
    summary_writer = tensorboard.SummaryWriter(
        os.path.join(FLAGS.save_dir, 'tb'))

  # TODO(jxihong): end-to-end loop is not batched right now.
  batch_size = 1
  io_shape = (batch_size, FLAGS.num_strings_per_task, FLAGS.max_characters)
  spec_target_shape = (batch_size, FLAGS.max_spec_target_length)
  program_shape = (batch_size, FLAGS.max_program_length)

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
    # TODO(jxihong): Scan is not handled yet.
    raise ValueError('Unhandled dataset_type: {}'.format(FLAGS.dataset_type))
  else:
    raise ValueError('Unhandled dataset_type: {}'.format(FLAGS.dataset_type))

  # Util functions for prediction
  # ---------------------------------------------------------------------------

  def decode_spec(target):
    """Convert from int tensor to a string."""
    if FLAGS.dataset_type == 'robust_fill':
      target = target[np.all([target != 0, target != bos_id, target != eos_id],
                             axis=0)].astype(np.int32)
      return ''.join([spec_id_token_table[t_id] for t_id in target if t_id > 0])
    else:
      raise ValueError('Unhandled dataset_type: {}'.format(FLAGS.dataset_type))

  def encode_spec(target, max_target_length, add_eos=True):
    if FLAGS.dataset_type == 'robust_fill':
      tokens = [spec_token_id_table[t] for t in target]
      if add_eos:
        tokens += [eos_id]
      return np.array(tokens + [0] * (max_target_length - len(tokens)))
    else:
      raise ValueError('Unhandled dataset_type: {}'.format(FLAGS.dataset_type))

  def split_spec(spec_parts, outputs, max_target_length):
    """Returns a tuple (valid, done, current_parts, remaining_parts)."""
    spec_parts_str = decode_spec(spec_parts).strip('|').split('|')
    outputs_str = [decode_spec(example_output) for example_output in outputs]

    current_parts_str = []
    remaining_parts_str = []
    for part, output in zip(spec_parts_str, outputs_str):
      if not output.startswith(part):
        return False, True, [], []
      current_parts_str.append(part)
      remaining_parts_str.append(output[len(part):])
    # Check termination condition
    remaining_is_empty = [not remaining for remaining in remaining_parts_str]
    done = any(remaining_is_empty)
    valid = all(remaining_is_empty) or not any(remaining_is_empty)
    # Build arrays
    current_parts = [
        encode_spec(
            spec_str, max_target_length=max_target_length, add_eos=False)
        for spec_str in current_parts_str
    ]
    remaining_parts = [
        encode_spec(
            spec_str, max_target_length=max_target_length, add_eos=True)
        for spec_str in remaining_parts_str
    ]
    return valid, done, np.array(current_parts), np.array(remaining_parts)

  def process_predicted_program(program, add_eos=True):
    """Decode program tokens."""
    program = program[:np.argmax(program == eos_id)].astype(np.int32)
    program = program[program != bos_id].tolist()
    return program + [eos_id] if add_eos else program

  def eval_predicted_program(program, inputs, outputs):
    """Evaluate predicted program."""
    inputs = [decode_spec(input) for input in inputs]
    outputs = [decode_spec(output) for output in outputs]
    try:
      program = robust_fill_dsl.decode_program(program, program_id_token_table)
      p_outs = [program(inp) for inp in inputs]
      score = np.sum([p_out == out for p_out, out in zip(p_outs, outputs)])
    except:  # pylint: disable=bare-except
      program = None
      score = -1
      p_outs = []
    return program, p_outs, score == len(inputs)

  # Load Dataset
  # ---------------------------------------------------------------------------
  logging.info('Initializing dataset.')
  if not FLAGS.dataset_filepattern:
    raise ValueError('Must specify filepattern to dataset.')

  # Training dataset.
  logging.info('Loading dataset from %s', FLAGS.test_dataset_filepattern)
  padded_shapes = {
      'inputs': io_shape[1:],
      'outputs': io_shape[1:],
      'target': program_shape[1:],
  }
  logging.info('padded_shapes: %s', padded_shapes)

  if FLAGS.dataset_type == 'robust_fill':
    create_dataset_fn = create_robust_fill_dataset
  elif FLAGS.dataset_type == 'scan':
    raise NotImplementedError()  # TODO(kshi): Implement.
    # create_dataset_fn = input_pipeline.create_scan_dataset_from_tf_record
  else:
    raise ValueError('Unhandled dataset_type: {}'.format(FLAGS.dataset_type))

  test_dataset = create_dataset_fn(FLAGS.test_dataset_filepattern,
                                   spec_token_id_table,
                                   FLAGS.num_strings_per_task)
  test_dataset = test_dataset.padded_batch(
      batch_size, padded_shapes=padded_shapes, drop_remainder=False)

  # TODO(jxihong): Implement fast decoding.
  assert FLAGS.slow_decode, 'Fast decoding is not implemented yet.'

  rng = jax.random.PRNGKey(0)
  rng, init_rng = jax.random.split(rng)

  # Load SpecDecomposer Model and Optimizer
  # ---------------------------------------------------------------------------

  # Custom hyper-parameters for SpecDecomposer
  num_position_buckets = FLAGS.spec_decomposer_num_position_buckets
  max_distance = FLAGS.spec_decomposer_max_distance
  max_program_cross_embed_distance = (
      FLAGS.spec_decomposer_max_program_cross_embed_distance)
  spec_decomposer_base_config = base_models.TransformerConfig(
      vocab_size=spec_vocab_size,
      output_vocab_size=spec_vocab_size,
      shift=False,
      emb_dim=FLAGS.embedding_dim,
      num_heads=FLAGS.num_heads,
      num_layers=FLAGS.num_layers,
      qkv_dim=FLAGS.embedding_dim,
      mlp_dim=FLAGS.hidden_dim,
      max_len=max(FLAGS.max_characters, FLAGS.max_spec_target_length),
      dropout_rate=FLAGS.dropout_rate,
      attention_dropout_rate=FLAGS.attention_dropout_rate,
      use_relative_attention=FLAGS.use_relative_attention,
      deterministic=True,
      decode=not FLAGS.slow_decode,
      bos_token=bos_id,
      num_input_relative_position_buckets=num_position_buckets,
      max_input_distance=max_distance,
      num_output_relative_position_buckets=num_position_buckets,
      max_output_distance=max_distance,
      num_input_cross_output_relative_position_buckets=num_position_buckets,
      max_input_cross_output_distance=max_distance,
      num_program_relative_position_buckets=num_position_buckets,
      max_program_distance=max_distance,
      num_program_cross_embed_relative_position_buckets=num_position_buckets,
      max_program_cross_embed_distance=max_program_cross_embed_distance,
      num_flat_encoding_relative_position_buckets=num_position_buckets,
      max_flat_encoding_distance=max_distance)
  spec_decomposer_predict_config = models.DecomposeAttentionTransformerConfig(
      base_config=spec_decomposer_base_config,
      flat_encoded_self_attention=FLAGS.spec_decomposer_encoded_self_attention,
      dataset_type=FLAGS.dataset_type)

  m = models.DecomposeAttentionTransformer(spec_decomposer_predict_config)
  initial_variables = jax.jit(m.init)(init_rng, jnp.ones(io_shape, jnp.float32),
                                      jnp.ones(io_shape, jnp.float32),
                                      jnp.ones(spec_target_shape, jnp.float32))

  optimizer_def = optim.Adam(
      1e-3, beta1=0.9, beta2=0.98, eps=1e-9, weight_decay=0.01)
  spec_decomposer_optimizer = optimizer_def.create(initial_variables['params'])

  del initial_variables  # Don't keep a copy of the initial model.

  spec_decomposer_optimizer = checkpoints.restore_checkpoint(
      FLAGS.spec_decomposer_path, spec_decomposer_optimizer)
  logging.info('Found spec decomposer checkpointed at step %d.',
               int(spec_decomposer_optimizer.state.step))

  # Load Synthesizer Model and Optimizer
  # ---------------------------------------------------------------------------

  # Custom hyper-parameters for Synthesizer
  num_position_buckets = FLAGS.synthesizer_num_position_buckets
  max_distance = FLAGS.synthesizer_max_distance
  max_program_cross_embed_distance = (
      FLAGS.synthesizer_max_program_cross_embed_distance)
  synthesizer_base_config = base_models.TransformerConfig(
      vocab_size=spec_vocab_size,
      output_vocab_size=program_vocab_size,
      shift=False,
      emb_dim=FLAGS.embedding_dim,
      num_heads=FLAGS.num_heads,
      num_layers=FLAGS.num_layers,
      qkv_dim=FLAGS.embedding_dim,
      mlp_dim=FLAGS.hidden_dim,
      max_len=max(FLAGS.max_characters, FLAGS.max_program_length),
      dropout_rate=FLAGS.dropout_rate,
      attention_dropout_rate=FLAGS.attention_dropout_rate,
      use_relative_attention=FLAGS.use_relative_attention,
      deterministic=True,
      decode=not FLAGS.slow_decode,
      bos_token=bos_id,
      num_input_relative_position_buckets=num_position_buckets,
      max_input_distance=max_distance,
      num_output_relative_position_buckets=num_position_buckets,
      max_output_distance=max_distance,
      num_input_cross_output_relative_position_buckets=num_position_buckets,
      max_input_cross_output_distance=max_distance,
      num_program_relative_position_buckets=num_position_buckets,
      max_program_distance=max_distance,
      num_program_cross_embed_relative_position_buckets=num_position_buckets,
      max_program_cross_embed_distance=max_program_cross_embed_distance,
      num_flat_encoding_relative_position_buckets=num_position_buckets,
      max_flat_encoding_distance=max_distance)
  synthesizer_predict_config = models.DecomposeAttentionTransformerConfig(
      base_config=synthesizer_base_config,
      flat_encoded_self_attention=FLAGS.synthesizer_encoded_self_attention,
      dataset_type=FLAGS.dataset_type)

  m = models.DecomposeAttentionTransformer(synthesizer_predict_config)
  initial_variables = jax.jit(m.init)(init_rng, jnp.ones(io_shape, jnp.float32),
                                      jnp.ones(io_shape, jnp.float32),
                                      jnp.ones(program_shape, jnp.float32))

  optimizer_def = optim.Adam(
      1e-3, beta1=0.9, beta2=0.98, eps=1e-9, weight_decay=0.01)
  synthesizer_optimizer = optimizer_def.create(initial_variables['params'])

  del initial_variables  # Don't keep a copy of the initial model.

  synthesizer_optimizer = checkpoints.restore_checkpoint(
      FLAGS.synthesizer_path, synthesizer_optimizer)
  logging.info('Found synthesizer checkpointed at step %d.',
               int(synthesizer_optimizer.state.step))

  # Main Prediction Loop
  # ---------------------------------------------------------------------------

  # TODO(jxihong): End-to-end loop needs to be batched.
  spec_decomposer_pred_step = functools.partial(
      predict_step,
      eos_token=eos_id,
      max_decode_len=FLAGS.max_spec_target_length,
      config=spec_decomposer_predict_config,
      slow_decode=FLAGS.slow_decode)
  synthesizer_pred_step = functools.partial(
      predict_step,
      eos_token=eos_id,
      max_decode_len=FLAGS.max_program_length,
      config=synthesizer_predict_config,
      slow_decode=FLAGS.slow_decode)

  num_success = 0
  total = 0
  results = collections.defaultdict(list)
  for batch in test_dataset.as_numpy_iterator():
    inputs, outputs = batch['inputs'], batch['outputs']
    results['inputs'].append(','.join(
        [decode_spec(input) for input in inputs[0]]))
    results['outputs'].append(','.join(
        [decode_spec(output) for output in outputs[0]]))

    predicted_program = []
    predicted_spec_parts = ''
    valid = True
    done = False
    remaining_outputs = np.copy(outputs)
    # End-to-end prediction loop.
    while valid and not done:
      predicted_spec_parts = spec_decomposer_pred_step(
          spec_decomposer_optimizer.target,
          inputs,
          remaining_outputs,
          cache=None,
          beam_size=FLAGS.beam_size)
      valid, done, current_outputs, remaining_outputs = split_spec(
          np.array(predicted_spec_parts)[0], remaining_outputs[0],
          max_target_length=FLAGS.max_characters)
      predicted_spec_parts += ','.join(
          [decode_spec(output) for output in current_outputs])
      predicted_spec_parts += ';'
      if not valid:
        break

      # Add back batch dimension.
      current_outputs, remaining_outputs = (current_outputs[None, Ellipsis],
                                            remaining_outputs[None, Ellipsis])
      predicted_program_parts = synthesizer_pred_step(
          synthesizer_optimizer.target,
          inputs,
          current_outputs,
          cache=None,
          beam_size=FLAGS.beam_size)
      predicted_program_parts = process_predicted_program(
          np.array(predicted_program_parts)[0], add_eos=False)
      predicted_program.extend(predicted_program_parts)

    program, predicted_outputs, success = eval_predicted_program(
        predicted_program + [eos_id], inputs[0], outputs[0])
    results['predictions'].append(program)
    results['predicted_spec_parts'].append(predicted_spec_parts)
    results['predicted_outputs'].append(','.join(predicted_outputs))
    ground_truth = process_predicted_program(batch['target'][0], add_eos=True)
    results['ground_truths'].append(ground_truth)
    if success:
      num_success += 1
    total += 1

  # Record beam search results as text summaries.
  message = []
  for n in np.random.choice(np.arange(total), 20):
    text = (f"inputs: {results['inputs'][n]}\n\n"
            f"outputs: {results['outputs'][n]}\n\n"
            f"ground_truth: {results['ground_truths'][n]}\n\n"
            f"predicted: {results['predictions'][n]}\n\n"
            f"predicted_spec_parts: {results['predicted_spec_parts'][n]}\n\n"
            f"predicted_outputs: {results['predicted_outputs'][n]}\n\n")
    message.append(text)

  # Write to tensorboard.
  if jax.host_id() == 0:
    accuracy = 100 * num_success / total
    summary_writer.scalar('accuracy', accuracy, 0)
    summary_writer.text('predictions', '\n------\n'.join(message), 0)
    summary_writer.flush()

if __name__ == '__main__':
  app.run(main)
