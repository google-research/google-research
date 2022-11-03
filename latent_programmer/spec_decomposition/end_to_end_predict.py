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

import functools
import itertools
import os
import statistics
import sys
import timeit

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

from latent_programmer import models as base_models
from latent_programmer.spec_decomposition import decode
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
flags.DEFINE_integer('beam_size', 10, 'Beam size')

flags.DEFINE_string('save_dir', None, 'Directory to save results to.')
flags.DEFINE_string('test_dataset', None,
                    'Filepattern for TFRecord test dataset.')
flags.DEFINE_integer('num_test_batches', 200, 'Number of test batches.')
flags.DEFINE_integer('num_examples', 4,
                     'Number of input/output strings per task.')
flags.DEFINE_integer('max_io_length', 120,
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

flags.DEFINE_enum(
    'prediction_type', 'separate',
    ['separate', 'joint'],
    'Whether to use separate models (SpecDecomposerModel and then '
    'SynthesizerModel) or one joint prediction model.')
flags.DEFINE_integer(
    'num_examples_to_log', 10,
    'Number of examples to log and save to TensorBoard text.')
flags.DEFINE_integer(
    'num_beam_elements_to_log', 4,
    'Number of beam elements to log and save to TensorBoard text.')
flags.DEFINE_bool(
    'detect_invalid', True,
    'Whether to detect invalid beam elements and mark them as finished.')
flags.DEFINE_bool(
    'change_invalid_scores', True,
    'Whether to change scores of invalid beam elements to NEG_INF.')
flags.DEFINE_bool(
    'use_execution', True,
    'Whether to guide beam search with program execution results.')
# Note: if detect_invalid=False, a prediction could be wrong but we must do our
# best to continue anyway. Specifically:
#   * If the current output is "abcde" and the next step prediction is "xy"
#     which isn't a prefix, future steps use the remaining output "cde",
#     obtained by remaining_output=current_output[len(next_step_output):].
#   * Similarly, when using execution to compute the remaining output but the
#     program's output isn't a prefix, we do the same as above.
#   * If the SpecDecomposerModel produces something malformed (wrong number of
#     strings compared to the number of examples), we'll trim or pad with empty
#     strings (which are actually valid outputs of certain partial programs).
#   * Similarly, when using execution to compute the remaining output but the
#     predicted program doesn't run, we pretend that the predicted program
#     produced an empty string.


_internal = False
if not _internal:
  flags.DEFINE_string('xm_parameters', None,
                      'String specifying hyperparamter search.')

# Test dataset input pipeline.
# -----------------------------------------------------------------------------


def create_robust_fill_dataset(file_pattern, spec_token_id_table,
                               num_examples):
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
    empty_default = [''] * num_examples
    feature_values = tf.io.parse_single_example(
        serialized=record,
        features={
            'inputs':
                tf.io.FixedLenFeature([num_examples],
                                      tf.string,
                                      default_value=empty_default),
            'outputs':
                tf.io.FixedLenFeature([num_examples],
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


def end_to_end_beam_init(batch_size,
                         beam_size,
                         max_decode_len,
                         encoded,  # Contains beam dimension.
                         encoded_padding_mask,  # Contains beam dimension.
                         cache,
                         aux,
                         bos_token=0):
  """Initializes the beam search state data structure."""
  cur_index0 = jnp.array(0)
  # If the beam isn't entirely finished (which is checked before we call beam
  # search anyway), make sure the last live score (>= NEG_INF / 2) is better
  # than the worst finished score (NEG_INF) for the beam search loop condition.
  live_logprobs0 = jnp.where(aux['finished'], decode.NEG_INF / 2, aux['scores'])
  finished_scores0 = jnp.where(aux['finished'], aux['scores'], decode.NEG_INF)
  live_seqs0 = jnp.concatenate(
      [jnp.full((batch_size, beam_size, 1), bos_token, jnp.int32),
       jnp.zeros((batch_size, beam_size, max_decode_len - 1), jnp.int32)],
      axis=-1)
  finished_seqs0 = jnp.concatenate(
      [jnp.full((batch_size, beam_size, 1), bos_token, jnp.int32),
       jnp.zeros((batch_size, beam_size, max_decode_len - 1), jnp.int32)],
      axis=-1)
  finished_flags0 = aux['finished']
  finished_aux = aux
  # add beam dimension to attention cache pytree elements
  beam_cache0 = jax.tree_map(lambda x: decode.add_beam_dim(x, beam_size), cache)
  return decode.BeamState(
      cur_index=cur_index0,
      cur_encoded=encoded,
      cur_encoded_padding_mask=encoded_padding_mask,
      live_logprobs=live_logprobs0,
      finished_scores=finished_scores0,
      live_seqs=live_seqs0,
      finished_seqs=finished_seqs0,
      finished_flags=finished_flags0,
      cache=beam_cache0,
      live_aux=aux,
      finished_aux=finished_aux)


def end_to_end_predict_step(params,
                            inputs,  # Contains beam dimension.
                            outputs,  # Contains beam dimension.
                            cache,
                            aux,
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
  batch_size = inputs.shape[0]
  encoded = decode.unflatten_beam_dim(
      models.DecomposeAttentionTransformer(config).apply(
          {'params': params},
          decode.flatten_beam_dim(inputs),
          decode.flatten_beam_dim(outputs),
          method=models.DecomposeAttentionTransformer.encode),
      batch_size,
      beam_size)
  encoded_padding_mask = jnp.where(
      outputs > 0, 1, 0).astype(jnp.float32)

  beam_init_state = end_to_end_beam_init(
      batch_size, beam_size, max_decode_len, encoded, encoded_padding_mask,
      cache, aux, bos_token=config.base_config.bos_token)

  if slow_decode:
    def tokens_ids_to_logits(flat_ids, flat_encoded, flat_encoded_padding_mask):
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
    def tokens_ids_to_logits(flat_ids, flat_encoded, flat_encoded_padding_mask,
                             flat_cache):
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
  return decode.beam_search(
      inputs,
      encoded,
      encoded_padding_mask,
      cache,
      tokens_ids_to_logits,
      beam_size=beam_size,
      alpha=0.0,  # If we use a brevity penalty, we'll adjust the running score
                  # multiple times, once for each model call, which isn't good.
      bos_token=config.base_config.bos_token,
      eos_token=eos_token,
      max_decode_len=max_decode_len,
      slow_decode=slow_decode,
      beam_search_init_state=beam_init_state)


def load_spec_decomposer_model(init_rng, spec_vocab_size, io_shape,
                               spec_target_shape, bos_id, eos_id):
  """Loads SpecDecomposerModel."""
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
      max_len=max(FLAGS.max_io_length, FLAGS.max_spec_part_length),
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
  spec_decomposer_optimizer = checkpoints.restore_checkpoint(
      FLAGS.spec_decomposer_path, spec_decomposer_optimizer)
  logging.info('Found spec decomposer checkpointed at step %d.',
               int(spec_decomposer_optimizer.state.step))

  spec_decomposer_pred_step = jax.jit(
      functools.partial(end_to_end_predict_step,
                        eos_token=eos_id,
                        max_decode_len=FLAGS.max_spec_part_length,
                        config=spec_decomposer_predict_config,
                        slow_decode=FLAGS.slow_decode),
      static_argnums=(5,),  # The `beam_size` argument.
  )

  return spec_decomposer_optimizer, spec_decomposer_pred_step


def load_synthesizer_model(init_rng, spec_vocab_size, program_vocab_size,
                           io_shape, program_shape, bos_id, eos_id):
  """Loads synthesizer or joint model."""
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
      max_len=max(FLAGS.max_io_length, FLAGS.max_program_length),
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
  synthesizer_optimizer = checkpoints.restore_checkpoint(
      FLAGS.synthesizer_path, synthesizer_optimizer)
  logging.info('Found synthesizer checkpointed at step %d.',
               int(synthesizer_optimizer.state.step))

  synthesizer_pred_step = jax.jit(
      functools.partial(end_to_end_predict_step,
                        eos_token=eos_id,
                        max_decode_len=FLAGS.max_program_length,
                        config=synthesizer_predict_config,
                        slow_decode=FLAGS.slow_decode),
      static_argnums=(5,),  # The `beam_size` argument.
  )

  return synthesizer_optimizer, synthesizer_pred_step


def main(_):
  tf.enable_v2_behavior()

  if FLAGS.change_invalid_scores and not FLAGS.detect_invalid:
    raise ValueError(
        'change_invalid_scores=True is incompatible with detect_invalid=False. '
        'We cannot change scores of invalid beam elements without detecting '
        'the invalid elements first.')
  if FLAGS.prediction_type == 'joint' and not FLAGS.use_execution:
    raise ValueError(
        'Joint prediction requires using execution to compute the remaining '
        'output.')

  if not gfile.isdir(FLAGS.save_dir):
    gfile.makedirs(FLAGS.save_dir)

  xm_client = xmanager_api.XManagerApi(xm_deployment_env='alphabet')
  work_unit = xm_client.get_current_work_unit()
  hparam_dict = work_unit.parameters['args']
  hparam_str = ','.join([f'{k}={v}' for k, v in hparam_dict.items()])

  if jax.host_id() == 0:
    summary_writer = tensorboard.SummaryWriter(
        os.path.join(FLAGS.save_dir, 'tb', hparam_str))

  # TODO(jxihong): end-to-end loop is not batched right now.
  batch_size = 1
  io_shape = (batch_size, FLAGS.num_examples, FLAGS.max_io_length)
  spec_target_shape = (batch_size, FLAGS.max_spec_part_length)
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
      target = np.array(target)  # JAX arrays will fail dict lookups.
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

  def split_spec(spec_parts, outputs, max_target_length, aux, i):
    """Returns a tuple (valid, last_step, current_parts, remaining_parts)."""
    num_examples = len(outputs)
    assert num_examples == FLAGS.num_examples

    if aux['finished'][0][i]:
      # Do nothing if it's finished.
      spec_parts_str = ['[finished]'] * num_examples
      current_parts = [
          encode_spec(spec_str, max_target_length=max_target_length,
                      add_eos=False)
          for spec_str in spec_parts_str]
      return (aux['valid'][0][i], aux['last_step'][0][i],
              np.array(current_parts), aux['remaining_outputs'][0][i])

    # If we pass in an already-decoded list of strings, use them directly.
    if isinstance(spec_parts, list) and isinstance(spec_parts[0], str):
      spec_parts_str = spec_parts
    else:
      # Decode the SpecDecomposerModel prediction and separate examples.
      spec_parts_str = decode_spec(spec_parts).strip('|').split('|')
    if isinstance(outputs, list) and isinstance(outputs[0], str):
      decoded_outputs = outputs
    else:
      decoded_outputs = [decode_spec(o) for o in outputs]

    valid = True

    if FLAGS.detect_invalid:
      # The prediction is invalid if it has an incorrect number of | characters
      # or all of the parts are empty. (Some but not all parts can be empty.)
      if len(spec_parts_str) != num_examples or all(
          [not part for part in spec_parts_str]):
        spec_parts_str = ['[invalid]'] * num_examples
        valid = False
    else:
      # Still need to handle an incorrect number of | characters. Do our best to
      # continue even if the prediction is malformed.
      spec_parts_str = spec_parts_str[:num_examples]
      spec_parts_str += [''] * (num_examples - len(spec_parts_str))
      assert len(spec_parts_str) == num_examples

    current_parts = [
        encode_spec(
            spec_str, max_target_length=max_target_length, add_eos=False)
        for spec_str in spec_parts_str
    ]

    remaining_parts_str = []
    for part, output in zip(spec_parts_str, decoded_outputs):
      if FLAGS.detect_invalid and not output.startswith(part):
        remaining_parts_str = ['[invalid]'] * num_examples
        valid = False
        break
      remaining_parts_str.append(output[len(part):])
    remaining_parts = [
        encode_spec(
            spec_str, max_target_length=max_target_length, add_eos=True)
        for spec_str in remaining_parts_str
    ]

    last_step = all([not remaining for remaining in remaining_parts_str])
    return valid, last_step, np.array(current_parts), np.array(remaining_parts)

  def process_predicted_program(program, add_eos=True):
    """Decode program tokens."""
    # Can have multiple EOS tokens, so need to take the latest appearence.
    max_len = program.shape[0]
    score = 1e4 * (program == eos_id) + np.arange(max_len)
    program = program[:np.argmax(score)].astype(np.int32)
    program = program[np.all([program != 0, program != bos_id,
                              program != eos_id], axis=0)].tolist()
    if add_eos:
      program += [eos_id]
    return program

  def process_and_decode_program(program_tokens):
    """Returns a pair (valid, program)."""
    try:
      program = robust_fill_dsl.decode_program(
          process_predicted_program(program_tokens, add_eos=True),
          program_id_token_table)
      # If the program can't be converted to string, it's invalid.
      str(program)
      return True, program
    except:  # pylint: disable=bare-except
      # It's not valid, but maybe we have to ignore that fact.
      valid = not FLAGS.detect_invalid
      return valid, '[invalid program]'

  def run_program(program, inputs):
    """Returns a pair (valid, outputs)."""
    # If the program cannot be run, we treat it as outputting an empty string.
    outputs = []
    valid = True
    for i in inputs:
      if program == '[invalid program]':
        outputs.append('')
        if FLAGS.detect_invalid:
          valid = False
      else:
        try:
          outputs.append(program(i))
        except:  # pylint: disable=bare-except
          outputs.append('')
          if FLAGS.use_execution and FLAGS.detect_invalid:
            valid = False
    return valid, outputs

  # Load Dataset
  # ---------------------------------------------------------------------------
  logging.info('Initializing dataset.')
  if not FLAGS.test_dataset:
    raise ValueError('Must specify filepattern to dataset.')

  # Training dataset.
  logging.info('Loading dataset from %s', FLAGS.test_dataset)
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

  test_dataset = create_dataset_fn(FLAGS.test_dataset,
                                   spec_token_id_table,
                                   FLAGS.num_examples)
  test_dataset = test_dataset.padded_batch(
      batch_size, padded_shapes=padded_shapes, drop_remainder=False)
  test_dataset = test_dataset.take(FLAGS.num_test_batches)

  # TODO(jxihong): Implement fast decoding.
  assert FLAGS.slow_decode, 'Fast decoding is not implemented yet.'

  rng = jax.random.PRNGKey(0)
  rng, init_rng = jax.random.split(rng)

  # Main Prediction Loop
  # ---------------------------------------------------------------------------

  if FLAGS.prediction_type == 'separate':
    spec_decomposer_optimizer, spec_decomposer_pred_step = (
        load_spec_decomposer_model(
            init_rng=init_rng,
            spec_vocab_size=spec_vocab_size,
            io_shape=io_shape,
            spec_target_shape=spec_target_shape,
            bos_id=bos_id,
            eos_id=eos_id))
  synthesizer_optimizer, synthesizer_pred_step = load_synthesizer_model(
      init_rng=init_rng,
      spec_vocab_size=spec_vocab_size,
      program_vocab_size=program_vocab_size,
      io_shape=io_shape,
      program_shape=program_shape,
      bos_id=bos_id,
      eos_id=eos_id)

  # ----------------------------------------------------------------------------
  # A discussion of metrics we collect.
  #
  # * Every test problem is either a success or failure at the end.
  # * Every test problem either encounters no errors, or the first error is
  #   caused by the SpecDecomposerModel or caused by the SynthesizerModel.
  # * Thus, every test problem contributes to one cell in the below table, such
  #   that A+B+C+D+E+F = (total number of test problems).
  #
  #                 |           | First error from    | First error from
  #                 | No errors | SpecDecomposerModel | SynthesizerModel
  # --------------------------------------------------------------------
  # Overall success |     A     |          B          |        C
  # Overall failure |   D = 0   |          E          |        F
  #
  # In the code we name these scenarios A-F.
  #
  # Metrics we send to TensorBoard:
  # * All of A-F (except D=0) individually
  # * Total success rate: (A + B + C) / (A + B + C + D + E + F)
  # * Fraction of failures caused by SpecDecomposerModel: E / (E + F)
  # * Fraction of failures caused by SynthesizerModel: F / (E + F)
  # * Recovery rate when SpecDecomposerModel errors: B / (B + E)
  # * Recovery rate when SynthesizerModel errors: C / (C + F)
  # * Overall recovery rate when an error occurs: (B + C) / (B + C + E + F)
  # * Fraction of recovered errors among successes: (B + C) / (A + B + C)
  # * Additionally, other things unrelated to these numbers, like timing info.
  # ----------------------------------------------------------------------------
  metric_a = 0
  metric_b = 0
  metric_c = 0
  metric_e = 0
  metric_f = 0

  beam_size = FLAGS.beam_size
  successes = []
  total_times = []
  spec_prediction_times = []
  spec_processing_times = []
  spec_analysis_times = []
  synthesis_prediction_times = []
  synthesis_processing_times = []
  synthesis_analysis_times = []
  num_steps = []
  num_ground_truth_steps = []

  for test_example_index, batch in enumerate(test_dataset.as_numpy_iterator()):
    if test_example_index % 10 == 0:
      logging.info('Processing test example #%s', test_example_index)
    do_logging = test_example_index < FLAGS.num_examples_to_log
    test_example_start_time = timeit.default_timer()

    inputs, outputs = batch['inputs'], batch['outputs']
    decoded_inputs = [decode_spec(i) for i in inputs[0]]
    decoded_outputs = [decode_spec(o) for o in outputs[0]]
    _, ground_truth = process_and_decode_program(batch['target'][0])
    ground_truth_length = len(ground_truth.expressions)

    if do_logging:
      inputs_str = '\n  '.join(decoded_inputs)
      outputs_str = '\n  '.join(decoded_outputs)
      log_message = (
          f'```\n'
          f'Problem #{test_example_index}\n'
          f'inputs:\n'
          f'  {inputs_str}\n'
          f'outputs:\n'
          f'  {outputs_str}\n'
          f'\n'
      )

    next_history_id = itertools.count().__next__
    beam_inputs = decode.add_beam_dim(inputs, beam_size)
    step_index = 0
    first_error = None

    # `valid` means whether we are in an unrecoverable state (or the chance of
    # recovering is so low that we'd rather allocate beam space to other
    # things). We manually change the scores of invalid beam states to NEG_INF
    # to make them drop out of the beam. The `use_execution` flag determines
    # whether we are allowed to use predicted programs' executions to inform
    # validity.

    # `last_step` is set during the SpecDecomposer step to signal that the
    # remaining_output is empty so there should be no further iterations. After
    # the Synthesizer step, this beam element should be marked as `finished`. If
    # `use_execution` is True, we use the predicted program's execution to see
    # if the beam element is truly finished or not.

    # `finished` means whether a beam element is completely done, i.e., its
    # score is final and there should be no more predictions by either model.
    # Invalid beam elements should also be marked as finished.
    aux = {
        'remaining_outputs': decode.add_beam_dim(outputs, beam_size),
        'programs': jnp.full((1, beam_size, 1), bos_id, jnp.int32),
        'scores': jnp.array([0.0] + [decode.NEG_INF] * (beam_size - 1))[None,
                                                                        Ellipsis],
        'valid': jnp.array([True] + [False] * (beam_size - 1))[None, Ellipsis],
        'last_step': jnp.array([False] + [True] * (beam_size - 1))[None, Ellipsis],
        'finished': jnp.array([False] + [True] * (beam_size - 1))[None, Ellipsis],
        'history': jnp.array([[[next_history_id()] for _ in range(beam_size)]]),
    }

    # End-to-end prediction loop.
    while jnp.any(~aux['finished']) and step_index < 20:
      ground_truth_program_part = (
          ground_truth.expressions[step_index]
          if step_index < ground_truth_length
          else '[step index out of bounds]')
      ground_truth_output_parts = (
          [ground_truth_program_part(i) for i in decoded_inputs]
          if step_index < ground_truth_length
          else '[step index out of bounds]')

      # Use the spec decomposition step when predicting with separate models,
      # but not when doing joint prediction.
      if FLAGS.prediction_type == 'separate':
        # Spec Decomposition Step.
        ##########################

        # Run the SpecDecomposerModel.
        start_time = timeit.default_timer()
        predicted_spec_parts, scores, aux = spec_decomposer_pred_step(
            params=spec_decomposer_optimizer.target,
            inputs=beam_inputs,
            outputs=aux['remaining_outputs'],
            cache=None,
            aux=aux,
            beam_size=beam_size)
        spec_prediction_times.append(timeit.default_timer() - start_time)

        # Process spec predictions.
        start_time = timeit.default_timer()
        spec_parts_batch = np.array(predicted_spec_parts)
        results = [
            split_spec(beam, aux['remaining_outputs'][0][i],
                       max_target_length=FLAGS.max_io_length, aux=aux, i=i)
            for i, beam in enumerate(spec_parts_batch[0])]
        valids, last_steps, current_outputs, remaining_outputs = zip(*results)

        current_outputs = jnp.array(current_outputs)[None, Ellipsis]
        aux['remaining_outputs'] = jnp.array(remaining_outputs)[None, Ellipsis]
        aux['scores'] = scores
        aux['valid'] = jnp.array(valids)[None, Ellipsis]
        aux['last_step'] = jnp.array(last_steps)[None, Ellipsis]
        current_history_ids = jnp.array(
            [next_history_id() for _ in range(beam_size)])[None, Ellipsis, None]
        aux['history'] = jnp.concatenate([aux['history'], current_history_ids],
                                         axis=-1)

        # Process invalid states.
        if FLAGS.detect_invalid:
          if FLAGS.change_invalid_scores:
            aux['scores'] += decode.NEG_INF * (1 - aux['valid'])
          aux['finished'] |= ~aux['valid']
        else:
          assert np.all(aux['valid'])
        spec_processing_times.append(timeit.default_timer() - start_time)

        # Analysis and logging.
        start_time = timeit.default_timer()
        assert len(current_outputs) == 1
        best_spec_prediction = [decode_spec(o) for o in current_outputs[0][-1]]
        if aux['finished'][0][-1] & aux['valid'][0][-1]:
          matches = 'N/A'
        else:
          matches = best_spec_prediction == ground_truth_output_parts
          if not matches and first_error is None:
            first_error = f'SpecDecomposerModel at step #{step_index + 1}'
        spec_analysis_times.append(timeit.default_timer() - start_time)

        if do_logging:
          log_message += '\n' + ('=' * 80) + '\n'
          log_message += (
              f'Spec Decomposition Step #{step_index + 1}:\n'
              f'  ground truth output parts:           {ground_truth_output_parts}\n'
              f'  SpecDecomposerModel best prediction: {best_spec_prediction}\n'
              f'    matches: {matches}\n'
              f'---------- Full beam: ----------\n'
          )
          for i in range(beam_size)[::-1][:FLAGS.num_beam_elements_to_log]:
            prediction_i = [decode_spec(o) for o in current_outputs[0][i]]
            score_i = aux['scores'][0][i]
            remaining_i = [decode_spec(o)
                           for o in aux['remaining_outputs'][0][i]]
            _, program_i = process_and_decode_program(aux['programs'][0][i])
            valid_i = aux['valid'][0][i]
            last_step_i = aux['last_step'][0][i]
            finished_i = aux['finished'][0][i]
            log_message += (
                f'Beam item {i}:\n'
                f'  prediction: {prediction_i}\n'
                f'  score: {score_i:.4f}\n'
                f'  remaining_outputs: {remaining_i}\n'
                f'  program: {program_i}\n'
                f'  valid: {valid_i}, last_step: {last_step_i}, finished: {finished_i}\n'
                f'  history: {aux["history"][0][i]}\n'
            )

        # Elements can become newly finished if they are invalid.
        if jnp.all(aux['finished']):
          step_index += 1  # Count this half-step.
          break

      # Synthesizer Step.
      ###################
      # (Used in both separate and joint prediction)

      # Run the SynthesizerModel.
      start_time = timeit.default_timer()
      if FLAGS.prediction_type == 'separate':
        # Use next output part as predicted by the SpecDecomposerModel.
        synthesizer_step_outputs = current_outputs
      elif FLAGS.prediction_type == 'joint':
        # Use the entire remaining output from the previous step.
        synthesizer_step_outputs = aux['remaining_outputs']
      else:
        raise ValueError(f'Unhandled prediction_type: {FLAGS.prediction_type}')
      predicted_program_parts, scores, aux = synthesizer_pred_step(
          params=synthesizer_optimizer.target,
          inputs=beam_inputs,
          outputs=synthesizer_step_outputs,
          cache=None,
          aux=aux,
          beam_size=beam_size)
      synthesis_prediction_times.append(timeit.default_timer() - start_time)

      # Process program predictions.
      start_time = timeit.default_timer()
      program_parts = jnp.array([
          np.zeros_like(beam) if aux['finished'][0][i] else beam
          for i, beam in enumerate(np.array(predicted_program_parts)[0])
      ])[None, Ellipsis]
      aux['programs'] = jnp.concatenate([aux['programs'], program_parts],
                                        axis=-1)
      aux['scores'] = scores
      current_history_ids = jnp.array(
          [next_history_id() for _ in range(beam_size)])[None, Ellipsis, None]
      aux['history'] = jnp.concatenate([aux['history'], current_history_ids],
                                       axis=-1)

      # Process invalid states.
      new_valids, new_last_steps, new_remaining = [], [], []
      for i in range(beam_size):
        valid_i = bool(aux['valid'][0][i])
        last_step_i = aux['last_step'][0][i]
        remaining_i = aux['remaining_outputs'][0][i]
        # Don't need to do any checking if the beam element is already finished
        # or already invalid.
        if not aux['finished'][0][i] and valid_i:
          # This is just a syntax check.
          valid_i, program_i = process_and_decode_program(aux['programs'][0][i])
          if FLAGS.use_execution:
            # Check for syntax, runtime errors, and output prefix matching.
            valid_i, program_outputs_i = run_program(program_i, decoded_inputs)
            # Must use execution (even if invalid) for joint prediction.
            if valid_i or FLAGS.prediction_type == 'joint':
              valid_i, last_step_i, _, remaining_i = split_spec(
                  program_outputs_i, decoded_outputs,
                  max_target_length=FLAGS.max_io_length, aux=aux, i=i)
        if not FLAGS.detect_invalid:
          # Even though we updated valid_i above, those helper functions should
          # not claim something is invalid if detect_invalid=False.
          assert valid_i
        new_valids.append(valid_i)
        new_last_steps.append(last_step_i)
        new_remaining.append(remaining_i)
      aux['valid'] = jnp.array(new_valids)[None, Ellipsis]
      aux['last_step'] = jnp.array(new_last_steps)[None, Ellipsis]
      aux['remaining_outputs'] = jnp.array(new_remaining)[None, Ellipsis]

      already_finished = aux['finished'][0]
      aux['finished'] |= aux['last_step']
      if FLAGS.detect_invalid:
        if FLAGS.change_invalid_scores:
          aux['scores'] += decode.NEG_INF * (1 - aux['valid'])
        aux['finished'] |= ~aux['valid']
      else:
        assert np.all(aux['valid'])
      synthesis_processing_times.append(timeit.default_timer() - start_time)

      # Analysis and logging.
      start_time = timeit.default_timer()
      _, best_program_prediction = process_and_decode_program(
          program_parts[0][-1])
      _, program_outputs = run_program(best_program_prediction, decoded_inputs)
      if already_finished[-1] & aux['valid'][0][-1]:
        functionally_correct = 'N/A'
      else:
        # Compare to the ground truth, not the spec prediction. The best-scoring
        # program didn't necessarily come from the best-scoring spec prediction.
        functionally_correct = program_outputs == ground_truth_output_parts
        if not functionally_correct and first_error is None:
          first_error = f'SynthesizerModel at step #{step_index + 1}'
      synthesis_analysis_times.append(timeit.default_timer() - start_time)

      if do_logging:
        log_message += '\n' + ('=' * 80) + '\n'
        log_message += (
            f'Synthesizer Step #{step_index + 1}:\n'
            f'  ground truth program part:        {ground_truth_program_part}\n'
            f'  SynthesizerModel best prediction: {best_program_prediction}\n'
            f'    functionally correct: {functionally_correct}\n'
            f'    ground truth output parts: {ground_truth_output_parts}\n'
            f"    best prediction's outputs: {program_outputs}\n"
            f'---------- Full beam: ----------\n'
        )

        for i in range(beam_size)[::-1][:FLAGS.num_beam_elements_to_log]:
          _, prediction_i = process_and_decode_program(program_parts[0][i])
          score_i = aux['scores'][0][i]
          remaining_i = [decode_spec(o) for o in aux['remaining_outputs'][0][i]]
          _, program_i = process_and_decode_program(aux['programs'][0][i])
          valid_i = aux['valid'][0][i]
          last_step_i = aux['last_step'][0][i]
          finished_i = aux['finished'][0][i]
          log_message += (
              f'Beam item {i}:\n'
              f'  prediction: {prediction_i}\n'
              f'  score: {score_i:.4f}\n'
              f'  remaining_outputs: {remaining_i}\n'
              f'  program: {program_i}\n'
              f'  valid: {valid_i}, last_step: {last_step_i}, finished: {finished_i}\n'
              f'  history: {aux["history"][0][i]}\n'
          )

      step_index += 1
    # End of step-by-step loop.

    if do_logging:
      log_message += '\nFinal Evaluation:\n'
    success = False
    for i in reversed(range(beam_size)):
      # The beam is ordered from worst to best score. Check the best scoring
      # program first.
      _, program_i = process_and_decode_program(aux['programs'][0][i])
      _, program_outputs = run_program(program_i, decoded_inputs)
      success_i = program_outputs == decoded_outputs
      if do_logging:
        log_message += (
            f'Program {i}: {program_i}\n'
            f'  success: {success_i}, score: {aux["scores"][0][i]:.4f}\n'
        )
      if success_i:
        success = True
        if not do_logging:  # Log all programs when desired.
          break

    successes.append(success)
    num_steps.append(step_index)
    num_ground_truth_steps.append(ground_truth_length)
    total_times.append(timeit.default_timer() - test_example_start_time)

    if first_error is None:
      if success:
        metric_a += 1
      else:
        if beam_size > 1:
          # We only check for errors in the best-scoring beam element. It's
          # possible that, for every step, the current predicted program part of
          # the best-scoring beam element is correct, but the best-scoring beam
          # elements are changing and no single beam element was correct at
          # every step. In this case... let's just say it's the SynthesizerModel
          # error? These metrics don't mean much if beam_size > 1 anyway.
          metric_f += 1
        else:
          # This shouldn't happen if beam_size == 1.
          raise ValueError('Test problem was failure but first_error is None')
    elif first_error.startswith('SpecDecomposerModel'):
      if success:
        metric_b += 1
      else:
        metric_e += 1
    elif first_error.startswith('SynthesizerModel'):
      if success:
        metric_c += 1
      else:
        metric_f += 1
    else:
      raise ValueError(f'Unhandled first_error: {first_error}')

    if do_logging:
      log_message += (
          f'\nground_truth: {ground_truth}\n'
          f'num steps taken:        {step_index}\n'
          f'num ground truth steps: {ground_truth_length}\n\n'
          f'overall success: {success}\n'
          f'first error: {first_error}\n'
          f'total time: {total_times[-1]:.1f} sec\n'
          f'```'
      )
      print(log_message)

      summary_writer.text(f'predictions_{test_example_index}',
                          log_message, 0)
      summary_writer.flush()

  # Compute overall metrics and write to tensorboard.
  num_success = sum(successes)
  total = len(successes)
  num_failure = total - num_success
  assert num_success == metric_a + metric_b + metric_c
  assert num_failure == metric_e + metric_f
  assert (len(total_times) == len(num_steps) == len(num_ground_truth_steps)
          == total)

  if jax.host_id() == 0:
    summary_writer.scalar('raw/# success & no errors', metric_a, 0)
    summary_writer.scalar('raw/# success & SpecDecomposerModel error', metric_b,
                          0)
    summary_writer.scalar('raw/# success & SynthesizerModel error', metric_c, 0)
    summary_writer.scalar('raw/# failure & SpecDecomposerModel error', metric_e,
                          0)
    summary_writer.scalar('raw/# failure & SynthesizerModel error', metric_f, 0)

    summary_writer.scalar('main/total success rate',
                          100 * num_success / total, 0)
    summary_writer.scalar(
        'main/failures from SpecDecomposerModel, among all failures',
        100 * metric_e / num_failure, 0)
    summary_writer.scalar(
        'main/failures from SynthesizerModel, among all failures',
        100 * metric_f / num_failure, 0)

    if metric_b + metric_e > 0:
      summary_writer.scalar(
          'error_recovery/specDecomposerModel error recovery rate',
          100 * metric_b / (metric_b + metric_e), 0)
    summary_writer.scalar(
        'error_recovery/synthesizerModel error recovery rate',
        100 * metric_c / (metric_c + metric_f), 0)
    summary_writer.scalar(
        'error_recovery/error recovery rate',
        100 * (metric_b + metric_c) / (
            metric_b + metric_c + metric_e + metric_f), 0)
    summary_writer.scalar(
        'error_recovery/recovered errors among successes',
        100 * (metric_b + metric_c) / num_success, 0)

    summary_writer.scalar('steps/avg. steps taken',
                          statistics.mean(num_steps), 0)
    summary_writer.scalar('steps/avg. ground-truth steps',
                          statistics.mean(num_ground_truth_steps), 0)
    summary_writer.scalar(
        'steps/success and (taken > ground truth steps), among all successes',
        len([0 for taken, gt, success in zip(num_steps,
                                             num_ground_truth_steps, successes)
             if taken > gt and success]) / num_success * 100, 0)
    summary_writer.scalar(
        'steps/success and (taken < ground truth steps), among all successes',
        len([0 for taken, gt, success in zip(num_steps,
                                             num_ground_truth_steps, successes)
             if taken < gt and success]) / num_success * 100, 0)
    summary_writer.scalar(
        'steps/failure and (taken > ground truth steps), among all failures',
        len([0 for taken, gt, success in zip(num_steps,
                                             num_ground_truth_steps, successes)
             if taken > gt and not success]) / num_failure * 100, 0)
    summary_writer.scalar(
        'steps/failure and (taken < ground truth steps), among all failures',
        len([0 for taken, gt, success in zip(num_steps,
                                             num_ground_truth_steps, successes)
             if taken < gt and not success]) / num_failure * 100, 0)

    summary_writer.scalar('time/total time per problem',
                          statistics.mean(total_times), 0)
    if spec_prediction_times:
      summary_writer.scalar('time/per SpecDecomposerModel call',
                            statistics.mean(spec_prediction_times), 0)
      summary_writer.scalar('time/per spec processing',
                            statistics.mean(spec_processing_times), 0)
      summary_writer.scalar('time/per spec analysis',
                            statistics.mean(spec_analysis_times), 0)
    summary_writer.scalar('time/per SynthesizerModel call',
                          statistics.mean(synthesis_prediction_times), 0)
    summary_writer.scalar('time/per synthesis processing',
                          statistics.mean(synthesis_processing_times), 0)
    summary_writer.scalar('time/per synthesis analysis',
                          statistics.mean(synthesis_analysis_times), 0)

    summary_writer.flush()

if __name__ == '__main__':
  app.run(main)
