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

"""Train seq-to-seq model on random supervised training tasks."""

# pytype: disable=wrong-arg-count
# pytype: disable=attribute-error

import collections
import functools
import itertools
import json
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
import tensorflow as tf

from latent_programmer import models as base_models
from latent_programmer.spec_decomposition import decode
from latent_programmer.spec_decomposition import decomposition_models as models
from latent_programmer.spec_decomposition import input_pipeline
from latent_programmer.tasks.deepcoder import deepcoder_dsl
from latent_programmer.tasks.robust_fill import dsl as robust_fill_dsl
from latent_programmer.tasks.robust_fill import tokens as dsl_tokens

sys.path.append('../../')
gfile = tf.io.gfile

# Experiment setup.
_SAVE_DIR = flags.DEFINE_string(
    'save_dir', None, 'Directory to save results to.')

# Flags for dataset info.
_DATASET_TYPE = flags.DEFINE_enum(
    'dataset_type', 'robustfill',
    ['robustfill', 'robustfill_base', 'deepcoder'],
    'The kind of dataset to use.')
_EXPERIMENT = flags.DEFINE_string(
    'experiment', 'NONE',
    'Kind of experiment.')
_TEST_DATASET_FORMAT = flags.DEFINE_string(
    'test_dataset_format', None, 'Filepattern for TFRecord test dataset.')
_NUM_TEST_BATCHES = flags.DEFINE_integer(
    'num_test_batches', 200, 'Number of test batches.')
_NUM_EXAMPLES = flags.DEFINE_integer(
    'num_examples', 4, 'Number of input/output strings per task.')
_MAX_IO_LENGTH = flags.DEFINE_integer(
    'max_io_length', 120,
    'Maximum number of characters in input/output strings.')
_MAX_PROGRAM_LENGTH = flags.DEFINE_integer(
    'max_program_length', 100, 'Maximum number of tokens in program.')
_MAX_SPEC_PART_LENGTH = flags.DEFINE_integer(
    'max_spec_part_length', 200, 'Maximum number of characters in spec part.')

# Model training hyperparameters.
_SPEC_DECOMPOSER_PATH_FORMAT = flags.DEFINE_string(
    'spec_decomposer_path_format', None,
    'Directory with saved weights for SpecDecomposer.')
_SYNTHESIZER_PATH_FORMAT = flags.DEFINE_string(
    'synthesizer_path_format', None,
    'Directory with saved weights for Synthesizer.')
_SEED = flags.DEFINE_integer(
    'seed', 0,
    'Seed used for training.')
_EMBEDDING_DIM = flags.DEFINE_integer(
    'embedding_dim', 256, 'Embedding dimension.')
_HIDDEN_DIM = flags.DEFINE_integer(
    'hidden_dim', 512, 'Hidden dimension.')
_NUM_HEADS = flags.DEFINE_integer(
    'num_heads', 4, 'Number of layers.')
_NUM_LAYERS = flags.DEFINE_integer(
    'num_layers', 3, 'Number of Transformer heads.')
_DROPOUT_RATE = flags.DEFINE_float(
    'dropout_rate', 0, 'Dropout rate')
_ATTENTION_DROPOUT_RATE = flags.DEFINE_float(
    'attention_dropout_rate', 0, 'Attention dropout rate')
_SPEC_DECOMPOSER_NUM_POSITION_BUCKETS = flags.DEFINE_integer(
    'spec_decomposer_num_position_buckets', 32,
    'Number of relative attention position buckets in SpecDecomposer.')
_SYNTHESIZER_NUM_POSITION_BUCKETS = flags.DEFINE_integer(
    'synthesizer_num_position_buckets', 16,
    'Number of relative attention position buckets in Synthesizer.')
_SPEC_DECOMPOSER_MAX_DISTANCE = flags.DEFINE_integer(
    'spec_decomposer_max_distance', 128,
    'Max distance for relative attention positions in SpecDecomposer.')
_SYNTHESIZER_MAX_DISTANCE = flags.DEFINE_integer(
    'synthesizer_max_distance', 20,
    'Max distance for relative attention positions in Synthesizer.')
_SPEC_DECOMPOSER_MAX_PROGRAM_CROSS_EMBED_DISTANCE = flags.DEFINE_integer(
    'spec_decomposer_max_program_cross_embed_distance', 800,
    'Max distance for relative attention positions in SpecDecomposer.')
_SYNTHESIZER_MAX_PROGRAM_CROSS_EMBED_DISTANCE = flags.DEFINE_integer(
    'synthesizer_max_program_cross_embed_distance', 100,
    'Max distance for relative attention positions in Synthesizer.')
_USE_RELATIVE_ATTENTION = flags.DEFINE_bool(
    'use_relative_attention', True,
    'Whether to use relative positonal embeddings.')
_ALIGNED_RELATIVE_ATTENTION = flags.DEFINE_bool(
    'aligned_relative_attention', True,
    'Whether to align relative attention positions between targets and encoded '
    'I/O examples, only relevant for the SpecDecomposerModel.')
_CORRUPTION_RATE = flags.DEFINE_float(
    'corruption_rate', 0.0,
    'Next part corruption rate for the SynthesizerModel.')

# Flags for end-to-end prediction settings.
_BEAM_SIZE = flags.DEFINE_integer(
    'beam_size', 10, 'Beam size')
_PREDICTION_TYPE = flags.DEFINE_enum(
    'prediction_type', 'separate',
    ['separate', 'joint'],
    'Whether to use separate models (SpecDecomposerModel and then '
    'SynthesizerModel) or one joint prediction model.')
_SLOW_DECODE = flags.DEFINE_boolean(
    'slow_decode', True, 'Use slow decoding for prediction?')

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
_DETECT_INVALID = flags.DEFINE_bool(
    'detect_invalid', True,
    'Whether to detect invalid beam elements and mark them as finished.')
_USE_EXECUTION = flags.DEFINE_bool(
    'use_execution', True,
    'Whether to guide beam search with program execution results.')
_DISCARD_REPEAT_FUNCTIONALITY = flags.DEFINE_bool(
    'discard_repeat_functionality', True,
    'Whether to mark duplicate program functionality in a beam as invalid.')

# Logging settings.
_NUM_EXAMPLES_TO_LOG = flags.DEFINE_integer(
    'num_examples_to_log', 10,
    'Number of examples to log and save to TensorBoard text.')
_NUM_BEAM_ELEMENTS_TO_LOG = flags.DEFINE_integer(
    'num_beam_elements_to_log', 4,
    'Number of beam elements to log and save to TensorBoard text.')

_internal = False
if not _internal:
  _ = flags.DEFINE_string(
      'xm_parameters', None, 'String specifying hyperparameter search.')

# Test dataset input pipeline.
# -----------------------------------------------------------------------------


def create_deepcoder_dataset(
    file_pattern, token_to_id, num_examples):
  """Loads a DeepCoder step-by-step dataset.

  Args:
    file_pattern: A file pattern for the TFRecord files to read.
    token_to_id: Mapping from tokens to token IDs for the DeepCoder vocabulary.
    num_examples: The number of examples in an I/O specification.

  Returns:
    A tf.data.Dataset containing dictionaries with keys 'inputs', 'outputs', and
    'target'.
  """
  filenames = gfile.glob(file_pattern)
  raw_dataset = tf.data.TFRecordDataset(filenames)

  vocab_table = tf.lookup.StaticVocabularyTable(
      tf.lookup.KeyValueTensorInitializer(
          list(token_to_id.keys()),
          list(token_to_id.values()),
          key_dtype=tf.string,
          value_dtype=tf.int64),
      len(token_to_id))
  eos_id = deepcoder_dsl.EOS_ID

  def _parse_fn(record):
    """Parses a record into a feature_dict."""
    empty_default = [''] * num_examples
    feature_values = tf.io.parse_single_example(
        serialized=record,
        features={
            'inputs':
                tf.io.FixedLenFeature([num_examples], tf.string,
                                      default_value=empty_default),
            'outputs':
                tf.io.FixedLenFeature([num_examples], tf.string,
                                      default_value=empty_default),
            'program':
                tf.io.FixedLenFeature([], tf.string, default_value=''),
        })

    # Map tokens to ids.
    inputs = tf.strings.split(feature_values['inputs'], sep=' ').to_tensor()
    inputs = vocab_table.lookup(inputs)

    outputs = tf.strings.split(feature_values['outputs'], sep=' ').to_tensor()
    outputs = vocab_table.lookup(outputs)

    program = tf.strings.split(feature_values['program'], sep=' ')
    program = vocab_table.lookup(program)
    program = tf.concat([program, [eos_id]], axis=-1)

    # inputs: [num_examples, max_length_of_input]
    # outputs: [num_examples, max_length_of_output]
    # program: [max_length_of_program + 1]
    return {
        'inputs': inputs,
        'outputs': outputs,
        'target': program,
    }

  dataset = raw_dataset.map(_parse_fn)
  return dataset


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
            'program':
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
        tf.strings.split(feature_values['program'], sep='|'), sep=' ')
    program = program.merge_dims(0, -1)
    program = tf.strings.to_number(program, out_type=tf.int32)
    program = tf.concat([program, [eos_id]], axis=-1)

    # inputs: [num_strings, max_length_of_input]
    # outputs: [num_strings, max_length_of_output]
    # program: [max_length_of_program + 1]
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
  beam_cache0 = jax.tree.map(lambda x: decode.add_beam_dim(x, beam_size), cache)
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


def format_path(model_or_dataset_path):
  format_dict = {
      'seed': _SEED.value,
      'experiment': _EXPERIMENT.value,
      'corruption_rate': _CORRUPTION_RATE.value,
      'aligned_relative_attention': _ALIGNED_RELATIVE_ATTENTION.value,
  }
  return model_or_dataset_path.format(**format_dict)


def load_spec_decomposer_model(init_rng, spec_vocab_size, io_shape,
                               spec_target_shape, bos_id, eos_id, sep_id):
  """Loads SpecDecomposerModel."""
  num_position_buckets = _SPEC_DECOMPOSER_NUM_POSITION_BUCKETS.value
  max_distance = _SPEC_DECOMPOSER_MAX_DISTANCE.value
  max_program_cross_embed_distance = (
      _SPEC_DECOMPOSER_MAX_PROGRAM_CROSS_EMBED_DISTANCE.value)
  spec_decomposer_base_config = base_models.TransformerConfig(
      vocab_size=spec_vocab_size,
      output_vocab_size=spec_vocab_size,
      shift=False,
      emb_dim=_EMBEDDING_DIM.value,
      num_heads=_NUM_HEADS.value,
      num_layers=_NUM_LAYERS.value,
      qkv_dim=_EMBEDDING_DIM.value,
      mlp_dim=_HIDDEN_DIM.value,
      max_len=max(_MAX_IO_LENGTH.value, _MAX_SPEC_PART_LENGTH.value),
      dropout_rate=_DROPOUT_RATE.value,
      attention_dropout_rate=_ATTENTION_DROPOUT_RATE.value,
      use_relative_attention=_USE_RELATIVE_ATTENTION.value,
      deterministic=True,
      decode=not _SLOW_DECODE.value,
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
      max_program_cross_embed_distance=max_program_cross_embed_distance)
  spec_decomposer_predict_config = models.DecomposeAttentionTransformerConfig(
      base_config=spec_decomposer_base_config,
      dataset_type=_DATASET_TYPE.value,
      aligned_relative_attention=_ALIGNED_RELATIVE_ATTENTION.value,
      separator_token_id=sep_id)

  m = models.DecomposeAttentionTransformer(spec_decomposer_predict_config)
  initial_variables = jax.jit(m.init)(init_rng,
                                      jnp.ones(io_shape, jnp.float32),
                                      jnp.ones(io_shape, jnp.float32),
                                      jnp.ones(spec_target_shape, jnp.float32))

  optimizer_def = optim.Adam(
      1e-3, beta1=0.9, beta2=0.98, eps=1e-9, weight_decay=0.01)
  spec_decomposer_optimizer = optimizer_def.create(initial_variables['params'])
  spec_decomposer_optimizer = checkpoints.restore_checkpoint(
      format_path(_SPEC_DECOMPOSER_PATH_FORMAT.value),
      spec_decomposer_optimizer)
  logging.info('Found spec decomposer checkpointed at step %d.',
               int(spec_decomposer_optimizer.state.step))

  spec_decomposer_pred_step = jax.jit(
      functools.partial(end_to_end_predict_step,
                        eos_token=eos_id,
                        max_decode_len=_MAX_SPEC_PART_LENGTH.value,
                        config=spec_decomposer_predict_config,
                        slow_decode=_SLOW_DECODE.value),
      static_argnums=(5,),  # The `beam_size` argument.
  )

  return spec_decomposer_optimizer, spec_decomposer_pred_step


def load_synthesizer_model(init_rng, spec_vocab_size, program_vocab_size,
                           io_shape, program_shape, bos_id, eos_id):
  """Loads synthesizer or joint model."""

  num_position_buckets = _SYNTHESIZER_NUM_POSITION_BUCKETS.value
  max_distance = _SYNTHESIZER_MAX_DISTANCE.value
  max_program_cross_embed_distance = (
      _SYNTHESIZER_MAX_PROGRAM_CROSS_EMBED_DISTANCE.value)
  synthesizer_base_config = base_models.TransformerConfig(
      vocab_size=spec_vocab_size,
      output_vocab_size=program_vocab_size,
      shift=False,
      emb_dim=_EMBEDDING_DIM.value,
      num_heads=_NUM_HEADS.value,
      num_layers=_NUM_LAYERS.value,
      qkv_dim=_EMBEDDING_DIM.value,
      mlp_dim=_HIDDEN_DIM.value,
      max_len=max(_MAX_IO_LENGTH.value, _MAX_PROGRAM_LENGTH.value),
      dropout_rate=_DROPOUT_RATE.value,
      attention_dropout_rate=_ATTENTION_DROPOUT_RATE.value,
      use_relative_attention=_USE_RELATIVE_ATTENTION.value,
      deterministic=True,
      decode=not _SLOW_DECODE.value,
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
      max_program_cross_embed_distance=max_program_cross_embed_distance)
  synthesizer_predict_config = models.DecomposeAttentionTransformerConfig(
      base_config=synthesizer_base_config,
      dataset_type=_DATASET_TYPE.value,
      # Synthesizer and Joint models don't use aligned_relative_attention.
      aligned_relative_attention=False,
      separator_token_id=-1)  # Not used for synthesizer or joint models.

  m = models.DecomposeAttentionTransformer(synthesizer_predict_config)
  initial_variables = jax.jit(m.init)(init_rng,
                                      jnp.ones(io_shape, jnp.float32),
                                      jnp.ones(io_shape, jnp.float32),
                                      jnp.ones(program_shape, jnp.float32))

  optimizer_def = optim.Adam(
      1e-3, beta1=0.9, beta2=0.98, eps=1e-9, weight_decay=0.01)
  synthesizer_optimizer = optimizer_def.create(initial_variables['params'])
  synthesizer_optimizer = checkpoints.restore_checkpoint(
      format_path(_SYNTHESIZER_PATH_FORMAT.value),
      synthesizer_optimizer)
  logging.info('Found synthesizer checkpointed at step %d.',
               int(synthesizer_optimizer.state.step))

  synthesizer_pred_step = jax.jit(
      functools.partial(end_to_end_predict_step,
                        eos_token=eos_id,
                        max_decode_len=_MAX_PROGRAM_LENGTH.value,
                        config=synthesizer_predict_config,
                        slow_decode=_SLOW_DECODE.value),
      static_argnums=(5,),  # The `beam_size` argument.
  )

  return synthesizer_optimizer, synthesizer_pred_step


def divide_no_nan(x, y, default=-1):
  return x/y if y > 0 else default


def main(_):
  if _PREDICTION_TYPE.value == 'joint' and not _USE_EXECUTION.value:
    raise ValueError(
        'Joint prediction requires using execution to compute the remaining '
        'output.')
  if not _USE_EXECUTION.value and _DISCARD_REPEAT_FUNCTIONALITY.value:
    raise ValueError(
        'discard_repeat_functionality requires using execution.')
  if not _DETECT_INVALID.value and _DISCARD_REPEAT_FUNCTIONALITY.value:
    raise ValueError(
        'discard_repeat_functionality requires detecting invalid states.')

  if not gfile.isdir(_SAVE_DIR.value):
    gfile.makedirs(_SAVE_DIR.value)

  xm_client = xmanager_api.XManagerApi(xm_deployment_env='alphabet')
  work_unit = xm_client.get_current_work_unit()
  hparam_dict = work_unit.parameters['args']
  hparam_str = 'hparams-' + ','.join(sorted([f'{k}={v}'
                                             for k, v in hparam_dict.items()]))

  write_summary = jax.host_id() == 0
  tb_dir = (os.path.join(_SAVE_DIR.value, 'tb', hparam_str) if write_summary
            else '')
  summary_writer = tensorboard.SummaryWriter(tb_dir)
  result_json_path = os.path.join(
      tb_dir, f'results-{_PREDICTION_TYPE.value}.json')

  # TODO(jxihong): end-to-end loop is not batched right now.
  batch_size = 1
  io_shape = (batch_size, _NUM_EXAMPLES.value, _MAX_IO_LENGTH.value)
  spec_target_shape = (batch_size, _MAX_SPEC_PART_LENGTH.value)
  program_shape = (batch_size, _MAX_PROGRAM_LENGTH.value)

  # Setup DSL
  # ---------------------------------------------------------------------------

  # Build token tables.
  if _DATASET_TYPE.value in ['robustfill', 'robustfill_base']:
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
    program_id_token_table, program_token_id_table = (
        dsl_tokens.build_token_tables())
    program_vocab_size = len(program_id_token_table) + 1
    sep_id = spec_token_id_table[input_pipeline.SEPARATOR_TOKEN]
  elif _DATASET_TYPE.value == 'deepcoder':
    id_to_token, token_to_id = deepcoder_dsl.vocab_tables()
    bos_id, eos_id = deepcoder_dsl.BOS_ID, deepcoder_dsl.EOS_ID
    vocab_size = len(id_to_token)  # Already includes padding.

    spec_vocab_size = program_vocab_size = vocab_size
    program_id_token_table = spec_id_token_table = id_to_token
    program_token_id_table = spec_token_id_table = token_to_id
    sep_id = deepcoder_dsl.SEP_ID
  else:
    raise ValueError('Unhandled dataset_type: {}'.format(_DATASET_TYPE.value))

  # Util functions for prediction
  # ---------------------------------------------------------------------------

  def decode_spec(target):
    """Convert from int tensor to a string."""
    if _DATASET_TYPE.value in ['robustfill', 'robustfill_base', 'deepcoder']:
      target = target[np.all([target != 0, target != bos_id, target != eos_id],
                             axis=0)].astype(np.int32)
      target = np.array(target)  # JAX arrays will fail dict lookups.
      separator = ' ' if _DATASET_TYPE.value == 'deepcoder' else ''
      return separator.join([spec_id_token_table[t_id]
                             for t_id in target if t_id > 0])
    else:
      raise ValueError('Unhandled dataset_type: {}'.format(_DATASET_TYPE.value))

  def encode_spec(target, max_target_length, add_eos=True):
    if _DATASET_TYPE.value in ['robustfill', 'robustfill_base', 'deepcoder']:
      tokens = (target.split(' ') if _DATASET_TYPE.value == 'deepcoder'
                else list(target))
      token_ids = [spec_token_id_table[t] for t in tokens]
      if add_eos:
        token_ids += [eos_id]
      return np.array(token_ids + [0] * (max_target_length - len(token_ids)))
    else:
      raise ValueError('Unhandled dataset_type: {}'.format(_DATASET_TYPE.value))

  def split_outputs(output_parts, outputs, max_target_length, aux, beam_i):
    """Returns a tuple (valid, last_step, step_outputs, current_outputs)."""
    num_examples = len(outputs)
    assert num_examples == _NUM_EXAMPLES.value

    if aux['finished'][0][beam_i]:
      # Do nothing if it's finished.
      finished_str = ('' if _DATASET_TYPE.value == 'deepcoder'
                      else '[finished]')
      spec_parts_str = [finished_str] * num_examples
      step_outputs = [
          encode_spec(spec_str, max_target_length=max_target_length,
                      add_eos=False)
          for spec_str in spec_parts_str]
      return (aux['valid'][0][beam_i], aux['last_step'][0][beam_i],
              np.array(step_outputs), aux['current_outputs'][0][beam_i])

    # If we pass in an already-decoded list of strings, use them directly.
    if isinstance(output_parts, list) and isinstance(output_parts[0], str):
      output_parts_str = output_parts
    else:
      # Decode the SpecDecomposerModel prediction and separate examples.
      output_parts_str = decode_spec(output_parts).strip('|').split('|')
    if isinstance(outputs, list) and isinstance(outputs[0], str):
      decoded_outputs = outputs
    else:
      decoded_outputs = [decode_spec(o) for o in outputs]

    valid = True

    if _DETECT_INVALID.value:
      # The prediction is invalid if it has an incorrect number of | characters
      # or all of the parts are empty. (Some but not all parts can be empty.)
      if len(output_parts_str) != num_examples or all(
          [not part for part in output_parts_str]):
        output_parts_str = ['' if _DATASET_TYPE.value == 'deepcoder'
                            else '[invalid]'] * num_examples
        valid = False
      if _DATASET_TYPE.value == 'deepcoder':
        try:
          for one_example in output_parts_str:
            deepcoder_dsl.ProgramState.from_str('x0 = ' + one_example)
        except deepcoder_dsl.ParseError:
          valid = False
    else:
      # Still need to handle an incorrect number of | characters. Do our best to
      # continue even if the prediction is malformed.
      output_parts_str = output_parts_str[:num_examples]
      output_parts_str += [''] * (num_examples - len(output_parts_str))
      assert len(output_parts_str) == num_examples
      # TODO(kshi): maybe for DeepCoder, pad with the output instead of empty
      # string. But this doesn't really matter, it's an ablation.

    step_outputs = [
        encode_spec(
            output_part_str, max_target_length=max_target_length, add_eos=False)
        for output_part_str in output_parts_str
    ]

    if _DATASET_TYPE.value in ['robustfill', 'robustfill_base']:
      current_outputs_str = []
      for part, output in zip(output_parts_str, decoded_outputs):
        if _DETECT_INVALID.value and not output.startswith(part):
          current_outputs_str = ['[invalid]'] * num_examples
          valid = False
          break
        current_outputs_str.append(output[len(part):])
      current_outputs = [
          encode_spec(current_output_str, max_target_length=max_target_length,
                      add_eos=True)
          for current_output_str in current_outputs_str
      ]
      last_step = all([not current_output_str
                       for current_output_str in current_outputs_str])
    elif _DATASET_TYPE.value == 'deepcoder':
      current_outputs = [
          encode_spec(
              decoded_output, max_target_length=max_target_length, add_eos=True)
          for decoded_output in decoded_outputs
      ]
      last_step = (output_parts_str == decoded_outputs)
    else:
      raise ValueError(f'Unsupported dataset type: {_DATASET_TYPE.value}')

    return valid, last_step, np.array(step_outputs), np.array(current_outputs)

  def join_inputs(input_parts, current_inputs, max_target_length,
                  variable_index):
    """Returns a new current_inputs."""
    if _DATASET_TYPE.value != 'deepcoder':
      # Do nothing for RobustFill.
      return current_inputs

    # If we pass in an already-decoded list of strings, use them directly.
    if isinstance(input_parts, list) and isinstance(input_parts[0], str):
      input_parts_str = input_parts
    else:
      # Decode the SpecDecomposerModel prediction and separate examples.
      input_parts_str = [decode_spec(part) for part in input_parts]
    decoded_current_inputs = [decode_spec(inp) for inp in current_inputs]
    assert len(input_parts_str) == len(decoded_current_inputs)

    current_inputs_str = [
        ' '.join([current_input, deepcoder_dsl.SEP,
                  deepcoder_dsl.variable_token(variable_index), '=', part])
        for current_input, part in zip(decoded_current_inputs, input_parts_str)
    ]

    current_inputs = [
        encode_spec(
            inp, max_target_length=max_target_length, add_eos=True)
        for inp in current_inputs_str
    ]
    return np.array(current_inputs)

  def join_programs(program_parts, programs, step=0):
    if _DATASET_TYPE.value == 'deepcoder':
      # Need to prepend variable to the predicted program parts, if the program
      # isn't all padding (which happens when that beam element was already
      # finished).
      # pylint: disable=g-complex-comprehension
      prepend_tokens = np.array([
          [deepcoder_dsl.SEP_ID,  # pylint: disable=g-long-ternary
           program_token_id_table[deepcoder_dsl.variable_token(step)],
           program_token_id_table['=']]
          if np.max(program_part) != 0 else [0] * 3
          for program_part in program_parts[0]])
      # pylint: enable=g-complex-comprehension
      program_parts = jnp.concatenate(
          [prepend_tokens[None, :], program_parts], axis=-1)
    return jnp.concatenate([programs, program_parts], axis=-1)

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

  def process_and_decode_program(program_token_ids):
    """Returns a pair (valid, program)."""
    try:
      if _DATASET_TYPE.value in ['robustfill', 'robustfill_base']:
        program = robust_fill_dsl.decode_program(
            process_predicted_program(program_token_ids, add_eos=True),
            program_id_token_table)
      elif _DATASET_TYPE.value == 'deepcoder':
        program_tokens = [program_id_token_table[int(p_id)]
                          for p_id in program_token_ids
                          if p_id > 0 and p_id != deepcoder_dsl.EOS_ID]
        program = deepcoder_dsl.Program.from_tokens(program_tokens)
      else:
        raise ValueError('Unhandled dataset_type: {}'.format(
            _DATASET_TYPE.value))
      # If the program can't be converted to string, it's invalid.
      str(program)
      return True, program
    except Exception:  # pylint: disable=broad-except
      # It's not valid, but maybe we have to ignore that fact.
      valid = not _DETECT_INVALID.value
      return valid, '[invalid program]'

  def run_program(program, inputs):
    """Returns a pair (valid, outputs)."""
    # If the program cannot be run, we treat it as outputting an empty string.
    outputs = []
    valid = True
    default_output = '' if _DATASET_TYPE.value == 'deepcoder' else ''
    for i in inputs:
      if program == '[invalid program]':
        outputs.append(default_output)
        if _DETECT_INVALID.value:
          valid = False
      else:
        try:
          if _DATASET_TYPE.value in ['robustfill', 'robustfill_base']:
            outputs.append(program(i))
          elif _DATASET_TYPE.value == 'deepcoder':
            initial_state = deepcoder_dsl.ProgramState.from_str(i)
            final_state = program.run(initial_state.state)
            outputs.append(deepcoder_dsl.result_to_str(
                final_state.get_output()))
          else:
            raise ValueError('Unhandled dataset_type: {}'.format(
                _DATASET_TYPE.value))

        except:  # pylint: disable=bare-except
          outputs.append(default_output)
          if _USE_EXECUTION.value and _DETECT_INVALID.value:
            valid = False
    return valid, outputs

  def run_program_execution_trace(program, inputs):
    """Returns an execution trace for the program (for all statements)."""
    traces = []
    default_output = '' if _DATASET_TYPE.value == 'deepcoder' else ''
    for i in inputs:
      if program == '[invalid program]':
        traces.append(default_output)
      else:
        try:
          if _DATASET_TYPE.value in ['robustfill', 'robustfill_base']:
            traces.append(program(i))
          elif _DATASET_TYPE.value == 'deepcoder':
            # Use entire program state (containing all local variables).
            initial_state = deepcoder_dsl.ProgramState.from_str(i)
            traces.append(str(program.run(initial_state.state)))
          else:
            raise ValueError('Unhandled dataset_type: {}'.format(
                _DATASET_TYPE.value))
        except:  # pylint: disable=bare-except
          traces.append(default_output)
    return traces

  # Load Dataset
  # ---------------------------------------------------------------------------
  logging.info('Initializing dataset.')
  if not _TEST_DATASET_FORMAT.value:
    raise ValueError('Must specify filepattern to dataset.')

  # Training dataset.
  test_dataset_path = format_path(_TEST_DATASET_FORMAT.value)
  logging.info('Loading dataset from %s', test_dataset_path)
  padded_shapes = {
      'inputs': io_shape[1:],
      'outputs': io_shape[1:],
      'target': program_shape[1:],
  }
  logging.info('padded_shapes: %s', padded_shapes)

  if _DATASET_TYPE.value == 'robustfill':
    create_dataset_fn = create_robust_fill_dataset
  elif _DATASET_TYPE.value == 'deepcoder':
    create_dataset_fn = create_deepcoder_dataset
  else:
    raise ValueError('Unhandled dataset_type: {}'.format(_DATASET_TYPE.value))

  test_dataset = create_dataset_fn(test_dataset_path,
                                   spec_token_id_table,
                                   _NUM_EXAMPLES.value)
  test_dataset = test_dataset.padded_batch(
      batch_size, padded_shapes=padded_shapes, drop_remainder=False)
  test_dataset = test_dataset.take(_NUM_TEST_BATCHES.value)

  # TODO(jxihong): Implement fast decoding.
  assert _SLOW_DECODE.value, 'Fast decoding is not implemented yet.'

  rng = jax.random.PRNGKey(0)
  rng, init_rng = jax.random.split(rng)  # pylint: disable=unused-variable

  # Main Prediction Loop
  # ---------------------------------------------------------------------------

  if _PREDICTION_TYPE.value == 'separate':
    spec_decomposer_optimizer, spec_decomposer_pred_step = (
        load_spec_decomposer_model(
            init_rng=init_rng,
            spec_vocab_size=spec_vocab_size,
            io_shape=io_shape,
            spec_target_shape=spec_target_shape,
            bos_id=bos_id,
            eos_id=eos_id,
            sep_id=sep_id))
  synthesizer_optimizer, synthesizer_pred_step = load_synthesizer_model(
      init_rng=init_rng,
      spec_vocab_size=spec_vocab_size,
      program_vocab_size=program_vocab_size,
      io_shape=io_shape,
      program_shape=program_shape,
      bos_id=bos_id,
      eos_id=eos_id)

  # ----------------------------------------------------------------------------
  # A discussion of metrics we collect (if beam_size=1).
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

  beam_size = _BEAM_SIZE.value
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

  result_json = []

  for test_example_index, batch in enumerate(test_dataset.as_numpy_iterator()):
    if test_example_index % 10 == 0:
      logging.info('Processing test example #%s', test_example_index)
    do_logging = test_example_index < _NUM_EXAMPLES_TO_LOG.value
    test_example_start_time = timeit.default_timer()

    inputs, outputs = batch['inputs'], batch['outputs']
    decoded_inputs = [decode_spec(i) for i in inputs[0]]
    if _DATASET_TYPE.value == 'deepcoder':
      deepcoder_num_inputs = decoded_inputs[0].count(deepcoder_dsl.SEP) + 1
    else:
      # This shouldn't be used for RobustFill but must be defined.
      deepcoder_num_inputs = 0
    decoded_outputs = [decode_spec(o) for o in outputs[0]]
    _, ground_truth = process_and_decode_program(batch['target'][0])
    ground_truth_length = len(
        ground_truth.statements if _DATASET_TYPE.value == 'deepcoder'
        else ground_truth.expressions)

    log_message = ''  # Avoid pytype errors.
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
          f'program: {ground_truth}\n'
          f'\n'
      )

    next_history_id = itertools.count().__next__
    step_index = 0
    first_error = None
    solution_program = None

    if _DATASET_TYPE.value == 'deepcoder':
      initial_program = []
      for input_index in range(deepcoder_num_inputs):
        if input_index > 0:
          initial_program.append(deepcoder_dsl.SEP_ID)
        initial_program.extend([
            program_token_id_table[deepcoder_dsl.variable_token(input_index)],
            program_token_id_table['='],
            program_token_id_table['INPUT']])
    else:
      initial_program = [bos_id]

    # `valid` means whether we are in an unrecoverable state (or the chance of
    # recovering is so low that we'd rather allocate beam space to other
    # things). We manually change the scores of invalid beam states to NEG_INF
    # to make them drop out of the beam. The `use_execution` flag determines
    # whether we are allowed to use predicted programs' executions to inform
    # validity.

    # `last_step` is set during the SpecDecomposer step to signal that it
    # predicts that we're on the last step so there should be no further
    # iterations. After the Synthesizer step, this beam element should be marked
    # as `finished`. If `use_execution` is True, we use the predicted program's
    # execution to see if the beam element is truly finished or not.

    # `finished` means whether a beam element is completely done, i.e., its
    # score is final and there should be no more predictions by either model.
    # Invalid beam elements should also be marked as finished.
    aux = {
        'current_inputs': decode.add_beam_dim(inputs, beam_size),
        'current_outputs': decode.add_beam_dim(outputs, beam_size),
        'step_outputs': decode.add_beam_dim(
            outputs, beam_size),  # Initial value doesn't matter.
        'programs': decode.add_beam_dim(
            np.array(initial_program)[None, Ellipsis], beam_size),
        'scores': jnp.array(
            [0.0] + [decode.NEG_INF] * (beam_size - 1))[None, Ellipsis],
        'valid': jnp.array([True] + [False] * (beam_size - 1))[None, Ellipsis],
        'last_step': jnp.array([False] + [True] * (beam_size - 1))[None, Ellipsis],
        'finished': jnp.array([False] + [True] * (beam_size - 1))[None, Ellipsis],
        'history': jnp.array([[[next_history_id()] for _ in range(beam_size)]]),
    }

    # End-to-end prediction loop.
    step_limit = 20
    if _DATASET_TYPE.value == 'deepcoder':
      step_limit = deepcoder_dsl.MAX_NUM_VARIABLES - deepcoder_num_inputs
    while jnp.any(~aux['finished']) and step_index < step_limit:

      # Get ground truth for logging and metrics purposes.
      if step_index >= ground_truth_length:
        ground_truth_program_part = '[step index out of bounds]'
        ground_truth_output_parts = '[step index out of bounds]'
      elif _DATASET_TYPE.value == 'deepcoder':
        ground_truth_program_part = ground_truth.statements[step_index]
        inputs_program_states = [deepcoder_dsl.ProgramState.from_str(i)
                                 for i in decoded_inputs]
        ground_truth_output_parts = [
            deepcoder_dsl.result_to_str(
                ground_truth.run(inputs_program_state.state).get_index(
                    step_index + deepcoder_num_inputs))
            for inputs_program_state in inputs_program_states
        ]
      else:
        ground_truth_program_part = ground_truth.expressions[step_index]
        ground_truth_output_parts = [ground_truth_program_part(i)
                                     for i in decoded_inputs]

      # Use the spec decomposition step when predicting with separate models,
      # but not when doing joint prediction.
      if _PREDICTION_TYPE.value == 'separate':
        # Spec Decomposition Step.
        ##########################

        # Run the SpecDecomposerModel.
        start_time = timeit.default_timer()
        predicted_spec_parts, scores, aux = spec_decomposer_pred_step(  # pylint: disable=undefined-variable
            params=spec_decomposer_optimizer.target,  # pylint: disable=undefined-variable
            inputs=aux['current_inputs'],
            outputs=aux['current_outputs'],
            cache=None,
            aux=aux,
            beam_size=beam_size)
        spec_prediction_times.append(timeit.default_timer() - start_time)

        # Process spec predictions.
        start_time = timeit.default_timer()
        spec_parts_batch = np.array(predicted_spec_parts)
        results = [
            split_outputs(beam, aux['current_outputs'][0][i],
                          max_target_length=_MAX_IO_LENGTH.value, aux=aux,
                          beam_i=i)
            for i, beam in enumerate(spec_parts_batch[0])]
        # split_outputs returns (valids, last_steps, step_outputs,
        # current_outputs)
        valids, last_steps, step_outputs, current_outputs = map(list,
                                                                zip(*results))

        aux['current_outputs'] = jnp.array(current_outputs)[None, Ellipsis]
        aux['step_outputs'] = jnp.array(step_outputs)[None, Ellipsis]
        aux['scores'] = scores
        aux['valid'] = jnp.array(valids)[None, Ellipsis]
        aux['last_step'] = jnp.array(last_steps)[None, Ellipsis]
        current_history_ids = jnp.array(
            [next_history_id() for _ in range(beam_size)])[None, Ellipsis, None]
        aux['history'] = jnp.concatenate([aux['history'], current_history_ids],
                                         axis=-1)

        # Process invalid states.
        if _DETECT_INVALID.value:
          aux['scores'] += decode.NEG_INF * (1 - aux['valid'])
          aux['finished'] |= ~aux['valid']
        spec_processing_times.append(timeit.default_timer() - start_time)

        # Analysis and logging.
        start_time = timeit.default_timer()
        assert len(aux['step_outputs']) == 1
        best_spec_prediction = [decode_spec(o)
                                for o in aux['step_outputs'][0][-1]]
        if aux['finished'][0][-1]:
          best_spec_prediction = '[finished]'
        if aux['finished'][0][-1] & aux['valid'][0][-1]:
          matches = 'N/A'
        else:
          matches = best_spec_prediction == ground_truth_output_parts
          if not matches and first_error is None:
            first_error = f'SpecDecomposerModel at step #{step_index + 1}'
        spec_analysis_times.append(timeit.default_timer() - start_time)

        if do_logging:
          # Prefer not to break these long lines because they reflect the lines
          # actually printed into the logs.
          # pylint: disable=line-too-long
          log_message += '\n' + ('=' * 80) + '\n'
          log_message += (
              f'Spec Decomposition Step #{step_index + 1}:\n'
              f'  ground truth output parts:           {ground_truth_output_parts}\n'
              f'  SpecDecomposerModel best prediction: {best_spec_prediction}\n'
              f'    matches: {matches}\n'
              f'  ground truth program part:           {ground_truth_program_part}\n'
              f'---------- Full beam: ----------\n'
          )
          for i in range(beam_size)[::-1][:_NUM_BEAM_ELEMENTS_TO_LOG.value]:
            prediction_i = [decode_spec(o) for o in aux['step_outputs'][0][i]]
            score_i = aux['scores'][0][i]
            current_inputs_i = [decode_spec(o)
                                for o in aux['current_inputs'][0][i]]
            current_outputs_i = [decode_spec(o)
                                 for o in aux['current_outputs'][0][i]]
            _, program_i = process_and_decode_program(aux['programs'][0][i])
            valid_i = aux['valid'][0][i]
            last_step_i = aux['last_step'][0][i]
            finished_i = aux['finished'][0][i]
            log_message += (
                f'Beam item {i}:\n'
                f'  prediction: {prediction_i}\n'
                f'  score: {score_i:.4f}\n'
                f'  current_inputs: {current_inputs_i}\n'
                f'  current_outputs: {current_outputs_i}\n'
                f'  program: {program_i}\n'
                f'  valid: {valid_i}, last_step: {last_step_i}, finished: {finished_i}\n'
                f'  history: {aux["history"][0][i]}\n'
            )
          # pylint: enable=line-too-long

        # Elements can become newly finished if they are invalid.
        if jnp.all(aux['finished']):
          step_index += 1  # Count this half-step.
          break

      # Synthesizer Step.
      ###################
      # (Used in both separate and joint prediction)

      # Run the SynthesizerModel.
      start_time = timeit.default_timer()
      if _PREDICTION_TYPE.value == 'separate':
        # Use next output part as predicted by the SpecDecomposerModel.
        synthesizer_step_outputs = aux['step_outputs']
      elif _PREDICTION_TYPE.value == 'joint':
        # Use the current output from the previous step.
        synthesizer_step_outputs = aux['current_outputs']
      else:
        raise ValueError(f'Unhandled prediction_type: {_PREDICTION_TYPE.value}')
      predicted_program_parts, scores, aux = synthesizer_pred_step(
          params=synthesizer_optimizer.target,
          inputs=aux['current_inputs'],
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
      aux['programs'] = join_programs(program_parts, aux['programs'],
                                      step=step_index + deepcoder_num_inputs)
      aux['scores'] = scores
      current_history_ids = jnp.array(
          [next_history_id() for _ in range(beam_size)])[None, Ellipsis, None]
      aux['history'] = jnp.concatenate([aux['history'], current_history_ids],
                                       axis=-1)

      # Process invalid states and set up current inputs/outputs for the next
      # step.
      new_valids, new_last_steps = [], []
      new_current_inputs, new_current_outputs = [], []
      programs = collections.defaultdict(list)  # Keep track of unique programs.
      for i in range(beam_size):
        valid_i = bool(aux['valid'][0][i])
        last_step_i = aux['last_step'][0][i]
        current_inputs_i = aux['current_inputs'][0][i]
        current_outputs_i = aux['current_outputs'][0][i]

        # Don't need to do any checking if the beam element is already finished
        # or already invalid.
        if not aux['finished'][0][i] and valid_i:
          # This is just a syntax check.
          valid_i, program_i = process_and_decode_program(aux['programs'][0][i])

          if _USE_EXECUTION.value:
            # Check for syntax, runtime errors, and output prefix matching.
            valid_i, program_outputs_i = run_program(program_i, decoded_inputs)
            # Must use execution (even if invalid) for joint prediction.
            if valid_i or _PREDICTION_TYPE.value == 'joint':
              valid_i, last_step_i, _, current_outputs_i = split_outputs(
                  program_outputs_i, decoded_outputs,
                  max_target_length=_MAX_IO_LENGTH.value, aux=aux,
                  beam_i=i)

              # For DeepCoder, update the program states by appending the
              # program's execution result to the current input.
              current_inputs_i = join_inputs(
                  program_outputs_i, current_inputs_i,
                  max_target_length=_MAX_IO_LENGTH.value,
                  variable_index=step_index + deepcoder_num_inputs)

          else:  # Not using execution.
            assert _PREDICTION_TYPE.value == 'separate'
            # For DeepCoder, append the SpecDecomposerModel's prediction to the
            # current input (program state).
            current_inputs_i = join_inputs(
                aux['step_outputs'][0][i], current_inputs_i,
                max_target_length=_MAX_IO_LENGTH.value,
                variable_index=step_index + deepcoder_num_inputs)

          if _DISCARD_REPEAT_FUNCTIONALITY.value:
            # Mark beam element as invalid if it share the same program
            # functionality as a beam element with higher score.
            program_traces_i = run_program_execution_trace(program_i,
                                                           decoded_inputs)
            program_traces_i_key = '~'.join(program_traces_i)
            for j in programs[program_traces_i_key]:
              new_valids[j] = False
            programs[program_traces_i_key].append(i)

        if not _DETECT_INVALID.value:
          # Even though we updated valid_i above, those helper functions should
          # not claim something is invalid if detect_invalid=False.
          assert valid_i
        new_valids.append(valid_i)
        new_last_steps.append(last_step_i)
        new_current_inputs.append(current_inputs_i)
        new_current_outputs.append(current_outputs_i)
      aux['valid'] = jnp.array(new_valids)[None, Ellipsis]
      aux['last_step'] = jnp.array(new_last_steps)[None, Ellipsis]
      aux['current_inputs'] = jnp.array(new_current_inputs)[None, Ellipsis]
      aux['current_outputs'] = jnp.array(new_current_outputs)[None, Ellipsis]

      already_finished = aux['finished'][0]
      aux['finished'] |= aux['last_step']
      if _DETECT_INVALID.value:
        aux['scores'] += decode.NEG_INF * (1 - aux['valid'])
        aux['finished'] |= ~aux['valid']
      else:
        assert np.all(aux['valid'])
      synthesis_processing_times.append(timeit.default_timer() - start_time)

      # Analysis and logging.
      start_time = timeit.default_timer()

      if already_finished[-1] & aux['valid'][0][-1]:
        best_program_prediction = '[already finished]'
        functionally_correct = 'N/A'
        program_outputs = 'N/A'
      else:
        # Compare to the ground truth, not the spec prediction. The best-scoring
        # program didn't necessarily come from the best-scoring spec prediction.
        program_tokens = (  # Use entire program for DeepCoder.
            aux['programs'][0][-1] if _DATASET_TYPE.value == 'deepcoder'
            else program_parts[0][-1])
        _, best_program_prediction = process_and_decode_program(program_tokens)
        _, program_outputs = run_program(best_program_prediction,
                                         decoded_inputs)
        functionally_correct = program_outputs == ground_truth_output_parts
        if not functionally_correct and first_error is None:
          first_error = f'SynthesizerModel at step #{step_index + 1}'
      synthesis_analysis_times.append(timeit.default_timer() - start_time)

      if do_logging:
        # Prefer not to break these long lines because they reflect the lines
        # actually printed into the logs.
        # pylint: disable=line-too-long
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

        for i in range(beam_size)[::-1][:_NUM_BEAM_ELEMENTS_TO_LOG.value]:
          if _DATASET_TYPE.value == 'deepcoder':
            # A program part isn't a valid whole program. Instead, just decode
            # tokens using the tables. Even though this is a program part not a
            # spec part, decode_spec is sufficient here.
            prediction_i = decode_spec(program_parts[0][i])
          else:
            _, prediction_i = process_and_decode_program(program_parts[0][i])
          score_i = aux['scores'][0][i]
          current_inputs_i = [decode_spec(o)
                              for o in aux['current_inputs'][0][i]]
          current_outputs_i = [decode_spec(o)
                               for o in aux['current_outputs'][0][i]]
          step_outputs_i = [decode_spec(o) for o in aux['step_outputs'][0][i]]
          _, program_i = process_and_decode_program(aux['programs'][0][i])
          valid_i = aux['valid'][0][i]
          last_step_i = aux['last_step'][0][i]
          finished_i = aux['finished'][0][i]
          log_message += (
              f'Beam item {i}:\n'
              f'  prediction: {prediction_i}\n'
              f'  score: {score_i:.4f}\n'
              f'  current_inputs: {current_inputs_i}\n'
              f'  current_outputs: {current_outputs_i}\n'
              f'  step_outputs: {step_outputs_i}\n'
              f'  program: {program_i}\n'
              f'  valid: {valid_i}, last_step: {last_step_i}, finished: {finished_i}\n'
              f'  history: {aux["history"][0][i]}\n'
          )
        # pylint: enable=line-too-long

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
        solution_program = program_i
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
      logging.info(log_message)

      if write_summary:
        summary_writer.text((f'predictions_{test_example_index}, '
                             f'beam_size={beam_size}'),
                            log_message, 0)
        summary_writer.flush()

    logging.info('Test example %d resulted in: %s', test_example_index,
                 'success' if success else 'failure')

    result_json.append({
        'test_example_index': test_example_index,
        'success': success,
        'inputs': decoded_inputs,
        'outputs': decoded_outputs,
        'ground_truth': str(ground_truth),
        'ground_truth_length': ground_truth_length,
        'solution': str(solution_program),
        'num_steps': step_index,
    })

  # Compute overall metrics and write to tensorboard.
  num_success = sum(successes)
  total = len(successes)
  num_failure = total - num_success
  assert num_success == metric_a + metric_b + metric_c
  assert num_failure == metric_e + metric_f
  assert (len(total_times) == len(num_steps) == len(num_ground_truth_steps)
          == total)

  if write_summary:
    with tf.io.gfile.GFile(result_json_path, 'w') as f:
      json.dump(result_json, f)

    bs = f'beam_size={beam_size}'
    summary_writer.scalar(f'raw/# success & no errors, {bs}',
                          metric_a, 0)
    summary_writer.scalar(f'raw/# success & SpecDecomposerModel error, {bs}',
                          metric_b, 0)
    summary_writer.scalar(f'raw/# success & SynthesizerModel error, {bs}',
                          metric_c, 0)
    summary_writer.scalar(f'raw/# failure & SpecDecomposerModel error, {bs}',
                          metric_e, 0)
    summary_writer.scalar(f'raw/# failure & SynthesizerModel error, {bs}',
                          metric_f, 0)

    summary_writer.scalar(f'main/total success rate, {bs}',
                          100 * num_success / total, 0)
    summary_writer.scalar(
        f'main/failures from SpecDecomposerModel, among all failures, {bs}',
        divide_no_nan(100 * metric_e, num_failure), 0)
    summary_writer.scalar(
        f'main/failures from SynthesizerModel, among all failures, {bs}',
        divide_no_nan(100 * metric_f, num_failure), 0)

    summary_writer.scalar(
        f'error_recovery/specDecomposerModel error recovery rate, {bs}',
        divide_no_nan(100 * metric_b, metric_b + metric_e), 0)
    summary_writer.scalar(
        f'error_recovery/synthesizerModel error recovery rate, {bs}',
        divide_no_nan(100 * metric_c, metric_c + metric_f), 0)
    summary_writer.scalar(
        f'error_recovery/error recovery rate, {bs}',
        divide_no_nan(100 * (metric_b + metric_c),
                      metric_b + metric_c + metric_e + metric_f), 0)
    summary_writer.scalar(
        f'error_recovery/recovered errors among successes, {bs}',
        divide_no_nan(100 * (metric_b + metric_c), num_success), 0)

    summary_writer.scalar(f'steps/avg. steps taken, {bs}',
                          statistics.mean(num_steps), 0)
    summary_writer.scalar(f'steps/avg. ground-truth steps, {bs}',
                          statistics.mean(num_ground_truth_steps), 0)
    summary_writer.scalar(
        (f'steps/success and (taken > ground truth steps), '
         f'among all successes, {bs}'),
        divide_no_nan(
            len([0 for taken, gt, success in zip(num_steps,
                                                 num_ground_truth_steps,
                                                 successes)
                 if taken > gt and success]) * 100,
            num_success), 0)
    summary_writer.scalar(
        (f'steps/success and (taken < ground truth steps), '
         f'among all successes, {bs}'),
        divide_no_nan(
            len([0 for taken, gt, success in zip(num_steps,
                                                 num_ground_truth_steps,
                                                 successes)
                 if taken < gt and success]) * 100,
            num_success), 0)
    summary_writer.scalar(
        (f'steps/failure and (taken > ground truth steps), '
         f'among all failures, {bs}'),
        divide_no_nan(
            len([0 for taken, gt, success in zip(num_steps,
                                                 num_ground_truth_steps,
                                                 successes)
                 if taken > gt and not success]) * 100,
            num_failure), 0)
    summary_writer.scalar(
        (f'steps/failure and (taken < ground truth steps), '
         f'among all failures, {bs}'),
        divide_no_nan(
            len([0 for taken, gt, success in zip(num_steps,
                                                 num_ground_truth_steps,
                                                 successes)
                 if taken < gt and not success]) * 100,
            num_failure), 0)

    summary_writer.scalar(f'time/total time per problem, {bs}',
                          statistics.mean(total_times), 0)
    if spec_prediction_times:
      summary_writer.scalar(f'time/per SpecDecomposerModel call, {bs}',
                            statistics.mean(spec_prediction_times), 0)
      summary_writer.scalar(f'time/per spec processing, {bs}',
                            statistics.mean(spec_processing_times), 0)
      summary_writer.scalar(f'time/per spec analysis, {bs}',
                            statistics.mean(spec_analysis_times), 0)
    summary_writer.scalar(f'time/per SynthesizerModel call, {bs}',
                          statistics.mean(synthesis_prediction_times), 0)
    summary_writer.scalar(f'time/per synthesis processing, {bs}',
                          statistics.mean(synthesis_processing_times), 0)
    summary_writer.scalar(f'time/per synthesis analysis, {bs}',
                          statistics.mean(synthesis_analysis_times), 0)

    summary_writer.flush()

if __name__ == '__main__':
  app.run(main)
