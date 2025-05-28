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

"""Defines `seqio.Task`s for DEDAL."""

import functools
from typing import Callable, Optional, Sequence

import gin
import seqio
import tensorflow as tf

from dedal.data import preprocessors


# Type aliases
DataSourceFn = Callable[[], seqio.DataSource]
NoiseMaskFn = preprocessors.NoiseMaskFn


@gin.configurable
def register_task(task):
  """Adds `task` to seqio.TaskRegistry, returning the task name."""
  if task.name in seqio.TaskRegistry.names():
    seqio.TaskRegistry.remove(task.name)
  seqio.TaskRegistry.add_provider(task.name, task)
  return task.name


@gin.configurable
def get_masked_lm_pretraining_task(
    task_name = gin.REQUIRED,
    source_fn = gin.REQUIRED,
    main_vocab_path = gin.REQUIRED,
    token_replace_vocab_path = gin.REQUIRED,
    num_reserved_tokens = 5,
    extra_ids = 95,
    noise_density = 0.15,
    mask_prob = 0.8,
    replace_prob = 0.1,
    noise_mask_fn = preprocessors.choose_k_noise_mask,
    exclude_reserved_tokens = True,
    passthrough_feature_keys = None,
    input_feature_key = 'inputs',
    target_feature_key = 'targets',
    noise_mask_feature_key = 'noise_mask',
):
  """Returns DEDAL's masked LM task as a `seqio.Task`."""
  main_vocabulary = seqio.SentencePieceVocabulary(
      main_vocab_path, extra_ids=extra_ids)
  replace_vocabulary = seqio.SentencePieceVocabulary(token_replace_vocab_path)

  return seqio.Task(
      name=task_name,
      source=source_fn(),
      output_features={
          input_feature_key: seqio.Feature(
              vocabulary=main_vocabulary,
              add_eos=True,
              dtype=tf.int32),
          target_feature_key: seqio.Feature(
              vocabulary=main_vocabulary,
              add_eos=True,
              dtype=tf.int32),
          noise_mask_feature_key: seqio.Feature(
              vocabulary=seqio.PassThroughVocabulary(size=3),
              add_eos=False,
              dtype=tf.int32),
      },
      preprocessors=[
          functools.partial(
              seqio.preprocessors.rekey,
              key_map={target_feature_key: 'sequence'}),
          functools.partial(
              seqio.preprocessors.tokenize,
              copy_pretokenized=False,
              with_eos=True),
          functools.partial(
              preprocessors.random_crop,
              feature_key=target_feature_key,
              add_eos=False),
          functools.partial(
              preprocessors.bert_denoising,
              noise_density=noise_density,
              mask_prob=mask_prob,
              replace_prob=replace_prob,
              noise_mask_fn=noise_mask_fn,
              exclude_reserved_tokens=exclude_reserved_tokens,
              num_reserved_tokens=num_reserved_tokens,
              replace_vocabulary=replace_vocabulary,
              passthrough_feature_keys=passthrough_feature_keys,
              input_feature_key=input_feature_key,
              target_feature_key=target_feature_key,
              noise_mask_feature_key=noise_mask_feature_key),
      ],
      metric_fns=[])


@gin.configurable
def get_dedal_alignment_task(
    task_name,
    source_fn = gin.REQUIRED,
    main_vocab_path = gin.REQUIRED,
    align_path_vocab_path = gin.REQUIRED,
):
  """Returns DEDAL's pairwise sequence alignment task as a `seqio.Task`."""
  main_vocabulary = seqio.SentencePieceVocabulary(main_vocab_path)
  align_path_vocabulary = seqio.SentencePieceVocabulary(align_path_vocab_path)

  return seqio.Task(
      name=task_name,
      source=source_fn(),
      output_features={
          'sequence_x': seqio.Feature(
              vocabulary=main_vocabulary,
              add_eos=True,
              dtype=tf.int32),
          'sequence_y': seqio.Feature(
              vocabulary=main_vocabulary,
              add_eos=True,
              dtype=tf.int32),
          'states': seqio.Feature(
              vocabulary=align_path_vocabulary,
              add_eos=False,
              dtype=tf.int32,
              required=False),
      },
      preprocessors=[
          functools.partial(
              preprocessors.cast_from_string,
              key_to_dtype_map={
                  'ali_start_x': tf.int32,
                  'ali_start_y': tf.int32,
                  'bos_x': tf.bool,
                  'bos_y': tf.bool,
                  'eos_x': tf.bool,
                  'eos_y': tf.bool,
                  'percent_identity': tf.float32,
                  'maybe_confounded': tf.bool,
                  'fallback': tf.bool,
                  'shares_clans_in_flanks': tf.bool,
                  'shares_coiled_coil_in_flanks': tf.bool,
                  'shares_disorder_in_flanks': tf.bool,
                  'shares_low_complexity_in_flanks': tf.bool,
                  'shares_sig_p_in_flanks': tf.bool,
                  'shares_transmembrane_in_flanks': tf.bool,
              },
          ),
          functools.partial(
              preprocessors.filter_by_bool_key,
              key='maybe_confounded',
              negate_predicate=True,
          ),
          preprocessors.uncompress_states,
          seqio.preprocessors.tokenize,
          functools.partial(
              preprocessors.maybe_append_eos,
              sequence_key='sequence_x',
              flag_key='eos_x',
          ),
          functools.partial(
              preprocessors.maybe_append_eos,
              sequence_key='sequence_y',
              flag_key='eos_y',
          ),
      ],
      metric_fns=[])


def get_dedal_homology_task(
    task_name,
    source_fn = gin.REQUIRED,
    main_vocab_path = gin.REQUIRED,
):
  """Returns DEDAL's pairwise homology detection task as a `seqio.Task`."""
  main_vocabulary = seqio.SentencePieceVocabulary(main_vocab_path)

  return seqio.Task(
      name=task_name,
      source=source_fn(),
      output_features={
          'sequence_x': seqio.Feature(
              vocabulary=main_vocabulary,
              add_eos=True,
              dtype=tf.int32),
          'sequence_y': seqio.Feature(
              vocabulary=main_vocabulary,
              add_eos=True,
              dtype=tf.int32),
      },
      preprocessors=[
          functools.partial(
              preprocessors.cast_from_string,
              key_to_dtype_map={
                  'bos_x': tf.bool,
                  'bos_y': tf.bool,
                  'eos_x': tf.bool,
                  'eos_y': tf.bool,
                  'homology_label': tf.int32,
                  'percent_identity': tf.float32,
                  'maybe_confounded': tf.bool,
                  'shares_clans': tf.bool,
                  'shares_coiled_coil': tf.bool,
                  'shares_disorder': tf.bool,
                  'shares_low_complexity': tf.bool,
                  'shares_sig_p': tf.bool,
                  'shares_transmembrane': tf.bool,
              },
          ),
          functools.partial(
              preprocessors.filter_by_bool_key,
              key='maybe_confounded',
              negate_predicate=True,
          ),
          seqio.preprocessors.tokenize,
          functools.partial(
              preprocessors.maybe_append_eos,
              sequence_key='sequence_x',
              flag_key='eos_x',
          ),
          functools.partial(
              preprocessors.maybe_append_eos,
              sequence_key='sequence_y',
              flag_key='eos_y',
          ),
      ],
      metric_fns=[])
