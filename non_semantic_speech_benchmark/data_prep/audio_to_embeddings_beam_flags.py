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

"""Flags for data prep.

A separate file to make them reusable.
"""

from absl import flags

flags.DEFINE_string('input_glob', None,
                    'Glob for input dir. XOR with `tfds_data`.')
flags.DEFINE_string(
    'tfds_dataset', None, 'Name of TFDS dataset. '
    'XOR with `input_glob`. Should be of the form ex "cifar".'
    'Exactly one of `sample_rate_key`, `sample_rate`, or '
    '`tfds_dataset` must be not None.')
flags.DEFINE_string(
    'tfds_data_dir', None,
    'An optional directory for the locally downloaded TFDS data. Should only '
    'be non-None when `tfds_dataset` is used. This is essential for data that '
    'needs to be manually downloaded.')
flags.DEFINE_string('output_filename', None, 'Output filename.')
flags.DEFINE_list(
    'embedding_names', None,
    'List of embedding module names. Used for logging, and as '
    'in the features key of the results tf.Example feature list.')
flags.DEFINE_list(
    'embedding_modules', None,
    'List of embedding modules to compute. Should be accepted '
    'by `hub.load`.`')
flags.DEFINE_list(
    'module_output_keys', None,
    'List of module output key. Must be the same length as '
    '`embedding_modules`.')
flags.DEFINE_enum('data_prep_behavior', 'many_models', [
    'many_models', 'many_embeddings_single_model', 'chunked_audio',
    'batched_single_model'
], 'Which metric to compute and report.')
# Extra data prep flags, needed for `many_embeddings_single_model` and
# `chunked_audio`.
flags.DEFINE_integer('chunk_len', None, 'Optional chunk len')
# Extra data prep flags, needed just for `many_embeddings_single_model`.
flags.DEFINE_integer(
    'embedding_length', None,
    'Expected length of the embedding. If present, must be this length.')
# Extra data prep flags, needed just for `chunked_audio`.
flags.DEFINE_bool(
    'compute_embeddings_on_chunked_audio', True,
    'Whether to compute targets on chunked audio or entire clip.')
# Extra data prep flags, needed just for ``.
flags.DEFINE_integer('batch_size', 1,
                     'Number of audio samples to compute embeddings at once.')

flags.DEFINE_string(
    'comma_escape_char', '?',
    'Sometimes we want commas to appear in `embedding_modules`, '
    '`embedding_names`, or `module_output_key`. However, commas get split out '
    'in Googles Python `DEFINE_list`. We compromise by introducing a special '
    'character, which we replace with commas.')
flags.DEFINE_string('audio_key', None, 'Key of audio.')
flags.DEFINE_integer(
    'sample_rate', None, 'Sample rate.'
    'Exactly one of `sample_rate_key`, `sample_rate`, or '
    '`tfds_dataset` must be not None.')
flags.DEFINE_string(
    'sample_rate_key', None, 'Key of sample rate. '
    'Exactly one of `sample_rate_key`, `sample_rate`, or '
    '`tfds_dataset` must be not None.')
flags.DEFINE_string(
    'label_key', None, 'Key for labels. If the feature value is an integer, '
    'convert to bytes.')
flags.DEFINE_string(
    'speaker_id_key', None,
    'Key for speaker_id, or `None`. If this flag is present, '
    'check that the key exists and is of type `bytes`.')
flags.DEFINE_bool('average_over_time', False,
                  'If true, return embeddings that are averaged over time.')
flags.DEFINE_bool(
    'delete_audio_from_output', True,
    'If true, remove audio from the output table. Can be helpful in keeping '
    'output tables small.')
flags.DEFINE_bool(
    'pass_through_normalized_audio', True,
    'When passing through audio, whether to pass through normalized audio.')
flags.DEFINE_bool(
    'split_embeddings_into_separate_tables', False,
    'If true, write each embedding to a separate table.')
flags.DEFINE_bool('debug', False, 'If True, run in debug model.')
# Do not use `use_frontend_fn` and `model_input_min_length > 0`.
flags.DEFINE_bool(
    'use_frontend_fn', False,
    'If `true`, call frontend fn on audio before passing to the model. Do not '
    'use if `model_input_min_length` is not `None`.')
flags.DEFINE_bool(
    'normalize_to_pm_one', True,
    'Whether to normalize input to +- 1 before passing to model.')
flags.DEFINE_integer(
    'model_input_min_length', None, 'Min length to the model. 0-pad inputs to '
    'this length, if necessary. Note that frontends usually contain their own '
    'length logic, unless the model is in TFLite format. Do not use if '
    '`use_frontend_fn` is `True`.')
