# coding=utf-8
# Copyright 2020 The Google Research Authors.
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
# pylint:disable=line-too-long
r"""Beam job to map to tf.Examples of embeddings.

This file has two modes:
1) Map from tf.Examples of audio to tf.Examples of embeddings.
2) Map from TFDS dataseet to tf.Examples of embeddings.

"""
# pylint:enable=line-too-long

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
from non_semantic_speech_benchmark.data_prep import audio_to_embeddings_beam_utils

flags.DEFINE_string('input_glob', None,
                    'Glob for input dir. XOR with `tfds_data`.')
flags.DEFINE_string(
    'tfds_dataset', None, 'Name of TFDS dataset. '
    'XOR with `input_glob`. Should be of the form ex "cifar".'
    'Exactly one of `sample_rate_key`, `sample_rate`, or '
    '`tfds_dataset` must be not None.')

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
flags.DEFINE_string('audio_key', None, 'Key of audio.')
flags.DEFINE_string(
    'sample_rate_key', None, 'Key of sample rate. '
    'Exactly one of `sample_rate_key`, `sample_rate`, or '
    '`tfds_dataset` must be not None.')
flags.DEFINE_integer(
    'sample_rate', None, 'Sample rate.'
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
    'If true, remove audio from the output table. Can be '
    'helpful in keeping output tables small.')
flags.DEFINE_bool('debug', False, 'If True, run in debug model.')

FLAGS = flags.FLAGS


def main(unused_argv):

  # Get input data location from flags. If we're reading a TFDS dataset, get
  # train, validation, and test.
  input_filenames_list, output_filenames, sample_rate = audio_to_embeddings_beam_utils.read_input_glob_and_sample_rate_from_flags(
      FLAGS.input_glob, FLAGS.sample_rate, FLAGS.tfds_dataset,
      FLAGS.output_filename)

  # Check that inputs and flags are formatted correctly.
  audio_to_embeddings_beam_utils.validate_inputs(input_filenames_list,
                                                 output_filenames,
                                                 FLAGS.embedding_modules,
                                                 FLAGS.embedding_names,
                                                 FLAGS.module_output_keys)

  input_format = 'tfrecord'
  output_format = 'tfrecord'

  # If you have custom beam options, add them here.
  beam_options = None

  logging.info('Starting to create flume pipeline...')
  with beam.Pipeline(beam_options) as root:
    for i, (input_filenames_or_glob, output_filename) in enumerate(
        zip(input_filenames_list, output_filenames)):
      audio_to_embeddings_beam_utils.make_beam_pipeline(
          root,
          input_filenames_or_glob,
          sample_rate,
          FLAGS.debug,
          FLAGS.embedding_names,
          FLAGS.embedding_modules,
          FLAGS.module_output_keys,
          FLAGS.audio_key,
          FLAGS.sample_rate_key,
          FLAGS.label_key,
          FLAGS.speaker_id_key,
          FLAGS.average_over_time,
          FLAGS.delete_audio_from_output,
          output_filename,
          input_format=input_format,
          output_format=output_format,
          suffix=i)


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'output_filename', 'embedding_names', 'embedding_modules',
      'module_output_keys', 'audio_key', 'label_key'
  ])
  flags.mark_flags_as_mutual_exclusive(['input_glob', 'tfds_dataset'],
                                       required=True)
  flags.mark_flags_as_mutual_exclusive(
      ['tfds_dataset', 'sample_rate_key', 'sample_rate'], required=True)
  app.run(main)
