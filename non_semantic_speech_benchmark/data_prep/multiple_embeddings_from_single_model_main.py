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
# pylint:disable=line-too-long
"""Beam job to map to tf.Examples of embeddings."""
# pylint:enable=line-too-long

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
import tensorflow as tf
# Import from main to force ourselves to use the same flags.
from non_semantic_speech_benchmark.data_prep import audio_to_embeddings_beam_main  # pylint:disable=unused-import
from non_semantic_speech_benchmark.data_prep import audio_to_embeddings_beam_utils as old_utils
from non_semantic_speech_benchmark.data_prep import data_prep_utils

flags.DEFINE_integer('chunk_len', None, 'Optional chunk len')
flags.DEFINE_integer(
    'embedding_length', None,
    'Expected length of the embedding. If present, must be this length.')

FLAGS = flags.FLAGS


def main(unused_argv):
  runner.program_started()

  # Get input data location from flags. If we're reading a TFDS dataset, get
  # train, validation, and test.
  input_filenames_list, output_filenames, beam_params = old_utils.get_beam_params_from_flags(
  )
  # Check that inputs and flags are formatted correctly.
  old_utils.validate_inputs(
      input_filenames_list=input_filenames_list,
      output_filenames=output_filenames,
      embedding_modules=beam_params['embedding_modules'],
      embedding_names=beam_params['embedding_names'],
      module_output_keys=beam_params['module_output_keys'])

  # If you have custom beam options, add them here.
  beam_options = None

  logging.info('Starting to create flume pipeline...')
  with beam.Pipeline(beam_options) as root:
    for i, (input_filenames_or_glob, output_filename) in enumerate(
        zip(input_filenames_list, output_filenames)):
      data_prep_utils.multiple_embeddings_from_single_model_pipeline(
          root,
          input_filenames=input_filenames_or_glob,
          sample_rate=beam_params['sample_rate'],
          debug=FLAGS.debug,
          embedding_names=beam_params['embedding_names'],
          embedding_modules=beam_params['embedding_modules'],
          module_output_keys=beam_params['module_output_keys'],
          sample_rate_key=beam_params['sample_rate_key'],
          audio_key=beam_params['audio_key'],
          label_key=beam_params['label_key'],
          speaker_id_key=beam_params['speaker_id_key'],
          average_over_time=beam_params['average_over_time'],
          delete_audio_from_output=beam_params['delete_audio_from_output'],
          output_filename=output_filename,
          chunk_len=FLAGS.chunk_len,
          embedding_length=FLAGS.embedding_length,
          input_format=beam_params['input_format'],
          output_format=beam_params['output_format'],
          suffix=str(i))


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'output_filename', 'embedding_names', 'embedding_modules',
      'module_output_keys', 'audio_key', 'label_key',
  ])
  flags.mark_flags_as_mutual_exclusive(['input_glob', 'tfds_dataset'],
                                       required=True)
  flags.mark_flags_as_mutual_exclusive(
      ['tfds_dataset', 'sample_rate_key', 'sample_rate'], required=True)
  tf.compat.v2.enable_v2_behavior()
  assert tf.executing_eagerly()
  app.run(main)
