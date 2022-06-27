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

# pylint:disable=line-too-long
r"""Beam job to map to tf.Examples of embeddings.

This file has two modes:
1) Map from tf.Examples of audio to tf.Examples of embeddings.
2) Map from TFDS dataseet to tf.Examples of embeddings.

"""
# pylint:enable=line-too-long

from typing import Any, Dict

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
import tensorflow as tf
from non_semantic_speech_benchmark.data_prep import audio_to_embeddings_beam_flags  # pylint:disable=unused-import
from non_semantic_speech_benchmark.data_prep import audio_to_embeddings_beam_utils as utils

FLAGS = flags.FLAGS


def main(_):

  input_filenames_list, output_filenames, beam_params = utils.get_beam_params_from_flags(
  )
  # Check that inputs and flags are formatted correctly.
  utils.validate_inputs(
      input_filenames_list=input_filenames_list,
      output_filenames=output_filenames,
      embedding_modules=beam_params['embedding_modules'],
      embedding_names=beam_params['embedding_names'],
      module_output_keys=beam_params['module_output_keys'])
  logging.info('main: input_filenames_list: %s', input_filenames_list)
  logging.info('main: output_filenames: %s', output_filenames)
  logging.info('main: beam_params: %s', beam_params)

  # If you have custom beam options, add them here.
  beam_options = None

  logging.info('Starting to create flume pipeline...')
  with beam.Pipeline(beam_options) as root:
    for i, (input_filenames_or_glob, output_filename) in enumerate(
        zip(input_filenames_list, output_filenames)):
      utils.data_prep_pipeline(
          root=root,
          input_filenames_or_glob=input_filenames_or_glob,
          output_filename=output_filename,
          data_prep_behavior=FLAGS.data_prep_behavior,
          beam_params=beam_params,
          suffix=str(i))


@flags.multi_flags_validator(
    ['use_frontend_fn', 'model_input_min_length'],
    message='Use only one of `use_frontend_fn` and `model_input_min_length`.'
)
def no_min_input_length_with_frontend_fn(flags_dict):
  return (not flags_dict['use_frontend_fn'] or
          not flags_dict['model_input_min_length'])

if __name__ == '__main__':
  flags.mark_flags_as_required([
      'output_filename', 'embedding_names', 'embedding_modules',
      'module_output_keys', 'audio_key',
  ])
  flags.mark_flags_as_mutual_exclusive(['input_glob', 'tfds_dataset'],
                                       required=True)
  flags.mark_flags_as_mutual_exclusive(
      ['tfds_dataset', 'sample_rate_key', 'sample_rate'], required=True)
  tf.compat.v2.enable_v2_behavior()
  assert tf.executing_eagerly()
  app.run(main)
