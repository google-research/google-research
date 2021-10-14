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

# pylint:disable=line-too-long
r"""For a given dataset, prep dataset and run sklearn eval.

This file has two modes:
1) Map from tf.Examples of audio to tf.Examples of embeddings.
2) Map from TFDS dataseet to tf.Examples of embeddings.

"""
# pylint:enable=line-too-long

from typing import Any, Dict, List, Sequence
from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
import tensorflow as tf


# Gets flags from data_prep's main.
from non_semantic_speech_benchmark.data_prep import audio_to_embeddings_beam_main as data_prep  # pylint:disable=unused-import
from non_semantic_speech_benchmark.data_prep import audio_to_embeddings_beam_utils as old_prep_utils
from non_semantic_speech_benchmark.data_prep import data_prep_utils as new_prep_utils
from non_semantic_speech_benchmark.eval_embedding.sklearn import train_and_eval_sklearn as sklearn_utils

# Flags needed for data prep. Data prep flags are imported directly from the
# main data prep beam file, but we need some custom behavior. XOR with
# tfds_dataset.
flags.DEFINE_string('train_input_glob', None, 'Glob for training data.')
flags.DEFINE_string('validation_input_glob', None, 'Glob for validation data.')
flags.DEFINE_string('test_input_glob', None, 'Glob for test data.')
flags.DEFINE_bool('skip_existing_error', False, 'Skip existing errors.')
flags.DEFINE_enum('data_prep_behavior', 'many_models', [
    'many_models', 'many_embeddings_single_model', 'chunked_audio'],
                  'Which metric to compute and report.')
# Extra data prep flags, needed for `many_embeddings_single_model` and
# `chunked_audio`.
flags.DEFINE_integer('chunk_len', None, 'Optional chunk len')
# Extra data prep flags, needed just for `many_embeddings_single_model`.
flags.DEFINE_integer(
    'embedding_length', None,
    'Expected length of the embedding. If present, must be this length.')


# Flags needed for sklearn eval.
flags.DEFINE_string('results_output_file', None, 'Output filename.')
flags.DEFINE_string('save_model_dir', None,
                    'If not `None`, write sklearn models to this directory.')
flags.DEFINE_string('save_predictions_dir', None,
                    'If not `None`, write numpy array of predictions on '
                    'train, eval, and test into this directory.')
flags.DEFINE_list('label_list', None, 'Python list of possible label values.')
flags.DEFINE_enum('eval_metric', 'accuracy', [
    'accuracy', 'balanced_accuracy', 'equal_error_rate',
    'unweighted_average_recall', 'auc'
], 'Which metric to compute and report.')

FLAGS = flags.FLAGS


def main(unused_argv):

  # Data prep setup.
  run_data_prep = True
  if FLAGS.train_input_glob:  # Explicitly pass globs.
    assert FLAGS.validation_input_glob
    assert FLAGS.test_input_glob
    input_filenames_list, output_filenames = [], []
    for input_glob in [
        FLAGS.train_input_glob, FLAGS.validation_input_glob,
        FLAGS.test_input_glob]:
      FLAGS.input_glob = input_glob
      cur_inputs, cur_outputs, prep_params = old_prep_utils.get_beam_params_from_flags(
      )
      input_filenames_list.extend(cur_inputs)
      output_filenames.extend(cur_outputs)
  else:  # Get params from a TFDS dataset.
    assert FLAGS.tfds_dataset
    input_filenames_list, output_filenames, prep_params = old_prep_utils.get_beam_params_from_flags(
    )
  assert input_filenames_list, input_filenames_list
  assert output_filenames, output_filenames
  try:
    # Check that inputs and flags are formatted correctly.
    old_prep_utils.validate_inputs(
        input_filenames_list, output_filenames,
        prep_params['embedding_modules'], prep_params['embedding_names'],
        prep_params['module_output_keys'])
  except ValueError:
    if FLAGS.skip_existing_error:
      run_data_prep = False
    else:
      raise
  logging.info('beam_params: %s', prep_params)

  # Generate sklearn eval experiment parameters based on data prep flags.
  if len(output_filenames) != 3:
    raise ValueError(f'Data prep output must be 3 files: {output_filenames}')
  # Make them globs.
  train_glob, eval_glob, test_glob = [f'{x}*' for x in output_filenames]
  sklearn_results_output_file = FLAGS.results_output_file
  exp_params = sklearn_utils.experiment_params(
      embedding_list=prep_params['embedding_names'],
      speaker_id_name=FLAGS.speaker_id_key,
      label_name=FLAGS.label_key,
      label_list=FLAGS.label_list,
      train_glob=train_glob,
      eval_glob=eval_glob,
      test_glob=test_glob,
      save_model_dir=FLAGS.save_model_dir,
      save_predictions_dir=FLAGS.save_predictions_dir,
      eval_metric=FLAGS.eval_metric,
  )
  logging.info('exp_params: %s', exp_params)

  # Make and run beam pipeline.
  beam_options = None

  if run_data_prep:
    logging.info('Data prep on: %s, %s...', input_filenames_list,
                 output_filenames)
    with beam.Pipeline(beam_options) as root:
      for i, (input_filenames_or_glob, output_filename) in enumerate(
          zip(input_filenames_list, output_filenames)):
        _data_prep(root, input_filenames_or_glob, output_filename, prep_params,
                   str(i), FLAGS.data_prep_behavior)

  # Check that previous beam pipeline wrote outputs.
  sklearn_utils.validate_flags(train_glob, eval_glob, test_glob,
                               sklearn_results_output_file)
  logging.info('Eval sklearn...')
  with beam.Pipeline(beam_options) as root:
    _ = (
        root
        | 'MakeCollection' >> beam.Create(exp_params)
        | 'CalcScores' >> beam.Map(
            lambda d: (d, sklearn_utils.train_and_get_score(**d)))
        | 'FormatText' >> beam.Map(sklearn_utils.format_text_line)
        | 'Reshuffle' >> beam.Reshuffle()
        | 'WriteOutput' >> beam.io.WriteToText(
            sklearn_results_output_file, num_shards=1))


def _data_prep(
    root,
    input_filenames_or_glob,
    output_filename,
    beam_params,
    suffix,
    data_prep_behavior,
    ):
  """Set up beam data prep pipeline based on `data_prep_behavior`."""
  if data_prep_behavior == 'many_models':
    old_prep_utils.make_beam_pipeline(
        root,
        input_filenames=input_filenames_or_glob,
        output_filename=output_filename,
        suffix=suffix,
        **beam_params)
  elif data_prep_behavior == 'many_embeddings_single_model':
    new_prep_utils.multiple_embeddings_from_single_model_pipeline(
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
        suffix=suffix)
  elif data_prep_behavior == 'chunked_audio':
    new_prep_utils.precompute_chunked_audio_pipeline(
        root,
        input_filenames=input_filenames_or_glob,
        sample_rate=beam_params['sample_rate'],
        debug=FLAGS.debug,
        embedding_names=beam_params['embedding_names'],
        embedding_modules=beam_params['embedding_modules'],
        module_output_keys=beam_params['module_output_keys'],
        audio_key=beam_params['audio_key'],
        sample_rate_key=beam_params['sample_rate_key'],
        label_key=beam_params['label_key'],
        speaker_id_key=beam_params['speaker_id_key'],
        average_over_time=beam_params['average_over_time'],
        delete_audio_from_output=beam_params['delete_audio_from_output'],
        output_filename=output_filename,
        chunk_len=FLAGS.chunk_len,
        embedding_length=FLAGS.embedding_length,
        input_format=beam_params['input_format'],
        output_format=beam_params['output_format'],
        suffix=suffix)
  else:
    raise ValueError(
        f'data_prep_behavior not recognized: {data_prep_behavior}')

if __name__ == '__main__':
  # From data prep.
  flags.mark_flags_as_required([
      'output_filename',
      'embedding_names',
      'embedding_modules',
      'module_output_keys',
      'audio_key',
  ])
  flags.mark_flags_as_mutual_exclusive(['train_input_glob', 'tfds_dataset'],
                                       required=True)
  flags.mark_flags_as_mutual_exclusive(
      ['validation_input_glob', 'tfds_dataset'], required=True)
  flags.mark_flags_as_mutual_exclusive(['test_input_glob', 'tfds_dataset'],
                                       required=True)
  flags.mark_flags_as_mutual_exclusive(
      ['tfds_dataset', 'sample_rate_key', 'sample_rate'], required=True)
  assert tf.executing_eagerly()

  # From sklearn eval.
  flags.mark_flags_as_required([
      'label_key',
      'label_list',
      'results_output_file',
  ])
  app.run(main)
