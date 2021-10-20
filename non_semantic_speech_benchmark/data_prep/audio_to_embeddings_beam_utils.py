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
r"""Construct a beam pipeline to map from audio to embeddings.

This file has two modes:
1) Map from tf.Examples of audio to tf.Examples of embeddings.
2) Map from TFDS dataseet to tf.Examples of embeddings.

It supports using a tf.hub module OR a TFLite model file to generate embeddings.
TFLite file should have the `.tflite` extension.
"""

import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from absl import flags
from absl import logging
import apache_beam as beam

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from non_semantic_speech_benchmark.data_prep import beam_dofns
from non_semantic_speech_benchmark.data_prep import data_prep_utils as utils

FLAGS = flags.FLAGS


def data_prep_pipeline(
    root,
    input_filenames_or_glob,
    output_filename,
    data_prep_behavior,
    beam_params,
    suffix,
    ):
  """Set up beam data prep pipeline based on `data_prep_behavior`."""
  if data_prep_behavior == 'many_models':
    make_many_models_beam_pipeline(
        root,
        input_filenames=input_filenames_or_glob,
        output_filename=output_filename,
        suffix=suffix,
        **beam_params)
  elif data_prep_behavior == 'many_embeddings_single_model':
    multiple_embeddings_from_single_model_pipeline(
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
    precompute_chunked_audio_pipeline(
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


def make_many_models_beam_pipeline(
    root,
    input_filenames,
    sample_rate,
    debug,
    embedding_names,
    embedding_modules,
    module_output_keys,
    audio_key,
    sample_rate_key,
    label_key,
    speaker_id_key,
    average_over_time,
    delete_audio_from_output,
    output_filename,
    split_embeddings_into_separate_tables = False,
    use_frontend_fn = False,
    normalize_to_pm_one = True,
    model_input_min_length = None,
    input_format = 'tfrecord',
    output_format = 'tfrecord',
    suffix = 'Main',
    module_call_fn = utils.samples_to_embedding_tfhub,
    setup_fn = hub.load):
  """Construct beam pipeline for mapping from audio to embeddings.

  Args:
    root: The beam root node.
    input_filenames: Python list. List of input files.
    sample_rate: Python int, or `None`. The sample rate for all embeddings, or
      `None` if this is a TFDS dataset, or if each example has its own sample
      rate.
    debug: Python bool. Whether to operate in debug mode.
    embedding_names: Python list of embeddings.
    embedding_modules: Python list of TF-Hub modules.
    module_output_keys: Python list of strings, names of output modules.
    audio_key: Python string, the key of the audio.
    sample_rate_key: Python string or `None`, the key for.
    label_key: Python string. Field for label.
    speaker_id_key: Python string or `None`. Key for speaker ID, or `None`.
    average_over_time: Python bool. If `True`, average over the time axis.
    delete_audio_from_output: Python bool. Whether to remove audio fromm
      outputs.
    output_filename: Python string. Output filename.
    split_embeddings_into_separate_tables: Python bool. If true, write each
      embedding to a separate table.
    use_frontend_fn: If `true`, call frontend fn on audio before passing to the
      model.
    normalize_to_pm_one: Whether to normalize input to +- 1 before passing to
      model.
    model_input_min_length: Min length to the model, or `None`. 0-pad inputs to
      this length, if necessary. Note that frontends usually contain their own
      length logic, unless the model is in TFLite format.
    input_format: Python string. Must correspond to a function in
      `reader_functions`.
    output_format: Python string. Must correspond to a function
      `writer_functions`.
    suffix: Python string. Suffix to stage names to make them unique.
    module_call_fn: Function for inference on audio.
    setup_fn: Function for creating audio inference model.
  """
  tf_examples_key_ = 'tf_examples'
  if tf_examples_key_ in embedding_names:
    raise ValueError(
        f'"{tf_examples_key_}" is reserved, cannot be embedding name.')
  s = suffix  # for code brevity.

  # Read from input.
  input_examples = _common_pipeline_beginning(
      root, input_format, input_filenames, s, debug)

  # Compute all the embeddings simultaneously.
  embedding_tables = {}
  for name, mod, out_key in zip(
      embedding_names, embedding_modules, module_output_keys):
    logging.info('Adding signal: %s %s, %s', name, mod, out_key)
    tbl = input_examples | f'ComputeEmbedding-{name}-{s}' >> beam.ParDo(
        beam_dofns.ComputeEmbeddingMapFn(
            name=name,
            module=mod,
            output_key=out_key,
            audio_key=audio_key,
            sample_rate_key=sample_rate_key,
            sample_rate=sample_rate,
            average_over_time=average_over_time,
            feature_fn=(utils.default_feature_fn if use_frontend_fn
                        else None),
            normalize_to_pm_one=normalize_to_pm_one,
            model_input_min_length=model_input_min_length,
            module_call_fn=module_call_fn,
            setup_fn=setup_fn))
    embedding_tables[name] = tbl
  assert tf_examples_key_ not in embedding_tables
  embedding_tables[tf_examples_key_] = input_examples
  logging.info('embedding_tables: %s', embedding_tables)

  # Either write to one table with all embeddings, or one table per embedding.
  if split_embeddings_into_separate_tables:
    output_table_dicts = [
        (k, {k: v, tf_examples_key_: input_examples}) for
        k, v in embedding_tables.items() if k != tf_examples_key_]
  else:
    output_table_dicts = [('all', embedding_tables)]

  # Combine embeddings and tf.train.Example, using the common key.
  writer_function = utils.writer_functions[output_format]
  for name, embedding_tables in output_table_dicts:
    if split_embeddings_into_separate_tables:
      cur_s = f'{name}-{s}'
      # Add `name` as a subdir.
      dirname, basename = os.path.split(output_filename)
      cur_output_filename = os.path.join(dirname, name, f'{basename}@*')
    else:
      cur_s = s
      cur_output_filename = f'{output_filename}@*'
    combined_tbl = (
        embedding_tables
        | f'CombineEmbeddingTables-{cur_s}' >> beam.CoGroupByKey()
        | f'AddEmbeddings-{cur_s}' >> beam.Map(
            utils.add_embeddings_to_tfex,
            original_example_key=tf_examples_key_,
            delete_audio_from_output=delete_audio_from_output,
            audio_key=audio_key,
            label_key=label_key,
            speaker_id_key=speaker_id_key))
    logging.info('Writing to %s', cur_output_filename)
    writer_function(combined_tbl, cur_output_filename, cur_s)


def multiple_embeddings_from_single_model_pipeline(
    root,
    input_filenames,
    sample_rate,
    debug,
    embedding_names,
    embedding_modules,
    module_output_keys,
    audio_key,
    sample_rate_key,
    label_key,
    speaker_id_key,
    average_over_time,
    delete_audio_from_output,
    output_filename,
    embedding_length = None,
    chunk_len = None,
    input_format = 'tfrecord',
    output_format = 'tfrecord',
    suffix = 'Main',
    setup_fn = hub.load):
  """Construct beam pipeline for mapping from audio to embeddings.

  Args:
    root: The beam root node.
    input_filenames: Python list. List of input files.
    sample_rate: Python int, or `None`. The sample rate for all embeddings, or
      `None` if this is a TFDS dataset, or if each example has its own sample
      rate.
    debug: Python bool. Whether to operate in debug mode.
    embedding_names: Python list of embeddings.
    embedding_modules: Python list of TF-Hub modules.
    module_output_keys: Python list of strings, names of output modules.
    audio_key: Python string, the key of the audio.
    sample_rate_key: Python string or `None`, the key for.
    label_key: Python string. Field for label.
    speaker_id_key: Python string or `None`. Key for speaker ID, or `None`.
    average_over_time: Python bool. If `True`, average over the time axis.
    delete_audio_from_output: Python bool. Whether to remove audio fromm
      outputs.
    output_filename: Python string. Output filename.
    embedding_length: None.
    chunk_len: Stuff
    input_format: Python string. Must correspond to a function in
      `reader_functions`.
    output_format: Python string. Must correspond to a function in
      `writer_functions`.
    suffix: Python string. Suffix to stage names to make them unique.
    setup_fn: Stuff.
  """
  # Common sanity checks and preprocessing.
  _common_pipeline_sanity_checks(
      embedding_modules, embedding_names, module_output_keys)
  input_examples = _common_pipeline_beginning(
      root, input_format, input_filenames, suffix, debug)
  s = suffix
  embedding_module = embedding_modules[0]

  # Compute all the embeddings simultaneously.
  logging.info('Adding all signals: %s', module_output_keys)
  tbl = (
      input_examples
      | f'Reshuffle1-{s}' >> beam.Reshuffle()
      | f'ComputeEmbedding-{s}' >> beam.ParDo(
          beam_dofns.ComputeMultipleEmbeddingsFromSingleModel(
              name='all',
              module=embedding_module,
              output_key=module_output_keys,
              audio_key=audio_key,
              sample_rate_key=sample_rate_key,
              sample_rate=sample_rate,
              average_over_time=average_over_time,
              feature_fn=None,
              embedding_names=embedding_names,
              embedding_length=embedding_length,
              chunk_len=chunk_len,
              setup_fn=setup_fn))
      | f'Reshuffle2-{s}' >> beam.Reshuffle()
      | f'ToTFExample-{s}' >> beam.Map(
          utils.combine_multiple_embeddings_to_tfex,
          delete_audio_from_output=delete_audio_from_output,
          audio_key=audio_key,
          label_key=label_key,
          speaker_id_key=speaker_id_key)
      | f'Reshuffle3-{s}' >> beam.Reshuffle())
  # Write embeddings to disk.
  writer_function = utils.writer_functions[output_format]
  cur_output_filename = f'{output_filename}@*'
  logging.info('Writing to %s', cur_output_filename)
  writer_function(tbl, cur_output_filename, s)


def precompute_chunked_audio_pipeline(root,
                                      input_filenames,
                                      sample_rate,
                                      debug,
                                      embedding_names,
                                      embedding_modules,
                                      module_output_keys,
                                      audio_key,
                                      sample_rate_key,
                                      output_filename,
                                      average_over_time = True,
                                      delete_audio_from_output = True,
                                      label_key = None,
                                      speaker_id_key = None,
                                      chunk_len = None,
                                      embedding_length = 1024,
                                      input_format = 'tfrecord',
                                      output_format = 'tfrecord',
                                      suffix = 'Main',
                                      setup_fn = hub.load):
  """Construct beam pipeline for mapping from audio to embeddings.

  Args:
    root: The beam root node.
    input_filenames: Python list. List of input files.
    sample_rate: Python int, or `None`. The sample rate for all embeddings, or
      `None` if this is a TFDS dataset, or if each example has its own sample
      rate.
    debug: Python bool. Whether to operate in debug mode.
    embedding_names: Python list of embeddings.
    embedding_modules: Python list of TF-Hub modules.
    module_output_keys: Python list of strings, names of output modules.
    audio_key: Python string, the key of the audio.
    sample_rate_key: Python string or `None`, the key for.
    output_filename: Python string. Output filename.
    average_over_time: Whether to average over time.
    delete_audio_from_output: Whether to remove audio.
    label_key: Python string. Field for label.
    speaker_id_key: Python string. Field for speaker id.
    chunk_len: stuff
    embedding_length: Length of embedding.
    input_format: Python string. Must correspond to a function in
      `reader_functions`.
    output_format: Python string. Must correspond to a function
      `writer_functions`.
    suffix: Python string. Suffix to stage names to make them unique.
    setup_fn: Stuff.
  """
  # Common sanity checks and preprocessing.
  _common_pipeline_sanity_checks(
      embedding_modules, embedding_names, module_output_keys)
  input_examples = _common_pipeline_beginning(
      root, input_format, input_filenames, suffix, debug)
  s = suffix
  embedding_module = embedding_modules[0]

  # Compute all the embeddings simultaneously.
  logging.info('Adding all signals: %s', module_output_keys)
  tbl = (
      input_examples
      | f'Reshuffle1-{s}' >> beam.Reshuffle()
      | f'ComputeEmbedding-{s}' >> beam.ParDo(
          beam_dofns.ChunkAudioAndComputeEmbeddings(
              name='all',
              module=embedding_module,
              output_key=module_output_keys,
              embedding_names=embedding_names,
              audio_key=audio_key,
              label_key=label_key,
              speaker_id_key=speaker_id_key,
              sample_rate_key=sample_rate_key,
              sample_rate=sample_rate,
              average_over_time=average_over_time,
              chunk_len=chunk_len,
              setup_fn=setup_fn))
      | f'Reshuffle2-{s}' >> beam.Reshuffle()
      | f'ToTFExample-{s}' >> beam.Map(
          utils.chunked_audio_to_tfex,
          delete_audio_from_output=delete_audio_from_output,
          chunk_len=chunk_len,
          speaker_id_key=speaker_id_key,
          embedding_length=embedding_length)
      | f'Reshuffle3-{s}' >> beam.Reshuffle())
  # Write embeddings to disk.
  writer_function = utils.writer_functions[output_format]
  cur_output_filename = f'{output_filename}@*'
  logging.info('Writing to %s', cur_output_filename)
  writer_function(tbl, cur_output_filename, s)


def _common_pipeline_sanity_checks(
    embedding_modules,
    embedding_names,
    module_output_keys):
  """Common sanity check for beam pipelines."""
  if len(set(embedding_modules)) != 1:
    raise ValueError(f'Too many modules: {set(embedding_modules)}')
  embedding_module = embedding_modules[0]
  if len(embedding_names) != len(module_output_keys):
    raise ValueError(f'Lens not the same: {len(embedding_names)} vs '
                     f'{len(module_output_keys)}')


def _common_pipeline_beginning(
    root,
    input_format,
    input_filenames,
    s,
    debug):
  """Common input reading for beam pipelines."""
  # Read from input.
  input_examples = utils.reader_functions[input_format](
      root, input_filenames, s)

  # In debug mode, take one input example.
  if debug:
    input_examples = (
        input_examples
        | f'TakeOne{s}' >> beam.transforms.combiners.Sample.FixedSizeGlobally(1)
        # Sampling generates lists, so flatten back into one collection.
        | f'DebugFlatten{s}' >> beam.FlatMap(lambda x: x))

  return input_examples


def get_beam_params_from_flags(
):
  """Parses flags and returns arguments for beam job."""
  # Get input data location from flags. If we're reading a TFDS dataset, get
  # train, validation, and test.
  input_filenames_list, output_filenames, sample_rate = utils.read_input_glob_and_sample_rate_from_flags(
      FLAGS.input_glob, FLAGS.sample_rate, FLAGS.tfds_dataset,
      FLAGS.output_filename, FLAGS.tfds_data_dir)

  # Sometimes we want commas to appear in `embedding_modules`,
  # `embedding_names`, or `module_output_key`. However, commas get split out in
  # Google's Python `DEFINE_list`. We compromise by introducing a special
  # character, which we replace with commas here.
  embedding_modules = _maybe_add_commas(FLAGS.embedding_modules,
                                        FLAGS.comma_escape_char)
  embedding_names = _maybe_add_commas(FLAGS.embedding_names,
                                      FLAGS.comma_escape_char)
  module_output_keys = _maybe_add_commas(FLAGS.module_output_keys,
                                         FLAGS.comma_escape_char)

  input_format = 'tfrecord'
  output_format = 'tfrecord'

  # All modules should be tflite or not tflite.
  tflite = [x.endswith('.tflite') for x in embedding_modules]
  if not np.all(tflite) and np.any(tflite):
    raise ValueError(
        f'Modules must all be tflite, or none: {embedding_modules}')
  is_tflite = np.any(tflite)
  logging.info('is_tflite: %s', is_tflite)

  # pylint:disable=line-too-long
  beam_params = dict(
      sample_rate=sample_rate,
      debug=FLAGS.debug,
      embedding_names=embedding_names,
      embedding_modules=embedding_modules,
      module_output_keys=module_output_keys,
      audio_key=FLAGS.audio_key,
      sample_rate_key=FLAGS.sample_rate_key,
      label_key=FLAGS.label_key,
      speaker_id_key=FLAGS.speaker_id_key,
      average_over_time=FLAGS.average_over_time,
      delete_audio_from_output=FLAGS.delete_audio_from_output,
      split_embeddings_into_separate_tables=FLAGS.split_embeddings_into_separate_tables,
      use_frontend_fn=FLAGS.use_frontend_fn,
      normalize_to_pm_one=FLAGS.normalize_to_pm_one,
      model_input_min_length=FLAGS.model_input_min_length,
      input_format=input_format,
      output_format=output_format,
      module_call_fn=(utils.samples_to_embedding_tflite if is_tflite
                      else utils.samples_to_embedding_tfhub),
      setup_fn=utils.build_tflite_interpreter if is_tflite else hub.load,
  )
  # pylint:enable=line-too-long

  logging.info('input_filenames_list: %s', input_filenames_list)
  logging.info('output_filenames: %s', output_filenames)


  return input_filenames_list, output_filenames, beam_params


def _maybe_add_commas(list_obj, comma_escape_char):
  return [x.replace(comma_escape_char, ',') for x in list_obj]


def validate_inputs(input_filenames_list,
                    output_filenames, embedding_modules,
                    embedding_names, module_output_keys):
  """Validate inputs and input flags."""
  for filename_list in input_filenames_list:
    for filename in filename_list:
      # It's either a filename or a glob. Try both.
      try:
        if not tf.io.gfile.exists(filename):
          raise ValueError(f'Files not found: {filename}')
      except (tf.errors.InvalidArgumentError, ValueError):  # was a glob.
        if not tf.io.gfile.glob(filename):
          raise ValueError(f'Files not found: {filename}')

  if len(input_filenames_list) != len(output_filenames):
    raise ValueError('Input/output filename lengths don\'t match: '
                     f'{input_filenames_list} vs {output_filenames}')

  # Make sure output files don't already exist.
  for output_filename in output_filenames:
    if tf.io.gfile.glob(f'{output_filename}*'):
      raise ValueError(f'Output file already exists: {output_filename}')

  # Make sure output file names are unique.
  if len(output_filenames) != len(set(output_filenames)):
    raise ValueError(f'Some output files are repeated: {output_filenames}')

  # Lengths of flag lists must be the same.
  if len(embedding_names) != len(embedding_modules):
    raise ValueError(
        f'Lengths don\'t match: {embedding_names} vs {embedding_modules}')
  if len(embedding_modules) != len(module_output_keys):
    raise ValueError(
        f'Lengths don\'t match: {embedding_modules} vs {module_output_keys}')
  # Shortnames must be unique.
  if len(set(embedding_names)) != len(embedding_names):
    raise ValueError(f'Shortnames must be unique: {embedding_names}')

  # Create output directory if it doesn't already exist.
  for output_filename in output_filenames:
    output_dir = output_filename.rsplit('/', 1)[0]
    if not tf.io.gfile.exists(output_dir):
      tf.io.gfile.makedirs(output_dir)
