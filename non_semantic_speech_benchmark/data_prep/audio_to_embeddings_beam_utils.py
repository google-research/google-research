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
r"""Construct a beam pipeline to map from audio to embeddings.

This file has two modes:
1) Map from tf.Examples of audio to tf.Examples of embeddings.
2) Map from TFDS dataseet to tf.Examples of embeddings.
"""

import copy
import numbers
import os
import random
import typing
from absl import logging
import apache_beam as beam
import librosa
import numpy as np
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
assert tf.executing_eagerly()
import tensorflow_datasets as tfds  # pylint: disable=g-import-not-at-top
import tensorflow_hub as hub
from non_semantic_speech_benchmark import file_utils


def _tfexample_audio_to_npfloat32(ex, audio_key):
  """Extract audio from tf.Example and convert it to np.float32."""
  audio_feats = ex.features.feature[audio_key]
  if audio_feats.int64_list.value:
    audio = np.array(audio_feats.int64_list.value)
    # Even though the data is in an int64 container, the data is actually int16.
    iinfo = np.iinfo(np.int16)
    assert np.logical_and(audio >= iinfo.min, audio <= iinfo.max).all(),\
        (np.min(audio), np.max(audio), iinfo.min, iinfo.max)
    audio = audio.astype(np.float32) / iinfo.max
  else:
    assert audio_feats.float_list.value
    audio = np.array(audio_feats.float_list.value, dtype=np.float32)
  return audio


def _samples_to_embedding(audio_samples, sample_rate, mod, output_key):
  """Run inference to map audio samples to an embedding."""
  tf_out = mod(tf.constant(audio_samples, tf.float32),
               tf.constant(sample_rate, tf.int32))
  return np.array(tf_out[output_key])


@beam.typehints.with_input_types(typing.Tuple[str, typing.Any])
@beam.typehints.with_output_types(typing.Tuple[str, typing.Any])
class ComputeEmbeddingMapFn(beam.DoFn):
  """Computes an embedding (key, tf.Example) from audio (key, tf.Example)."""

  def __init__(self, name, module, output_key, audio_key, sample_rate_key,
               sample_rate, average_over_time):
    self._name = name
    self._module = module
    self._output_key = output_key
    self._audio_key = audio_key
    self._sample_rate_key = sample_rate_key
    self._sample_rate = sample_rate
    self._average_over_time = average_over_time

  def setup(self):
    self.module = hub.load(self._module)

  def process(self, k_v):
    k, ex = k_v

    # Only one of `sample_rate_key` and `sample_rate` should be not None.
    assert (self._sample_rate_key is None) ^ (self._sample_rate is None),\
        (self._sample_rate_key, self._sample_rate)

    # Read the input example audio and assert input format sanity.
    assert self._audio_key in ex.features.feature, ex.features.feature.keys()
    audio = _tfexample_audio_to_npfloat32(ex, self._audio_key)
    assert audio.size > 0, k
    beam.metrics.Metrics.distribution(
        'computed-embedding-audio', 'length').update(audio.size)

    # Read the sample rate, if a key to do so has been provided.
    sample_rate = self._sample_rate
    if self._sample_rate_key:
      assert self._sample_rate_key in ex.features.feature
      sample_rate = ex.features.feature[
          self._sample_rate_key].int64_list.value[0]
    logging.info(
        'len(audio): %s / %s / %s', len(audio), sample_rate, self._name)

    # Resample, if necessary.
    if sample_rate != 16000:
      audio = librosa.core.resample(
          audio, orig_sr=sample_rate, target_sr=16000, res_type='kaiser_best')
      sample_rate = 16000

    # Calculate the 2D embedding.
    embedding_2d = _samples_to_embedding(
        audio, sample_rate, self.module, self._output_key)
    assert isinstance(embedding_2d, np.ndarray)
    assert embedding_2d.ndim == 2
    assert embedding_2d.dtype == np.float32
    beam.metrics.Metrics.counter('computed-embedding', self._name).inc()
    beam.metrics.Metrics.distribution(f'computed-embedding-{self._name}',
                                      'length').update(embedding_2d.shape[0])

    # Average over time, if required.
    if self._average_over_time:
      embedding = np.mean(embedding_2d, axis=0, keepdims=True)
    else:
      embedding = embedding_2d

    yield (k, embedding)


def _add_embedding_to_tfexample(ex, embedding, name):
  """Add a 2D embedding to a tf.train.Example."""
  # Store the embedding 2D shape and store the 1D embedding. The original
  # embedding can be recovered with `emb.reshape(feature['shape'])`.
  f = ex.features.feature[f'{name}/shape']
  f.int64_list.value.extend(embedding.shape)
  f = ex.features.feature[name]
  f.float_list.value.extend(embedding.reshape([-1]))
  return ex


def _add_embedding_column_map_fn(k_v, original_example_key,
                                 delete_audio_from_output, audio_key,
                                 label_key, speaker_id_key):
  """Combine a dictionary of named embeddings with a tf.train.Example."""
  k, v_dict = k_v

  assert original_example_key in v_dict, (original_example_key, v_dict.keys())
  ex_l = v_dict[original_example_key]
  assert len(ex_l) == 1, (len(ex_l), k_v[0], ex_l)
  ex = copy.deepcopy(ex_l[0])  # Beam does not allow modifying the input.
  assert isinstance(ex, tf.train.Example), type(ex)

  for name, embedding_l in v_dict.items():
    if name == original_example_key:
      continue
    assert len(embedding_l) == 1, embedding_l
    embedding = embedding_l[0]
    assert isinstance(embedding, np.ndarray)
    assert embedding.ndim == 2, embedding.ndim

    # Store the embedding 2D shape and store the 1D embedding. The original
    # embedding can be recovered with `emb.reshape(feature['shape'])`.
    ex = _add_embedding_to_tfexample(ex, embedding, f'embedding/{name}')

  if delete_audio_from_output:
    ex.features.feature.pop(audio_key, None)

  # Assert that the label is present. If it's a integer, convert it to bytes.
  assert label_key in ex.features.feature
  lbl_feat = ex.features.feature[label_key]
  if lbl_feat.int64_list.value:
    lbl_val_as_bytes = str(lbl_feat.int64_list.value[0]).encode('utf-8')
    ex.features.feature.pop(label_key, None)
    ex.features.feature[label_key].bytes_list.value.append(lbl_val_as_bytes)

  # If provided, assert that the speaker_id field is present, and of type
  # `bytes`.
  if speaker_id_key:
    feats = ex.features.feature
    assert speaker_id_key in feats, (speaker_id_key, feats.keys())
    assert feats[speaker_id_key].bytes_list.value, feats[speaker_id_key]

  return k, ex


def _tfds_filenames(dataset_name, split_name):
  """Returns filenames for a TFDS dataset."""
  data_dir = tfds.builder(dataset_name).data_dir
  return [os.path.join(data_dir, x) for x in
          tfds.builder(dataset_name).info.splits[split_name].filenames]


def _tfds_sample_rate(dataset_name):
  return tfds.builder(dataset_name).info.features['audio'].sample_rate


def read_input_glob_and_sample_rate_from_flags(
    input_glob_flag, sample_rate_flag, tfds_dataset_flag, output_filename_flag):
  """Read flags for input data and sample rate.

  Args:
    input_glob_flag: String flag. The input file glob.
    sample_rate_flag: String flag. The sample rate.
    tfds_dataset_flag: String flag. The TFDS dataset.
    output_filename_flag: String flag. The output filename.

  Returns:
    (input_filenames, output_filenames, sample_rate)
    `input_filenames` is a list of list of filenames. `output_filenames` is a
    list of the same length.
  """
  if input_glob_flag:
    assert file_utils.Glob(input_glob_flag), input_glob_flag
    input_filenames = [file_utils.Glob(input_glob_flag)]
    output_filenames = [output_filename_flag]
    sample_rate = sample_rate_flag
  else:
    assert tfds_dataset_flag
    dataset_name = tfds_dataset_flag
    tfds.load(dataset_name)  # download dataset, if necessary.
    sample_rate = _tfds_sample_rate(dataset_name)
    assert sample_rate, sample_rate

    input_filenames = []
    output_filenames = []
    for split_name in ('train', 'validation', 'test'):
      input_filenames.append(_tfds_filenames(dataset_name, split_name))
      output_filenames.append(output_filename_flag + f'.{split_name}')

    logging.info('TFDS input filenames: %s', input_filenames)
    logging.info('sample rate: %s', sample_rate)

  if sample_rate:
    assert isinstance(sample_rate, numbers.Number)

  for filename_list in input_filenames:
    for filename in filename_list:
      assert tf.io.gfile.exists(filename), filename
  assert len(input_filenames) == len(output_filenames)

  return input_filenames, output_filenames, sample_rate


def validate_inputs(
    input_filenames_list, output_filenames, embedding_modules, embedding_names,
    module_output_keys):
  """Validate inputs and input flags."""
  for filename_list in input_filenames_list:
    for filename in filename_list:
      assert tf.io.gfile.exists(filename), filename
  assert len(input_filenames_list) == len(output_filenames)

  # Make sure output files don't already exist.
  for output_filename in output_filenames:
    assert not file_utils.Glob(f'{output_filename}*'), output_filename

  # Lengths of flag lists must be the same.
  assert len(embedding_names) == len(embedding_modules),\
         (embedding_names, embedding_modules)
  assert len(embedding_modules) == len(module_output_keys),\
         (embedding_modules, module_output_keys)
  # Shortnames must be unique.
  assert len(set(embedding_names)) == len(embedding_names), embedding_names

  # Create output directory if it doesn't already exist.
  for output_filename in output_filenames:
    output_dir = output_filename.rsplit('/', 1)[0]
    file_utils.MaybeMakeDirs(output_dir)


def _read_from_tfrecord(root, input_filenames, suffix):
  """Reads from a Python list of TFRecord files."""
  assert isinstance(input_filenames, list), input_filenames
  return (root
          | f'MakeFilenames{suffix}' >> beam.Create(input_filenames)
          | f'ReadTFRecords{suffix}' >> beam.io.tfrecordio.ReadAllFromTFRecord(
              coder=beam.coders.ProtoCoder(tf.train.Example))
          | f'AddKeys{suffix}' >> beam.Map(
              lambda x: (str(random.getrandbits(128)), x)))




def _write_to_tfrecord(combined_tbl, output_filename, suffix):
  _ = (combined_tbl
       | f'RemoveKey{suffix}' >> beam.Map(lambda k_v: k_v[1])
       | f'Write{suffix}' >> beam.io.WriteToTFRecord(
           output_filename, coder=beam.coders.ProtoCoder(tf.train.Example)))




# Possible input formats. If you want to read from a different input format,
# add your read function here. Function should take (root, input_filenames) and
# map to input_examples.
reader_functions = {
    'tfrecord': _read_from_tfrecord
}


# Write output to disk.
writer_functions = {
    'tfrecord': _write_to_tfrecord,
}


def make_beam_pipeline(
    root, input_filenames, sample_rate, debug, embedding_names,
    embedding_modules, module_output_keys, audio_key, sample_rate_key,
    label_key, speaker_id_key, average_over_time, delete_audio_from_output,
    output_filename, input_format='tfrecord', output_format='tfrecord',
    suffix='Main'):
  """Construct beam pipeline for mapping from audio to embeddings.

  Args:
    root: The beam root node.
    input_filenames: Python list. List of input files.
    sample_rate: Python int, or `None`. The sample rate for all embeddings,
      or `None` if this is a TFDS dataset, or if each example has its own sample
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
    input_format: Python string. Must correspond to a function in
      `reader_functions`.
    output_format: Python string. Must correspond to a function
      `writer_functions`.
    suffix: Python string. Suffix to stage names to make them unique.
  """
  tf_examples_key_ = 'tf_examples'
  assert tf_examples_key_ not in embedding_names
  s = suffix  # for code brevity.

  # Read from input.
  input_examples = reader_functions[input_format](root, input_filenames, s)

  # In debug mode, take one input example.
  if debug:
    input_examples = (
        input_examples
        | f'TakeOne{s}' >> beam.transforms.combiners.Sample.FixedSizeGlobally(1)
        # Sampling generates lists, so flatten back into one collection.
        | f'DebugFlatten{s}' >> beam.FlatMap(lambda x: x))

  # Compute all the embeddings simultaneously.
  embedding_tables = {}
  for name, mod, out_key in zip(
      embedding_names, embedding_modules, module_output_keys):
    logging.info('Adding signal: %s %s, %s', name, mod, out_key)
    tbl = input_examples | f'ComputeEmbedding-{name}-{s}' >> beam.ParDo(
        ComputeEmbeddingMapFn(
            name=name,
            module=mod,
            output_key=out_key,
            audio_key=audio_key,
            sample_rate_key=sample_rate_key,
            sample_rate=sample_rate,
            average_over_time=average_over_time))
    embedding_tables[name] = tbl
  assert tf_examples_key_ not in embedding_tables
  embedding_tables[tf_examples_key_] = input_examples
  logging.info('embedding_tables: %s', embedding_tables)

  # Combine embeddings and tf.train.Example, using the common key.
  combined_tbl = (
      embedding_tables
      | f'CombineEmbeddingTables-{s}' >> beam.CoGroupByKey()
      | f'AddEmbeddings-{s}' >> beam.Map(
          _add_embedding_column_map_fn,
          original_example_key=tf_examples_key_,
          delete_audio_from_output=delete_audio_from_output,
          audio_key=audio_key,
          label_key=label_key,
          speaker_id_key=speaker_id_key))

  output_filename = f'{output_filename}@*'
  logging.info('Writing to %s', output_filename)
  writer_functions[output_format](combined_tbl, output_filename, s)
