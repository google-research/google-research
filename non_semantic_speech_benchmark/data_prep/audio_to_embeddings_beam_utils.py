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

import copy
import numbers
import os
import random
import typing
from absl import logging
import apache_beam as beam
import librosa
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from non_semantic_speech_benchmark import file_utils
from non_semantic_speech_benchmark.export_model import tf_frontend


def tfexample_audio_to_npfloat32(ex, audio_key, normalize_to_pm_one):
  """Extract audio from tf.Example and convert it to np.float32."""
  audio_feats = ex.features.feature[audio_key]
  iinfo = np.iinfo(np.int16)
  if audio_feats.int64_list.value:
    audio = np.array(audio_feats.int64_list.value)
    # Even though the data is in an int64 container, the data is actually int16.
    assert np.logical_and(audio >= iinfo.min, audio <= iinfo.max).all(),\
        (np.min(audio), np.max(audio), iinfo.min, iinfo.max)
    audio = audio.astype(np.float32)
    if normalize_to_pm_one:
      audio /= iinfo.max
  else:
    assert audio_feats.float_list.value
    audio = np.array(audio_feats.float_list.value, dtype=np.float32)
    if not normalize_to_pm_one:
      audio *= iinfo.max
  return audio


def samples_to_embedding_tfhub(model_input, sample_rate, mod, output_key, name):
  """Run inference to map a single audio sample to an embedding."""
  logging.info('[%s] Module input shape: %s', name, model_input.shape)
  # Some modules have signatures. If they do, they should only have one valid
  # signature, and we should use that one. Otherwise, raise an error.
  if callable(mod):
    logging.info('[%s] is callable.', name)
    sig = None
  else:
    logging.info('[%s] has signatures.', name)
    if not hasattr(mod, 'signatures'):
      raise ValueError(f'[{name}] Not callable and no signatures.')
    if not mod.signatures:
      raise ValueError(f'[{name}] Expected signatures, but they were empty.')
    all_sigs = [s for s in mod.signatures if not s.startswith('_')]
    valid_sigs = [s for s in all_sigs if not s.startswith('_')]
    if len(valid_sigs) != 1:
      raise ValueError(
          f'[{name}] Didn\'t find exactly one valid signature: {all_sigs}')
    sig = valid_sigs[0]
    logging.info('[%s] Using signatures, and found: %s', name, sig)
  # Models either take 2 args (input, sample_rate) or 1 arg (input).
  # The first argument is either 1 dimensional (samples) or 2 dimensional
  # (batch, samples).
  # Try all. Order here matters. We must try "2 args" before "1 arg", otherwise
  # models that use sample rate might ignore it.
  errors = []  # Track errors. Display if none of them work.
  tf_out = None
  for num_args, add_batch_dim in [(2, False), (1, False), (2, True), (1, True)]:
    cur_model_input = (tf.expand_dims(model_input, 0) if add_batch_dim
                       else model_input)
    func_args = ((cur_model_input,) if num_args == 1 else
                 (cur_model_input, sample_rate))
    try:
      if sig:
        tf_out = mod.signatures[sig](*func_args)
      else:
        tf_out = mod(*func_args)
    except (ValueError, TypeError,
            tf.errors.InvalidArgumentError) as e:
      # Track errors and print them only if none of the expected signatures
      # work.
      errors.append(e)
      continue
    logging.info('[%s] Succeeded with num args %i, add_batch_dim %s', name,
                 num_args, add_batch_dim)
    break
  if tf_out is None:
    raise ValueError(f'[{name}] None of the signatures worked: {errors}')
  if isinstance(tf_out, dict):
    if output_key not in tf_out:
      raise ValueError(
          f'[{name}] Key not recognized: "{output_key}" vs {tf_out.keys()}')
    ret = tf_out[output_key]
  else:
    ret = tf_out
  ret = np.array(ret)
  if ret.ndim > 2:
    # Batch-flatten in numpy.
    ret = np.reshape(ret, [ret.shape[0], -1])
  return ret


def build_tflite_interpreter(tflite_model_path):
  model_content = None
  with tf.io.gfile.GFile(tflite_model_path, 'rb') as model_file:
    model_content = model_file.read()
  interpreter = tf.lite.Interpreter(model_content=model_content)
  interpreter.allocate_tensors()
  return interpreter


def _default_feature_fn(samples, sample_rate):
  frontend_args = tf_frontend.frontend_args_from_flags()
  feats = tf_frontend.compute_frontend_features(
      samples, sample_rate, **frontend_args)
  return tf.expand_dims(feats, axis=-1).numpy().astype(np.float32)


def samples_to_embedding_tflite(model_input, sample_rate, interpreter,
                                output_key, name):
  """Run TFLite inference to map audio samples to an embedding."""
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  # Resize TFLite input size based on length of sample.
  # Ideally, we should explore if we can use fixed-size input here, and
  # tile the sample to meet TFLite input size.
  if not np.array_equal(model_input.shape, input_details[0]['shape']):
    logging.info('[%s] TFLite input, actual vs expected: %s vs %s', name,
                 model_input.shape, input_details[0]['shape'])
  interpreter.resize_tensor_input(input_details[0]['index'], model_input.shape)
  interpreter.allocate_tensors()
  interpreter.set_tensor(input_details[0]['index'], model_input)
  # Models either take 2 args (input, sample_rate) or 1 arg (input). Try both.
  if len(input_details) > 1:
    interpreter.set_tensor(input_details[1]['index'],
                           np.array(sample_rate).astype(np.int32))

  interpreter.invoke()
  embedding_2d = interpreter.get_tensor(
      output_details[int(output_key)]['index'])
  return np.array(embedding_2d, dtype=np.float32)


@beam.typehints.with_input_types(typing.Tuple[typing.Union[str], typing.Any])
@beam.typehints.with_output_types(typing.Tuple[str, typing.Any])
class ComputeEmbeddingMapFn(beam.DoFn):
  """Computes an embedding (key, tf.Example) from audio (key, tf.Example)."""

  def __init__(self,
               name,
               module,
               output_key,
               audio_key,
               sample_rate_key,
               sample_rate,
               average_over_time,
               feature_fn=None,
               model_input_min_length=None,
               target_sample_rate=16000,
               module_call_fn=samples_to_embedding_tfhub):
    self._name = name
    # If TFLite should be used, `module` should point to a flatbuffer model.
    self._module = module
    self._use_tflite = self._module.endswith('.tflite')
    # For TFLite, `output_key` is the index of the embedding output from TFLite
    # model (Usually 0).
    self._output_key = output_key
    self._audio_key = audio_key
    self._sample_rate_key = sample_rate_key
    self._sample_rate = sample_rate
    self._average_over_time = average_over_time
    self._feature_fn = feature_fn
    self._model_input_min_length = model_input_min_length
    self._target_sample_rate = target_sample_rate
    self._mod_call_fn = module_call_fn

    # Only one of `sample_rate_key` and `sample_rate` should be not None.
    assert (self._sample_rate_key is None) ^ (self._sample_rate is None),\
        (self._sample_rate_key, self._sample_rate)

  def setup(self):
    if self._use_tflite:
      self.interpreter = build_tflite_interpreter(self._module)
    else:
      self.module = hub.load(self._module)

  def read_audio_from_tfexample(self, ex, k, normalize_to_pm_one=True):
    """Reads the audio samples from a tf.Example, and assert input sanity."""
    if self._audio_key not in ex.features.feature:
      raise ValueError(f'Audio key `{self._audio_key}` not found: '
                       f'{list(ex.features.feature.keys())}')
    audio = tfexample_audio_to_npfloat32(ex, self._audio_key,
                                         normalize_to_pm_one)
    assert audio.ndim == 1, audio.ndim
    if audio.size == 0:
      raise ValueError(f'No audio found: {self._audio_key}, {audio.size} {k}')
    beam.metrics.Metrics.distribution(
        'computed-embedding-audio', 'length').update(audio.size)

    return audio

  def read_sample_rate_from_tfexample(self, ex):
    """Reads the sample rate from a tf.Example."""
    if self._sample_rate_key:
      if self._sample_rate_key not in ex.features.feature:
        raise ValueError(f'Sample rate key not found: {self._sample_rate_key}')
      sr_feat = ex.features.feature[self._sample_rate_key]
      # Use `sample_rate` in `float_list` or `int64_list`. Either way, convert
      # to an integer for downstream use.
      if not len(sr_feat.float_list.value) ^ len(sr_feat.int64_list.value):
        raise ValueError(
            f'Expected exactly one of `float_list` and `int64_list`: {sr_feat}')
      if sr_feat.float_list.value:
        sample_rate = int(sr_feat.float_list.value[0])
      else:
        sample_rate = sr_feat.int64_list.value[0]
    else:
      if not self._sample_rate:
        raise ValueError('If `sample_rate_key` not provided, must provide '
                         '`sample_rate`.')
      sample_rate = self._sample_rate

    return sample_rate

  def resample(self, audio, sample_rate, target_sr):
    """Resample audio to target."""
    return librosa.core.resample(
        audio, orig_sr=sample_rate, target_sr=target_sr, res_type='kaiser_best')

  def audio_to_features(self, audio, sample_rate):
    """Convert audio to features, if required."""
    if self._feature_fn:
      model_input = self._feature_fn(audio, sample_rate)
      if not isinstance(model_input, np.ndarray):
        raise ValueError(f'Expected ndarray, got {type(model_input)}')
      if model_input.dtype != np.float32:
        raise ValueError(f'Should be float32, was: {model_input.dtype}')
    else:
      model_input = audio
      if self._model_input_min_length and model_input.size < self._model_input_min_length:
        delta = self._model_input_min_length - model_input.size
        model_input = np.pad(model_input, [0, delta], mode='constant')
      if self._use_tflite:
        model_input = np.expand_dims(model_input, axis=0)
    logging.info('`model_input` shape is: %s', model_input.shape)

    return model_input

  def process(self, k_v):
    k, ex = k_v

    # Read the input example audio and assert input format sanity.
    audio = self.read_audio_from_tfexample(ex, k)

    # Read the sample rate, if a key to do so has been provided.
    sample_rate = self.read_sample_rate_from_tfexample(ex)

    logging.info('len(audio): %s / %s / %s', len(audio), sample_rate,
                 self._name)

    # Resample, if necessary.
    if sample_rate != self._target_sample_rate:
      audio = self.resample(
          audio, sample_rate, target_sr=self._target_sample_rate)
      sample_rate = self._target_sample_rate

    # Convert audio to features, if required.
    model_input = self.audio_to_features(audio, sample_rate)

    # Calculate the 2D embedding.
    if self._use_tflite:
      embedding_2d = samples_to_embedding_tflite(
          model_input, sample_rate, self.interpreter, self._output_key,
          self._name)
    else:
      # A custom call function with the same input and output signature as
      # _sample_to_embedding_tfhub can be used
      # (_sample_to_embedding_tfhub is default).
      embedding_2d = self._mod_call_fn(model_input, sample_rate, self.module,
                                       self._output_key, self._name)
    assert isinstance(embedding_2d, np.ndarray)
    assert embedding_2d.ndim == 2
    assert embedding_2d.dtype == np.float32
    logging.info('[%s] `embedding_2d` shape: %s', self._name,
                 embedding_2d.shape)
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


def _tfds_filenames(dataset_name, split_name, data_dir=None):
  """Returns filenames for a TFDS dataset."""
  data_dir = tfds.builder(dataset_name, data_dir=data_dir).data_dir
  return [os.path.join(data_dir, x) for x in
          tfds.builder(dataset_name).info.splits[split_name].filenames]


def _tfds_sample_rate(dataset_name, data_dir=None):
  return tfds.builder(dataset_name, data_dir=data_dir).info.features[
      'audio'].sample_rate


def read_input_glob_and_sample_rate_from_flags(
    input_glob_flag, sample_rate_flag, tfds_dataset_flag, output_filename_flag,
    tfds_data_dir_flag):
  """Read flags for input data and sample rate.

  Args:
    input_glob_flag: String flag. The input file glob.
    sample_rate_flag: String flag. The sample rate.
    tfds_dataset_flag: String flag. The TFDS dataset.
    output_filename_flag: String flag. The output filename.
    tfds_data_dir_flag: String flag. Optional location of local TFDS data.

  Returns:
    (input_filenames, output_filenames, sample_rate)
    `input_filenames` is a list of list of filenames. `output_filenames` is a
    list of the same length.
  """
  if input_glob_flag:
    assert file_utils.Glob(input_glob_flag), input_glob_flag
    assert not tfds_data_dir_flag
    input_filenames = [file_utils.Glob(input_glob_flag)]
    output_filenames = [output_filename_flag]
    sample_rate = sample_rate_flag
  else:
    assert tfds_dataset_flag
    dataset_name = tfds_dataset_flag
    # Download dataset, if necessary.
    tfds.load(dataset_name, data_dir=tfds_data_dir_flag)
    sample_rate = _tfds_sample_rate(dataset_name, tfds_data_dir_flag)
    assert sample_rate, sample_rate

    input_filenames = []
    output_filenames = []
    for split_name in ('train', 'validation', 'test'):
      input_filenames.append(
          _tfds_filenames(dataset_name, split_name, tfds_data_dir_flag))
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
         (len(embedding_modules), len(module_output_keys))
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
    split_embeddings_into_separate_tables=False,
    use_frontend_fn=False,
    model_input_min_length=None,
    input_format='tfrecord',
    output_format='tfrecord',
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
    split_embeddings_into_separate_tables: Python bool. If true, write each
      embedding to a separate table.
    use_frontend_fn: If `true`, call frontend fn on audio before passing to the
      model.
    model_input_min_length: Min length to the model, or `None`. 0-pad inputs to
      this length, if necessary. Note that frontends usually contain their own
      length logic, unless the model is in TFLite format.
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
            average_over_time=average_over_time,
            feature_fn=_default_feature_fn if use_frontend_fn else None,
            model_input_min_length=model_input_min_length))
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
  writer_function = writer_functions[output_format]
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
            _add_embedding_column_map_fn,
            original_example_key=tf_examples_key_,
            delete_audio_from_output=delete_audio_from_output,
            audio_key=audio_key,
            label_key=label_key,
            speaker_id_key=speaker_id_key))
    logging.info('Writing to %s', cur_output_filename)
    writer_function(combined_tbl, cur_output_filename, cur_s)
