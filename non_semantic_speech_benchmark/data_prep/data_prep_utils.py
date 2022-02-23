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

# Lint as: python3
"""Utils for data prep beam jobs.

1) `data_prep_utils.py` contains low-level utils.
2) `beam_dofns` contain the complex beam DoFns, using 1)
3) `audio_to_embeddings_beam_utils` contains python functions for full beam
    pipelines using 1) and 2)
"""

import copy
import numbers
import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from absl import logging
import apache_beam as beam
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from non_semantic_speech_benchmark.export_model import tf_frontend


KEY_FIELD = 'key_adhoc'


#### Some inference functions, such as for `TFHub` and `TFLite` formats.


def samples_to_embedding_tfhub(
    model_input,
    sample_rate,
    mod,  # pylint:disable=g-bare-generic
    output_key,
    name):
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
    # Squeeze all possible dimensions, and hope the dimension is correct.
    ret = np.squeeze(ret)
  return ret


def samples_to_embedding_tflite(model_input, sample_rate,
                                interpreter,
                                output_key, name):
  """Run TFLite inference to map audio samples to an embedding."""
  if model_input.ndim == 1:
    model_input = np.expand_dims(model_input, axis=0)

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


def samples_to_embedding_tfhub_w2v2(
    model_input, mod):
  """Run inference to map audio samples to an embedding."""
  # There are 2 w2v2 model input signatures. Try both.
  try:
    tf_out = mod.signatures['waveform'](
        waveform=tf.constant(model_input),
        paddings=tf.zeros_like(model_input, dtype=tf.float32))
  except TypeError:
    tf_out = mod.signatures['waveform'](
        waveform=tf.constant(model_input),
        waveform_paddings=tf.zeros_like(model_input, dtype=tf.float32))

  return {k: v.numpy() for k, v in tf_out.items()}


#### Setup functions. `TFLite` requires special code, but `TFHub` is trivial.


def build_tflite_interpreter(tflite_model_path):
  model_content = None
  with tf.io.gfile.GFile(tflite_model_path, 'rb') as model_file:
    model_content = model_file.read()
  interpreter = tf.lite.Interpreter(model_content=model_content)
  interpreter.allocate_tensors()
  return interpreter


#### Preprocessing helpers.


def tfexample_audio_to_npfloat32(ex, audio_key,
                                 normalize_to_pm_one,
                                 key_field = None):
  """Extract audio from tf.Example and convert it to np.float32."""
  audio_feats = ex.features.feature[audio_key]
  iinfo = np.iinfo(np.int16)
  if audio_feats.int64_list.value:
    audio = np.array(audio_feats.int64_list.value)
    # Even though the data is in an int64 container, the data is actually int16.
    if np.logical_or(audio < iinfo.min, audio > iinfo.max).any():
      raise ValueError(
          f'Audio doesn\'t conform to int16: {np.min(audio)}, {np.max(audio)}')
    audio = audio.astype(np.float32)
    if normalize_to_pm_one:
      audio /= iinfo.max
  elif audio_feats.float_list.value:
    audio = np.array(audio_feats.float_list.value, dtype=np.float32)
    if not normalize_to_pm_one:
      audio *= iinfo.max
  else:
    if key_field:
      if key_field not in ex.features.feature:
        raise ValueError('Tried to raise error with ident, but had no key '
                         f'field: {key_field} {ex}')
      ident = ex.features.feature[key_field].bytes_list.value[0]
      assert ident, (key_field, ex)
      raise ValueError(f'Did not find any audio: {ident} {ex}')
    else:
      raise ValueError(f'Did not find any audio: {ex}')
  return audio


def get_chunked_audio_fn(model_input, chunk_len):
  """Do some chunking."""
  assert model_input.ndim == 1, model_input.ndim
  audio_size = (model_input.shape[0] // chunk_len) * chunk_len
  usable_audio = model_input[:audio_size]
  chunked_audio = np.reshape(usable_audio, [-1, chunk_len])
  return chunked_audio


def add_key_to_audio(ex,
                     audio_key,
                     key_field = KEY_FIELD):
  """Add hash of audio to tf.Example."""
  if key_field in ex.features.feature:
    raise ValueError(f'`{key_field}` is protected, can\'t be in tf.Train.')

  # Compute the key.
  # Note: Computing the key from the audio means keys won't be preserved when
  # chunking audio.
  samples = tfexample_audio_to_npfloat32(
      ex, audio_key, normalize_to_pm_one=True, key_field=key_field)
  samples = samples[:16000]
  key = round(np.mean(samples), 5)  # Round so it's stable.
  key = str(key).encode('utf-8')

  # Add the key to the tf.Example.
  ex.features.feature[key_field].bytes_list.value.append(key)
  return ex


def add_embedding_to_tfexample(ex, embedding,
                               name):
  """Add a 2D embedding to a tf.train.Example."""
  # Store the embedding 2D shape and store the 1D embedding. The original
  # embedding can be recovered with `emb.reshape(feature['shape'])`.
  f = ex.features.feature[f'{name}/shape']
  f.int64_list.value.extend(embedding.shape)
  f = ex.features.feature[name]
  f.float_list.value.extend(embedding.reshape([-1]))
  return ex


#### Utils for other stages of the pipeline.


def add_embeddings_to_tfex(
    k_v,
    original_example_key,
    delete_audio_from_output,
    pass_through_normalized_audio,
    audio_key,
    label_key,
    speaker_id_key):
  """Combine a dictionary of named embeddings with a tf.train.Example."""
  k, v_dict = k_v

  if original_example_key not in v_dict:
    raise ValueError(
        f'Original key not found: {original_example_key} vs {v_dict.keys()}')
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
    ex = add_embedding_to_tfexample(ex, embedding, f'embedding/{name}')

  # Add the hash of the audio as a key.
  ex = add_key_to_audio(ex, audio_key)

  if delete_audio_from_output:
    ex.features.feature.pop(audio_key, None)
  else:
    # If audio is an int, store a normalized version of it instead.
    if (pass_through_normalized_audio and
        ex.features.feature[audio_key].int64_list):
      audio_int = np.array(ex.features.feature[audio_key].int64_list.value)
      ex.features.feature.pop(audio_key, None)
      ex.features.feature[audio_key].float_list.value.extend(
          audio_int.astype(np.float32) / np.iinfo(np.int16).max)

  # Assert that the label is present. If it's a integer, convert it to bytes.
  if label_key:
    if label_key not in ex.features.feature:
      raise ValueError(f'Label not found: {label_key} vs {ex.features.feature}')
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


def combine_multiple_embeddings_to_tfex(
    k_v,
    delete_audio_from_output,
    pass_through_normalized_audio,
    audio_key,
    label_key,
    speaker_id_key):
  """Combine a dictionary of named embeddings with a tf.train.Example."""
  k, ex, out_dict = k_v
  assert isinstance(k, str), type(k)

  ex = copy.deepcopy(ex)  # Beam does not allow modifying the input.
  assert isinstance(ex, tf.train.Example), type(ex)

  # Add the hash of the audio as a key.
  ex = add_key_to_audio(ex, audio_key)

  if delete_audio_from_output:
    ex.features.feature.pop(audio_key, None)
  else:
    # If audio is an int, store a normalized version of it instead.
    if (pass_through_normalized_audio and
        ex.features.feature[audio_key].int64_list):
      audio_int = np.array(ex.features.feature[audio_key].int64_list.value)
      ex.features.feature.pop(audio_key, None)
      ex.features.feature[audio_key].float_list.value.extend(
          audio_int.astype(np.float32) / np.iinfo(np.int16).max)

  for name, embedding in out_dict.items():
    assert isinstance(embedding, np.ndarray)
    assert embedding.ndim == 2, embedding.ndim

    # Store the embedding 2D shape and store the 1D embedding. The original
    # embedding can be recovered with `emb.reshape(feature['shape'])`.
    ex = add_embedding_to_tfexample(ex, embedding, f'embedding/{name}')

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


def chunked_audio_to_tfex(
    k_v,
    delete_audio_from_output,
    pass_through_normalized_audio,
    chunk_len,
    audio_key = 'processed/audio_samples',
    label_key = 'label',
    speaker_id_key = 'speaker_id',
    embedding_length = 1024):
  """Combine a dictionary of named embeddings with a tf.train.Example."""
  k, audio, lbl, speaker_id, embs_dict = k_v

  # Sanity checks.
  if not isinstance(k, str):
    raise ValueError(f'Key was wrong type: {type(k)}')
  for emb in embs_dict.values():
    if emb.ndim != 2:
      raise ValueError(f'Embedding dims wrong: {emb.ndim}')
    if embedding_length and emb.shape[1] != embedding_length:
      raise ValueError(
          f'Feature dim wrong: {emb.shape[1]} vs {embedding_length}')
  if audio.ndim != 1:
    raise ValueError(f'Audio wrong shape: {audio.shape}')
  if chunk_len and audio.shape != (chunk_len,):
    raise ValueError(f'Audio len wrong: {chunk_len} vs {audio.shape}')

  ex = tf.train.Example()
  if not delete_audio_from_output:
    assert audio.dtype == np.float32
    if pass_through_normalized_audio:
      audio = audio.astype(np.float32) / np.iinfo(np.int16).max
    ex.features.feature[audio_key].float_list.value.extend(
        audio.reshape([-1]))
    # Add the hash of the audio as a key.
    ex = add_key_to_audio(ex, audio_key)

  for name, emb in embs_dict.items():
    ex = add_embedding_to_tfexample(ex, emb, f'embedding/{name}')

  # Pass the label through, if it exists.
  if lbl:
    ex.features.feature[label_key].bytes_list.value.append(lbl)

  # Pass the speaker_id through, if it exists.
  if speaker_id:
    ex.features.feature[speaker_id_key].bytes_list.value.append(speaker_id)

  return k, ex


def single_audio_emb_to_tfex(
    k_v,
    embedding_name,
    audio_key = 'audio',
    embedding_length = 1024):
  """Make simple (audio, embedding) pair into a tf.Example."""
  k, audio, emb = k_v

  # Sanity checks.
  if emb.ndim != 1:
    raise ValueError(f'Embedding dims wrong: {emb.ndim}')
  if embedding_length and emb.shape[0] != embedding_length:
    raise ValueError(
        f'Feature dim wrong: {emb.shape[0]} vs {embedding_length}')
  if audio.ndim != 1:
    raise ValueError(f'Audio wrong shape: {audio.shape}')

  ex = tf.train.Example()
  ex.features.feature[audio_key].float_list.value.extend(audio.reshape([-1]))
  ex = add_embedding_to_tfexample(ex, emb, f'embedding/{embedding_name}')

  # Add the hash of the audio as a key.
  ex = add_key_to_audio(ex, audio_key)

  return k, ex


def _read_from_tfrecord(root, input_filenames,
                        suffix):
  """Reads from a Python list of TFRecord files."""
  if not isinstance(input_filenames, list):
    raise ValueError(f'Expected list: {type(input_filenames)}')
  return (root
          | f'MakeFilenames{suffix}' >> beam.Create(input_filenames)
          | f'ReadTFRecords{suffix}' >> beam.io.tfrecordio.ReadAllFromTFRecord(
              coder=beam.coders.ProtoCoder(tf.train.Example))
          | f'AddKeys{suffix}' >> beam.Map(
              lambda x: (str(random.getrandbits(128)), x)))




def _write_to_tfrecord(combined_tbl, output_filename,
                       suffix):
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


def default_feature_fn(samples, sample_rate):
  frontend_args = tf_frontend.frontend_args_from_flags()
  feats = tf_frontend.compute_frontend_features(
      samples, sample_rate, **frontend_args)
  logging.info('Feats shape: %s', feats.shape)
  return tf.expand_dims(feats, axis=-1).numpy().astype(np.float32)


def tfds_filenames(dataset_name,
                   split_name,
                   data_dir = None):
  """Returns filenames for a TFDS dataset."""
  data_dir = tfds.builder(dataset_name, data_dir=data_dir).data_dir
  return [os.path.join(data_dir, x) for x in
          tfds.builder(dataset_name).info.splits[split_name].filenames]


def _tfds_sample_rate(dataset_name, data_dir = None):
  return tfds.builder(dataset_name, data_dir=data_dir).info.features[
      'audio'].sample_rate


def read_input_glob_and_sample_rate_from_flags(
    input_glob_flag, sample_rate_flag, tfds_dataset_flag,
    output_filename_flag, tfds_data_dir_flag
):
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
    if not tf.io.gfile.glob(input_glob_flag):
      raise ValueError(f'Files not found: {input_glob_flag}')
    if tfds_data_dir_flag:
      raise ValueError(
          f'`tfds_data_dir_flag` should be None: {tfds_data_dir_flag}')
    input_filenames = [tf.io.gfile.glob(input_glob_flag)]
    output_filenames = [output_filename_flag]
    sample_rate = int(sample_rate_flag) if sample_rate_flag else None
  else:
    assert tfds_dataset_flag
    dataset_name = tfds_dataset_flag
    # Download dataset, if necessary.
    tfds.load(dataset_name, data_dir=tfds_data_dir_flag)
    sample_rate = _tfds_sample_rate(dataset_name, tfds_data_dir_flag)
    if not sample_rate:
      raise ValueError(f'Must have sample rate: {sample_rate}')

    input_filenames = []
    output_filenames = []
    for split_name in ('train', 'validation', 'test'):
      input_filenames.append(
          tfds_filenames(dataset_name, split_name, tfds_data_dir_flag))
      output_filenames.append(output_filename_flag + f'.{split_name}')

    logging.info('TFDS input filenames: %s', input_filenames)
    logging.info('sample rate: %s', sample_rate)

  if sample_rate and not isinstance(sample_rate, numbers.Number):
    raise ValueError(f'Sample rate must be number: {type(sample_rate)}')

  for filename_list in input_filenames:
    for filename in filename_list:
      if not tf.io.gfile.exists(filename):
        raise ValueError(f'File doesn\'t exist: {filename}')
  if len(input_filenames) != len(output_filenames):
    raise ValueError('Lengths not equal.')

  logging.info('input_filenames: %s', input_filenames)
  logging.info('output_filenames: %s', output_filenames)

  return input_filenames, output_filenames, sample_rate
