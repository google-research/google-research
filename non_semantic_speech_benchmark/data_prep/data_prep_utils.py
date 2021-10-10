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
"""Construct a beam pipeline to map from audio to embeddings.

Long audio can be 182000 samples (11 seconds)
"""

import copy
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from absl import logging
import apache_beam as beam
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from non_semantic_speech_benchmark.data_prep import audio_to_embeddings_beam_utils as old_utils


def _samples_to_embedding_tfhub(model_input,
                                mod):
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


def _get_chunked_audio_fn(model_input, chunk_len):  # pylint:disable=g-bare-generic
  """Do some chunking in a tf.cond."""
  def chunked_audio():
    assert model_input.ndim == 1, model_input.ndim
    audio_size = (
        tf.shape(model_input)[0] // chunk_len) * chunk_len
    usable_audio = model_input[:audio_size]
    chunked_audio = tf.reshape(usable_audio, [-1, chunk_len])
    return chunked_audio
  return chunked_audio


@beam.typehints.with_input_types(Tuple[str, tf.train.Example])
@beam.typehints.with_output_types(
    Tuple[str, tf.train.Example, Dict[str, np.ndarray]])
class ComputeMultipleEmbeddingsFromSingleModel(old_utils.ComputeEmbeddingMapFn):
  """Computes an embedding (key, tf.Example) from audio (key, tf.Example)."""

  def __init__(self,
               *args,
               embedding_names,
               chunk_len = None,
               embedding_length = None,
               **kwargs):
    super(ComputeMultipleEmbeddingsFromSingleModel, self).__init__(
        *args, **kwargs)
    self._chunk_len = chunk_len
    self._output_keys = self._output_key
    self._embedding_names = embedding_names
    self._embedding_len = embedding_length
    assert isinstance(self._output_keys, (tuple, list))

  def tfex_to_emb_from_chunked_audio(
      self, k,
      ex):
    # Read the input example audio and assert input format sanity.
    audio = self.read_audio_from_tfexample(ex, k, normalize_to_pm_one=False)

    # Read the sample rate, if a key to do so has been provided.
    sample_rate = self.read_sample_rate_from_tfexample(ex)

    logging.info(
        'len(audio): %s / %s / %s', len(audio), sample_rate, self._name)

    # Resample, if necessary.
    if sample_rate != 16000:
      audio = self.resample(audio, sample_rate, target_sr=16000)
      sample_rate = 16000

    # Convert audio to features, if required.
    model_input = audio
    logging.info('`model_input` shape is: %s', model_input.shape)

    # Do some chunking.
    if self._chunk_len:
      logging.info('Chunk len: %s', self._chunk_len)
      model_input = tf.cond(
          tf.shape(model_input)[0] >= self._chunk_len,
          _get_chunked_audio_fn(model_input, self._chunk_len),
          lambda: model_input)

    # Calculate the 3D embedding.
    if model_input.ndim == 1:
      model_input = np.expand_dims(model_input, axis=0)
    tf_out = _samples_to_embedding_tfhub(model_input, self.post_setup_module)

    return tf_out, model_input

  def process(self, k_v):
    k, ex = k_v
    if not isinstance(k, str):
      raise ValueError(f'Expected str: {type(k)}')

    # Get dictionary of 3D embeddings.
    tf_out, _ = self.tfex_to_emb_from_chunked_audio(k, ex)

    out_dict = {}
    for name, output_key in zip(self._embedding_names, self._output_keys):
      assert isinstance(name, str)
      if output_key not in tf_out:
        raise ValueError(
            f'Output key not recognized: {output_key} vs {tf_out.keys()}')
      cur_emb = np.array(tf_out[output_key])
      if cur_emb.ndim != 3:  #  (chunk size, time, embedding dim)
        raise ValueError(f'Wrong output dim size: {cur_emb.ndim}')
      if cur_emb.dtype != np.float32:
        raise ValueError(f'Wrong dtype: {cur_emb.dtype}')

      embedding_2d = np.mean(cur_emb, axis=0, keepdims=False)
      embedding_2d = np.mean(embedding_2d, axis=0, keepdims=True)
      assert isinstance(embedding_2d, np.ndarray)
      assert embedding_2d.ndim == 2, embedding_2d.shape
      assert embedding_2d.dtype == np.float32
      if self._average_over_time and embedding_2d.shape[0] != 1:
        raise ValueError(f'Wrong batch dim: {embedding_2d.shape[0]} vs {1}')
      if self._embedding_len and embedding_2d.shape[1] != self._embedding_len:
        raise ValueError(f'Wrong output dim: {embedding_2d.shape[1]}')
      out_dict[name] = embedding_2d
    print(f'Out k: {type(k)}')
    print(f'Out ex: {type(ex)}')
    print(f'Out out_dict: {type(out_dict)}')
    yield (k, ex, out_dict)


def add_embedding_fn(
    k_v,
    delete_audio_from_output,
    audio_key,
    label_key,
    speaker_id_key):
  """Combine a dictionary of named embeddings with a tf.train.Example."""
  k, ex, out_dict = k_v
  assert isinstance(k, str), type(k)

  ex = copy.deepcopy(ex)  # Beam does not allow modifying the input.
  assert isinstance(ex, tf.train.Example), type(ex)

  # Add the hash of the audio as a key.
  ex = old_utils.add_key_to_audio(ex, audio_key)

  if delete_audio_from_output:
    ex.features.feature.pop(audio_key, None)

  for name, embedding in out_dict.items():
    assert isinstance(embedding, np.ndarray)
    assert embedding.ndim == 2, embedding.ndim

    # Store the embedding 2D shape and store the 1D embedding. The original
    # embedding can be recovered with `emb.reshape(feature['shape'])`.
    ex = old_utils._add_embedding_to_tfexample(  # pylint:disable=protected-access
        ex, embedding, f'embedding/{name}')

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


@beam.typehints.with_input_types(Tuple[str, tf.train.Example])
@beam.typehints.with_output_types(Tuple[
    str, np.ndarray, Optional[bytes], Optional[bytes],
    Dict[str, np.ndarray]])
class ChunkAudioAndComputeEmbeddings(ComputeMultipleEmbeddingsFromSingleModel):
  """Computes an embedding (key, tf.Example) from audio (key, tf.Example)."""

  def __init__(self,
               *args,
               label_key=None,
               speaker_id_key=None,
               **kwargs):
    super(ChunkAudioAndComputeEmbeddings, self).__init__(*args, **kwargs)
    self._label_key = label_key
    self._speaker_id_key = speaker_id_key
    logging.info('chunk_len: %s', self._chunk_len)
    logging.info('label_key: %s', self._label_key)
    logging.info('speaker_id_key: %s', self._speaker_id_key)

  def process(
      self, k_v):
    k, ex = k_v

    if not isinstance(k, str):
      raise ValueError(f'Wrong type: {type(k)}')

    # Get dictionary of 3D embeddings.
    tf_out, model_input = self.tfex_to_emb_from_chunked_audio(k, ex)
    assert model_input.ndim == 2

    cur_embs = [np.array(tf_out[okey]) for okey in self._output_key]
    for emb in cur_embs:
      if emb.ndim != 3:  # (chunk, time, emb dim)
        raise ValueError(f'Wrong output dims: {emb.shape}')
    if self._average_over_time:
      embedding_3ds = [
          np.mean(x, axis=1, keepdims=True) for x in cur_embs
      ]
    else:
      embedding_3ds = cur_embs

    for x in embedding_3ds:
      assert isinstance(x, np.ndarray)
      assert x.ndim == 3
      assert x.dtype == np.float32
      assert x.shape[0] == model_input.shape[0], (x.shape, model_input.shape)
      if self._embedding_len:
        assert x.shape[2] == self._embedding_len, x.shape
      if self._average_over_time:
        assert x.shape[1] == 1, x.shape

    # Get the label, if a key to do so has been provided.
    label = _get_label(self._label_key, ex) if self._label_key else None
    logging.info('`label` is: %s', label)
    if label:
      assert isinstance(label, bytes)

    # Get the speaker ID, if a label has been provided.
    speaker_id = (
        _get_speaker_id(self._speaker_id_key, ex)
        if self._speaker_id_key else None)
    logging.info('`speaker_id` is: %s', speaker_id)
    if speaker_id:
      assert isinstance(speaker_id, bytes)

    for i in range(model_input.shape[0]):
      cur_k = f'{k}_{i}'
      cur_audio = np.array(model_input[i])
      out_dict = {
          name: x[i] for name, x in zip(self._embedding_names, embedding_3ds)}
      yield (cur_k, cur_audio, label, speaker_id, out_dict)


def _get_label(label_key, ex):
  """Gets a label."""
  if label_key not in ex.features.feature:
    raise ValueError(
        f'label key not found: {label_key} vs {ex.features.feature}')
  lbl_feat = ex.features.feature[label_key]
  if lbl_feat.int64_list.value:
    label = str(lbl_feat.int64_list.value[0]).encode('utf-8')
  else:
    assert lbl_feat.bytes_list.value
    label = lbl_feat.bytes_list.value[0]
  assert label
  assert isinstance(label, bytes)
  return label


def _get_speaker_id(speaker_id_key, ex):
  """Get speakerID from tf.Example."""
  if speaker_id_key not in ex.features.feature:
    raise ValueError(f'speaker_id key not found: {speaker_id_key} vs '
                     f'{ex.features.feature}')
  speaker_id = ex.features.feature[speaker_id_key].bytes_list.value[0]
  assert speaker_id
  return speaker_id


def chunked_audio_to_tfex(
    k_v,
    delete_audio_from_output,
    chunk_len,
    audio_key = 'audio',
    label_key = 'label',
    speaker_id_key = 'speaker_id',
    embedding_dimension = 1024):
  """Combine a dictionary of named embeddings with a tf.train.Example."""
  k, audio, lbl, speaker_id, embs_dict = k_v

  # Sanity checks.
  if not isinstance(k, str):
    raise ValueError(f'Key was wrong type: {type(k)}')
  for emb in embs_dict.values():
    if emb.ndim != 2:
      raise ValueError(f'Embedding dims wrong: {emb.ndim}')
    if emb.shape[1] != embedding_dimension:
      raise ValueError(
          f'Feature dim wrong: {emb.shape[1]} vs {embedding_dimension}')
  if audio.ndim != 1:
    raise ValueError(f'Audio wrong shape: {audio.shape}')
  if chunk_len and audio.shape != (chunk_len,):
    raise ValueError(f'Audio len wrong: {chunk_len} vs {audio.shape}')

  ex = tf.train.Example()

  ex.features.feature[audio_key].float_list.value.extend(audio.reshape([-1]))

  for name, emb in embs_dict.items():
    ex = old_utils._add_embedding_to_tfexample(  # pylint:disable=protected-access
        ex, emb, f'embedding/{name}')

  # Add the hash of the audio as a key.
  ex = old_utils.add_key_to_audio(ex, audio_key)

  if delete_audio_from_output:
    ex.features.feature.pop(audio_key, None)

  # Pass the label through, if it exists.
  if lbl:
    ex.features.feature[label_key].bytes_list.value.append(lbl)

  # Pass the speaker_id through, if it exists.
  if speaker_id:
    ex.features.feature[speaker_id_key].bytes_list.value.append(speaker_id)

  return k, ex


def common_sanity_checks(
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


def common_pipeline_beginning(
    root,
    input_format,
    input_filenames,
    s,
    debug):
  """Common input reading for beam pipelines."""
  # Read from input.
  input_examples = old_utils.reader_functions[input_format](
      root, input_filenames, s)

  # In debug mode, take one input example.
  if debug:
    input_examples = (
        input_examples
        | f'TakeOne{s}' >> beam.transforms.combiners.Sample.FixedSizeGlobally(1)
        # Sampling generates lists, so flatten back into one collection.
        | f'DebugFlatten{s}' >> beam.FlatMap(lambda x: x))

  return input_examples


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
  common_sanity_checks(embedding_modules, embedding_names, module_output_keys)
  input_examples = common_pipeline_beginning(
      root, input_format, input_filenames, suffix, debug)
  s = suffix
  embedding_module = embedding_modules[0]

  # Compute all the embeddings simultaneously.
  logging.info('Adding all signals: %s', module_output_keys)
  tbl = (
      input_examples
      | f'Reshuffle1-{s}' >> beam.Reshuffle()
      | f'ComputeEmbedding-{s}' >> beam.ParDo(
          ComputeMultipleEmbeddingsFromSingleModel(
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
          add_embedding_fn,
          delete_audio_from_output=delete_audio_from_output,
          audio_key=audio_key,
          label_key=label_key,
          speaker_id_key=speaker_id_key)
      | f'Reshuffle3-{s}' >> beam.Reshuffle())
  # Write embeddings to disk.
  writer_function = old_utils.writer_functions[output_format]
  cur_output_filename = f'{output_filename}@*'
  logging.info('Writing to %s', cur_output_filename)
  writer_function(tbl, cur_output_filename, s)


def precompute_chunked_audio_pipeline(
    root,
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
  common_sanity_checks(embedding_modules, embedding_names, module_output_keys)
  input_examples = common_pipeline_beginning(
      root, input_format, input_filenames, suffix, debug)
  s = suffix
  embedding_module = embedding_modules[0]

  # Compute all the embeddings simultaneously.
  logging.info('Adding all signals: %s', module_output_keys)
  tbl = (
      input_examples
      | f'Reshuffle1-{s}' >> beam.Reshuffle()
      | f'ComputeEmbedding-{s}' >> beam.ParDo(
          ChunkAudioAndComputeEmbeddings(
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
          chunked_audio_to_tfex,
          delete_audio_from_output=delete_audio_from_output,
          chunk_len=chunk_len,
          speaker_id_key=speaker_id_key,
          embedding_dimension=embedding_length)
      | f'Reshuffle3-{s}' >> beam.Reshuffle())
  # Write embeddings to disk.
  writer_function = old_utils.writer_functions[output_format]
  cur_output_filename = f'{output_filename}@*'
  logging.info('Writing to %s', cur_output_filename)
  writer_function(tbl, cur_output_filename, s)
