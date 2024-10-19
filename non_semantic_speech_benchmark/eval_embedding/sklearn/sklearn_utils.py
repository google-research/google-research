# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Utilities for evaluating on sklearn models."""

from typing import Any, List, Optional, Tuple
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf


def tfexamples_to_nps(
    path,
    embedding_name,
    label_name,
    label_list,
    l2_normalization,
    speaker_name = None,
    key_field = None,
):
  """Reads tf.Examples to numpy arrays.

  Args:
    path: Python string. Path to tf.Example data.
    embedding_name: Python string. Name of tf.Example field with embedding.
      Field name must be f"embedding/{embedding_name}".
    label_name: Python string. Name of tf.Example field with label.
    label_list: Python list of strings. List of valid label values.
    l2_normalization: Python bool. If `True`, normalize embeddings by L2 norm.
    speaker_name: Python string or None. If present, the tf.Example field with
      the speaker ID.
    key_field: Optional field to return. This should be a unique per-example
      identifier. For practical applications, can use the `audio_key`.

  Returns:
    (numpy array of embeddings, numpy array of labels, Optional array of keys)
  """
  if embedding_name.startswith('embedding/'):
    raise ValueError(f'Don\'t prepend embedding name: {embedding_name}')

  # Read data from disk.
  logging.info('About to read from "%s"...', path)
  itervalues_fn = get_itervalues_fn(path)
  logging.info('Successfully created iterator.')
  embeddings, labels, speaker_ids, keys = [], [], [], []
  for ex in itervalues_fn():
    feats = ex.features.feature

    # Read embeddings.
    cur_emb = feats[f'embedding/{embedding_name}'].float_list.value
    if not bool(cur_emb):
      raise ValueError(f'Embeddings empty: embedding/{embedding_name} {path}')
    embeddings.append(cur_emb)

    # Read labels.
    if label_name not in feats:
      raise ValueError(
          f'`label_name` not in feats: {label_name} vs {list(feats.keys())}')
    if feats[label_name].bytes_list.value:
      cur_lbl = feats[label_name].bytes_list.value[0]
      assert isinstance(cur_lbl, bytes)
      if cur_lbl.decode('utf-8') not in label_list:
        raise ValueError(
            f'Current label not found in label list: {cur_lbl} vs {label_list}')
      label = cur_lbl.decode('utf-8')
    elif feats[label_name].int64_list.value:
      cur_lbl = feats[label_name].int64_list.value[0]
      label = str(cur_lbl)
    else:
      raise ValueError('Invalid type for cur_lbl.')
    labels.append(label_list.index(label))

    # Read speaker ID, if necessary.
    if speaker_name:
      if speaker_name not in feats:
        raise ValueError(
            f'`speaker_name` not in feats: {speaker_name} vs {feats.keys()}')
      cur_spkr = feats[speaker_name].bytes_list.value[0]
      if not cur_spkr:
        raise ValueError('speaker_name is empty')
      speaker_ids.append(cur_spkr)

    # Read key, if necessary.
    if key_field:
      if key_field not in feats:
        raise ValueError(
            f'`key_field` not in feats: {key_field} vs {feats.keys()}')
      cur_key = None
      for key_type in ['bytes_list', 'int64_list', 'float_list']:
        cur_key = getattr(feats[key_field], key_type).value
        if bool(cur_key):
          if len(cur_key) == 1:
            cur_key = cur_key[0]
          break
      if not bool(cur_key):
        raise ValueError('`key_field` is empty.')
      keys.append(cur_key)

  if not embeddings:
    raise ValueError(f'No embeddings found in {path}')

  try:
    embeddings = np.array(embeddings, np.float32)
    labels = np.array(labels, np.int16)
    # TODO(joelshor): Consider adding a uniqueness check for keys.
  except ValueError:
    logging.warning(
        '`tfexamples_to_nps` failed with the following inputs: %s, %s, %s, %s %s',
        path, embedding_name, label_name, speaker_name, key_field)
    raise

  # Perform L2 normalization.
  if l2_normalization:
    embeddings /= np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)

  # Perform speaker normalization, if appropriate.
  if speaker_name:
    speaker_ids = np.array(speaker_ids, str)
    embeddings = _speaker_normalization(embeddings, speaker_ids)

  if not key_field:
    assert not bool(keys)
    keys = None

  return embeddings, labels, keys


def _speaker_normalization(embeddings,
                           speaker_ids):
  """Normalize embedding features by per-speaker statistics."""
  all_speaker_ids = np.unique(speaker_ids)
  for speaker in all_speaker_ids:
    speaker_features = speaker_ids == speaker

    # Normalize feature mean.
    embeddings[speaker_features] -= embeddings[speaker_features].mean(axis=0)

    # Normalize feature variance.
    stds = embeddings[speaker_features].std(axis=0)
    stds[stds == 0] = 1
    embeddings[speaker_features] /= stds

  return embeddings




# TODO(joelshor): Add typing when you figure out how to type yields and
# generators.
def get_tfrecord_iterator(glob, proto=tf.train.Example):
  def itervalues():
    for path in tf.io.gfile.glob(glob):
      for raw_str in tf.python_io.tf_record_iterator(path):
        example = proto()
        example.ParseFromString(raw_str)
        yield example
  return itervalues


# TODO(joelshor): Add typing when you figure out how to type yields and
# generators.
def get_itervalues_fn(path):
  return get_tfrecord_iterator(path)
