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
"""Utilities for evaluating on sklearn models."""

from typing import Any, List, Optional, Tuple
import numpy as np
import tensorflow.compat.v1 as tf
from non_semantic_speech_benchmark import file_utils


def tfexamples_to_nps(
    path,
    embedding_name,
    label_name,
    label_list,
    l2_normalization,
    speaker_name = None):
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

  Returns:
    (numpy array of embeddings, numpy array of labels)
  """
  # Read data from disk.
  itervalues_fn = get_itervalues_fn(path)
  embeddings, labels, speaker_ids = [], [], []
  for ex in itervalues_fn():
    feats = ex.features.feature

    # Read embeddings.
    cur_emb = feats[f'embedding/{embedding_name}'].float_list.value
    assert cur_emb, (f'embedding/{embedding_name}', path)
    embeddings.append(cur_emb)

    # Read labels.
    assert label_name in feats, (label_name, feats.keys())
    cur_lbl = feats[label_name].bytes_list.value[0]
    assert isinstance(cur_lbl, bytes)
    if cur_lbl.decode('utf-8') not in label_list:
      raise ValueError(
          f'Current label not found in label list: {cur_lbl} vs {label_list}')
    labels.append(label_list.index(cur_lbl.decode('utf-8')))

    # Read speaker ID, if necessary.
    if speaker_name:
      assert speaker_name in feats
      cur_spkr = feats[speaker_name].bytes_list.value[0]
      assert cur_spkr
      speaker_ids.append(cur_spkr)

  if not embeddings:
    raise ValueError(f'No embeddings found in {path}')

  embeddings = np.array(embeddings, np.float32)
  labels = np.array(labels, np.int16)

  # Perform L2 normalization.
  if l2_normalization:
    embeddings /= np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)

  # Perform speaker normalization, if appropriate.
  if speaker_name:
    speaker_ids = np.array(speaker_ids, np.str)
    embeddings = _speaker_normalization(embeddings, speaker_ids)

  return embeddings, labels


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
    for path in file_utils.Glob(glob):
      for raw_str in tf.python_io.tf_record_iterator(path):
        example = proto()
        example.ParseFromString(raw_str)
        yield example
  return itervalues


# TODO(joelshor): Add typing when you figure out how to type yields and
# generators.
def get_itervalues_fn(path):
  return get_tfrecord_iterator(path)
