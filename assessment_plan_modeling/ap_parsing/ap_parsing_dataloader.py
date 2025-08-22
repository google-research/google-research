# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Loads dataset for AP parsing TF-NLP tagging task."""

import dataclasses
import random
from typing import Dict, Mapping, NamedTuple, Optional, Tuple

import numpy as np
import tensorflow as tf

from assessment_plan_modeling.ap_parsing import constants
from official.core import config_definitions as cfg
from official.core import input_reader
from official.nlp.data import data_loader

_RecordDataType = Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]


@dataclasses.dataclass
class APParsingDataConfig(cfg.DataConfig):
  """Data config for AP parsing task."""
  seq_length: Optional[int] = 4096


def get_name_to_feature_mapping(
    seq_length):
  """TF example feature mapping dictionary."""

  def get_feat_by_size(size, dtype):
    if size:
      return tf.io.FixedLenFeature((seq_length), dtype)
    return tf.io.VarLenFeature(dtype)

  name_to_features = {}
  # Add metadata features.
  name_to_features.update({
      k: tf.io.FixedLenFeature((1), tf.int64)
      for k in constants.METADATA_FEATURES
  })

  # Add features and labels.
  name_to_features.update({
      k: get_feat_by_size(seq_length, tf.int64)
      for k in constants.MODEL_FEATURES
  })

  return name_to_features  # pytype: disable=bad-return-type


def test_dataset(seq_length):
  """Returns a test dataset with the right features and shapes."""

  def test_data(unused_args):
    del unused_args
    test_features = {}
    true_seq_length = random.randint(1, seq_length)
    for k in constants.FEATURE_NAMES:
      test_features[k] = np.ones((1, seq_length), dtype=np.int32)
      test_features[k][true_seq_length:seq_length] = 0

    test_labels = {}
    for k, v in [(k, len(v)) for k, v in constants.CLASS_NAMES.items()]:
      test_labels[k] = np.random.randint(0, v, (1, seq_length), dtype=np.int32)
      test_labels[k][true_seq_length:seq_length] = -1

    return ({**test_features, **test_labels}, test_labels)

  dataset = tf.data.Dataset.range(100)
  dataset = dataset.repeat()
  dataset = dataset.map(
      test_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  return dataset


class APParsingDataLoader(data_loader.DataLoader):
  """A class to load dataset for AP parsing (sequence tagging) task."""

  def __init__(self, params):
    self._params = params

  def _decode(self, record):
    """Decodes a serialized tf.Example."""

    name_to_features = get_name_to_feature_mapping(self._params.seq_length)
    name_to_features = {
        k: v
        for k, v in name_to_features.items()
        if k in constants.MODEL_FEATURES
    }
    example = tf.io.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in example:
      t = example[name]
      if isinstance(t, tf.SparseTensor):
        t = tf.sparse.to_dense(t)
      if t.dtype == tf.int64:
        t = tf.cast(t, tf.int32)
      example[name] = t

    return example

  def _parse(self, record):
    """Parses raw tensors into a dict of tensors to be consumed by the model."""
    x = {k: record[k] for k in constants.MODEL_FEATURES}
    y = {k: record[k] for k in constants.LABEL_NAMES}
    return (x, y)

  def load(
      self,
      input_context = None
  ):
    """Returns a tf.dataset.Dataset."""
    if self._params.input_path == "test":
      return test_dataset(self._params.seq_length)

    reader = input_reader.InputReader(
        params=self._params, decoder_fn=self._decode, parser_fn=self._parse)
    return reader.read(input_context)

  def decode_parse_single_record(self, record):
    """Processes a single record."""
    decoded = self._decode(record)
    x, y = self._parse(decoded)

    # add batch dimension
    x = {k: tf.expand_dims(v, 0) for k, v in x.items()}
    y = {k: tf.expand_dims(v, 0) for k, v in y.items()}

    return (x, y)
