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

"""Loads dataset for the BERT pretraining task."""
from typing import Mapping, Optional

import tensorflow as tf

from tensorflow_models.official.core import input_reader
from tensorflow_models.official.nlp.data import data_loader


class PretrainDataLoader(data_loader.DataLoader):
  """A class to load dataset for bert pretraining task."""

  def __init__(self, params):
    """Inits `PretrainDataLoader` class.

    Args:
      params: A `BertPretrainDataConfig` object.
    """
    self._params = params
    self._seq_length = params.seq_length
    self._max_predictions_per_seq = params.max_predictions_per_seq
    self._use_next_sentence_label = params.use_next_sentence_label
    self._use_position_id = params.use_position_id

  def _decode(self, record):
    """Decodes a serialized tf.Example."""
    name_to_features = {
        'new_input_ids':
            tf.io.FixedLenFeature([self._seq_length], tf.int64),
        'new_input_masks':
            tf.io.FixedLenFeature([self._seq_length], tf.int64),
        'new_segment_ids':
            tf.io.FixedLenFeature([self._seq_length], tf.int64),
        'new_input_positions':
            tf.io.FixedLenFeature([self._seq_length], tf.int64),
        'masked_lm_positions':
            tf.io.FixedLenFeature([self._max_predictions_per_seq], tf.int64),
        'masked_lm_ids':
            tf.io.FixedLenFeature([self._max_predictions_per_seq], tf.int64),
        'masked_lm_weights':
            tf.io.FixedLenFeature([self._max_predictions_per_seq], tf.float32),
        'masked_input_ids':
            tf.io.FixedLenFeature([self._max_predictions_per_seq], tf.int64),
        'masked_segment_ids':
            tf.io.FixedLenFeature([self._max_predictions_per_seq], tf.int64),
    }
    if self._use_next_sentence_label:
      name_to_features['next_sentence_labels'] = tf.io.FixedLenFeature([1],
                                                                       tf.int64)
    if self._use_position_id:
      name_to_features['position_ids'] = tf.io.FixedLenFeature(
          [self._seq_length], tf.int64)

    example = tf.io.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.cast(t, tf.int32)
      example[name] = t

    return example

  def _parse(self, record):
    """Parses raw tensors into a dict of tensors to be consumed by the model."""
    x = {
        'input_word_ids': record['new_input_ids'],
        'input_mask': record['new_input_masks'],
        'input_type_ids': record['new_segment_ids'],
        'input_positions': record['new_input_positions'],
        'masked_lm_positions': record['masked_lm_positions'],
        'masked_lm_ids': record['masked_lm_ids'],
        'masked_lm_weights': record['masked_lm_weights'],
        'masked_input_ids': record['masked_input_ids'],
        'masked_segment_ids': record['masked_segment_ids'],
    }
    if self._use_next_sentence_label:
      x['next_sentence_labels'] = record['next_sentence_labels']
    if self._use_position_id:
      x['position_ids'] = record['position_ids']

    return x

  def load(self, input_context = None):
    """Returns a tf.dataset.Dataset."""
    reader = input_reader.InputReader(
        params=self._params, decoder_fn=self._decode, parser_fn=self._parse)
    return reader.read(input_context)
