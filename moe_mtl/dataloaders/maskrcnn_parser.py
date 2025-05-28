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


"""Data parser for Mask R-CNN model."""
from typing import Any, Dict

import gin
import tensorflow.compat.v1 as tf

Parser = maskrcnn_parser.Parser
# gin.external_configurable(Parser, name='maskrcnn_map_fn')


@gin.configurable
class TfdsMaskRCNNParser(Parser):
  """A subclass to parse without tf.ExampleDecoder."""

  def __call__(self, value):
    with tf.name_scope('parser'):
      return self._parse_fn(value)


gin.external_configurable(TfdsMaskRCNNParser, name='tfds_map_fn')


@gin.configurable
class VALMaskRCNNParser(Parser):
  """A subclass to supply source id along with each parsed example."""

  def __call__(self, value):
    with tf.name_scope('parser'):
      data = self._example_decoder.decode(value)
      if 'source_id' not in data:
        raise KeyError('Source id must be in data because this parser is'
                       ' designed for source id filtering. Use another '
                       'parser when this filtering is not needed.')
      source_id = tf.strings.to_number(data['source_id'], tf.int64)
      return self._parse_fn(data), source_id


@gin.configurable(denylist=['value'])
def ours_maskrcnn_parser_fn(value, parser_fn,
                            is_training):
  """Wrapper around mask rcnn parser to standardize its output to a dictionary.

  Args:
    value: a string tensor holding a serialized tf.Example proto.
    parser_fn: a function to parse data for training and testing.
    is_training: a bool to indicate whether it's in training or testing.

  Returns:
    A dictionary {'images': image, 'labels': labels} whether it's in training
    or prediction mode.
  """
  data = parser_fn(value)
  if is_training:
    images, labels = data
    data = {'images': images, 'labels': labels}

  return data
