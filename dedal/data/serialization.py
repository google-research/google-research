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

"""Serializer/deserializers for biological sequence data.

Inspired by learning/brain/research/wavesplit/wavesplit_v2/serialization.py

This module defines Coder that are object that turn dictionary of features,
keyed by string and with tensor values, into binary strings and vice-versa.

Different serialization protocols are implemented to perform this conversion.
"""

import json
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import gin
import numpy as np
import tensorflow as tf

FeaturesDict = Dict[str, tf.Tensor]


@gin.configurable
class JSONCoder:
  """A JSON coder for beam."""

  def encode(self, features):
    return json.dumps(features)

  def decode(self, line):
    return json.loads(line)


@gin.configurable
class Coder:
  """Encodes / decodes FeaturesDict into / from strings."""

  def __init__(self,
               specs = None,
               shapes = None,
               to_numpy = False):
    self._specs = specs
    self._shapes = shapes if shapes is not None else {}
    self._to_numpy = to_numpy

  def encode(self, features):
    raise NotImplementedError()

  def raw_features(self, serialized_example):
    return tf.io.parse_single_example(serialized_example, self.specs)

  def decode(self, serialized_example):
    """Reads a tf.Example and turns it into a string."""
    sparse = self.raw_features(serialized_example)
    features = {}
    for k, v in sparse.items():
      is_sparse = isinstance(self.specs.get(k, None), tf.io.VarLenFeature)
      features[k] = tf.sparse.to_dense(v) if is_sparse else v

    result = {}
    for k, v in features.items():
      if v.dtype == tf.string and v.shape.rank > 0 and v.shape[0] == 1:
        parsed = v[0]
      else:
        parsed = v
      parsed = parsed.numpy() if self._to_numpy else parsed
      parsed = parsed.decode() if isinstance(parsed, bytes) else parsed
      # Enforces the final shapes if possible.
      shape = self._shapes.get(k, None)
      parsed = tf.ensure_shape(parsed, shape) if shape is not None else parsed
      result[k] = parsed
    return result

  @property
  def specs(self):
    result = {}
    for k, v in self._specs.items():
      if isinstance(v, tf.dtypes.DType):
        v = tf.io.VarLenFeature(v)
      result[k] = v
    return result


@gin.configurable
class FlatCoder(Coder):
  """Encode and decode strings into tf.Example with flat tensors."""

  def encode(self, features):
    """Turns a features dictionary into a serialized tf.Example."""
    data = {}
    for k, v in features.items():
      curr_dtype = self._specs.get(k, None)
      if curr_dtype is None:
        continue
      if curr_dtype == tf.float32:
        data[k] = tf.train.Feature(float_list=tf.train.FloatList(value=v))
      elif curr_dtype == tf.int64:
        data[k] = tf.train.Feature(int64_list=tf.train.Int64List(value=v))
      elif curr_dtype == tf.string:
        v = v.numpy() if isinstance(v, tf.Tensor) else v.encode()
        data[k] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))
    example = tf.train.Example(features=tf.train.Features(feature=data))
    return example.SerializeToString()


@gin.configurable
class SequenceCoder(Coder):
  """Use a chunked sequence serialization.

  To be encoded, the sequences are first chunked into subsequences of fix size.
  In case of multi channels, all the channels are interleaved into a single one.
  Decoding is done by `tf.io.parse_single_sequence_example`.
  """

  def __init__(self, sequence_keys = None, **kwargs):
    super().__init__(**kwargs)
    self.sequence_keys = {} if sequence_keys is None else set(sequence_keys)

  def raw_features(self, serialized_example):
    """Returns the decoded sparse features."""
    ctx_specs = {}
    seq_specs = {}
    for k, v in self._specs.items():
      target_specs = seq_specs if k in self.sequence_keys else ctx_specs
      v = tf.io.VarLenFeature(v) if isinstance(v, tf.dtypes.DType) else v
      target_specs[k] = v
    context, sparse = tf.io.parse_single_sequence_example(
        serialized_example, ctx_specs, seq_specs)

    sparse.update(context)
    return sparse

  def encode(self, features):
    """Encodes a Dict of Tensors into a string."""
    example = tf.train.SequenceExample()
    for key, tensor in features.items():
      if key not in self._specs:
        continue

      spec = self.specs.get(key, tf.io.VarLenFeature(tf.float32))
      if key in self.sequence_keys:
        feature = example.feature_lists.feature_list[key].feature
        sequence = np.array(tensor) if isinstance(tensor, list) else tensor
        for i in range(sequence.shape[0]):
          if spec.dtype == tf.float32:
            if len(sequence.shape) > 1:
              feature.add().float_list.value.extend(sequence[i])
            else:
              feature.add().float_list.value.append(sequence[i])
          else:
            if len(sequence.shape) > 1:
              feature.add().int64_list.value.extend(sequence[i])
            else:
              feature.add().int64_list.value.append(sequence[i])

      else:
        tensor = [tensor] if isinstance(spec, tf.io.FixedLenFeature) else tensor
        if spec.dtype == tf.string:
          tensor = tensor.encode() if isinstance(tensor, str) else tensor
          tensor = tensor.numpy() if isinstance(tensor, tf.Tensor) else tensor
          example.context.feature[key].bytes_list.value.append(tensor)
        if spec.dtype == tf.int64:
          example.context.feature[key].int64_list.value.extend(tensor)
        if spec.dtype == tf.float32:
          example.context.feature[key].float_list.value.extend(tensor)
    return example.SerializeToString()
