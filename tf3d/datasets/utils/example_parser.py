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

"""Parsing library for tf.train.Example protos."""

import tensorflow_datasets as tfds


def decode_serialized_example(serialized_example, features):
  """Decodes a serialized Example proto as dictionary of tensorflow tensors.

  Args:
    serialized_example: serialized tf.train.Example proto.
    features: tfds.features.FeatureConnector which provides the decoding
      specification.

  Returns:
    Decoded tf.Tensor or dictionary of tf.Tensor, stored in the Example proto.
  """
  example_specs = features.get_serialized_info()
  parser = tfds.core.example_parser.ExampleParser(example_specs)
  tfexample_data = parser.parse_example(serialized_example)
  return features.decode_example(tfexample_data)


def decode_serialized_example_as_numpy(serialized_example, features):
  """Decodes a serialized Example proto as dictionary of numpy tensors.

  Note: This function is eager only.

  Args:
    serialized_example: serialized tf.train.Example proto.
    features: tfds.features.FeatureConnector which provides the decoding
      specification.

  Returns:
    Decoded tf.Tensor or dictionary of tf.Tensor, stored in the Example proto.
  """
  tensor_dict = decode_serialized_example(serialized_example, features)
  return tfds.core.utils.map_nested(lambda x: x.numpy(), tensor_dict)


def encode_serialized_example(example_data, features):
  """Encode the feature dict into a tf.train.Eexample proto string.

  The input example_data can be anything that the provided feature specification
  can consume.

  Example:
  example_data =  {
        'image': 'path/to/img.png',
        'rotation': np.eye(3, dtype=np.float32)
  }
  features={
        'image': tfds.features.Image(),
        'rotation': tfds.features.Tensor(shape=(3,3), dtype=tf.float64),
  }
  example_proto_string = encode_serialized_example(example_data, features)

  Args:
    example_data: Value or dictionary of feature values to convert into Example
      proto.
    features: tfds.features.FeatureConnector which provides the encoding
      specification.

  Returns:
    Serialized Example proto storing encoded example_data as per specification
    provided by features.
  """
  encoded = features.encode_example(example_data)
  example_specs = features.get_serialized_info()
  serializer = tfds.core.example_serializer.ExampleSerializer(example_specs)
  return serializer.serialize_example(encoded)
