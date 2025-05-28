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

"""Utils for reading and writing `tfrec` of `tfds.features.FeaturesDict`."""
import os

import tensorflow as tf
import tensorflow_datasets as tfds


def get_tfex_feature_dict_parse_fn(spec):
  parser = tfds.core.example_parser.ExampleParser(spec.get_serialized_info())
  return parser.parse_example


def get_tfex_feature_dict_serialize_fn(spec):
  serializer_spec = spec.get_serialized_info()
  serializer = tfds.core.example_serializer.ExampleSerializer(serializer_spec)
  def serialize_fn(rec):
    return serializer.serialize_example(spec.encode_example(rec))

  return serialize_fn


def tfrec_to_tuple(rec):
  key = rec['keys'][0]
  return (key, rec)


def read_tfrec_feature_dict_ds(dataset_root):
  input_spec = tfds.features.FeaturesDict.from_config(dataset_root)
  parse_fn = get_tfex_feature_dict_parse_fn(input_spec)
  tfrec_path = os.path.join(dataset_root, 'dataset.tfrec')
  tfrec_ds = tf.data.TFRecordDataset(tfrec_path)
  ds = tfrec_ds.map(lambda x: tfrec_to_tuple(parse_fn(x)))
  return ds


def write_tfrec_feature_dict_ds(ds, output_spec, output_path):
  tf.io.gfile.makedirs(output_path)
  output_spec.save_config(output_path)
  tfrec_path = os.path.join(output_path, 'dataset.tfrec')
  serialize_fn = get_tfex_feature_dict_serialize_fn(output_spec)

  with tf.io.TFRecordWriter(tfrec_path) as writer:
    for example in ds:
      writer.write(serialize_fn(example))
