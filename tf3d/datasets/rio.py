# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""RIO Scene dataset."""

import os
import tensorflow_datasets as tfds

from tf3d.datasets.specs import rio_specs
from tf3d.datasets.utils import example_parser

_FILE_PATTERN = '%s-*.sstable'
_FILE_PATTERN_TFRECORD = '%s*.tfrecords'
DATASET_FORMAT = 'sstable'

DATASET_DIR = None



def _get_feature_label_keys():
  """Extracts and returns the dataset feature and label keys."""
  feature_spec = (
      rio_specs.scene_feature_spec(
          with_annotations=True).get_serialized_info())
  feature_dict = tfds.core.utils.flatten_nest_dict(feature_spec)
  feature_keys = []
  label_keys = []
  for key in sorted(feature_dict):
    if 'labels' in key:
      label_keys.append(key)
    else:
      feature_keys.append(key)
  return feature_keys, label_keys


def get_feature_keys():
  return _get_feature_label_keys()[0]


def get_label_keys():
  return _get_feature_label_keys()[1]


def get_file_pattern(split_name, dataset_dir=DATASET_DIR,
                     dataset_format=DATASET_FORMAT):
  if dataset_format == DATASET_FORMAT:
    return os.path.join(dataset_dir, _FILE_PATTERN % split_name)
  elif dataset_format == 'tfrecord':
    return os.path.join(dataset_dir, _FILE_PATTERN_TFRECORD % split_name)


def get_decode_fn():
  """Returns a tfds decoder.

  Returns:
    A tf.data decoder.
  """

  def decode_fn(value):
    tensors = example_parser.decode_serialized_example(
        serialized_example=value,
        features=rio_specs.scene_feature_spec(with_annotations=True))
    tensor_dict = tfds.core.utils.flatten_nest_dict(tensors)
    return tensor_dict

  return decode_fn
