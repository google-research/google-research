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

"""Dataset class for parsing tfrecord files."""
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import nest


class Dataset(object):
  """Dataset class for parsing tfrecords files."""

  _feature_description = {
      'image_raw': tf.FixedLenFeature([], tf.string),
      'height': tf.FixedLenFeature([], tf.int64),
      'width': tf.FixedLenFeature([], tf.int64),
      'depth': tf.FixedLenFeature([], tf.int64),
      'label': tf.FixedLenFeature([], tf.int64)
  }

  def __init__(self, tfrecord_files, img_shape, labeled=False):
    super(Dataset, self).__init__()
    self._tfrecords = nest.flatten(tfrecord_files)
    self._img_shape = img_shape
    tf.logging.info(tfrecord_files)
    tf.logging.info(labeled)
    if labeled:
      self._feature_description['labeled'] = tf.FixedLenFeature([], tf.int64)

  def __call__(self):
    ds = tf.data.TFRecordDataset(self._tfrecords)
    return ds.map(self._parse_example)

  def _parse_example(self, example_proto):
    d = tf.parse_single_example(example_proto, self._feature_description)

    img = tf.decode_raw(d['image_raw'], tf.uint8)
    img = tf.reshape(img, self._img_shape)
    d['image'] = tf.to_float(img) / 255.
    del d['image_raw']

    return d
