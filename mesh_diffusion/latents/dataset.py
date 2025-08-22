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

"""Parse .tfrecord files w.r.t. to mesh data."""

import json

import numpy as np
import tensorflow as tf
from tensorflow.io import gfile


def load_dict(file_name):
  """Load dict from json file."""
  with gfile.GFile(file_name) as json_file:
    data = json.load(json_file)
  return data


def encode_example(geom):
  """Encode example."""
  features = {}
  key_list = {}

  for k in geom:
    feat = geom[k]
    if feat.dtype.char in np.typecodes['AllFloat']:
      feat = tf.convert_to_tensor(feat.astype(np.float32), dtype=tf.float32)
      key_list[k] = ['float', geom[k].shape]

    else:
      if k == 'texture' or k == 'image':
        feat = tf.convert_to_tensor(feat.astype(np.uint8), dtype=tf.uint8)
        key_list[k] = ['uint8', geom[k].shape]
      else:
        feat = tf.convert_to_tensor(feat.astype(np.int32), dtype=tf.int32)
        key_list[k] = ['int', geom[k].shape]
    feat_serial = tf.io.serialize_tensor(feat)
    features[k] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[feat_serial.numpy()])
    )

  return (
      tf.train.Example(
          features=tf.train.Features(feature=features)
      ).SerializeToString(),
      key_list,
  )


def encode_shape_example(geom):
  """"Encode shape example."""
  features = {}

  key_list = {}
  for k in geom:
    if k == 'num_levels':
      features['num_levels'] = tf.train.Feature(
          int64_list=tf.train.Int64List(value=[geom['num_levels']])
      )
    else:
      feat = geom[k]

      if feat.dtype.char in np.typecodes['AllFloat']:
        feat = tf.convert_to_tensor(feat.astype(np.float32), dtype=tf.float32)
        key_list[k] = ['float', geom[k].shape]

      else:
        if k == 'texture' or k == 'image':
          feat = tf.convert_to_tensor(feat.astype(np.uint8), dtype=tf.uint8)
          key_list[k] = ['uint8', geom[k].shape]
        else:
          feat = tf.convert_to_tensor(feat.astype(np.int32), dtype=tf.int32)
          key_list[k] = ['int', geom[k].shape]

      feat_serial = tf.io.serialize_tensor(feat)
      features[k] = tf.train.Feature(
          bytes_list=tf.train.BytesList(value=[feat_serial.numpy()])
      )

  return (
      tf.train.Example(
          features=tf.train.Features(feature=features)
      ).SerializeToString(),
      key_list,
  )


def load_tfr_dataset(files, ordered=True):  # pylint: disable=unused-argument
  """Load TFRecords dataset."""
  dataset = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
  return dataset


# ------------------------------------------------------------------------------
# Use for shapes.
# ------------------------------------------------------------------------------


def load_shape_tfrecords(path):
  """Load TFRecords dataset."""
  shape_keys = load_dict(path + '/shape_keys.json')
  files = gfile.glob(path + '*.tfrecords')
  return load_tfr_dataset(files), shape_keys


def get_des(key_list):
  """Get shape_des."""
  shape_des = {}
  for k in key_list:
    shape_des[k] = tf.io.FixedLenFeature([], tf.string)

  return shape_des


def parse_data(ex, key_list, shape_des):
  """Parse data."""
  example = tf.io.parse_single_example(ex, shape_des)
  shape = {}

  for k in key_list:
    dat = example[k]
    if key_list[k][0] == 'float':
      feat = tf.io.parse_tensor(dat, tf.float32)
      feat = tf.ensure_shape(feat, key_list[k][1])
    else:
      if key_list[k][0] == 'uint8':
        feat = tf.io.parse_tensor(dat, tf.uint8)
      else:
        feat = tf.io.parse_tensor(dat, tf.int32)

      feat = tf.ensure_shape(feat, key_list[k][1])

    shape[k] = feat

  return shape


def parser(key_list):
  """Parser."""
  shape_des = get_des(key_list)
  return lambda ex: parse_data(ex, key_list, shape_des)


def get_num_repeat(num_el, num_steps, batch_dim=1):
  """Get number of dataset repeats."""
  num_repeat = (num_steps * batch_dim // (num_el)) + 1
  return num_repeat


# ------------------------------------------------------------------------------
# Use for shapes.
# ------------------------------------------------------------------------------


def parse_image(ex):
  """Convert image to float in [0, 1]."""
  im = ex['image']
  im_rescale = tf.cast(im, tf.float32) / 255.0
  return {'image': im, 'im_rescale': im_rescale}


def resize_512(ex):
  """Thin wrapper around tf.image.resize."""
  im = ex['image']
  im = tf.image.resize(im, [512, 512], method='nearest')
  return {'image': im}


def crop_512(ex):
  """Thin wrapper around tf.image.resize_with_crop_or_pad."""
  im = ex['image']
  im = tf.image.resize_with_crop_or_pad(im, 512, 512)
  return {'image': im}
