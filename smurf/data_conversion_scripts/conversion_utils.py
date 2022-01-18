# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Shared utils for converting datasets."""


import numpy as np
import tensorflow as tf


def generate_sharded_filenames(filename):
  name, num_shards = filename.split('@')
  return [f'{name}@{num}.tfrecord' for num in range(int(num_shards))]


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def read_flow(filename):
  """Read .flo file in Middlebury format."""
  with open(filename, 'rb') as f:
    magic_no = np.fromfile(f, np.float32, count=1)
    if magic_no != 202021.25:
      raise ValueError('Magic no. incorrect. {} is invalid.'.format(filename))
    else:
      width = np.fromfile(f, np.int32, count=1)
      height = np.fromfile(f, np.int32, count=1)
      data = np.fromfile(f, np.float32, count=2 * int(width) * int(height))
      return np.resize(data, (int(height), int(width), 2))


def write_flow(filename, uv):
  """Write optical flow to file using .flo Middlebury format.

  Args:
    filename: str, where to write .flo file
    uv: np.array, predicted flow in u, v coordinates (horizontal, vertical)
  """
  n_bands = 2
  tag_char = np.array([202021.25], np.float32)
  assert uv.ndim == 3
  assert uv.shape[2] == 2
  u = uv[:, :, 0]
  v = uv[:, :, 1]
  assert u.shape == v.shape
  height, width = u.shape
  with open(filename, 'wb') as f:
    f.write(tag_char)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    flow = np.zeros((height, width * n_bands))
    flow[:, np.arange(width) * 2] = u
    flow[:, np.arange(width) * 2 + 1] = v
    flow.astype(np.float32).tofile(f)
