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

"""Build norb tf records from norb files."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow.compat.v1 as tf  # tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', '/tmp/tensorflow/mnist/input_data',
                           'Directory for storing input data')
tf.app.flags.DEFINE_bool('debug', False, 'If set pring extra debug info.')


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_record(dataset, filename):
  """Write a tf record for norb."""
  writer = tf.python_io.TFRecordWriter(filename)
  print('result dim:')
  print(dataset['images'].shape)
  for image, label, meta in zip(dataset['images'], dataset['labels'],
                                dataset['meta']):
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'height': _int64_feature(image.shape[1]),
                'width': _int64_feature(image.shape[2]),
                'depth': _int64_feature(2),
                'label': _int64_feature(label),
                'meta': _int64_list_feature(meta),
                'image_raw': _bytes_feature(image.tostring()),
            }))
    writer.write(example.SerializeToString())
  writer.close()


def get_path(split, filetype):
  if split == 'train':
    instance_list = '46789'
  else:
    instance_list = '01235'
  filename = 'smallnorb-5x%sx9x18x6x2x96x96-%s-%s.mat' % (
      instance_list, split + 'ing', filetype)
  return os.path.join(FLAGS.data_dir, filename)


def read_nums(file_handle, num_type, count):
  """Reads 4 bytes from file, returns it as a 32-bit integer."""
  num_bytes = count * np.dtype(num_type).itemsize
  string = file_handle.read(num_bytes)
  return np.fromstring(string, dtype=num_type)


def read_header(file_handle):
  """Read the header of the norb files."""
  key_to_type = {
      0x1E3D4C51: ('float32', 4),
      0x1E3D4C53: ('float64', 8),
      0x1E3D4C54: ('int32', 4),
      0x1E3D4C55: ('uint8', 1),
      0x1E3D4C56: ('int16', 2)
  }

  type_key = read_nums(file_handle, 'int32', 1)[0]
  elem_type, elem_size = key_to_type[type_key]
  if FLAGS.debug:
    print("header's type key, type, type size: {}, {}, {} ".format(
        type_key, elem_type, elem_size))

  num_dims = read_nums(file_handle, 'int32', 1)[0]
  if FLAGS.debug:
    print('# of dimensions, according to header: {}'.format(num_dims))
  shape = np.fromfile(
      file_handle, dtype='int32', count=max(num_dims, 3))[:num_dims]

  if FLAGS.debug:
    print('Tensor shape, as listed in header: {}'.format(shape))

  return elem_type, elem_size, shape


def parse_norb_file(file_handle):
  elem_type, _, shape = read_header(file_handle)
  file_handle.tell()

  num_elems = np.prod(shape)

  result = np.fromfile(
      file_handle, dtype=elem_type, count=num_elems).reshape(shape)
  return result


def main(_):
  dataset = {}
  for s in ['train', 'test']:
    file_handle = open(get_path(s, 'dat'))
    dataset['images'] = parse_norb_file(file_handle)
    file_handle = open(get_path(s, 'cat'))
    dataset['labels'] = parse_norb_file(file_handle)
    file_handle = open(get_path(s, 'info'))
    dataset['meta'] = parse_norb_file(file_handle)
    write_record(dataset,
                 os.path.join(FLAGS.data_dir, '{}duo.tfrecords'.format(s)))


if __name__ == '__main__':
  tf.app.run()
