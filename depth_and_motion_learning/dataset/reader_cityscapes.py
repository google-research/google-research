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

"""Reader functions for Cityscapes/KITTI struct2depth data.

  This implements the interface functions for reading Cityscapes data.

  Each image file consists of 3 consecutive frames concatenated along the width
  dimension, stored in png format. The camera intrinsics are stored in a file
  that has the same name as the image, with a 'txt' extension and the
  coefficients flattened inside. The 'train.txt' file lists the training
  samples.

"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import os


import tensorflow.compat.v1 as tf
from depth_and_motion_learning import parameter_container


FORMAT_NAME = 'STRUCT2DEPTH'
IMAGE_WIDTH = 416
IMAGE_HEIGHT = 128
IMAGES_PER_SEQUENCE = 3
CHANNELS = 3

# One of the intrinsics files on KITTI is missing, and since fixing it requires
# re-copying the entire dataset to Placer, we meanwhile make this quick patch
# (otherwise KITTI training crashes).
KITTI_CORRUPT_FILE = '2011_09_26_drive_0001_sync_02/0000000001'
KITTI_CORRUPT_FILE_INTRINSICS = ('241.67446312399355,0.0,204.16801030595812,'
                                 '0.0,246.28486826666665,59.000832,'
                                 '0.0,0.0,1.0')


READ_FRAME_PAIRS_FROM_DATA_PATH_PARAMS = {

    # Number of parallel threads for reading.
    'num_parallel_calls': 64,

}


def read_frame_pairs_from_data_path(train_file_path, params=None):
  """Reads frame pairs from a text file in the struct2depth format.

  Args:
    train_file_path: A string, file path to the text file listing the training
      examples.
    params: A dictionary or a ParameterContainer with overrides for the default
      params (READ_FRAME_PAIRS_FROM_DATA_PATH_PARAMS)

  Returns:
    A dataset object.
  """
  return read_frame_sequence_from_data_path(
      train_file_path, sequence_length=2, params=params)


def read_frame_sequence_from_data_path(train_file_path,
                                       sequence_length=IMAGES_PER_SEQUENCE,
                                       params=None):
  """Reads frames sequences from a text file in the struct2depth format.

  Args:
    train_file_path: A string, file path to the text file listing the training
      examples.
    sequence_length: Number of images in the output sequence (1, 2, or 3).
    params: A dictionary or a ParameterContainer with overrides for the default
      params (READ_FRAME_PAIRS_FROM_DATA_PATH_PARAMS)

  Returns:
    A dataset object.
  """
  if sequence_length not in (1, 2, 3):
    raise ValueError('sequence_length must be in (1, 2, 3), not %d.' %
                     sequence_length)
  params = parameter_container.ParameterContainer(
      READ_FRAME_PAIRS_FROM_DATA_PATH_PARAMS, params)

  with tf.gfile.Open(train_file_path) as f:
    lines = f.read().split('\n')

  lines = list(filter(None, lines))  # Filter out empty strings.

  directory = os.path.dirname(train_file_path)

  def make_filename(line):
    words = line.split(' ')
    if len(words) != 2:
      raise RuntimeError('Invalid train file: Each lines are expected to have '
                         'exactly two words, and the line "%s" does not '
                         'conform.' % line)
    return os.path.join(directory, words[0], words[1])

  files = [make_filename(line) for line in lines]

  ds = tf.data.Dataset.from_tensor_slices(files)
  ds = ds.repeat()

  def parse_fn_for_pairs(filename):
    return parse_fn(filename, output_sequence_length=sequence_length)

  num_parallel_calls = min(len(lines), params.num_parallel_calls)
  ds = ds.map(parse_fn_for_pairs, num_parallel_calls=num_parallel_calls)

  return ds


def parse_fn(filename,
             output_sequence_length=IMAGES_PER_SEQUENCE):
  """Read data from single files stored in directories.

  Args:
    filename: the filename of the set of files to be loaded.
    output_sequence_length: Length of the output sequence. If less than
      IMAGES_PER_SEQUENCE, only the first `output_sequence_length` frames will
      be kept.

  Returns:
    A dictionary that maps strings to tf.Tensors of type float32:

    'rgb': an RGB image of shape H, W, 3. Each channel value is between 0.0 and
           1.0.
    'intrinsics': a list of intrinsics values.
  """
  if output_sequence_length > IMAGES_PER_SEQUENCE or output_sequence_length < 1:
    raise ValueError('Invalid output_sequence_length %d: must be within [1, '
                     '%d].' % (output_sequence_length, IMAGES_PER_SEQUENCE))
  image_file = tf.strings.join([filename, '.png'])
  intrinsics_file = tf.strings.join([filename, '_cam.txt'])
  mask_file = tf.strings.join([filename, '-fseg.png'])

  # Read files.
  encoded_image = tf.io.read_file(image_file)
  encoded_mask = tf.io.read_file(mask_file)
  intrinsics_content = tf.io.read_file(intrinsics_file)
  content_is_empty = tf.math.equal(intrinsics_content, '')
  filename_matches = tf.strings.regex_full_match(filename,
                                                 '.*%s$' % KITTI_CORRUPT_FILE)
  file_is_corrupt = tf.math.logical_and(content_is_empty, filename_matches)

  intrinsics_content = tf.cond(file_is_corrupt,
                               lambda: KITTI_CORRUPT_FILE_INTRINSICS,
                               lambda: intrinsics_content)

  # Parse intrinsics data to a tensor representing a 3x3 matrix.
  intrinsics = tf.strings.split([intrinsics_content], ',').values
  intrinsics = tf.strings.to_number(intrinsics)
  intrinsics.set_shape([9])

  fx, _, x0, _, fy, y0, _, _, _ = tf.unstack(intrinsics)
  intrinsics = tf.stack([IMAGE_WIDTH, IMAGE_HEIGHT, fx, fy, x0, y0])

  # Decode and normalize images.
  decoded_image = tf.image.decode_png(encoded_image, channels=3)
  decoded_image = tf.to_float(decoded_image) * (1 / 255.0)
  split_image_sequence = tf.split(decoded_image, IMAGES_PER_SEQUENCE, axis=1)

  decoded_mask = tf.image.decode_png(encoded_mask, channels=3)
  mask_r, mask_g, mask_b = tf.unstack(tf.to_int32(decoded_mask), axis=-1)
  # Since TPU does not support images of type uint8, we encode the 3 RGB uint8
  # values into one int32 value.
  mask = mask_r * (256 * 256) + mask_g * 256 + mask_b
  # All images in our pipeline have 3 dimensions (height, width, channels), so
  # we add a third dimension to the mask too.
  mask = tf.expand_dims(mask, -1)
  split_mask_sequence = tf.split(mask, IMAGES_PER_SEQUENCE, axis=1)

  return {
      'rgb': tf.stack(split_image_sequence[:output_sequence_length]),
      'intrinsics': tf.stack([intrinsics] * output_sequence_length),
      'mask': tf.stack(split_mask_sequence[:output_sequence_length]),
  }


def read_and_parse_data(files):
  """Default reader and parser for reading Cityscapes/KITTI data from files.

  Args:
    files: a list of filenames. Each filename is extended to image and camera
      intrinsics filenames in parse_fn.

  Returns:
    A preprocessing function representing data stored as a collection of files.
  """
  ds = tf.data.Dataset.from_tensor_slices(files)
  ds = ds.repeat()
  ds = ds.map(parse_fn)

  return ds
