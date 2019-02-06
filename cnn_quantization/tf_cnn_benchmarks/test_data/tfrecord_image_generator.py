# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Generate black and white test TFRecords with Example protos.

Each record within the TFRecord file is a
serialized Example proto. The Example proto contains the following fields:

  image/encoded: string containing JPEG encoded image in RGB colorspace
  image/height: integer, image height in pixels
  image/width: integer, image width in pixels
  image/colorspace: string, specifying the colorspace, always 'RGB'
  image/channels: integer, specifying the number of channels, always 3
  image/format: string, specifying the format, always'JPEG'

  image/filename: string containing the basename of the image file
            e.g. 'n01440764_10026.JPEG' or 'ILSVRC2012_val_00000293.JPEG'
  image/class/label: integer specifying the index in a classification layer.
    The label ranges from [1, 1000] where 0 is not used.
  image/class/synset: string specifying the unique ID of the label,
    e.g. 'n01440764'
  image/class/text: string specifying the human-readable version of the label
    e.g. 'red fox, Vulpes vulpes'

  image/object/bbox/xmin: list of integers specifying the 0+ human annotated
    bounding boxes
  image/object/bbox/xmax: list of integers specifying the 0+ human annotated
    bounding boxes
  image/object/bbox/ymin: list of integers specifying the 0+ human annotated
    bounding boxes
  image/object/bbox/ymax: list of integers specifying the 0+ human annotated
    bounding boxes
  image/object/bbox/label: integer specifying the index in a classification
    layer. The label ranges from [1, 1000] where 0 is not used. Note this is
    always identical to the image label.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

import numpy as np
import tensorflow as tf


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
  """Wrapper for inserting float features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, label, synset, human, bbox,
                        height, width):
  """Build an Example proto for an example.

  Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
    label: integer, identifier for the ground truth for the network
    synset: string, unique WordNet ID specifying the label, e.g., 'n02323233'
    human: string, human-readable label, e.g., 'red fox, Vulpes vulpes'
    bbox: list of bounding boxes; each box is a list of integers
      specifying [xmin, ymin, xmax, ymax]. All boxes are assumed to belong to
      the same label as the image label.
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto
  """
  xmin = []
  ymin = []
  xmax = []
  ymax = []
  for b in bbox:
    assert len(b) == 4
    # pylint: disable=expression-not-assigned
    [l.append(point) for l, point in zip([xmin, ymin, xmax, ymax], b)]
    # pylint: enable=expression-not-assigned

  colorspace = 'RGB'
  channels = 3
  image_format = 'JPEG'

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/colorspace': _bytes_feature(colorspace),
      'image/channels': _int64_feature(channels),
      'image/class/label': _int64_feature(label),
      'image/class/synset': _bytes_feature(synset),
      'image/class/text': _bytes_feature(human),
      'image/object/bbox/xmin': _float_feature(xmin),
      'image/object/bbox/xmax': _float_feature(xmax),
      'image/object/bbox/ymin': _float_feature(ymin),
      'image/object/bbox/ymax': _float_feature(ymax),
      'image/object/bbox/label': _int64_feature([label] * len(xmin)),
      'image/format': _bytes_feature(image_format),
      'image/filename': _bytes_feature(os.path.basename(filename)),
      'image/encoded': _bytes_feature(image_buffer)}))
  return example


class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that converts PNG to JPEG data.
    self._image = tf.placeholder(dtype=tf.uint8)
    self._encode_jpeg = tf.image.encode_jpeg(
        self._image, format='rgb', quality=100)

  def encode_jpeg(self, image):
    jpeg_image = self._sess.run(self._encode_jpeg,
                                feed_dict={self._image: image})
    return jpeg_image


def _process_image(coder, name):
  """Process a single image file.

  If name is "train", a black image is returned. Otherwise, a white image is
  returned.

  Args:
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    name: string, unique identifier specifying the data set.
  Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
  """
  # Read the image file.
  value = 0 if name == 'train' else 255
  height = random.randint(30, 299)
  width = random.randint(30, 299)
  image = np.full((height, width, 3), value, np.uint8)

  jpeg_data = coder.encode_jpeg(image)

  return jpeg_data, height, width


def _process_dataset(output_directory, num_classes, coder, name, num_images,
                     num_shards):
  """Process a complete data set and save it as a TFRecord.

  Args:
    output_directory: Where to put outputs.
    num_classes: number of classes.
    coder: Instance of an ImageCoder.
    name: string, unique identifier specifying the data set.
    num_images: number of images to generate.
    num_shards: integer number of shards to create.
  """
  files_per_shard = num_images // num_shards
  for shard in range(num_shards):
    output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
    output_file = os.path.join(output_directory, output_filename)
    with tf.python_io.TFRecordWriter(output_file) as writer:
      for i in range(files_per_shard):
        index = shard * files_per_shard + i
        image_buffer, height, width = _process_image(coder, name)

        filename = '{}_{}_{}'.format(name, shard, i)
        label = index % num_classes
        synset = str(index)
        human = name
        bbox = [[0.1, 0.1, 0.9, 0.9]]
        example = _convert_to_example(filename, image_buffer, label,
                                      synset, human, bbox,
                                      height, width)
        writer.write(example.SerializeToString())


def write_black_and_white_tfrecord_data(
    output_directory, num_classes, num_train_images=512,
    num_validation_images=128, train_shards=8, validation_shards=2):
  """Writes black and white images in tfrecord format.

  Training images are black and validation images are white.

  Args:
    output_directory: Where to put outputs.
    num_classes: number of classes.
    num_train_images: number of training images to generate.
    num_validation_images: number of validation images to generate.
    train_shards: integer number of training shards to create.
    validation_shards: integer number of validation shards to create.
  """

  coder = ImageCoder()
  _process_dataset(output_directory, num_classes, coder, 'validation',
                   num_validation_images, validation_shards)
  _process_dataset(output_directory, num_classes, coder, 'train',
                   num_train_images, train_shards)
