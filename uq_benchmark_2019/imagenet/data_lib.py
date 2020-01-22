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

# Lint as: python2, python3
"""Library for constructing data for imagenet experiments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
from absl import flags
from absl import logging

import attr
import robustness_dhtd
from six.moves import range
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from uq_benchmark_2019 import image_data_utils
from uq_benchmark_2019.imagenet import imagenet_input

flags.DEFINE_string('imagenet_dir', None, 'Path to IMAGENET data tables.')
flags.DEFINE_string('imagenet_c_dir', None, 'Path to IMAGENET-C data tables.')
FLAGS = flags.FLAGS
gfile = tf.io.gfile

IMAGENET_SHAPE = (224, 224, 3)

# Imagenet training and test data sets.
IMAGENET_NUM_CLASSES = 1000
IMAGE_SIZE = 224
APPROX_IMAGENET_TRAINING_IMAGES = 1280000  # Approximate number of images.
IMAGENET_VALIDATION_IMAGES = 50000  # Number of images.


def _download_alt_dataset(config, shuffle_files):
  dataset_builder = tfds.builder(config.alt_dataset_name)
  dataset_builder.download_and_prepare()
  return dataset_builder.as_dataset(split=config.split,
                                    shuffle_files=shuffle_files)


def build_dataset(
    config, batch_size, is_training=False, fake_data=False, use_bfloat16=False):
  """Returns a tf.data.Dataset with <image, label> pairs.

  Args:
    config: DataConfig instance.
    batch_size: Dataset batch size.
    is_training: Whether to build a dataset for training
        (with shuffling and image distortions).
    fake_data: If True, use randomly generated data.
    use_bfloat16: If True, use bfloat16. If False, use float32.
  Returns:
    tf.data.Dataset
  """
  if fake_data:
    logging.info('Generating fake data for config: %s', config)
    return image_data_utils.make_fake_data(IMAGENET_SHAPE).batch(batch_size)

  if config.alt_dataset_name:
    dataset = _download_alt_dataset(config, shuffle_files=is_training)

    def prep_fn(image_input):
      image = tf.image.convert_image_dtype(image_input['image'], tf.float32)
      image = tf.image.crop_to_bounding_box(image, 20, 0, 178, 178)
      image = tf.image.resize(image, (224, 224))

      # omit CelebA labels
      return image, -1

    return dataset.map(prep_fn).batch(batch_size)

  logging.info('Building dataset for config:\n%s', attr.asdict(config))
  if config.corruption_type and config.corruption_static:
    return image_data_utils.make_static_dataset(
        config, _get_static_imagenet_c).batch(batch_size)

  dataset_builder = imagenet_input.ImageNetInput(
      is_training=is_training, data_dir=FLAGS.imagenet_dir,
      batch_size=batch_size, dataset_split=config.split,
      use_bfloat16=use_bfloat16)

  dataset = dataset_builder.input_fn()

  if config.corruption_type:
    assert (config.corruption_value is not None) != (
        config.corruption_level > 0)

    # NOTE: dhtd corruptions expect to be applied before float32 conversion.
    def apply_corruption(image, label):
      """Apply the corruption function to the image."""
      image = tf.image.convert_image_dtype(image, tf.uint8)
      corruption_fn = functools.partial(
          robustness_dhtd.corrupt,
          severity=config.corruption_level,
          severity_value=config.corruption_value, dim=224,
          corruption_name=config.corruption_type, dataset_name='imagenet')

      def apply_to_batch(ims):
        ims_numpy = ims.numpy()
        for i in range(ims_numpy.shape[0]):
          ims_numpy[i] = corruption_fn(ims_numpy[i])
        return ims_numpy

      image = tf.py_function(func=apply_to_batch, inp=[image], Tout=tf.float32)
      image = tf.clip_by_value(image, 0., 255.) / 255.
      return image, label

    dataset = dataset.map(apply_corruption)

  if config.roll_pixels:
    def roll_fn(image, label):
      """Function to roll pixels."""
      image = tf.roll(image, config.roll_pixels, -2)
      return image, label
    dataset = dataset.map(roll_fn)

  return dataset


def _get_static_imagenet_c(corruption_type, corruption_level,
                           num_parallel_reads=1):
  """Load static imagenet-C images to tf.dataset."""

  imagenet_c_images = os.path.join(
      FLAGS.imagenet_c_dir, corruption_type, str(corruption_level), 'val/*')
  filenames = gfile.glob(imagenet_c_images)

  def parse(serialized):
    """Parses a serialized tf.Example."""
    keys_to_features = {
        'image': tf.io.FixedLenFeature([], dtype=tf.string),
        'label': tf.io.FixedLenFeature([], dtype=tf.int64),
    }
    out = tf.io.parse_single_example(serialized, keys_to_features)
    image = tf.image.decode_image(out['image'], channels=3)
    image.set_shape([None, None, 3])
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, out['label']

  dataset = tf.data.TFRecordDataset(
      filenames, num_parallel_reads=num_parallel_reads)

  return dataset.map(parse, num_parallel_calls=1)
