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

"""Efficient input pipeline using tf.data.Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import functools

from absl import flags
from inputs import resnet_preprocessing
import tensorflow as tf
from tensorflow.contrib import data as contrib_data

FLAGS = flags.FLAGS

flags.DEFINE_boolean(
    "condv2", default=False, help="Use CondV2 as part of the input pipeline.")
flags.DEFINE_integer("subset_example_per_class", 0, "")
flags.DEFINE_integer("subset_idx", 0, "")

# BELOW WILL BE SIMPLIFIED.

# The input tensor is in the range of [0, 255], we need to scale them to the
# range of [0, 1]
MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]

num_classes_map = {"pets": 37}


def filenames_for_oxford_pets(mode):
  """The list of filenames for Oxford Pets based on 'mode'."""

  num_train_images = 3680
  num_valid_images = 740
  num_classes = 37
  filenames = ""  # TO BE MODIFIED

  if mode == "train":
    num_images = num_train_images

  elif mode == "l2l_train":
    if FLAGS.subset_example_per_class:
      num_images = num_classes * FLAGS.subset_example_per_class
    else:
      num_images = num_train_images - num_valid_images
  elif mode == "l2l_valid":
    num_images = num_valid_images
  elif mode == "test":
    num_images = 3669
  else:
    raise ValueError("Invalid mode: %s" % mode)

  return filenames, num_images


def get_filenames_func():
  """return the dataset function for different datasets."""
  if FLAGS.target_dataset == "pets":
    get_filenames = filenames_for_oxford_pets
  return get_filenames


class ImageNetTFExampleInput(object):
  """Base class for ImageNet input_fn generator."""
  __metaclass__ = abc.ABCMeta

  def __init__(self,
               is_training,
               use_bfloat16,
               image_size=224,
               transpose_input=False,
               task_id=0,
               dataset_name="",
               num_classes=-1,
               num_parallel_calls=8):
    self.num_classes = num_classes
    self.task_id = task_id
    self.dataset_name = dataset_name
    self.is_training = is_training
    self.use_bfloat16 = use_bfloat16
    self.transpose_input = transpose_input
    self.image_size = image_size
    self.num_parallel_calls = num_parallel_calls

  def set_shapes(self, batch_size, images, labels):
    """Statically set the batch_size dimension."""
    if self.transpose_input:
      images.set_shape(images.get_shape().merge_with(
          tf.TensorShape([None, None, None, batch_size])))
      images = tf.reshape(images, [-1])
      labels.set_shape(labels.get_shape().merge_with(
          tf.TensorShape([batch_size])))
    else:
      images.set_shape(images.get_shape().merge_with(
          tf.TensorShape([batch_size, None, None, None])))
      labels.set_shape(labels.get_shape().merge_with(
          tf.TensorShape([batch_size])))

    return images, labels

  def dataset_parser(self, value):
    return self.dataset_parser_normal(value)

  def dataset_parser_normal(self, value):
    """Parses an image and its label from a serialized TFExample.

    Args:
      value: serialized string containing an ImageNet TFExample.

    Returns:
      Returns a tuple of (image, label) from the TFExample.
    """
    keys_to_features = {
        "image/encoded": tf.FixedLenFeature((), tf.string, ""),
    }
    label_key = "image/class/label"
    keys_to_features[label_key] = tf.FixedLenFeature([], tf.int64, -1)

    parsed = tf.parse_single_example(value, keys_to_features)
    image_bytes = tf.reshape(parsed["image/encoded"], shape=[])

    image = resnet_preprocessing.preprocess_image(
        image_bytes=image_bytes,
        is_training=self.is_training,
        image_size=self.image_size,
        use_bfloat16=self.use_bfloat16)
    image -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=image.dtype)
    image /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=image.dtype)

    # Subtract one so that labels are in [0, 1000).
    if self.task_id == 0 and FLAGS.model_type == "resnet":
      label_subtract = 1
    else:
      label_subtract = 0

    label = tf.cast(
        tf.reshape(parsed[label_key], shape=[]),
        dtype=tf.int32) - label_subtract

    return image, label

  @abc.abstractmethod
  def make_source_dataset(self, index, num_hosts):
    """Makes dataset of serialized TFExamples.

    The returned dataset will contain `tf.string` tensors, but these strings are
    serialized `TFExample` records that will be parsed by `dataset_parser`.

    If self.is_training, the dataset should be infinite.

    Args:
      index: current host index.
      num_hosts: total number of hosts.

    Returns:
      A `tf.data.Dataset` object.
    """
    return

  def input_fn(self, params):
    """Input function which provides a single batch for train or eval."""

    batch_size = params["batch_size"]

    dataset = self.make_source_dataset

    parser = self.dataset_parser_ss

    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            parser,
            batch_size=batch_size,
            num_parallel_batches=self.num_parallel_calls,
            drop_remainder=True))

    dataset = dataset.map(
        lambda images, labels: (tf.transpose(images, [1, 2, 3, 0]), labels),
        num_parallel_calls=self.num_parallel_calls)

    # Assign static batch size dimension
    dataset = dataset.map(functools.partial(self.set_shapes, batch_size))

    return dataset


class ImageNetInput(ImageNetTFExampleInput):
  """Generates ImageNet input_fn from a series of TFRecord files.

  The training data is assumed to be in TFRecord format with keys as specified
  in the dataset_parser below, sharded across 1024 files, named sequentially:

      train-00000-of-01024
      train-00001-of-01024
      ...
      train-01023-of-01024

  The validation data is in the same format but sharded in 128 files.

  The format of the data required is created by the script at:
      https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py
  """

  def __init__(self,
               is_training,
               use_bfloat16,
               transpose_input,
               data_dir,
               image_size=224,
               num_parallel_calls=8,
               cache=False,
               task_id=0,
               num_classes=-1,
               dataset_name="",
               dataset_split=None,
               shuffle_shards=False):
    """Creates an input from TFRecord files.

    Args:
      is_training: `bool` for whether the input is for training
      use_bfloat16: If True, use bfloat16 precision; else use float32.
      transpose_input: 'bool' for whether to use the double transpose trick
      data_dir: `str` for the directory of the training and validation data; if
        'null' (the literal string 'null') or implicitly False then construct a
        null pipeline, consisting of empty images and blank labels.
      image_size: `int` image height and width.
      num_parallel_calls: concurrency level to use when reading data from disk.
      cache: if true, fill the dataset by repeating from its cache
      task_id: task id.
      num_classes: number of classes.
      dataset_name: dataset name.
      dataset_split: If provided, must be one of 'train' or 'validation' and
        specifies the dataset split to read, overriding the default set by
        is_training. In this case, is_training specifies whether the data is
        augmented.
      shuffle_shards: Whether to shuffle the dataset shards.
    """
    super(ImageNetInput, self).__init__(
        is_training=is_training,
        image_size=image_size,
        use_bfloat16=use_bfloat16,
        transpose_input=transpose_input,
        dataset_name=dataset_name,
        num_classes=num_classes,
        task_id=task_id)
    self.data_dir = data_dir
    if self.data_dir == "null" or not self.data_dir:
      self.data_dir = None
    self.num_parallel_calls = num_parallel_calls
    self.cache = cache
    self.dataset_split = dataset_split
    self.shuffle_shards = shuffle_shards

  def _get_null_input(self, data):
    """Returns a null image (all black pixels)."""
    del data  # Unused since output is constant regardless of input
    return tf.zeros([self.image_size, self.image_size, 3],
                    tf.bfloat16 if self.use_bfloat16 else tf.float32)

  def dataset_parser(self, value):
    """See base class."""
    if not self.data_dir:
      return value, tf.constant(0, tf.int32)
    return super(ImageNetInput, self).dataset_parser(value)

  def dataset_parser_ss(self, unused, value):  # pylint: disable=unused-argument
    if not self.data_dir:
      return value, tf.constant(0, tf.int32)
    return super(ImageNetInput, self).dataset_parser(value)

  def make_source_dataset(self, index=0, num_hosts=1):
    """See base class."""

    if not self.data_dir:
      tf.logging.info("Undefined data_dir implies null input")
      return tf.data.Dataset.range(1).repeat().map(self._get_null_input)

    get_filenames = get_filenames_func()
    filenames, _ = get_filenames(self.dataset_split)
    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    if self.is_training and not self.cache:
      if filenames is not None:
        dataset = dataset.shuffle(len(filenames))
      dataset = dataset.repeat()

    def fetch_dataset(filename):
      buffer_size = 8 * 1024 * 1024  # 8 MB per file
      dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
      return dataset

    cycle_length = 64
    shuffle_size = 1024

    # Read the data from disk in parallel
    if self.is_training:
      dataset = dataset.apply(
          contrib_data.parallel_interleave(
              fetch_dataset, cycle_length=cycle_length, sloppy=True))
    else:
      dataset = dataset.apply(
          contrib_data.parallel_interleave(
              fetch_dataset, cycle_length=1, sloppy=False))

    if self.cache:
      dataset = dataset.cache().apply(
          contrib_data.shuffle_and_repeat(shuffle_size))
    else:
      if self.is_training:
        dataset = dataset.shuffle(shuffle_size)
    return dataset
