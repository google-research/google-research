# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Data provider with an argument to control data augmentation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import functools
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

from low_rank_local_connectivity import utils


def extract_data(data, preprocess_image):
  """Extracts image, label and create a mask."""
  image = data["image"]
  # Reserve label 0 for background
  label = tf.cast(data["label"], dtype=tf.int32)
  # Create a mask variable to track the real vs padded data in the last batch.
  mask = 1.
  image = preprocess_image(image)
  return image, label, mask


def construct_iterator(dataset_builder,
                       split,
                       preprocess_fn,
                       batch_size,
                       is_training):
  """Constructs data iterator.

  Args:
    dataset_builder: tensorflow_datasets data builder.
    split: tensorflow_datasets data split.
    preprocess_fn: Function that preprocess each data example.
    batch_size: (Integer) Batch size.
    is_training: (boolean) Whether training or inference mode.
  Returns:
    Data iterator.

  """
  dataset = dataset_builder.as_dataset(split=split, shuffle_files=True)
  dataset = dataset.map(preprocess_fn,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
  if is_training:
    # 4096 is ~0.625 GB of RAM. Reduce if memory issues encountered.
    dataset = dataset.shuffle(buffer_size=4096)
  dataset = dataset.repeat(-1 if is_training else 1)
  dataset = dataset.batch(batch_size, drop_remainder=is_training)

  if not is_training:
    # Pad the remainder of the last batch to make batch size fixed.
    dataset = utils.pad_to_batch(dataset, batch_size)

  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  return tf.compat.v1.data.make_one_shot_iterator(dataset)


class MNISTDataProvider(object):
  """MNIST Data Provider.

  Attributes:
    images: (4-D tensor) Images of shape (batch, height, width, channels).
    labels: (1-D tensor) Data labels of size (batch,).
    mask: (1-D boolean tensor) Data mask. Used when data is not repeated to
      indicate the fraction of the batch with true data in the final batch.
    num_classes: (Integer) Number of classes in the dataset.
    num_examples: (Integer) Number of examples in the dataset.
    class_names: (List of Strings) MNIST id for class labels.
    num_channels: (integer) Number of image color channels.
    image_size: (Integer) Size of the image.
    iterator: Tensorflow data iterator.
  """

  def __init__(self,
               subset,
               batch_size,
               is_training,
               data_dir=None):
    dataset_builder = tfds.builder("mnist", data_dir=data_dir)
    dataset_builder.download_and_prepare(download_dir=data_dir)
    self.image_size = 28
    if subset == "train":
      split = tfds.core.ReadInstruction("train", from_=8, to=100, unit="%")
    elif subset == "valid":
      split = tfds.core.ReadInstruction("train", from_=0, to=8, unit="%")
    elif subset == "test":
      split = tfds.Split.TEST
    else:
      raise ValueError("subset %s is undefined " % subset)
    self.num_channels = 1
    iterator = construct_iterator(
        dataset_builder, split, self._preprocess_fn(), batch_size, is_training)

    info = dataset_builder.info
    self.iterator = iterator
    self.images, self.labels, self.mask = iterator.get_next()
    self.num_classes = info.features["label"].num_classes
    self.class_names = info.features["label"].names
    self.num_examples = info.splits[split].num_examples

  def _preprocess_fn(self):
    """Preprocessing function."""
    image_size = self.image_size
    def preprocess_image(image):
      """Preprocessing."""
      # Normalize to 0-1 range.
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
      image = 2 * image - 1
      return tf.image.resize_image_with_crop_or_pad(
          image, image_size, image_size)

    return functools.partial(extract_data, preprocess_image=preprocess_image)


class CIFAR10DataProvider(object):
  """CIFAR10 Data Provider.

  Attributes:
    images: (4-D tensor) Images of shape (batch, height, width, channels).
    labels: (1-D tensor) Data labels of size (batch,).
    mask: (1-D boolean tensor) Data mask. Used when data is not repeated to
      indicate the fraction of the batch with true data in the final batch.
    num_classes: (Integer) Number of classes in the dataset.
    num_examples: (Integer) Number of examples in the dataset.
    class_names: (List of Strings) CIFAR10 id for class labels.
    num_channels: (integer) Number of image color channels.
    image_size: (Integer) Size of the image.
    iterator: Tensorflow data iterator.
  """

  def __init__(self,
               subset,
               batch_size,
               is_training,
               data_dir=None):
    dataset_builder = tfds.builder("cifar10", data_dir=data_dir)
    dataset_builder.download_and_prepare(download_dir=data_dir)
    self.image_size = 32

    if subset == "train":
      split = tfds.core.ReadInstruction("train", from_=10, to=100, unit="%")
    elif subset == "valid":
      split = tfds.core.ReadInstruction("train", from_=0, to=10, unit="%")
    elif subset == "test":
      split = tfds.Split.TEST
    else:
      raise ValueError("subset %s is undefined " % subset)
    self.num_channels = 3
    iterator = construct_iterator(
        dataset_builder, split, self._preprocess_fn(), batch_size, is_training)
    info = dataset_builder.info
    self.iterator = iterator
    self.images, self.labels, self.mask = iterator.get_next()
    self.num_classes = info.features["label"].num_classes
    self.class_names = info.features["label"].names
    self.num_examples = info.splits[split].num_examples

  def _preprocess_fn(self):
    """Preprocessing function."""
    image_size = self.image_size
    def preprocess_image(image):
      """Preprocessing."""
      # Normalize to 0-1 range.
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
      return tf.image.resize_image_with_crop_or_pad(
          image, image_size, image_size)

    return functools.partial(extract_data, preprocess_image=preprocess_image)


def extract_data_celeba(data, preprocess_image, attribute="Male"):
  """Extracts image, label and create a mask (used by CelebA data provider)."""
  image = data["image"]
  # Reserve label 0 for background
  label = tf.cast(data["attributes"][attribute], dtype=tf.int32)
  # Create a mask variable to track the real vs padded data in the last batch.
  mask = 1.
  image = preprocess_image(image)
  return image, label, mask


class CelebADataProvider(object):
  """CelebA Data Provider.

  Attributes:
    images: (4-D tensor) Images of shape (batch, height, width, channels).
    labels: (1-D tensor) Data labels of size (batch,).
    mask: (1-D boolean tensor) Data mask. Used when data is not repeated to
      indicate the fraction of the batch with true data in the final batch.
    num_classes: (integer) Number of classes in the dataset.
    num_examples: (integer) Number of examples in the dataset.
    num_channels: (integer) Number of image color channels.
    image_size: (Integer) Size of the image.
    iterator: Tensorflow data iterator.
    class_names: (List of strings) Name of classes in the order of the labels.

  """

  image_size = 32

  def __init__(self,
               subset,
               batch_size,
               is_training,
               data_dir=None):

    dataset_builder = tfds.builder("celeb_a",
                                   data_dir=data_dir)
    dataset_builder.download_and_prepare(download_dir=data_dir)
    if subset == "train":
      split = tfds.Split.TRAIN

    elif subset == "valid":
      split = tfds.Split.VALIDATION

    elif subset == "test":
      split = tfds.Split.TEST

    else:
      raise ValueError(
          "subset %s is undefined for the dataset" % subset)
    self.num_channels = 3
    iterator = construct_iterator(
        dataset_builder, split, self._preprocess_fn(), batch_size, is_training)
    info = dataset_builder.info
    self.iterator = iterator
    self.images, self.labels, self.mask = iterator.get_next()
    self.num_classes = 2
    self.class_names = ["Female", "Male"]
    self.num_examples = info.splits[split].num_examples

  def _preprocess_fn(self):
    """Preprocessing."""
    crop = True
    image_size = self.image_size
    def preprocess_image(image):
      """Preprocesses the given image.

      Args:
        image: Tensor `image` representing a single image example of
          arbitrary size.
      Returns:
        Preprocessed image.
      """
      # Normalize to 0-1 range.
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
      if crop:
        image = tf.image.crop_to_bounding_box(image, 40, 20, 218 - 80, 178 - 40)

      return tf.image.resize_bicubic([image], [image_size, image_size])[0]

    return functools.partial(extract_data_celeba,
                             preprocess_image=preprocess_image,
                             attribute="Male")


def _random_translate(image, new_image_size, noise_fill=True):
  """Randomly translate image and pad with noise."""
  image_shape = image.shape.as_list()
  image_size = image_shape[0]

  mask = tf.ones(image_shape)
  mask = tf.image.resize_image_with_crop_or_pad(
      mask, 2 * new_image_size - image_size, 2 * new_image_size - image_size)
  image = tf.image.resize_image_with_crop_or_pad(
      image, 2 * new_image_size - image_size, 2 * new_image_size - image_size)

  # Range of bounding boxes is from [0, new_image_size-image_size).
  offset_height = tf.random_uniform(
      shape=(), minval=0, maxval=new_image_size - image_size, dtype=tf.int32)
  offset_width = tf.random_uniform(
      shape=(), minval=0, maxval=new_image_size - image_size, dtype=tf.int32)

  image = tf.image.crop_to_bounding_box(image, offset_height, offset_width,
                                        new_image_size, new_image_size)
  if noise_fill:
    mask = tf.image.crop_to_bounding_box(mask, offset_height, offset_width,
                                         new_image_size, new_image_size)
    image += tf.random_uniform(
        (new_image_size, new_image_size, image_shape[-1]), 0, 1.0) * (1 - mask)
  return image


class TranslatedCelebADataProvider(CelebADataProvider):
  """Celeb A Data Provider with images translated randomly.

  Attributes:
    init_op: Initialization operation for the data provider.
    images: (4-D tensor) Images of shape (batch, height, width, channels).
    labels: (1-D tensor) Data labels of size (batch,).
    mask: (1-D boolean tensor) Data mask. Used when data is not repeated to
      indicate the fraction of the batch with true data in the final batch.
    num_classes: (integer) Number of classes in the dataset.
    num_examples: (integer) Number of examples in the dataset.
    num_channels: (integer) Number of image color channels.
    use_augmentation: (boolean) Whether to use data augmentation or not.
    image_size: (Integer) Size of the image.
    iterator: Tensorflow data iterator.
    class_names: (List of strings) Name of classes in the order of the labels.

  """
  image_size = 48

  def _preprocess_fn(self):
    """Preprocessing."""
    crop = True
    image_size = self.image_size
    def preprocess_image(image):
      """Preprocesses the given image.

      Args:
        image: Tensor `image` representing a single image example of
          arbitrary size.
      Returns:
        Preprocessed image.
      """
      # Normalize to 0-1 range.
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
      if crop:
        image = tf.image.crop_to_bounding_box(image, 40, 20, 218 - 80, 178 - 40)

      image = tf.image.resize_bicubic([image], [32, 32])[0]
      return _random_translate(image, image_size, noise_fill=True)

    return functools.partial(extract_data_celeba,
                             preprocess_image=preprocess_image,
                             attribute="Male")


# ===== Function that provides data. ======
_DATASETS = {
    "cifar10": CIFAR10DataProvider,
    "mnist": MNISTDataProvider,
    "celeba32": CelebADataProvider,
    "trans_celeba48": TranslatedCelebADataProvider,
}


def get_data_provider(dataset_name):
  """Returns dataset by name."""
  return _DATASETS[dataset_name]
