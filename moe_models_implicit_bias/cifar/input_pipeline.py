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

"""Code for generating the training/evaluation datasets for CIFAR-10 data.
"""

import jax
import tensorflow as tf
import tensorflow_datasets as tfds

IMAGE_SIZE = 32
CROP_PADDING = 4
MEAN_RGB = [0.4914 * 255, 0.4822 * 255, 0.4465 * 255]
STDDEV_RGB = [0.2023 * 255, 0.1994 * 255, 0.2010 * 255]


def normalize_image(image):
  image = tf.cast(image, dtype=tf.float32)
  image -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=image.dtype)
  image /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=image.dtype)
  return image


def preprocess_for_train(image_bytes, dtype=tf.float32):
  """Preprocesses the given image for training.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    dtype: data type of the image.

  Returns:
    A preprocessed image `Tensor`.
  """
  image = tf.io.decode_image(image_bytes, channels=3)
  image = tf.image.resize_with_crop_or_pad(image, IMAGE_SIZE + CROP_PADDING,
                                           IMAGE_SIZE + CROP_PADDING)
  image = tf.image.random_crop(image, size=[IMAGE_SIZE, IMAGE_SIZE, 3])
  image = tf.image.random_flip_left_right(image)
  image = normalize_image(image)
  image = tf.image.convert_image_dtype(image, dtype=dtype)
  return image


def preprocess_for_eval(image_bytes, dtype=tf.float32):
  """Preprocesses the given image for evaluation.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    dtype: data type of the image.

  Returns:
    A preprocessed image `Tensor`.
  """
  image = tf.io.decode_image(image_bytes, channels=3)
  image = normalize_image(image)
  image = tf.image.convert_image_dtype(image, dtype=dtype)
  return image


def create_split(dataset_builder,
                 batch_size,
                 train,
                 dtype=tf.float32,
                 cache=False):
  """Creates a split from the ImageNet dataset using TensorFlow Datasets.

  Args:
    dataset_builder: TFDS dataset builder for ImageNet.
    batch_size: the batch size returned by the data pipeline.
    train: Whether to load the train or evaluation split.
    dtype: data type of the image.
    cache: Whether to cache the dataset.

  Returns:
    A `tf.data.Dataset`.
  """
  if train:
    train_examples = dataset_builder.info.splits['train'].num_examples
    split_size = train_examples // jax.process_count()
    start = jax.process_index() * split_size
    split = f'train[{start}:{start + split_size}]'
  else:
    validate_examples = dataset_builder.info.splits['test'].num_examples
    split_size = validate_examples // jax.process_count()
    start = jax.process_index() * split_size
    split = f'test[{start}:{start + split_size}]'

  def decode_example(example):
    if train:
      image = preprocess_for_train(example['image'], dtype)
    else:
      image = preprocess_for_eval(example['image'], dtype)
    return {'image': image, 'label': example['label']}

  ds = dataset_builder.as_dataset(
      split=split, decoders={
          'image': tfds.decode.SkipDecoding(),
      })
  options = tf.data.Options()
  options.autotune.enabled = True
  ds = ds.with_options(options)
  if cache:
    ds = ds.cache()
  if train:
    ds = ds.repeat()
    ds = ds.shuffle(16 * batch_size, seed=0)
  ds = ds.map(decode_example, num_parallel_calls=tf.data.AUTOTUNE)
  ds = ds.batch(batch_size, drop_remainder=True)
  if not train:
    ds = ds.repeat()
  ds = ds.prefetch(tf.data.AUTOTUNE)
  return ds
