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

# Lint as: python3
"""Basic Image Process."""

from absl import logging
import numpy as np
import tensorflow as tf
from deep_representation_one_class.data.augment import apply_augment
from deep_representation_one_class.data.augment import compose_augment_seq


class BasicImageProcess():
  """Basic Image Process."""

  def __init__(self, data, input_shape=(256, 256, 3)):
    self.input_shape = input_shape
    self.data = data

  def image_normalize(self, image, do_mean=True, do_std=True):
    channel_means = [0.485, 0.456, 0.406]
    channel_stds = [0.229, 0.224, 0.225]
    if do_mean:
      means = tf.broadcast_to(channel_means, tf.shape(image))
      image = image - means
    if do_std:
      stds = tf.broadcast_to(channel_stds, tf.shape(image))
      image = image / stds
    return image

  def preprocess_image(self, image, dtype=tf.float32, aug_ops_list=None):
    """Preprocess images."""
    image = tf.cast(image, dtype) / 255.0
    image = tf.reshape(
        image, shape=tf.stack(self.input_shape))  # may become problematic
    images = apply_augment(image, ops_list=aug_ops_list)
    return images

  def parse_record_fn(self, raw_record, is_training, dtype, aug_list=None):
    """Parse record function."""
    # create augmentation list
    aug_ops_list = [
        compose_augment_seq(aug_type, is_training=is_training)
        for aug_type in aug_list
    ]
    # do preprocessing
    image, label, image_id = raw_record
    images = self.preprocess_image(
        image, dtype=dtype, aug_ops_list=aug_ops_list)
    label = tf.cast(tf.reshape(label, shape=[1]), dtype=tf.float32)
    return images + (label, image_id)

  def process_record_dataset(self,
                             dataset,
                             aug_list,
                             is_training,
                             batch_size,
                             shuffle_buffer,
                             num_batch_per_epoch=1,
                             dtype=tf.float32,
                             datasets_num_private_threads=None,
                             force_augment=False,
                             drop_remainder=False):
    """Process record dataset."""

    # Defines a specific size thread pool for tf.data operations.
    if datasets_num_private_threads:
      options = tf.data.Options()
      options.experimental_threading.private_threadpool_size = (
          datasets_num_private_threads)
      dataset = dataset.with_options(options)
      logging.info('datasets_num_private_threads: %s',
                   datasets_num_private_threads)

    if is_training:
      # multiplier if original dataset is too small
      num_data = len([1 for _ in dataset.enumerate()])
      multiplier = np.maximum(1, np.int(np.ceil(batch_size / num_data)))
      if multiplier > 1:
        dataset = dataset.repeat(multiplier)
      # Shuffles records before repeating to respect epoch boundaries.
      dataset = dataset.shuffle(
          buffer_size=shuffle_buffer, reshuffle_each_iteration=True)

    # Parses the raw records into images and labels.
    dataset = dataset.map(
        lambda *args: self.parse_record_fn(
            args,
            is_training=is_training or force_augment,
            dtype=dtype,
            aug_list=aug_list),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    if not is_training:
      num_batch_per_epoch = len([1 for _ in dataset.enumerate()])
    else:
      if num_batch_per_epoch <= 0:
        num_batch_per_epoch = len([1 for _ in dataset.enumerate()])
    dataset = dataset.repeat()

    # Prefetch.
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return [dataset, num_batch_per_epoch]

  def input_fn(self,
               is_training,
               batch_size,
               aug_list=None,
               num_batch_per_epoch=1,
               dtype=tf.float32,
               datasets_num_private_threads=None,
               input_context=None,
               force_augment=False,
               training_dataset_cache=False):
    """Creates an input function from the dataset."""
    dataset = self.make_dataset(
        is_training=is_training, input_context=input_context)

    if is_training and training_dataset_cache:
      # Improve training performance when training data is in remote storage and
      # can fit into worker memory.
      dataset = dataset.cache()

    # Aug_list should be a list of list of tuples
    if not isinstance(aug_list, list):
      raise TypeError('augmentation list should be a list')
    if isinstance(aug_list, list):
      if not isinstance(aug_list[0], list):
        aug_list = [aug_list]

    return self.process_record_dataset(
        dataset=dataset,
        aug_list=aug_list,
        is_training=is_training,
        batch_size=batch_size,
        shuffle_buffer=1000,
        num_batch_per_epoch=num_batch_per_epoch,
        dtype=dtype,
        datasets_num_private_threads=datasets_num_private_threads,
        force_augment=force_augment,
        drop_remainder=True if is_training else False)

  def make_dataset(self, is_training, input_context=None):
    """Makes a dataset."""
    dataset = tf.data.Dataset.from_tensor_slices(self.data)

    if input_context:
      logging.info(
          'Sharding the dataset: input_pipeline_id=%d num_input_pipelines=%d',
          input_context.input_pipeline_id, input_context.num_input_pipelines)
      dataset = dataset.shard(input_context.num_input_pipelines,
                              input_context.input_pipeline_id)
    if is_training:
      # Shuffle the input files
      dataset = dataset.shuffle(buffer_size=len(self.data[0]))
    return dataset
