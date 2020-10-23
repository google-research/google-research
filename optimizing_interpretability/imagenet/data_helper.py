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

"""Efficient input pipeline using tf.data.Dataset."""

import tensorflow.compat.v1 as tf
from optimizing_interpretability.imagenet import preprocessing_helper


def input_fn(params):
  """Input function from cloud tpu resnet model pre-processing."""

  def parser_tfrecords(serialized_example):
    """Parses an ImageNet image and its label from a serialized TFExample."""
    data = {}
    if params['test_small_sample']:
      data['images_batch'] = serialized_example
      data['labels_batch'] = tf.constant(0, tf.int32)
      return data
    else:
      features = tf.parse_single_example(
          serialized_example,
          features={
              'image/encoded': tf.io.FixedLenFeature((), tf.string, ''),
              'image/class/label': tf.io.FixedLenFeature([], tf.int64, -1),
          })
      image_bytes = tf.reshape(features['image/encoded'], shape=[])
      if params['mode'] == 'train':
        data['images_batch'] = preprocessing_helper.preprocess_image(
            image_bytes=image_bytes, is_training=True, image_size=224)
      else:
        data['images_batch'] = preprocessing_helper.preprocess_image(
            image_bytes=image_bytes, is_training=False, image_size=224)

      label = tf.cast(
          tf.reshape(features['image/class/label'], shape=[]),
          dtype=tf.int32) - 1
      data['labels_batch'] = label

      return data

  def _get_null_input(data):
    """Returns a zero matrix image.

    Args:
      data: element of a dataset, ignored in this method, since it produces the
        same null image regardless of the element.

    Returns:
      a tensor representing a null image.
    """
    del data  # Unused since output is constant regardless of input
    return tf.zeros([224, 224, 3], tf.float32)

  def predicate(x):
    label = x['labels_batch']
    isallowed = tf.equal(params['filter_label'], label)
    reduced = tf.reduce_sum(tf.cast(isallowed, tf.float32))
    return tf.greater(reduced, tf.constant(0.))

  data_dir = params['data_dir']

  batch_size = params['batch_size']
  sloppy_shuffle = params['sloppy_shuffle']

  if params['test_small_sample']:
    dataset = tf.data.Dataset.range(1).repeat().map(_get_null_input)
  else:
    dataset = tf.data.Dataset.list_files(data_dir, shuffle=False)

    def fetch_dataset(filename):
      dataset = tf.data.TFRecordDataset(filename)
      return dataset

    dataset = dataset.apply(
        tf.data.experimental.parallel_interleave(
            fetch_dataset,
            cycle_length=params['num_cores'],
            sloppy=sloppy_shuffle))

  if params['visualize_image']:
    dataset = dataset.map(parser_tfrecords)
    if params['filter_label']:
      dataset = dataset.filter(predicate).batch(params['batch_size'])
    else:
      dataset.batch(params['batch_size'])
    dataset = dataset.make_one_shot_iterator().get_next()
    return dataset

  if params['mode'] == 'train':
    dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.repeat()

  dataset = dataset.map(parser_tfrecords, num_parallel_calls=64)
  dataset = dataset.prefetch(batch_size)
  dataset = dataset.batch(batch_size, drop_remainder=sloppy_shuffle)
  dataset = dataset.make_one_shot_iterator().get_next()

  return dataset
