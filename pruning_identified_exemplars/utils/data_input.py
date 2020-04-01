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
"""Helper functions to pre-process data inputs and create data iterator.

Predictions must be stored in the same sequence for each checkpoint.
This requires a tailored data input script to preserve ordering.
"""

import numpy as np
import tensorflow.compat.v1 as tf
from pruning_identified_exemplars.utils import preprocessing_helper


def input_fn(params):
  """Input function from cloud tpu resnet model pre-processing."""

  def parser_tfrecords(serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    data = {}
    if params['test_small_sample']:
      data['image_raw'] = serialized_example
      data['label'] = tf.constant(0, tf.int32)
      data['human_label'] = tf.constant('human_label', tf.string)
      data['key_'] = tf.constant('key', tf.string)
      return data
    else:
      features = tf.parse_single_example(
          serialized_example,
          features={
              'key_':
                  tf.FixedLenFeature([], dtype=tf.string, default_value=''),
              'image/encoded':
                  tf.FixedLenFeature((), tf.string, default_value=''),
              'image/class/label':
                  tf.FixedLenFeature([1], dtype=tf.int64, default_value=-1),
              'image/class/text':
                  tf.FixedLenFeature([], dtype=tf.string, default_value=''),
          })

      image_bytes = tf.reshape(features['image/encoded'], shape=[])
      label = tf.cast(
          tf.reshape(features['image/class/label'], shape=[]),
          dtype=tf.int32) - 1
      image = tf.image.decode_jpeg(features['image/encoded'], 3)

      human_label = tf.cast(
          tf.reshape(features['image/class/text'], shape=[]), dtype=tf.string)
      if params['task'] == 'imagenet_training':
        # training is set to false in prediction mode
        image = preprocessing_helper.preprocess_image(
            image=image, image_size=224, is_training=True)
      else:
        # training is set to false in prediction mode
        image = preprocessing_helper.preprocess_image(
            image=image, image_size=224, is_training=False)

      if params['task'] == 'pie_dataset_gen':
        data['image_raw'] = image_bytes
      else:
        data['image_raw'] = image
      data['label'] = label
      data['human_label'] = human_label
      data['key_'] = tf.reshape(features['key_'], shape=[])
      return data

  def _get_null_input(data):
    """Returns a null image (all black pixels)."""
    del data
    return tf.zeros([224, 224, 3], tf.float32)

  data_dir = params['data_dir']
  # batch_size is set to 1 to avoid any images being dropped
  if params['task'] in ['imagenet_predictions', 'pie_dataset_gen']:
    batch_size = 1
    sloppy_shuffle = False
  else:
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

  if params['mode'] == 'train':
    dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.repeat()

  dataset = dataset.map(parser_tfrecords, num_parallel_calls=64)
  dataset = dataset.prefetch(batch_size)
  dataset = dataset.batch(batch_size, drop_remainder=sloppy_shuffle)
  dataset = dataset.make_one_shot_iterator().get_next()

  if params['task'] == 'imagenet_predictions':
    return ([
        tf.reshape(dataset['image_raw'], [batch_size, 224, 224, 3]),
        tf.reshape(dataset['label'], [batch_size])
    ], tf.reshape(dataset['label'], [batch_size]))
  else:
    return dataset


def image_to_tfexample(key,
                       raw_image,
                       label,
                       stored_class_label,
                       image_index=None,
                       baseline_n=None,
                       variant_n=None,
                       pruning_fraction=None,
                       baseline_mean_probability=None,
                       variant_mean_probability=None,
                       variant_mode_label=None,
                       baseline_mode_label=None,
                       test_small_sample=False):
  """Generates a serialized TF-example."""

  if test_small_sample:
    type_raw_image = tf.train.Feature(
        float_list=tf.train.FloatList(
            value=np.reshape(raw_image.astype(np.float32), [-1])))
  else:
    type_raw_image = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[raw_image]))
  return tf.train.Example(
      features=tf.train.Features(
          feature={
              'key':
                  tf.train.Feature(bytes_list=tf.train.BytesList(value=[key])),
              'image/encoded':
                  type_raw_image,
              'image/class/label':
                  tf.train.Feature(
                      int64_list=tf.train.Int64List(value=[label])),
              'stored_class_label':
                  tf.train.Feature(
                      int64_list=tf.train.Int64List(
                          value=[stored_class_label])),
              'image_index':
                  tf.train.Feature(
                      int64_list=tf.train.Int64List(value=[image_index])),
              'baseline_n':
                  tf.train.Feature(
                      int64_list=tf.train.Int64List(value=[baseline_n])),
              'variant_n':
                  tf.train.Feature(
                      int64_list=tf.train.Int64List(value=[variant_n])),
              'pruning_fraction':
                  tf.train.Feature(
                      float_list=tf.train.FloatList(value=[pruning_fraction])),
              'baseline_mean_probability':
                  tf.train.Feature(
                      float_list=tf.train.FloatList(
                          value=[baseline_mean_probability])),
              'variant_mean_probability':
                  tf.train.Feature(
                      float_list=tf.train.FloatList(
                          value=[variant_mean_probability])),
              'variant_mode_label':
                  tf.train.Feature(
                      int64_list=tf.train.Int64List(
                          value=[variant_mode_label])),
              'baseline_mode_label':
                  tf.train.Feature(
                      int64_list=tf.train.Int64List(
                          value=[baseline_mode_label])),
          }))
