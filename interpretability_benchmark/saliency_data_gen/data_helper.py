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

# Lint as: python3
"""Copyright 2018 The ROAR Authors.

All rights reserved.

Helper library with functions for data import and creation of TF-Records.
"""
from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf

from interpretability_benchmark.utils.preprocessing_helper import preprocess_image

FLAGS = flags.FLAGS

# image size config by model.
SALIENCY_MODEL_CONFIG = {'resnet_50': [224, 224, 3]}
SALIENCY_BASELINE = {
    'resnet_50':
        np.zeros(SALIENCY_MODEL_CONFIG['resnet_50']) + np.array(
            [[[-2.11790395, -2.03571415, -1.80444443]]])
}


class DataIterator(object):
  """Data input pipeline class.

  Attributes:
    filename: string indicating the shard being processed.
    mode: boolean to indicate whether training or eval is occuring.
    dataset: string indicating the images and labels being processed.
    test_small_sample: boolean for whether to test workflow using small sample.

  Returns:
    feature_ranking: feature_ranking estimate based upon saliency_method.
  """

  def __init__(self,
               filename,
               dataset,
               test_small_sample=False,
               image_size=224,
               preprocessing=False):
    self._filename = filename
    self._dataset_name = dataset
    self._preprocessing = preprocessing
    self._test_small_sample = test_small_sample
    self._image_size = image_size

  def parser(self, serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    if self._test_small_sample:
      image_raw = serialized_example
      image_preprocessing = serialized_example
      label = tf.constant(0, tf.int32)
    else:
      features = tf.parse_single_example(
          serialized_example,
          features={
              'image/encoded':
                  tf.FixedLenFeature([], tf.string, default_value=''),
              'image/class/label': (tf.FixedLenFeature([], tf.int64)),
          })
      image_raw = tf.reshape(features['image/encoded'], shape=[])

      image_preprocessing = tf.image.decode_image(image_raw, dtype=tf.float32)

      image_preprocessing = preprocess_image(
          image=image_preprocessing, image_size=224, is_training=False)

      if self._dataset_name == 'imagenet':
        # Subtract one so that labels are in [0, 1000).
        label = tf.cast(
            tf.reshape(features['image/class/label'], shape=[]),
            dtype=tf.int32) - 1
      else:
        label = tf.cast(
            tf.reshape(features['image/class/label'], shape=[]), dtype=tf.int32)

    return (image_raw, image_preprocessing, label)

  def _get_null_input(self, data):
    """Returns a null image (all black pixels)."""
    del data
    return tf.zeros([self._image_size, self._image_size, 3], tf.float32)

  def input_fn(self):
    """Input function from cloud tpu resnet model pre-processing."""

    if self._test_small_sample:
      dataset = tf.data.Dataset.range(1).repeat().map(self._get_null_input)
    else:
      dataset = tf.data.Dataset.list_files(self._filename, shuffle=False)

      def fetch_dataset(filename):
        dataset = tf.data.TFRecordDataset(filename)
        return dataset

      dataset = dataset.apply(
          tf.data.experimental.parallel_interleave(
              fetch_dataset, cycle_length=8, sloppy=False))

    dataset = dataset.map(self.parser)
    dataset = dataset.batch(1)
    image_raw, image_processed, label = dataset.make_one_shot_iterator(
    ).get_next()
    image_processed = tf.reshape(image_processed, [1, 224, 224, 3])

    label = tf.reshape(label, [1])

    return image_raw, image_processed, label


def image_to_tfexample(raw_image, maps, label):
  """Generates a serialized TF-example."""
  saliency_dict = {
      'IG_SG': 'ig_smooth',
      'IG': 'ig_image',
      'SH': 'gradient_image',
      'SH_SG': 'gradient_smooth',
      'GB': 'gb_image',
      'GB_SG': 'gb_smooth',
      'GB_SG_2': 'gb_smooth_2',
      'IG_SG_2': 'ig_smooth_2',
      'SH_SG_2': 'gradient_smooth_2',
      'SH_V': 'gradient_vargrad',
      'IG_V': 'IG_vargrad',
      'GB_V': 'GB_vargrad',
      'SOBEL': 'sobel'
  }
  saliency_name = saliency_dict[FLAGS.saliency_method]

  # our synthetic data is float32, since this is for test we just switch type
  if FLAGS.test_small_sample:
    type_raw_image = tf.train.Feature(
        float_list=tf.train.FloatList(
            value=np.reshape(raw_image.astype(np.float32), [-1])))
  else:
    type_raw_image = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[raw_image]))
  return tf.train.Example(
      features=tf.train.Features(
          feature={
              'raw_image':
                  type_raw_image,
              saliency_name:
                  tf.train.Feature(float_list=tf.train.FloatList(value=maps)),
              'label':
                  tf.train.Feature(
                      int64_list=tf.train.Int64List(value=[label]))
          }))
