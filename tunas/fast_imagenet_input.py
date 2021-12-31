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

# Lint as: python2, python3
"""Optimized input pipeline for ImageNet model training.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from typing import Any, Callable, Dict, Optional, Text, Tuple

import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

from tunas import imagenet_preprocessing

IMAGE_SIZE = 224

# ResNet-based preprocessing
MEAN_RGB = (0.485 * 255, 0.456 * 255, 0.406 * 255)
STDDEV_RGB = (0.229 * 255, 0.224 * 255, 0.225 * 255)


# TPUs require that all batches have the same shape. For the final batch (which
# can have fewer examples than the other batches), we have two options:
# (1) Drop (ignore) the batch.
# (2) Zero-pad the batch with all-zero "dummy" examples.
class FinalBatchMode(object):
  DROP = 'drop'
  PAD = 'pad'

_ALL_FINAL_BATCH_MODES = (FinalBatchMode.DROP, FinalBatchMode.PAD)


def tfds_split_for_mode(mode):
  """Return the TFDS split to use for a given input dataset."""
  if mode == 'test':
    # The labels for the real ImageNet test set were never released. So we
    # follow the standard (although admitted confusing) practice of obtain
    # "test set" accuracy numbers from the ImageNet validation.
    return 'validation'
  elif mode == 'train':
    return 'train'
  elif mode == 'l2l_valid':
    # To prevent overfitting to the test set, we  use a held-out portion of the
    # training set for model validation. We use the same number of examples here
    # as we did for the TuNAS paper. However, the validation set contains
    # different examples than it did in the paper, and we should expect small
    # deviations from the reported results for this reason.
    return 'train[:50046]'
  elif mode == 'l2l_train':
    return 'train[50046:]'
  else:
    raise ValueError('Invalid mode: {!r}'.format(mode))


def dataset_size_for_mode(mode):
  """Returns the number of training examples in the input dataset."""
  if mode == 'test':
    return 50000
  elif mode == 'train':
    return 1281167
  elif mode == 'l2l_valid':
    return 50046
  elif mode == 'l2l_train':
    return 1281167 - 50046
  else:
    raise ValueError('Invalid mode: {!r}'.format(mode))


def _parse_example(features,
                   training,
                   use_bfloat16,
                   image_size,
                   image_resize_method):
  """Convert a serialized tf.Example to an (image_tensor, label_tensor) pair."""
  # Preprocess the input image
  image = imagenet_preprocessing.preprocess_image(
      features['image'],
      is_training=training,
      use_bfloat16=use_bfloat16,
      image_size=image_size,
      image_resize_method=image_resize_method)

  # Normalize the input image
  image -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=image.dtype)
  image /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=image.dtype)

  # Preprocess the input label
  one_hot_label = tf.one_hot(features['label'], 1001)

  # That's all, folks
  return (image, one_hot_label)


def make_dataset(
    dataset_dir,
    dataset_mode,
    training,
    shuffle_and_repeat = None,
    use_bfloat16 = False,
    image_size = IMAGE_SIZE,
    image_resize_method=tf.image.resize_bicubic,
    transpose_input = True,
    final_batch_mode = FinalBatchMode.DROP,
):
  """Set up a pipeline for reading/preprocessing data from ImageNet.

  WARNING: BY DEFAULT, INPUT IMAGES WILL BE RETURNED IN HWCN FORMAT, NOT NHWC.

  Args:
    dataset_dir: Path of the TFDS data directory to load ImageNet data from.
    dataset_mode: String controlling the training or evaluation dataset to use.
        For example: 'l2l_train', 'l2l_valid', 'train', or 'test'.
    training: Boolean, true for model training, false for evaluation. Controls
        the default value of 'shuffle_and_repeat', as well as whether or not
        we use data augmentation.
    shuffle_and_repeat: Boolean or None. If true, we will loop through the
        dataset indefinitely, visiting examples in a different (random) order
        at each epoch. If false, we will loop through examples once in a
        deterministic order. If None, we will automatically infer the value
        from `training`.
    use_bfloat16: Boolean. If true, use bfloat16 for input images.
    image_size: Integer, the output size of image.
    image_resize_method: Function. For example, it can be
        tf.image.resize_bicubic or tf.image.resize_bilinear.
    transpose_input: Whether to transpose the input from NHWC to HWCN format,
        which would improve in-feed throughput on TPUs . Default True.
    final_batch_mode: How to handle the final batch, which can potentially have
        fewer examples than the other batches. If set to FinalBatchMode.PAD, we
        will return a struct `((features, mask), labels)`. If set to
        FinalBatchMode.DROP, we will return a pair `(features, labels)`, and
        will drop the final batch if it has fewer elements than the others.

  Returns:
    A pair (dataset_fn, dataset_size), where dataset_fn is a function that
    produces a tf.data.Dataset object, and dataset_size is the integer size
    of the selected dataset.
  """
  if shuffle_and_repeat is None:
    shuffle_and_repeat = training

  if final_batch_mode not in _ALL_FINAL_BATCH_MODES:
    raise ValueError('Invalid final_batch_mode: {}. Permitted values: {}'
                     .format(final_batch_mode, _ALL_FINAL_BATCH_MODES))

  def dataset_fn(params):
    """Construct a tf.Dataset with image-label pairs from ImageNet."""
    # Read examples from disk.
    dataset = tfds.load(
        'imagenet2012',
        data_dir=dataset_dir,
        decoders={'image': tfds.decode.SkipDecoding()},
        split=tfds_split_for_mode(dataset_mode),
        shuffle_files=shuffle_and_repeat,
        try_gcs=True)

    if shuffle_and_repeat:
      dataset = dataset.repeat().shuffle(1024)

    # Parse the examples into (image, label) pairs.
    max_batch_size = params['batch_size']

    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            functools.partial(
                _parse_example, training=training, use_bfloat16=use_bfloat16,
                image_size=image_size, image_resize_method=image_resize_method),
            batch_size=max_batch_size,
            num_parallel_batches=8,
            drop_remainder=(final_batch_mode == FinalBatchMode.DROP)))

    def _pad_batch(features, labels):
      """Zero-pad features and labels along the `batch` dimension."""
      current_batch_size = tf.shape(features)[0]
      padding = max_batch_size - current_batch_size

      padded_features = tf.pad(features, [[0, padding], [0, 0], [0, 0], [0, 0]])
      padded_features.set_shape([max_batch_size, None, None, None])

      padded_labels = tf.pad(labels, [[0, padding], [0, 0]])
      padded_labels.set_shape([max_batch_size, None])

      mask = tf.sequence_mask(current_batch_size, max_batch_size, tf.float32)
      return (padded_features, mask), padded_labels

    if final_batch_mode == FinalBatchMode.PAD:
      dataset = dataset.map(_pad_batch)

    # Transpose input from NHWC to HWCN format.
    def _transpose_features(features, labels):
      if final_batch_mode == FinalBatchMode.PAD:
        features, mask = features
      features = tf.transpose(features, [1, 2, 3, 0])
      if final_batch_mode == FinalBatchMode.PAD:
        features = (features, mask)
      return (features, labels)

    if transpose_input:
      dataset = dataset.map(_transpose_features, num_parallel_calls=8)

    dataset = dataset.prefetch(1)
    return dataset

  dataset_size = dataset_size_for_mode(dataset_mode)
  return dataset_fn, dataset_size
