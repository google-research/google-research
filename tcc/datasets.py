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

"""Datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
from absl import logging

import tensorflow.compat.v2 as tf

from tcc.config import CONFIG
from tcc.dataset_splits import DATASETS
from tcc.preprocessors import preprocess_sequence

FLAGS = flags.FLAGS

flags.DEFINE_integer('num_parallel_calls', 60, 'Number of parallel calls while'
                     'preprocessing data on CPU.')


def normalize_input(frame, new_max=1., new_min=0.0, old_max=255.0, old_min=0.0):
  x = tf.cast(frame, tf.float32)
  x = (x - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
  return x


def preprocess_input(frames, augment=True):
  """Preprocesses raw frames and optionally performs data augmentation."""

  preprocessing_ranges = {
      preprocess_sequence.IMAGE_TO_FLOAT: (),
      preprocess_sequence.RESIZE: {
          'new_size': [CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE],
      },
      preprocess_sequence.CLIP: {
          'lower_limit': 0.0,
          'upper_limit': 1.0,
      },
      preprocess_sequence.NORMALIZE_MEAN_STDDEV: {
          'mean': 0.5,
          'stddev': 0.5,
      }
  }

  if augment:
    if CONFIG.AUGMENTATION.RANDOM_FLIP:
      preprocessing_ranges[preprocess_sequence.FLIP] = {
          'dim': 2,
          'probability': 0.5,
      }
    if CONFIG.AUGMENTATION.RANDOM_CROP:
      preprocessing_ranges[preprocess_sequence.RANDOM_CROP] = {
          'image_size': tf.shape(frames)[1:4],
          'min_scale': 0.8,
      }
    if CONFIG.AUGMENTATION.BRIGHTNESS:
      preprocessing_ranges[preprocess_sequence.BRIGHTNESS] = {
          'max_delta': CONFIG.AUGMENTATION.BRIGHTNESS_MAX_DELTA,
      }
    if CONFIG.AUGMENTATION.CONTRAST:
      preprocessing_ranges[preprocess_sequence.CONTRAST] = {
          'lower': CONFIG.AUGMENTATION.CONTRAST_LOWER,
          'upper': CONFIG.AUGMENTATION.CONTRAST_UPPER
      }
    if CONFIG.AUGMENTATION.HUE:
      preprocessing_ranges[preprocess_sequence.HUE] = {
          'max_delta': CONFIG.AUGMENTATION.HUE_MAX_DELTA,
      }
    if CONFIG.AUGMENTATION.SATURATION:
      preprocessing_ranges[preprocess_sequence.SATURATION] = {
          'lower': CONFIG.AUGMENTATION.SATURATION_LOWER,
          'upper': CONFIG.AUGMENTATION.SATURATION_UPPER
      }
  else:
    if CONFIG.AUGMENTATION.RANDOM_CROP:
      preprocessing_ranges[preprocess_sequence.CENTRAL_CROP] = {
          'image_size': tf.shape(frames)[1:3]
      }

  frames, = preprocess_sequence.preprocess_sequence(
      ((frames, preprocess_sequence.IMAGE),), preprocessing_ranges)

  return frames


def decode(serialized_example):
  """Decode serialized SequenceExample."""

  context_features = {
      'name': tf.io.FixedLenFeature([], dtype=tf.string),
      'len': tf.io.FixedLenFeature([], dtype=tf.int64),
      'label': tf.io.FixedLenFeature([], dtype=tf.int64),
  }

  seq_features = {}

  seq_features['video'] = tf.io.FixedLenSequenceFeature([], dtype=tf.string)

  if CONFIG.DATA.FRAME_LABELS:
    seq_features['frame_labels'] = tf.io.FixedLenSequenceFeature(
        [], dtype=tf.int64)

  # Extract features from serialized data.
  context_data, sequence_data = tf.io.parse_single_sequence_example(
      serialized=serialized_example,
      context_features=context_features,
      sequence_features=seq_features)

  seq_len = context_data['len']
  seq_label = context_data['label']

  video = sequence_data.get('video', [])
  frame_labels = sequence_data.get('frame_labels', [])
  name = tf.cast(context_data['name'], tf.string)
  return video, frame_labels, seq_label, seq_len, name


def get_steps(step):
  """Sample multiple context steps for a given step."""
  num_steps = CONFIG.DATA.NUM_STEPS
  stride = CONFIG.DATA.FRAME_STRIDE
  if num_steps < 1:
    raise ValueError('num_steps should be >= 1.')
  if stride < 1:
    raise ValueError('stride should be >= 1.')
  # We don't want to encode information from the future.
  steps = tf.range(step - (num_steps - 1) * stride, step + stride, stride)
  return steps


def sample_and_preprocess(video,
                          labels,
                          seq_label,
                          seq_len,
                          name,
                          num_steps,
                          augment,
                          sample_all=False,
                          sample_all_stride=1,
                          add_shape=False):
  """Samples frames and prepares them for training."""

  if sample_all:
    # When dealing with very long videos we can choose to sub-sample to fit
    # data in memory. But be aware this also evaluates over a subset of frames.
    # Subsampling the validation set videos when reporting performance is not
    # recommended.
    steps = tf.range(0, seq_len, sample_all_stride)
    seq_len = tf.shape(steps)[0]
    chosen_steps = steps
  else:
    stride = CONFIG.DATA.STRIDE
    sampling_strategy = CONFIG.DATA.SAMPLING_STRATEGY

    # TODO(debidatta) : More flexible sampling
    if sampling_strategy == 'stride':
      # Offset can be set between 0 and maximum location from which we can get
      # total coverage of the video without having to pad.
      # This handles sampling over longer sequences.
      offset = tf.random.uniform(
          (), 0, tf.maximum(tf.cast(1, tf.int64), seq_len - stride * num_steps),
          dtype=tf.int64)
      # This handles sampling over shorter sequences by padding the last frame
      # many times. This is not ideal for the way alignment training batches are
      # created.
      steps = tf.minimum(
          seq_len - 1,
          tf.range(offset, offset + num_steps * stride + 1, stride))
      steps = steps[:num_steps]
    elif sampling_strategy == 'offset_uniform':
      # Sample a random offset less than a provided max offset. Among all frames
      # higher than the chosen offset, randomly sample num_frames
      check1 = tf.debugging.assert_greater_equal(
          seq_len,
          tf.cast(CONFIG.DATA.RANDOM_OFFSET, tf.int64),
          message='Random offset is more than sequence length.')
      check2 = tf.less_equal(
          tf.cast(num_steps, tf.int64),
          seq_len - tf.cast(CONFIG.DATA.RANDOM_OFFSET, tf.int64),
      )

      def _sample_random():
        with tf.control_dependencies([tf.identity(check1.outputs[0])]):
          offset = CONFIG.DATA.RANDOM_OFFSET
          steps = tf.random.shuffle(tf.range(offset, seq_len))
          steps = tf.gather(steps, tf.range(0, num_steps))
          steps = tf.gather(steps,
                            tf.nn.top_k(steps, k=num_steps).indices[::-1])
          return steps

      def _sample_all():
        return tf.range(0, num_steps, dtype=tf.int64)

      steps = tf.cond(check2, _sample_random, _sample_all)

    else:
      raise ValueError('Sampling strategy %s is unknown. Supported values are '
                       'stride, offset_uniform .' % sampling_strategy)

    if not sample_all and 'tcn' in CONFIG.TRAINING_ALGO:
      pos_window = CONFIG.TCN.POSITIVE_WINDOW
      # pylint: disable=g-long-lambda
      pos_steps = tf.map_fn(
          lambda step: tf.random.uniform((),
                                         minval=step - pos_window,
                                         maxval=step, dtype=tf.int64),
          steps)
      # pylint: enable=g-long-lambda
      steps = tf.stack([pos_steps, steps])
      steps = tf.reshape(tf.transpose(steps), (-1,))

    # Store chosen indices.
    chosen_steps = steps
    # Get multiple context steps depending on config at selected steps.
    steps = tf.reshape(tf.map_fn(get_steps, steps), [-1])
    steps = tf.maximum(tf.cast(0, tf.int64), steps)
    steps = tf.minimum(seq_len - 1, steps)

  shape_all_steps = CONFIG.DATA.NUM_STEPS * num_steps
  if not sample_all and 'tcn' in CONFIG.TRAINING_ALGO:
    shape_all_steps *= 2

  # Select data based on steps/
  video = tf.gather(video, steps)
  # Decode the encoded JPEG images
  video = tf.map_fn(
      tf.image.decode_jpeg,
      video,
      parallel_iterations=FLAGS.num_parallel_calls,
      dtype=tf.uint8)
  # Take images in range [0, 255] and normalize to [0, 1]
  video = tf.map_fn(
      normalize_input,
      video,
      parallel_iterations=FLAGS.num_parallel_calls,
      dtype=tf.float32)
  # Perform data-augmentation and return images in range [-1, 1]
  video = preprocess_input(video, augment)
  if add_shape:
    video.set_shape([shape_all_steps, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE, 3])

  if CONFIG.DATA.FRAME_LABELS:
    labels = tf.gather(labels, steps)
    if add_shape:
      labels.set_shape([shape_all_steps])

  return {
      'frames': video,
      'frame_labels': labels,
      'chosen_steps': chosen_steps,
      'seq_lens': seq_len,
      'seq_labels': seq_label,
      'name': name
  }


def get_tfrecords(dataset, split, path, per_class=False):
  """Get TFRecord files based on dataset and split."""

  if per_class:
    path_to_tfrecords = os.path.join(path % dataset, '*%s*'%split)
    logging.info('Loading %s data from: %s', split, path_to_tfrecords)
    tfrecord_files = sorted(tf.io.gfile.glob(path_to_tfrecords))
  else:
    path_to_tfrecords = os.path.join(path % dataset,
                                     '%s_%s*' % (dataset, split))

    logging.info('Loading %s data from: %s', split, path_to_tfrecords)
    tfrecord_files = sorted(tf.io.gfile.glob(path_to_tfrecords))

  if not tfrecord_files:
    raise ValueError('No tfrecords found at path %s' % path_to_tfrecords)

  return tfrecord_files


def create_dataset(split, mode, batch_size=None, return_iterator=True):
  """Creates a single-class dataset iterator based on config and split."""

  per_class = CONFIG.DATA.PER_CLASS
  # pylint: disable=g-long-lambda
  if mode == 'train':
    if not batch_size:
      batch_size = CONFIG.TRAIN.BATCH_SIZE
    num_steps = CONFIG.TRAIN.NUM_FRAMES
    preprocess_fn = (
        lambda video, labels, seq_label, seq_len, name: sample_and_preprocess(
            video,
            labels,
            seq_label,
            seq_len,
            name,
            num_steps,
            augment=True,
            add_shape=True))
  elif mode == 'eval':
    if not batch_size:
      batch_size = CONFIG.EVAL.BATCH_SIZE
    num_steps = CONFIG.EVAL.NUM_FRAMES
    preprocess_fn = (
        lambda video, labels, seq_label, seq_len, name: sample_and_preprocess(
            video,
            labels,
            seq_label,
            seq_len,
            name,
            num_steps,
            augment=False,
            add_shape=True))
  else:
    raise ValueError('Unidentified mode: %s. Use either train or eval.' % mode)
  # pylint: enable=g-long-lambda

  fraction = CONFIG.DATA.PER_DATASET_FRACTION

  datasets = []
  with tf.device('/cpu:0'):
    for dataset_name in CONFIG.DATASETS:
      tfrecord_files = get_tfrecords(
          dataset_name, split, CONFIG.PATH_TO_TFRECORDS, per_class=per_class)
      dataset = tf.data.TFRecordDataset(
          tfrecord_files, num_parallel_reads=FLAGS.num_parallel_calls)

      if (fraction != 1.0 and mode == 'train'):
        num_samples = max(1, int(fraction * DATASETS[dataset_name][split]))
        dataset = dataset.take(num_samples)
      else:
        num_samples = DATASETS[dataset_name][split]
      if CONFIG.DATA.SHUFFLE_QUEUE_SIZE <= 0:
        dataset = dataset.shuffle(num_samples)
      else:
        dataset = dataset.shuffle(CONFIG.DATA.SHUFFLE_QUEUE_SIZE)
      dataset = dataset.repeat()
      dataset = dataset.batch(batch_size)
      datasets.append(dataset)

    dataset = tf.data.experimental.sample_from_datasets(datasets,
                                                        len(datasets) * [1.0])

    dataset = dataset.apply(tf.data.experimental.unbatch())

    dataset = dataset.map(decode,
                          num_parallel_calls=FLAGS.num_parallel_calls)

    dataset = dataset.map(preprocess_fn,
                          num_parallel_calls=FLAGS.num_parallel_calls)

    # drop_remainder adds batch size in shape else first dim remains as None.
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # Prefetch batches
    dataset = dataset.prefetch(1)

    if return_iterator:
      return tf.compat.v1.data.make_one_shot_iterator(dataset)
    else:
      return dataset


def create_one_epoch_dataset(dataset, split, mode, path_to_tfrecords):
  """Creates a dataset iterator that gives one epoch of dataset."""
  batch_size = 1
  sample_all_stride = CONFIG.DATA.SAMPLE_ALL_STRIDE
  tfrecord_files = get_tfrecords(dataset, split, path_to_tfrecords)

  with tf.device('/cpu:0'):
    dataset = tf.data.TFRecordDataset(
        tfrecord_files,
        num_parallel_reads=FLAGS.num_parallel_calls)

    dataset = dataset.map(decode, num_parallel_calls=FLAGS.num_parallel_calls)

    # pylint: disable=g-long-lambda
    if mode == 'train':
      num_steps = CONFIG.TRAIN.NUM_FRAMES
      preprocess_fn = (
          lambda video, labels, seq_label, seq_len, name: sample_and_preprocess(
              video,
              labels,
              seq_label,
              seq_len,
              name,
              num_steps,
              augment=True,
              sample_all=True,
              sample_all_stride=sample_all_stride))
    else:
      num_steps = CONFIG.EVAL.NUM_FRAMES
      preprocess_fn = (
          lambda video, labels, seq_label, seq_len, name: sample_and_preprocess(
              video,
              labels,
              seq_label,
              seq_len,
              name,
              num_steps,
              augment=False,
              sample_all=True,
              sample_all_stride=sample_all_stride))

    # pylint: enable=g-long-lambda
    dataset = dataset.map(preprocess_fn,
                          num_parallel_calls=FLAGS.num_parallel_calls)

    dataset = dataset.batch(batch_size)
    # Prefetch batches
    dataset = dataset.prefetch(1)

  return tf.compat.v1.data.make_one_shot_iterator(dataset)
