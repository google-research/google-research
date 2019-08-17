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

"""Visualzie TFRecords."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os

from absl import app
from absl import flags
from absl import logging

import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top

import tensorflow.compat.v2 as tf


flags.DEFINE_string('path_to_tfrecord', None, 'Path to TFRecords.')
flags.DEFINE_integer('num_vids', 1, 'Number of videos to visualize.')
flags.DEFINE_integer('num_skip_frames', 10, 'Number of frames to skip while'
                     'visualizing.')

flags.mark_flag_as_required('path_to_tfrecord')
FLAGS = flags.FLAGS


def decode(serialized_example):
  """Decode a serialized a SequenceExample string.

  Args:
    serialized_example: SequenceExample, A SequenceExample from a TFRecord.

  Returns:
    frames: list, A list of frames in the video in SequenceExample.
    name: string, Name of the sequence.
  """

  context_features = {
      'label': tf.io.FixedLenFeature([], dtype=tf.int64),
      'len': tf.io.FixedLenFeature([], dtype=tf.int64)
  }
  sequence_features = {
      'video': tf.io.FixedLenSequenceFeature([], dtype=tf.string)
  }

  # Extract features from serialized data.
  context_data, sequence_data = tf.io.parse_single_sequence_example(
      serialized=serialized_example,
      context_features=context_features,
      sequence_features=sequence_features)

  # Decode the encoded JPG images.
  frames = tf.map_fn(tf.image.decode_jpeg, sequence_data['video'],
                     dtype=tf.uint8, parallel_iterations=12)
  label = context_data['label']

  return frames, label


def visualize_tfrecords(path_to_tfrecord, num_vids, num_skip_frames):
  """Visualizes TFRecords in given path.

  Args:
    path_to_tfrecord: string, Path to TFRecords. Provide search pattern in
    string.
    num_vids: integer, Number of videos to visualize.
    num_skip_frames: integer, Number of frames to skip while visualzing.
  """
  tfrecord_files = glob.glob(path_to_tfrecord)
  tfrecord_files.sort()
  dataset = tf.data.TFRecordDataset(tfrecord_files)
  dataset = dataset.map(decode)
  dataset = dataset.batch(1)

  ctr = 0
  for batch_videos, batch_labels in dataset:
    batch_videos = batch_videos.numpy()
    batch_labels = batch_labels.numpy()
    logging.info('Class label for %s = %d',
                 tfrecord_files[ctr],
                 batch_labels[0])
    plt.show()
    plt.title(os.path.splitext(os.path.basename(tfrecord_files[ctr]))[0])
    for frame_idx in xrange(0, len(batch_videos[0]), num_skip_frames):
      plt.imshow(batch_videos[0, frame_idx])
      plt.pause(0.1)
    plt.close()
    ctr += 1
    if ctr == num_vids:
      break


def main(_):
  tf.enable_v2_behavior()
  visualize_tfrecords(FLAGS.path_to_tfrecord, FLAGS.num_vids,
                      FLAGS.num_skip_frames)

if __name__ == '__main__':
  app.run(main)
