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
"""Parses a video into a tf.data.dataset of TFRecords."""

import os

from absl import app
from absl import flags
import cv2
import tensorflow as tf

from uflow.data_conversion_scripts import conversion_utils

FLAGS = flags.FLAGS

flags.DEFINE_string('video_path', '', 'Location of the mp4 video file.')
flags.DEFINE_string('output_path', '', 'Location to write the video dataset.')


def write_data_example(record_writer, image1, image2):
  """Write data example to disk."""
  assert image1.shape[0] == image2.shape[0]
  assert image1.shape[1] == image2.shape[1]
  assert image1.shape[2] == image2.shape[2]

  feature = {
      'height': conversion_utils.int64_feature(image1.shape[0]),
      'width': conversion_utils.int64_feature(image1.shape[1]),
  }
  example = tf.train.SequenceExample(
      context=tf.train.Features(feature=feature),
      feature_lists=tf.train.FeatureLists(
          feature_list={
              'images':
                  tf.train.FeatureList(feature=[
                      conversion_utils.bytes_feature(
                          image1.astype('uint8').tobytes()),
                      conversion_utils.bytes_feature(
                          image2.astype('uint8').tobytes())
                  ]),
          }))
  record_writer.write(example.SerializeToString())


def convert_video(video_file_path, output_folder):
  """Converts video at video_file_path to a tf.data.dataset at output_folder."""
  if not tf.io.gfile.exists(output_folder):
    print('Making new plot directory', output_folder)
    tf.io.gfile.makedirs(output_folder)
  filename = os.path.join(output_folder, 'fvideo@1')
  with tf.io.TFRecordWriter(filename) as record_writer:
    vidcap = cv2.VideoCapture(video_file_path)
    success = True
    count = 0
    success, image1 = vidcap.read()
    while 1:
      success, image2 = vidcap.read()
      if not success:
        break
      tf.compat.v1.logging.info('Read a new frame: %d', count)
      write_data_example(record_writer, image1, image2)
      image1 = image2
      count += 1


def main(unused_argv):
  convert_video(FLAGS.video_file, FLAGS.output_folder)


if __name__ == '__main__':
  app.run(main)
