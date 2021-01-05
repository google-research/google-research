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

"""Converts the KITTI image pair data to the TFRecords format."""

import os
from absl import app
from absl import flags
import tensorflow as tf

FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '.', 'Dataset folder.')
flags.DEFINE_string('subdirs', 'training,testing', 'training or testing')


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def convert_dataset(data_dir):
  """Convert the data to the TFRecord format."""

  for subdir in FLAGS.subdirs.split(','):
    # Make a directory to save the tfrecords to.
    output_dir = data_dir + '_' + subdir + '-tfrecords'
    # Directory with images.
    image_dir = os.path.join(data_dir, subdir + '/image_2')
    num_images = len(tf.io.gfile.listdir(image_dir)) // 2

    if not os.path.exists(output_dir):
      os.mkdir(output_dir)

    for i in range(num_images):
      image_files = ['{0:06d}_{1:}.png'.format(i, j) for j in [10, 11]]

      # Collect RGB images.
      image_bytes_list = []
      for image_file in image_files:
        # Read the image.
        image_path = os.path.join(image_dir, image_file)
        image_data = tf.compat.v1.gfile.FastGFile(image_path, 'r').read()
        image_tensor = tf.image.decode_png(image_data, channels=3)
        height, width, _ = image_tensor.shape
        # Encode image as byte list again.
        image_bytes_list.append(image_tensor.numpy().tobytes())

      # if subdir == 'testing':
      if subdir == 'training':
        # Collect flow.
        # Flow in the first image points to the second one; including occluded
        # regions (occ), or not including occluded regions (noc).
        flow_uv_bytes = dict()
        flow_valid_bytes = dict()
        for version in ['noc', 'occ']:
          flow_path = os.path.join(data_dir, FLAGS.subdir, 'flow_' + version,
                                   image_files[0])
          flow_data = tf.compat.v1.gfile.FastGFile(flow_path, 'r').read()
          flow_tensor = tf.image.decode_png(
              flow_data, channels=3, dtype=tf.uint16)
          # Recover flow vectors from flow image according to KITTI README.
          flow_uv = (tf.cast(flow_tensor[Ellipsis, :2], tf.float32) - 2**15) / 64.0
          flow_valid = tf.cast(flow_tensor[Ellipsis, 2:3], tf.uint8)

          # Encode image as byte list again.
          flow_uv_bytes[version] = flow_uv.numpy().tobytes()
          flow_valid_bytes[version] = flow_valid.numpy().tobytes()

        # Build a tf sequence example.
        example = tf.train.SequenceExample(
            context=tf.train.Features(
                feature={
                    'height': int64_feature(height),
                    'width': int64_feature(width),
                    'flow_uv_occ': bytes_feature(flow_uv_bytes['occ']),
                    'flow_uv_noc': bytes_feature(flow_uv_bytes['noc']),
                    'flow_valid_occ': bytes_feature(flow_valid_bytes['occ']),
                    'flow_valid_noc': bytes_feature(flow_valid_bytes['noc']),
                }),
            feature_lists=tf.train.FeatureLists(
                feature_list={
                    'images':
                        tf.train.FeatureList(feature=[
                            bytes_feature(b) for b in image_bytes_list
                        ])
                }))
      elif subdir == 'testing':
        # Build a tf sequence example.
        example = tf.train.SequenceExample(
            context=tf.train.Features(feature={
                'height': int64_feature(height),
                'width': int64_feature(width),
            }),
            feature_lists=tf.train.FeatureLists(
                feature_list={
                    'images':
                        tf.train.FeatureList(feature=[
                            bytes_feature(b) for b in image_bytes_list
                        ])
                }))

      # Create a tfrecord file to save this sequence to.
      output_filename = data_dir.split(
          '/')[-1] + '_' + subdir + '_{0:06d}.tfrecord'.format(i)
      output_file = os.path.join(output_dir, output_filename)
      with tf.io.TFRecordWriter(output_file) as record_writer:
        record_writer.write(example.SerializeToString())
        record_writer.flush()
    print('Saved results to', output_dir)


def main(unused_argv):
  convert_dataset(FLAGS.data_dir)


if __name__ == '__main__':
  app.run(main)
