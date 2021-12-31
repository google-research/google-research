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

"""This script converts Kitti data to the TFRecords format."""


import os
from absl import app
from absl import flags
import tensorflow as tf

from smurf.data_conversion_scripts import conversion_utils


FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '', 'Location of the raw KITTI data folder.')
flags.DEFINE_string('subdirs', 'training,testing', 'training or testing')


def convert_dataset(data_dir):
  """Convert the data to the TFRecord format."""

  for subdir in FLAGS.subdirs.split(','):
    # Make a directory to save the tfrecords to.
    output_dir = data_dir + '_' + subdir + '-tfrecords'
    # Directory with images.
    image_dir = os.path.join(data_dir, subdir + '/image_2')
    image_dir_right = os.path.join(data_dir, subdir + '/image_3')
    num_images = len(tf.io.gfile.listdir(image_dir)) // 2

    if not os.path.exists(output_dir):
      os.mkdir(output_dir)

    for i in range(num_images):
      image_files = ['{0:06d}_{1:}.png'.format(i, j) for j in [10, 11]]

      # Collect RGB images.
      image_bytes_list_left = []
      image_bytes_list_right = []
      for image_file in image_files:
        image_path_left = os.path.join(image_dir, image_file)
        image_path_right = os.path.join(image_dir_right, image_file)
        image_data_left = tf.io.gfile.GFile(image_path_left, 'rb').read()
        image_data_right = tf.io.gfile.GFile(image_path_right, 'rb').read()
        image_tensor_left = tf.image.decode_png(image_data_left, channels=3)
        image_tensor_right = tf.image.decode_png(image_data_right, channels=3)
        height, width, _ = image_tensor_left.shape
        # Encode image as byte list again.
        image_bytes_list_left.append(image_tensor_left.numpy().tobytes())
        image_bytes_list_right.append(image_tensor_right.numpy().tobytes())

      if subdir == 'training':
        # Collect flow.
        # Flow in the first image points to the second one; including occluded
        # regions (occ), or not including occluded regions (noc).

        # NOTE: disp0 corresponds to disparity at time 0
        # and disp1 to disparity at time 1. All disparities are given
        # in the frame of the left image.
        flow_uv_bytes = dict()
        flow_valid_bytes = dict()
        disp0_bytes = dict()
        disp0_valid_bytes = dict()
        disp1_bytes = dict()
        disp1_valid_bytes = dict()
        for version in ['noc', 'occ']:
          flow_path = os.path.join(data_dir, subdir, 'flow_' + version,
                                   image_files[0])
          disp_path0 = os.path.join(data_dir, subdir,
                                    'disp_' + version + '_0', image_files[0])
          disp_path1 = os.path.join(data_dir, subdir,
                                    'disp_' + version + '_1', image_files[0])
          flow_data = tf.io.gfile.GFile(flow_path, 'rb').read()
          disp_data0 = tf.io.gfile.GFile(disp_path0, 'rb').read()
          disp_data1 = tf.io.gfile.GFile(disp_path1, 'rb').read()
          flow_tensor = tf.image.decode_png(
              flow_data, channels=3, dtype=tf.uint16)
          disp0_tensor = tf.image.decode_png(
              disp_data0, channels=1, dtype=tf.uint16)
          disp1_tensor = tf.image.decode_png(
              disp_data1, channels=1, dtype=tf.uint16)
          # Recover flow vectors from flow image according to KITTI README.
          flow_uv = (tf.cast(flow_tensor[Ellipsis, :2], tf.float32) - 2**15) / 64.0
          flow_valid = tf.cast(flow_tensor[Ellipsis, 2:3], tf.uint8)
          # Recover disp according to the KITTI README.
          disp0 = tf.cast(disp0_tensor, tf.float32) / 256.
          disp0_valid = tf.cast(disp0 > 0, tf.uint8)
          disp1 = tf.cast(disp1_tensor, tf.float32) / 256.
          disp1_valid = tf.cast(disp1 > 0, tf.uint8)

          # Encode image as byte list again.
          flow_uv_bytes[version] = flow_uv.numpy().tobytes()
          flow_valid_bytes[version] = flow_valid.numpy().tobytes()
          disp0_bytes[version] = disp0.numpy().tobytes()
          disp0_valid_bytes[version] = disp0_valid.numpy().tobytes()
          disp1_bytes[version] = disp1.numpy().tobytes()
          disp1_valid_bytes[version] = disp1_valid.numpy().tobytes()

        # Build a tf sequence example.
        example = tf.train.SequenceExample(
            context=tf.train.Features(
                feature={
                    'height':
                        conversion_utils.int64_feature(height),
                    'width':
                        conversion_utils.int64_feature(width),
                    'flow_uv_occ':
                        conversion_utils.bytes_feature(flow_uv_bytes['occ']),
                    'flow_uv_noc':
                        conversion_utils.bytes_feature(flow_uv_bytes['noc']),
                    'flow_valid_occ':
                        conversion_utils.bytes_feature(flow_valid_bytes['occ']),
                    'flow_valid_noc':
                        conversion_utils.bytes_feature(flow_valid_bytes['noc']),
                    'disp0_occ':
                        conversion_utils.bytes_feature(disp0_bytes['occ']),
                    'disp0_noc':
                        conversion_utils.bytes_feature(disp0_bytes['noc']),
                    'disp1_occ':
                        conversion_utils.bytes_feature(disp1_bytes['occ']),
                    'disp1_noc':
                        conversion_utils.bytes_feature(disp1_bytes['noc']),
                    'disp0_valid_occ':
                        conversion_utils.bytes_feature(disp0_valid_bytes['occ']
                                                      ),
                    'disp0_valid_noc':
                        conversion_utils.bytes_feature(disp0_valid_bytes['noc']
                                                      ),
                    'disp1_valid_occ':
                        conversion_utils.bytes_feature(disp1_valid_bytes['occ']
                                                      ),
                    'disp1_valid_noc':
                        conversion_utils.bytes_feature(disp1_valid_bytes['noc']
                                                      ),
                }),
            feature_lists=tf.train.FeatureLists(
                feature_list={
                    'images':
                        tf.train.FeatureList(feature=[
                            conversion_utils.bytes_feature(b)
                            for b in image_bytes_list_left
                        ]),
                    'images_right':
                        tf.train.FeatureList(feature=[
                            conversion_utils.bytes_feature(b)
                            for b in image_bytes_list_right
                        ])
                }))
      elif subdir == 'testing':
        # Build a tf sequence example.
        example = tf.train.SequenceExample(
            context=tf.train.Features(feature={
                'height': conversion_utils.int64_feature(height),
                'width': conversion_utils.int64_feature(width),
            }),
            feature_lists=tf.train.FeatureLists(
                feature_list={
                    'images':
                        tf.train.FeatureList(feature=[
                            conversion_utils.bytes_feature(b)
                            for b in image_bytes_list_left
                        ]),
                    'images_right':
                        tf.train.FeatureList(feature=[
                            conversion_utils.bytes_feature(b)
                            for b in image_bytes_list_right
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
  if not FLAGS.data_dir:
    raise ValueError('Must pass kitti root directory as '
                     '--data_dir=<path to raw data>.')
  convert_dataset(FLAGS.data_dir)

if __name__ == '__main__':
  app.run(main)
