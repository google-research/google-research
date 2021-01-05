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

"""Converts KITTI multiview extension data to the TFRecords format."""


import os
from absl import app
from absl import flags
import tensorflow as tf

FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '.', 'Dataset folder.')
flags.DEFINE_integer('height', 384, '')
flags.DEFINE_integer('width', 1280, '')
flags.DEFINE_bool('entire_sequence', False,
                  'Train on the full sequence, otherwise skip frames 9-12.')


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def convert_dataset(data_dir):
  """Convert the data to the TFRecord format."""

  for subdir in ['training', 'testing']:

    if FLAGS.entire_sequence:
      sequences = [list(range(21))]
      output_dir = data_dir + '_{}_{}x{}_fullseq-tfrecords'.format(
          subdir[:-3], FLAGS.height, FLAGS.width)
    else:
      # Of the 21 frames, ignore frames 9-12 because we will test on those.
      sequences = [[0, 1, 2, 3, 4, 5, 6, 7, 8],
                   [13, 14, 15, 16, 17, 18, 19, 20]]
      # Make a directory to save the tfrecords to.
      output_dir = data_dir + '_{}_{}x{}-tfrecords'.format(
          subdir[:-3], FLAGS.height, FLAGS.width)

    # Directory with images.
    image_dir = os.path.join(data_dir, subdir + '/image_2')
    num_images = int(tf.io.gfile.listdir(image_dir)[-1][:-7])

    if not os.path.exists(output_dir):
      os.mkdir(output_dir)

    for i in range(num_images):
      # Don't use frames 9-12 because those will be tested.
      for js in sequences:
        image_files = ['{0:06d}_{1:02d}.png'.format(i, j) for j in js]

        try:
          # Collect RGB images.
          image_bytes_list = []
          for image_file in image_files:
            # Read the image.
            image_path = os.path.join(image_dir, image_file)
            image_data = tf.compat.v1.gfile.FastGFile(image_path, 'r').read()

            image_tensor = tf.image.decode_png(image_data, channels=3)
            image_resized = tf.image.resize(
                image_tensor[None], [FLAGS.height, FLAGS.width],
                method=tf.image.ResizeMethod.BILINEAR)[0]
            # Undo the implicit cast of resize_images to tf.float32
            image_resized = tf.cast(image_resized, tf.uint8)
            # Encode image as byte list again.
            # image_bytes_list.append(image_resized.numpy().tobytes())
            image_bytes_list.append(tf.image.encode_png(image_resized).numpy())

          # Build a tf sequence example.
          example = tf.train.SequenceExample(
              context=tf.train.Features(
                  feature={
                      'height': int64_feature(FLAGS.height),
                      'width': int64_feature(FLAGS.width),
                  }),
              feature_lists=tf.train.FeatureLists(
                  feature_list={
                      'images':
                          tf.train.FeatureList(feature=[
                              bytes_feature(b) for b in image_bytes_list
                          ])
                  }))
          output_filename = data_dir.split('/')[
              -1] + '_' + subdir + '_{0:06d}_{1:02d}-{2:02d}.tfrecord'.format(
                  i, js[0], js[-1])
          output_file = os.path.join(output_dir, output_filename)
          with tf.io.TFRecordWriter(output_file) as record_writer:
            record_writer.write(example.SerializeToString())
            record_writer.flush()

        except tf.errors.NotFoundError:
          print('Skipping {} because the file is not found.'.format(image_path))

  print('Saved results to', output_dir)


def main(unused_argv):
  convert_dataset(FLAGS.data_dir)


if __name__ == '__main__':
  app.run(main)
