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

"""This script converts Flying Chairs data to the TFRecords format."""

# pylint:disable=g-bad-import-order
import glob
import os


from absl import app
from absl import flags
import scipy.misc
import scipy.ndimage
import tensorflow as tf

from smurf.data_conversion_scripts import conversion_utils

FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir',
                    '',
                    'Dataset location of raw Flying chairs data. E.g., '
                    '/tmp/FlyingChairs/data')
flags.DEFINE_string(
    'output_dir', '',
    'Location to export to.')
flags.DEFINE_integer('shard', None, 'Which shard this is. Pass None to write'
                     ' all shards.')
flags.DEFINE_integer('num_shards', 100, 'How many total shards there are.')



def convert_dataset(shard):
  """Convert the data to the TFRecord format."""

  # Make a directory to save the tfrecords to.
  if not os.path.exists(FLAGS.output_dir):
    os.mkdir(FLAGS.output_dir)

  train_dir = os.path.join(FLAGS.output_dir, 'train')
  test_dir = os.path.join(FLAGS.output_dir, 'test')
  if not os.path.exists(train_dir):
    os.mkdir(train_dir)
  if not os.path.exists(test_dir):
    os.mkdir(test_dir)

  # Directory with images.
  images = sorted(glob.glob(FLAGS.data_dir + '/*.ppm'))
  flow_list = sorted(glob.glob(FLAGS.data_dir + '/*.flo'))
  assert len(images) // 2 == len(flow_list)
  image_list = []
  for i in range(len(flow_list)):
    im1 = images[2 * i]
    im2 = images[2 * i + 1]
    image_list.append((im1, im2))
  assert len(image_list) == len(flow_list)

  train_filenames = conversion_utils.generate_sharded_filenames(
      os.path.join(train_dir, 'flying_chairs@{}'.format(FLAGS.num_shards)))
  test_filenames = conversion_utils.generate_sharded_filenames(
      os.path.join(test_dir, 'flying_chairs@{}'.format(FLAGS.num_shards)))
  train_record_writer = tf.io.TFRecordWriter(
      train_filenames[shard])
  test_record_writer = tf.io.TFRecordWriter(test_filenames[shard])
  total = len(image_list)
  images_per_shard = total // FLAGS.num_shards
  start = images_per_shard * shard
  smurf_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  filepath = 'files/chairs_train_val.txt'
  filepath = os.path.join(smurf_dir, filepath)
  with open(filepath, mode='r') as f:
    train_val = f.readlines()
    train_val = [int(x.strip()) for x in train_val]
  if shard == FLAGS.num_shards - 1:
    end = len(image_list)
  else:
    end = start + images_per_shard
  assert len(train_val) == len(image_list)
  assert len(flow_list) == len(train_val)
  image_list = image_list[start:end]
  train_val = train_val[start:end]
  flow_list = flow_list[start:end]

  tf.compat.v1.logging.info('Writing %d images per shard', images_per_shard)
  tf.compat.v1.logging.info('Writing range %d to %d of %d total.', start, end,
                            total)

  for i, (images, flow,
          assignment) in enumerate(zip(image_list, flow_list, train_val)):
    image1_data = scipy.ndimage.imread(images[0])
    image2_data = scipy.ndimage.imread(images[1])
    flow_data = conversion_utils.read_flow(flow)

    height = image1_data.shape[0]
    width = image1_data.shape[1]

    assert height == image2_data.shape[0] == flow_data.shape[0]
    assert width == image2_data.shape[1] == flow_data.shape[1]

    example = tf.train.SequenceExample(
        context=tf.train.Features(
            feature={
                'height': conversion_utils.int64_feature(height),
                'width': conversion_utils.int64_feature(width),
                'flow_uv': conversion_utils.bytes_feature(flow_data.tobytes()),
            }),
        feature_lists=tf.train.FeatureLists(
            feature_list={
                'images':
                    tf.train.FeatureList(feature=[
                        conversion_utils.bytes_feature(image1_data.tobytes()),
                        conversion_utils.bytes_feature(image2_data.tobytes())
                    ])
            }))
    if i % 10 == 0:
      tf.compat.v1.logging.info('Writing %d out of %d total.', i,
                                len(image_list))
    if assignment == 1:
      train_record_writer.write(example.SerializeToString())
    elif assignment == 2:
      test_record_writer.write(example.SerializeToString())
    else:
      assert False, 'There is an error in the chairs_train_val.txt'

  train_record_writer.close()
  test_record_writer.close()
  tf.compat.v1.logging.info('Saved results to %s', FLAGS.output_dir)


def main(_):
  if FLAGS.shard is None:
    shards = list(range(FLAGS.num_shards))
  else:
    shards = [FLAGS.shard]
  for shard in shards:
    convert_dataset(shard)


if __name__ == '__main__':
  app.run(main)
