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

"""This script converts Sintel data to the TFRecords format."""

import glob
import os

from absl import app
from absl import flags
import numpy as np
import scipy.misc
import scipy.ndimage
import tensorflow as tf

from data_conversion_scripts import conversion_utils

# EAGER
tf.compat.v1.enable_eager_execution()

FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '', 'Dataset folder.')
flags.DEFINE_string('segment_data_dir', '', 'Dataset folder for segments.')
flags.DEFINE_string('output_dir', '', 'Location to export to.')
flags.DEFINE_integer('shard', None, 'Which shard this is. Pass None to write '
                     'all shards.')
flags.DEFINE_integer('num_shards', 100, 'How many total shards there are.')


def convert_dataset():
  """Convert the data to the TFRecord format."""

  if not os.path.exists(FLAGS.output_dir):
    os.mkdir(FLAGS.output_dir)

  for data_split in ['training', 'test']:
    split_folder = os.path.join(FLAGS.output_dir, data_split)
    if not os.path.exists(split_folder):
      os.mkdir(split_folder)

    for data_type in ['clean', 'final']:
      output_folder = os.path.join(FLAGS.output_dir, data_split, data_type)
      if not os.path.exists(output_folder):
        os.mkdir(output_folder)

      output_folder = os.path.join(FLAGS.output_dir, data_split, data_type)
      input_folder = os.path.join(FLAGS.data_dir, data_split, data_type)
      flow_folder = os.path.join(FLAGS.data_dir, data_split, 'flow')
      occlusion_folder = os.path.join(FLAGS.data_dir, data_split, 'occlusions')
      invalid_folder = os.path.join(FLAGS.data_dir, data_split, 'invalid')
      if data_split == 'training':
        segment_folder = os.path.join(FLAGS.segment_data_dir, data_split,
                                      'segmentation')
        segments_invalid_folder = os.path.join(FLAGS.segment_data_dir,
                                               data_split,
                                               'segmentation_invalid')
      else:
        segment_folder = None
        segments_invalid_folder = None

      # Directory with images.
      image_folders = sorted(glob.glob(input_folder + '/*'))

      if data_split == 'training':
        occlusion_folders = sorted(glob.glob(occlusion_folder + '/*'))
        invalid_folders = sorted(glob.glob(invalid_folder + '/*'))
        flow_folders = sorted(glob.glob(flow_folder + '/*'))
        segment_folders = sorted(glob.glob(segment_folder + '/*'))
        segment_invalid_folders = sorted(
            glob.glob(segments_invalid_folder + '/*'))
        assert len(image_folders) == len(flow_folders)
        assert len(flow_folders) == len(invalid_folders)
        assert len(invalid_folders) == len(occlusion_folders)
        assert len(segment_folders) == len(flow_folders)
        assert len(segment_invalid_folders) == len(segment_folders)
      else:  # Test has no ground truth flow.
        flow_folders = occlusion_folders = invalid_folders = segment_folders = segment_invalid_folders = [
            None for _ in image_folders
        ]

      data_list = []
      for image_folder, flow_folder, occlusion_folder, invalid_folder, segment_folder, segment_invalid_folder in zip(
          image_folders, flow_folders, occlusion_folders, invalid_folders,
          segment_folders, segment_invalid_folders):
        images = glob.glob(image_folder + '/*png')

        # pylint:disable=g-long-lambda
        sort_by_frame_index = lambda x: int(
            os.path.basename(x).split('_')[1].split('.')[0])
        images = sorted(images, key=sort_by_frame_index)

        if data_split == 'training':
          flows = glob.glob(flow_folder + '/*flo')
          flows = sorted(flows, key=sort_by_frame_index)
          occlusions = glob.glob(occlusion_folder + '/*png')
          occlusions = sorted(occlusions, key=sort_by_frame_index)
          invalids = glob.glob(invalid_folder + '/*png')
          invalids = sorted(invalids, key=sort_by_frame_index)
          segments = glob.glob(segment_folder + '/*png')
          segments = sorted(segments, key=sort_by_frame_index)
          segments_invalid = glob.glob(segment_invalid_folder + '/*png')
          segments_invalid = sorted(segments_invalid, key=sort_by_frame_index)
        else:  # Test has no ground truth flow.
          flows = occlusions = [None for _ in range(len(images) - 1)]
          invalids = segments = segments_invalid = [None for _ in images]

        image_pairs = list(zip(images[:-1], images[1:]))
        segment_pairs = list(zip(segments[:-1], segments[1:]))
        segment_invalid_pairs = list(
            zip(segments_invalid[:-1], segments_invalid[1:]))
        invalid_pairs = list(zip(invalids[:-1], invalids[1:]))
        # there should be 1 fewer flow images than video frames
        assert len(flows) == len(images) - 1 == len(occlusions)
        assert len(invalids) == len(images) == len(segments)
        data_list.extend(
            list(
                zip(image_pairs, flows, occlusions, invalid_pairs,
                    segment_pairs, segment_invalid_pairs)))
      if FLAGS.shards is None:  # Write all shards in this case.
        shards = list(range(FLAGS.num_shards))
      else:
        shards = [FLAGS.shard]
      for shard in shards:
        write_records(data_list, output_folder, shard)


def write_records(data_list, output_folder, shard):
  """Takes in list: [((im1_path, im2_path), flow_path)] and writes records."""
  filenames = conversion_utils.generate_sharded_filenames(
      os.path.join(output_folder, 'sintel@{}'.format(FLAGS.num_shards)))
  with tf.io.TFRecordWriter(filenames[shard]) as record_writer:
    total = len(data_list)
    images_per_shard = total // FLAGS.num_shards
    start = images_per_shard * shard
    end = start + images_per_shard
    # Account for num images not being divisible by num shards.
    if shard == FLAGS.num_shards - 1:
      data_list = data_list[start:]
    else:
      data_list = data_list[start:end]

    tf.compat.v1.logging.info('Writing %d images per shard', images_per_shard)
    tf.compat.v1.logging.info('Writing range %d to %d of %d total.', start,
                              end, total)

    for i, (images, flow, occlusion, invalids, segments,
            segments_invalid) in enumerate(data_list):

      image1_data = scipy.ndimage.imread(images[0])
      image2_data = scipy.ndimage.imread(images[1])
      if flow is not None:
        assert occlusion is not None
        assert segments is not None
        assert segments_invalid is not None

        flow_data = conversion_utils.read_flow(flow)
        # Make binary
        occlusion_data = np.expand_dims(
            scipy.ndimage.imread(occlusion) // 255, axis=-1)
        invalid1_data = np.expand_dims(
            scipy.ndimage.imread(invalids[0]) // 255, axis=-1)
        invalid2_data = np.expand_dims(
            scipy.ndimage.imread(invalids[1]) // 255, axis=-1)
        segment1_data = np.expand_dims(
            scipy.ndimage.imread(segments[0]), axis=-1)
        segment2_data = np.expand_dims(
            scipy.ndimage.imread(segments[1]), axis=-1)
        segment_invalid1_data = np.expand_dims(
            scipy.ndimage.imread(segments_invalid[0]), axis=-1)
        segment_invalid2_data = np.expand_dims(
            scipy.ndimage.imread(segments_invalid[1]), axis=-1)
      else:  # Test has no flow data, spoof flow data.
        flow_data = np.zeros((image1_data.shape[0], image1_data.shape[1], 2),
                             np.float32)
        occlusion_data = invalid1_data = invalid2_data = np.zeros(
            (image1_data.shape[0], image1_data.shape[1], 1), np.uint8)
        segment1_data = segment2_data = occlusion_data
        segment_invalid1_data = segment_invalid2_data = segment1_data
      height = image1_data.shape[0]
      width = image1_data.shape[1]

      assert height == image2_data.shape[0] == flow_data.shape[0]
      assert width == image2_data.shape[1] == flow_data.shape[1]
      assert height == occlusion_data.shape[0] == invalid1_data.shape[0]
      assert width == occlusion_data.shape[1] == invalid1_data.shape[1]
      assert invalid1_data.shape == invalid2_data.shape
      feature = {
          'height':
              conversion_utils.int64_feature(height),
          'width':
              conversion_utils.int64_feature(width),
          'image1_path':
              conversion_utils.bytes_feature(str.encode(images[0])),
          'image2_path':
              conversion_utils.bytes_feature(str.encode(images[1])),
      }
      feature_list = {}
      if flow is not None:
        feature.update({
            'flow_uv':
                conversion_utils.bytes_feature(flow_data.tobytes()),
            'occlusion_mask':
                conversion_utils.bytes_feature(occlusion_data.tobytes()),
            'flow_path':
                conversion_utils.bytes_feature(str.encode(flow)),
            'occlusion_path':
                conversion_utils.bytes_feature(str.encode(occlusion))})
      if segments[0] is not None:
        feature.update({
            'segment1_path':
                conversion_utils.bytes_feature(str.encode(segments[0])),
            'segment2_path':
                conversion_utils.bytes_feature(str.encode(segments[1])),
            'segment_invalid1_path':
                conversion_utils.bytes_feature(
                    str.encode(segments_invalid[0])),
            'segment_invalid2_path':
                conversion_utils.bytes_feature(
                    str.encode(segments_invalid[1])),
        })
        feature_list.update({
            'segments':
                tf.train.FeatureList(feature=[
                    conversion_utils.bytes_feature(segment1_data.tobytes()),
                    conversion_utils.bytes_feature(segment2_data.tobytes()),
                ]),
            'segments_invalid':
                tf.train.FeatureList(feature=[
                    conversion_utils.bytes_feature(
                        segment_invalid1_data.tobytes()),
                    conversion_utils.bytes_feature(
                        segment_invalid2_data.tobytes()),
                ]),
        })
      feature_list.update({
          'images':
              tf.train.FeatureList(feature=[
                  conversion_utils.bytes_feature(image1_data.tobytes()),
                  conversion_utils.bytes_feature(image2_data.tobytes())
              ]),
          'invalid_masks':
              tf.train.FeatureList(feature=[
                  conversion_utils.bytes_feature(invalid1_data.tobytes()),
                  conversion_utils.bytes_feature(invalid2_data.tobytes())
              ])
      })
      example = tf.train.SequenceExample(
          context=tf.train.Features(feature=feature),
          feature_lists=tf.train.FeatureLists(feature_list=feature_list))
      if i % 10 == 0:
        tf.compat.v1.logging.info('Writing %d out of %d total.', i,
                                  len(data_list))
      record_writer.write(example.SerializeToString())

  tf.compat.v1.logging.info('Saved results to %s', FLAGS.output_dir)


def main(_):
  convert_dataset()


if __name__ == '__main__':
  app.run(main)
