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

"""Tests for tf.Example input layer."""

import os

from absl import flags
import tensorflow as tf

from poem.core import keypoint_profiles
from poem.core import tfe_input_layer

FLAGS = flags.FLAGS


class TfeInputLayerTest(tf.test.TestCase):

  def testReadFromTable(self):
    testdata_dir = 'poem/testdata'  # Assumes $PWD == "google_research/".
    table_path = os.path.join(FLAGS.test_srcdir, testdata_dir,
                              'tfe-2.tfrecords')
    parser_fn = tfe_input_layer.create_tfe_parser(
        keypoint_names_2d=(
            keypoint_profiles.LegacyCoco13KeypointProfile2D().keypoint_names),
        keypoint_names_3d=(
            keypoint_profiles.LegacyH36m17KeypointProfile3D().keypoint_names),
        include_keypoint_scores_2d=True,
        include_keypoint_scores_3d=False,
        num_objects=2)
    inputs = tfe_input_layer.read_from_table(table_path, parser_fn=parser_fn)
    inputs = next(iter(inputs))

    self.assertCountEqual(
        inputs.keys(),
        ['image_sizes', 'keypoints_2d', 'keypoint_scores_2d', 'keypoints_3d'])
    self.assertEqual(inputs['image_sizes'].shape, [2, 2])
    self.assertEqual(inputs['keypoints_2d'].shape, [2, 13, 2])
    self.assertEqual(inputs['keypoint_scores_2d'].shape, [2, 13])
    self.assertEqual(inputs['keypoints_3d'].shape, [2, 17, 3])

  def testReadBatchFromOneTable(self):
    testdata_dir = 'poem/testdata'  # Assumes $PWD == "google_research/".
    table_path = os.path.join(FLAGS.test_srcdir, testdata_dir,
                              'tfe-2.tfrecords')
    parser_fn = tfe_input_layer.create_tfe_parser(
        keypoint_names_2d=(
            keypoint_profiles.LegacyCoco13KeypointProfile2D().keypoint_names),
        keypoint_names_3d=(
            keypoint_profiles.LegacyH36m17KeypointProfile3D().keypoint_names),
        include_keypoint_scores_2d=True,
        include_keypoint_scores_3d=False,
        num_objects=2)
    inputs = tfe_input_layer.read_batch_from_tables([table_path],
                                                    batch_sizes=[4],
                                                    drop_remainder=True,
                                                    shuffle=True,
                                                    num_shards=2,
                                                    shard_index=1,
                                                    parser_fn=parser_fn)
    inputs = next(iter(inputs))

    self.assertCountEqual(
        inputs.keys(),
        ['image_sizes', 'keypoints_2d', 'keypoint_scores_2d', 'keypoints_3d'])
    self.assertEqual(inputs['image_sizes'].shape, [4, 2, 2])
    self.assertEqual(inputs['image_sizes'].shape, [4, 2, 2])
    self.assertEqual(inputs['keypoints_2d'].shape, [4, 2, 13, 2])
    self.assertEqual(inputs['keypoint_scores_2d'].shape, [4, 2, 13])
    self.assertEqual(inputs['keypoints_3d'].shape, [4, 2, 17, 3])

  def testReadBatchFromThreeTables(self):
    testdata_dir = 'poem/testdata'  # Assumes $PWD == "google_research/".
    table_path = os.path.join(FLAGS.test_srcdir, testdata_dir,
                              'tfe-2.tfrecords')
    parser_fn = tfe_input_layer.create_tfe_parser(
        keypoint_names_2d=(
            keypoint_profiles.LegacyCoco13KeypointProfile2D().keypoint_names),
        keypoint_names_3d=(
            keypoint_profiles.LegacyH36m17KeypointProfile3D().keypoint_names),
        include_keypoint_scores_2d=True,
        include_keypoint_scores_3d=False,
        num_objects=2)
    inputs = tfe_input_layer.read_batch_from_tables(
        [table_path, table_path, table_path],
        batch_sizes=[4, 2, 3],
        drop_remainder=True,
        shuffle=True,
        parser_fn=parser_fn)
    inputs = next(iter(inputs))

    self.assertCountEqual(
        inputs.keys(),
        ['image_sizes', 'keypoints_2d', 'keypoint_scores_2d', 'keypoints_3d'])
    self.assertEqual(inputs['image_sizes'].shape, [9, 2, 2])
    self.assertEqual(inputs['image_sizes'].shape, [9, 2, 2])
    self.assertEqual(inputs['keypoints_2d'].shape, [9, 2, 13, 2])
    self.assertEqual(inputs['keypoint_scores_2d'].shape, [9, 2, 13])
    self.assertEqual(inputs['keypoints_3d'].shape, [9, 2, 17, 3])


if __name__ == '__main__':
  tf.test.main()
