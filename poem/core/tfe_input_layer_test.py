# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

  def testGenerateClassTargets(self):
    label_ids = tf.constant([0, 1, 3], dtype=tf.int64)
    label_confidences = tf.constant([0.9, 0.3, 0.7], dtype=tf.float32)
    class_targets, class_weights = (
        tfe_input_layer.generate_class_targets(
            label_ids, label_confidences, num_classes=5))
    self.assertAllEqual(class_targets, [1, 0, 0, 1, 0])
    self.assertAllClose(class_weights, [1.0, 1.0, 0.0, 1.0, 0.0])

  def testReadFromTable(self):
    testdata_dir = 'poem/testdata'  # Assume $PWD == "google_research/".
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
    testdata_dir = 'poem/testdata'  # Assume $PWD == "google_research/".
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
    testdata_dir = 'poem/testdata'  # Assume $PWD == "google_research/".
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

  def testReadFromTableWithoutSequenceDim(self):
    testdata_dir = 'poem/testdata'  # Assume $PWD == "google_research/".
    table_path = os.path.join(FLAGS.test_srcdir, testdata_dir,
                              'tfe-1.tfrecords')
    parser_fn = tfe_input_layer.create_tfe_parser(
        keypoint_names_2d=(
            keypoint_profiles.LegacyCoco13KeypointProfile2D().keypoint_names),
        include_keypoint_scores_2d=True,
        feature_dim=32,
        num_classes=6,
        num_objects=1)
    inputs = tfe_input_layer.read_from_table(
        [table_path],
        dataset_class=tf.data.TFRecordDataset,
        parser_fn=parser_fn)
    inputs = next(iter(inputs))

    self.assertCountEqual(inputs.keys(), [
        'image_sizes', 'keypoints_2d', 'keypoint_scores_2d', 'features',
        'class_targets', 'class_weights'
    ])
    self.assertEqual(inputs['image_sizes'].shape, [1, 2])
    self.assertEqual(inputs['features'].shape, [1, 32])
    self.assertEqual(inputs['keypoints_2d'].shape, [1, 13, 2])
    self.assertEqual(inputs['keypoint_scores_2d'].shape, [1, 13])
    self.assertEqual(inputs['class_targets'].shape, [1, 6])
    self.assertEqual(inputs['class_weights'].shape, [1, 6])

  def testReadFromTableWithSequenceDim(self):
    testdata_dir = 'poem/testdata'  # Assume $PWD == "google_research/".
    table_path = os.path.join(FLAGS.test_srcdir, testdata_dir,
                              'tfe-1.tfrecords')
    parser_fn = tfe_input_layer.create_tfe_parser(
        keypoint_names_2d=(
            keypoint_profiles.LegacyCoco13KeypointProfile2D().keypoint_names),
        include_keypoint_scores_2d=True,
        feature_dim=32,
        num_classes=6,
        num_objects=1,
        sequence_length=1)
    inputs = tfe_input_layer.read_from_table(
        [table_path],
        dataset_class=tf.data.TFRecordDataset,
        parser_fn=parser_fn)
    inputs = next(iter(inputs))

    self.assertCountEqual(inputs.keys(), [
        'image_sizes', 'keypoints_2d', 'keypoint_scores_2d', 'features',
        'class_targets', 'class_weights'
    ])
    self.assertEqual(inputs['image_sizes'].shape, [1, 1, 2])
    self.assertEqual(inputs['features'].shape, [1, 1, 32])
    self.assertEqual(inputs['keypoints_2d'].shape, [1, 1, 13, 2])
    self.assertEqual(inputs['keypoint_scores_2d'].shape, [1, 1, 13])
    self.assertEqual(inputs['class_targets'].shape, [1, 6])
    self.assertEqual(inputs['class_weights'].shape, [1, 6])

  def testReadSequenceBatchFromTable(self):
    testdata_dir = 'poem/testdata'  # Assume $PWD == "google_research/".
    table_path = os.path.join(FLAGS.test_srcdir, testdata_dir,
                              'tfe-1-seq.tfrecords')
    parser_fn = tfe_input_layer.create_tfe_parser(
        keypoint_names_2d=(
            keypoint_profiles.LegacyCoco13KeypointProfile2D().keypoint_names),
        include_keypoint_scores_2d=True,
        num_classes=6,
        num_objects=1,
        sequence_length=5)
    inputs = tfe_input_layer.read_from_table(
        [table_path],
        dataset_class=tf.data.TFRecordDataset,
        parser_fn=parser_fn)
    inputs = next(iter(inputs))

    self.assertCountEqual(inputs.keys(), [
        'image_sizes', 'keypoints_2d', 'keypoint_scores_2d', 'class_targets',
        'class_weights'
    ])
    self.assertEqual(inputs['image_sizes'].shape, [1, 5, 2])
    self.assertEqual(inputs['keypoints_2d'].shape, [1, 5, 13, 2])
    self.assertEqual(inputs['keypoint_scores_2d'].shape, [1, 5, 13])
    self.assertEqual(inputs['class_targets'].shape, [1, 6])
    self.assertEqual(inputs['class_weights'].shape, [1, 6])


if __name__ == '__main__':
  tf.test.main()
