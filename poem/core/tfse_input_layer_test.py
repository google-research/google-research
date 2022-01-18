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

"""Tests for tf.SequenceExample input layer."""

import os

from absl import flags

import tensorflow as tf
from poem.core import keypoint_profiles
from poem.core import tfse_input_layer

FLAGS = flags.FLAGS


class TfseInputLayerInternalTest(tf.test.TestCase):

  def testReadFromTable(self):
    testdata_dir = 'poem/testdata'  # Assume $PWD == "google_research/".
    table_path = os.path.join(FLAGS.test_srcdir, testdata_dir,
                              'tfse-2.tfrecords')
    parser_fn = tfse_input_layer.create_tfse_parser(
        keypoint_names_2d=(
            keypoint_profiles.Std13KeypointProfile2D().keypoint_names),
        keypoint_names_3d=(
            keypoint_profiles.Std16KeypointProfile3D().keypoint_names),
        num_objects=2,
        sequence_length=5)
    inputs = tfse_input_layer.read_from_table(table_path, parser_fn=parser_fn)
    inputs = next(iter(inputs))

    self.assertCountEqual(
        inputs.keys(),
        ['image_sizes',
         'keypoints_2d', 'keypoint_scores_2d',
         'keypoints_3d'])
    self.assertEqual(inputs['image_sizes'].shape, [2, 5, 2])
    self.assertEqual(inputs['keypoints_2d'].shape, [2, 5, 13, 2])
    self.assertEqual(inputs['keypoint_scores_2d'].shape, [2, 5, 13])
    self.assertEqual(inputs['keypoints_3d'].shape, [2, 5, 16, 3])

if __name__ == '__main__':
  tf.test.main()
