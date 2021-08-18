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

"""Tests for pipeline utility functions."""

import os

from absl import flags
import tensorflow.compat.v1 as tf

from poem.core import keypoint_profiles
from poem.core import models
from poem.core import pipeline_utils
tf.disable_v2_behavior()

FLAGS = flags.FLAGS


class PipelineUtilsTest(tf.test.TestCase):

  def test_read_batch_from_dataset_tables(self):
    testdata_dir = 'poem/testdata'  # Assume $PWD == "google_research/".
    table_path = os.path.join(FLAGS.test_srcdir, testdata_dir,
                              'tfe-2.tfrecords')
    inputs = pipeline_utils.read_batch_from_dataset_tables(
        [table_path, table_path],
        batch_sizes=[4, 2],
        num_instances_per_record=2,
        shuffle=True,
        num_epochs=None,
        keypoint_names_3d=keypoint_profiles.create_keypoint_profile_or_die(
            'LEGACY_3DH36M17').keypoint_names,
        keypoint_names_2d=keypoint_profiles.create_keypoint_profile_or_die(
            'LEGACY_2DCOCO13').keypoint_names,
        seed=0)

    self.assertCountEqual(inputs.keys(), [
        'image_sizes', 'keypoints_2d', 'keypoint_scores_2d',
        'keypoint_masks_2d', 'keypoints_3d'
    ])
    self.assertEqual(inputs['image_sizes'].shape, [6, 2, 2])
    self.assertEqual(inputs['keypoints_2d'].shape, [6, 2, 13, 2])
    self.assertEqual(inputs['keypoint_scores_2d'].shape, [6, 2, 13])
    self.assertEqual(inputs['keypoint_masks_2d'].shape, [6, 2, 13])
    self.assertEqual(inputs['keypoints_3d'].shape, [6, 2, 17, 3])

  def test_add_moving_average(self):
    inputs = tf.zeros([4, 2, 3])
    output_sizes = {'a': 8, 'b': 4}
    models.simple_model(
        inputs,
        output_sizes,
        sequential_inputs=False,
        is_training=True,
        name='M')
    pipeline_utils.add_moving_average(decay=0.9999)

    expected_global_variable_shapes = {
        'M/InputFC/Linear/weight:0': ([3, 1024]),
        'M/InputFC/Linear/bias:0': ([1024]),
        'M/InputFC/BatchNorm/gamma:0': ([1024]),
        'M/InputFC/BatchNorm/beta:0': ([1024]),
        'M/InputFC/BatchNorm/moving_mean:0': ([1024]),
        'M/InputFC/BatchNorm/moving_variance:0': ([1024]),
        'M/FullyConnectedBlock_0/FC_0/Linear/weight:0': ([1024, 1024]),
        'M/FullyConnectedBlock_0/FC_0/Linear/bias:0': ([1024]),
        'M/FullyConnectedBlock_0/FC_0/BatchNorm/gamma:0': ([1024]),
        'M/FullyConnectedBlock_0/FC_0/BatchNorm/beta:0': ([1024]),
        'M/FullyConnectedBlock_0/FC_0/BatchNorm/moving_mean:0': ([1024]),
        'M/FullyConnectedBlock_0/FC_0/BatchNorm/moving_variance:0': ([1024]),
        'M/FullyConnectedBlock_0/FC_1/Linear/weight:0': ([1024, 1024]),
        'M/FullyConnectedBlock_0/FC_1/Linear/bias:0': ([1024]),
        'M/FullyConnectedBlock_0/FC_1/BatchNorm/gamma:0': ([1024]),
        'M/FullyConnectedBlock_0/FC_1/BatchNorm/beta:0': ([1024]),
        'M/FullyConnectedBlock_0/FC_1/BatchNorm/moving_mean:0': ([1024]),
        'M/FullyConnectedBlock_0/FC_1/BatchNorm/moving_variance:0': ([1024]),
        'M/FullyConnectedBlock_1/FC_0/Linear/weight:0': ([1024, 1024]),
        'M/FullyConnectedBlock_1/FC_0/Linear/bias:0': ([1024]),
        'M/FullyConnectedBlock_1/FC_0/BatchNorm/gamma:0': ([1024]),
        'M/FullyConnectedBlock_1/FC_0/BatchNorm/beta:0': ([1024]),
        'M/FullyConnectedBlock_1/FC_0/BatchNorm/moving_mean:0': ([1024]),
        'M/FullyConnectedBlock_1/FC_0/BatchNorm/moving_variance:0': ([1024]),
        'M/FullyConnectedBlock_1/FC_1/Linear/weight:0': ([1024, 1024]),
        'M/FullyConnectedBlock_1/FC_1/Linear/bias:0': ([1024]),
        'M/FullyConnectedBlock_1/FC_1/BatchNorm/gamma:0': ([1024]),
        'M/FullyConnectedBlock_1/FC_1/BatchNorm/beta:0': ([1024]),
        'M/FullyConnectedBlock_1/FC_1/BatchNorm/moving_mean:0': ([1024]),
        'M/FullyConnectedBlock_1/FC_1/BatchNorm/moving_variance:0': ([1024]),
        'M/OutputLogits/a/weight:0': ([1024, 8]),
        'M/OutputLogits/a/bias:0': ([8]),
        'M/OutputLogits/b/weight:0': ([1024, 4]),
        'M/OutputLogits/b/bias:0': ([4]),
        'M/InputFC/Linear/weight/ExponentialMovingAverage:0': ([3, 1024]),
        'M/InputFC/Linear/bias/ExponentialMovingAverage:0': ([1024]),
        'M/InputFC/BatchNorm/gamma/ExponentialMovingAverage:0': ([1024]),
        'M/InputFC/BatchNorm/beta/ExponentialMovingAverage:0': ([1024]),
        'M/FullyConnectedBlock_0/FC_0/Linear/weight/ExponentialMovingAverage:0':
            ([1024, 1024]),
        'M/FullyConnectedBlock_0/FC_0/Linear/bias/ExponentialMovingAverage:0':
            ([1024]),
        'M/FullyConnectedBlock_0/FC_0/BatchNorm/gamma/ExponentialMovingAverage:0':
            ([1024]),
        'M/FullyConnectedBlock_0/FC_0/BatchNorm/beta/ExponentialMovingAverage:0':
            ([1024]),
        'M/FullyConnectedBlock_0/FC_1/Linear/weight/ExponentialMovingAverage:0':
            ([1024, 1024]),
        'M/FullyConnectedBlock_0/FC_1/Linear/bias/ExponentialMovingAverage:0':
            ([1024]),
        'M/FullyConnectedBlock_0/FC_1/BatchNorm/gamma/ExponentialMovingAverage:0':
            ([1024]),
        'M/FullyConnectedBlock_0/FC_1/BatchNorm/beta/ExponentialMovingAverage:0':
            ([1024]),
        'M/FullyConnectedBlock_1/FC_0/Linear/weight/ExponentialMovingAverage:0':
            ([1024, 1024]),
        'M/FullyConnectedBlock_1/FC_0/Linear/bias/ExponentialMovingAverage:0':
            ([1024]),
        'M/FullyConnectedBlock_1/FC_0/BatchNorm/gamma/ExponentialMovingAverage:0':
            ([1024]),
        'M/FullyConnectedBlock_1/FC_0/BatchNorm/beta/ExponentialMovingAverage:0':
            ([1024]),
        'M/FullyConnectedBlock_1/FC_1/Linear/weight/ExponentialMovingAverage:0':
            ([1024, 1024]),
        'M/FullyConnectedBlock_1/FC_1/Linear/bias/ExponentialMovingAverage:0':
            ([1024]),
        'M/FullyConnectedBlock_1/FC_1/BatchNorm/gamma/ExponentialMovingAverage:0':
            ([1024]),
        'M/FullyConnectedBlock_1/FC_1/BatchNorm/beta/ExponentialMovingAverage:0':
            ([1024]),
        'M/OutputLogits/a/weight/ExponentialMovingAverage:0': ([1024, 8]),
        'M/OutputLogits/a/bias/ExponentialMovingAverage:0': ([8]),
        'M/OutputLogits/b/weight/ExponentialMovingAverage:0': ([1024, 4]),
        'M/OutputLogits/b/bias/ExponentialMovingAverage:0': ([4]),
        'global_step:0': ([]),
    }
    self.assertDictEqual(
        {var.name: var.shape.as_list() for var in tf.global_variables()},
        expected_global_variable_shapes)

  def test_get_moving_average_variables_to_restore(self):
    inputs = tf.zeros([4, 2, 3])
    output_sizes = {'a': 8, 'b': 4}
    models.simple_model(
        inputs,
        output_sizes,
        sequential_inputs=False,
        is_training=False,
        name='M')
    variables_to_restore = (
        pipeline_utils.get_moving_average_variables_to_restore())

    expected_variable_to_restore_names = {
        'M/InputFC/Linear/weight/ExponentialMovingAverage':
            'M/InputFC/Linear/weight:0',
        'M/InputFC/Linear/bias/ExponentialMovingAverage':
            'M/InputFC/Linear/bias:0',
        'M/InputFC/BatchNorm/gamma/ExponentialMovingAverage':
            'M/InputFC/BatchNorm/gamma:0',
        'M/InputFC/BatchNorm/beta/ExponentialMovingAverage':
            'M/InputFC/BatchNorm/beta:0',
        'M/InputFC/BatchNorm/moving_mean':
            'M/InputFC/BatchNorm/moving_mean:0',
        'M/InputFC/BatchNorm/moving_variance':
            'M/InputFC/BatchNorm/moving_variance:0',
        'M/FullyConnectedBlock_0/FC_0/Linear/weight/ExponentialMovingAverage':
            'M/FullyConnectedBlock_0/FC_0/Linear/weight:0',
        'M/FullyConnectedBlock_0/FC_0/Linear/bias/ExponentialMovingAverage':
            'M/FullyConnectedBlock_0/FC_0/Linear/bias:0',
        'M/FullyConnectedBlock_0/FC_0/BatchNorm/gamma/ExponentialMovingAverage':
            'M/FullyConnectedBlock_0/FC_0/BatchNorm/gamma:0',
        'M/FullyConnectedBlock_0/FC_0/BatchNorm/beta/ExponentialMovingAverage':
            'M/FullyConnectedBlock_0/FC_0/BatchNorm/beta:0',
        'M/FullyConnectedBlock_0/FC_0/BatchNorm/moving_mean':
            'M/FullyConnectedBlock_0/FC_0/BatchNorm/moving_mean:0',
        'M/FullyConnectedBlock_0/FC_0/BatchNorm/moving_variance':
            'M/FullyConnectedBlock_0/FC_0/BatchNorm/moving_variance:0',
        'M/FullyConnectedBlock_0/FC_1/Linear/weight/ExponentialMovingAverage':
            'M/FullyConnectedBlock_0/FC_1/Linear/weight:0',
        'M/FullyConnectedBlock_0/FC_1/Linear/bias/ExponentialMovingAverage':
            'M/FullyConnectedBlock_0/FC_1/Linear/bias:0',
        'M/FullyConnectedBlock_0/FC_1/BatchNorm/gamma/ExponentialMovingAverage':
            'M/FullyConnectedBlock_0/FC_1/BatchNorm/gamma:0',
        'M/FullyConnectedBlock_0/FC_1/BatchNorm/beta/ExponentialMovingAverage':
            'M/FullyConnectedBlock_0/FC_1/BatchNorm/beta:0',
        'M/FullyConnectedBlock_0/FC_1/BatchNorm/moving_mean':
            'M/FullyConnectedBlock_0/FC_1/BatchNorm/moving_mean:0',
        'M/FullyConnectedBlock_0/FC_1/BatchNorm/moving_variance':
            'M/FullyConnectedBlock_0/FC_1/BatchNorm/moving_variance:0',
        'M/FullyConnectedBlock_1/FC_0/Linear/weight/ExponentialMovingAverage':
            'M/FullyConnectedBlock_1/FC_0/Linear/weight:0',
        'M/FullyConnectedBlock_1/FC_0/Linear/bias/ExponentialMovingAverage':
            'M/FullyConnectedBlock_1/FC_0/Linear/bias:0',
        'M/FullyConnectedBlock_1/FC_0/BatchNorm/gamma/ExponentialMovingAverage':
            'M/FullyConnectedBlock_1/FC_0/BatchNorm/gamma:0',
        'M/FullyConnectedBlock_1/FC_0/BatchNorm/beta/ExponentialMovingAverage':
            'M/FullyConnectedBlock_1/FC_0/BatchNorm/beta:0',
        'M/FullyConnectedBlock_1/FC_0/BatchNorm/moving_mean':
            'M/FullyConnectedBlock_1/FC_0/BatchNorm/moving_mean:0',
        'M/FullyConnectedBlock_1/FC_0/BatchNorm/moving_variance':
            'M/FullyConnectedBlock_1/FC_0/BatchNorm/moving_variance:0',
        'M/FullyConnectedBlock_1/FC_1/Linear/weight/ExponentialMovingAverage':
            'M/FullyConnectedBlock_1/FC_1/Linear/weight:0',
        'M/FullyConnectedBlock_1/FC_1/Linear/bias/ExponentialMovingAverage':
            'M/FullyConnectedBlock_1/FC_1/Linear/bias:0',
        'M/FullyConnectedBlock_1/FC_1/BatchNorm/gamma/ExponentialMovingAverage':
            'M/FullyConnectedBlock_1/FC_1/BatchNorm/gamma:0',
        'M/FullyConnectedBlock_1/FC_1/BatchNorm/beta/ExponentialMovingAverage':
            'M/FullyConnectedBlock_1/FC_1/BatchNorm/beta:0',
        'M/FullyConnectedBlock_1/FC_1/BatchNorm/moving_mean':
            'M/FullyConnectedBlock_1/FC_1/BatchNorm/moving_mean:0',
        'M/FullyConnectedBlock_1/FC_1/BatchNorm/moving_variance':
            'M/FullyConnectedBlock_1/FC_1/BatchNorm/moving_variance:0',
        'M/OutputLogits/a/weight/ExponentialMovingAverage':
            'M/OutputLogits/a/weight:0',
        'M/OutputLogits/a/bias/ExponentialMovingAverage':
            'M/OutputLogits/a/bias:0',
        'M/OutputLogits/b/weight/ExponentialMovingAverage':
            'M/OutputLogits/b/weight:0',
        'M/OutputLogits/b/bias/ExponentialMovingAverage':
            'M/OutputLogits/b/bias:0',
    }
    self.assertDictEqual(
        {key: var.name for key, var in variables_to_restore.items()},
        expected_variable_to_restore_names)

  def test_get_sigmoid_parameters(self):
    raw_a, a, b = pipeline_utils.get_sigmoid_parameters(
        name='test',
        raw_a_initial_value=1.0,
        b_initial_value=2.0,
        a_range=(-0.5, 1.2),
        b_range=(3.0, 5.0))

    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      raw_a_result, a_result, b_result = sess.run([raw_a, a, b])

    self.assertAlmostEqual(raw_a_result, 1.0)
    self.assertAlmostEqual(a_result, 1.2)
    self.assertAlmostEqual(b_result, 3.0)


if __name__ == '__main__':
  tf.test.main()
