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

"""Tests for smurf_utils."""

# pylint:skip-file

from absl.testing import absltest
import numpy as np
import tensorflow as tf

from smurf import smurf_utils


class SMURFUtilsTest(absltest.TestCase):

  def _assertAllEqual(self, t1, t2):
    all_equal = (t1.numpy() == t2.numpy()).all()
    self.assertTrue(all_equal)

  def test_fb_consistency_no_occlusion(self):
    # flows points right and up by 4
    flow_01 = np.ones((4, 64, 64, 2), np.float32) * 4.
    # flow points left and down by 4
    flow_10 = -flow_01
    flow_01 = tf.convert_to_tensor(value=flow_01)
    flow_10 = tf.convert_to_tensor(value=flow_10)
    occlusion_mask = smurf_utils.compute_occlusions_brox(flow_01,
                                                         flow_10).numpy()
    is_zeros = np.equal(occlusion_mask[:, 4:-4, 4:-4, :],
                        np.zeros_like(occlusion_mask[:, 4:-4, 4:-4, :])).all()
    is_ones1 = np.equal(occlusion_mask[:, -4:, :, :],
                        np.ones_like(occlusion_mask[:, -4:, :, :])).all()
    is_ones2 = np.equal(occlusion_mask[:, :, -4:, :],
                        np.ones_like(occlusion_mask[:, :, -4:, :])).all()
    self.assertTrue(is_ones1)
    self.assertTrue(is_ones2)
    self.assertTrue(is_zeros)

  def test_fb_consistency_no_occlusion_no_borders(self):
    # Test that fb_consistency with boundaries_are_occluded=False
    # marks everything unoccluded when flows are fb-consistent.

    # Flows points right and up by 4.
    flow_01 = tf.ones((4, 64, 64, 2), tf.float32) * 4.
    # Flow points left and down by 4.
    flow_10 = -flow_01
    occlusion_mask = smurf_utils.compute_occlusions(
        flow_01,
        flow_10,
        occlusion_estimation='brox',
        boundaries_occluded=False,
        occlusions_are_zeros=False)
    self._assertAllEqual(occlusion_mask, tf.zeros_like(occlusion_mask))

  def test_fb_consistency_occlusion_no_borders(self):
    # Test that fb_consistency with boundaries_are_occluded=False marks
    # everything except boundaries as occluded when flows are not fb-consistent.

    # Flows points right and up by 4
    flow_01 = tf.ones((4, 64, 64, 2), tf.float32) * 4.
    # Flows points left and down by 2
    flow_10 = -flow_01 * .5
    occlusion_mask = smurf_utils.compute_occlusions(
        flow_01,
        flow_10,
        occlusion_estimation='brox',
        boundaries_occluded=False,
        occlusions_are_zeros=False)
    self._assertAllEqual(occlusion_mask[:, 4:-4, 4:-4, :],
                         tf.ones_like(occlusion_mask[:, 4:-4, 4:-4, :]))
    self._assertAllEqual(occlusion_mask[:, -4:, :, :],
                         tf.zeros_like(occlusion_mask[:, -4:, :, :]))
    self._assertAllEqual(occlusion_mask[:, :, -4:, :],
                         tf.zeros_like(occlusion_mask[:, :, -4:, :]))

  def test_wang_no_occlusion_no_borders(self):
    # Test that wang with boundaries_are_occluded=False marks everything
    # unoccluded when backward flow is perfectly smooth.

    # Flows points right and up by 4.
    flow_01 = tf.ones((4, 64, 64, 2), tf.float32) * 4.
    # Flow points left and down by 2
    flow_10 = -flow_01 * .5
    occlusion_mask = smurf_utils.compute_occlusions(
        flow_01,
        flow_10,
        occlusion_estimation='wang',
        boundaries_occluded=False,
        occlusions_are_zeros=False)
    # NOTE(austinstone): Wang marks things as unoccluded if there is a backward
    # flow vector pointing to it, even if the forward / backward flow is
    # highly inconsistent. In this case, everything is marked as unoccluded
    # because every pixel either has a backward flow pixel pointing to it
    # or the forward flow vector points off the edge of the image.
    self._assertAllEqual(occlusion_mask, tf.zeros_like(occlusion_mask))

  def test_wang_no_occlusion(self):
    # Test that wang with boundaries_are_occluded=True marks everything
    # unoccluded except for parts where flow vectors point off the edges
    # when the backward flow is perfectly smooth.

    # flows points right and up by 4
    flow_01 = tf.ones((4, 64, 64, 2), tf.float32) * 4.
    # flow points left and down by 2
    flow_10 = -flow_01 * .5
    occlusion_mask = smurf_utils.compute_occlusions(
        flow_01,
        flow_10,
        occlusion_estimation='wang',
        boundaries_occluded=True,
        occlusions_are_zeros=False)
    self._assertAllEqual(occlusion_mask[:, :, -2:, :],
                         tf.ones_like(occlusion_mask[:, :, -2:, :]))
    self._assertAllEqual(occlusion_mask[:, -2:, :, :],
                         tf.ones_like(occlusion_mask[:, -2:, :, :]))
    self._assertAllEqual(occlusion_mask[:, :-2, :-2, :],
                         tf.zeros_like(occlusion_mask[:, :-2, :-2, :]))

  def test_fb_consistency_with_occlusion(self):
    # Flows points right and up by 5.
    flow_01 = np.ones((4, 64, 64, 2), np.float32) * 5
    # Flow points left and down by 2.5.
    flow_10 = -flow_01 * .5
    flow_01 = tf.convert_to_tensor(value=flow_01)
    flow_10 = tf.convert_to_tensor(value=flow_01)
    occlusion_mask = smurf_utils.compute_occlusions_brox(flow_01,
                                                         flow_10).numpy()
    is_ones = np.equal(occlusion_mask, np.ones_like(occlusion_mask)).all()
    self.assertTrue(is_ones)

  def test_resize_sparse_flow(self):
    flow = tf.constant(
        [[[1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
         [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
         [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
         [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
         [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
         [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
         [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
         [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]],
        dtype=tf.float32)
    mask = tf.constant([[[1], [0], [0], [0], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [0], [0], [0], [0]]],
                       dtype=tf.float32)
    flow_result = tf.constant([[[0.25, 0], [0, 0]], [[0, 0], [0, 0]]],
                              dtype=tf.float32)
    mask_result = tf.constant([[[1], [0]], [[0], [0]]], dtype=tf.float32)
    flow_resized, mask_resized = smurf_utils.resize(
        flow, 2, 2, is_flow=True, mask=mask)
    flow_okay = tf.reduce_all(tf.math.equal(flow_resized, flow_result)).numpy()
    mask_okay = tf.reduce_all(tf.math.equal(mask_resized, mask_result)).numpy()
    self.assertTrue(flow_okay)
    self.assertTrue(mask_okay)

  def test_resampler_flat_gather(self):
    data = [[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            [[[10.0, 20.0], [30.0, 40.0]], [[50.0, 60.0], [70.0, 80.0]]]]
    warp = [[[[1.0, 1.5], [1.0, 0.0]], [[0.4, 1.0], [1.0, 0.0]]],
            [[[10.0, 20.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]]]
    expected = [[[[3.5, 4.], [3., 4.]], [[5.8, 6.8], [3., 4.]]],
                [[[0., 0.], [70., 80.]], [[10., 20.], [10., 20.]]]]
    data_tensor = tf.convert_to_tensor(data)
    warp_tensor = tf.convert_to_tensor(warp)
    expected = tf.convert_to_tensor(expected)
    outputs = smurf_utils.resampler_flat_gather(data_tensor, warp_tensor)
    self.assertAlmostEqual(np.max(np.abs(expected - outputs)), 0., places=5)


if __name__ == '__main__':
  absltest.main()
