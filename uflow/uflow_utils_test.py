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

"""Tests for uflow_utils."""

from absl.testing import absltest
import numpy as np
import tensorflow as tf

from uflow import uflow_utils


class UflowUtilsTest(absltest.TestCase):

  def test_fb_consistency_no_occlusion(self):
    batch_size = 4
    height = 64
    width = 64
    # flows points right and up by 4
    flow_01 = np.ones((batch_size, height, width, 2)) * 4.
    # flow points left and down by 4
    perfect_flow_10 = -flow_01
    flow_01 = tf.convert_to_tensor(value=flow_01.astype(np.float32))
    flow_01_level1 = tf.image.resize(flow_01, (height // 2, width // 2)) / 2.
    perfect_flow_10 = tf.convert_to_tensor(
        value=perfect_flow_10.astype(np.float32))
    perfect_flow_10_level1 = -flow_01_level1
    flows = {}
    flows[(0, 1, 0)] = [flow_01, flow_01_level1]
    flows[(1, 0, 0)] = [perfect_flow_10, perfect_flow_10_level1]
    _, _, _, not_occluded_masks, _, _ = \
        uflow_utils.compute_warps_and_occlusion(
            flows, occlusion_estimation='brox')
    # assert that nothing is occluded
    is_ones_01 = np.equal(
        np.ones((batch_size, height - 8, width - 8, 1)),
        not_occluded_masks[(0, 1, 0)][0][:, 4:-4, 4:-4, :]).all()
    is_ones_10 = np.equal(
        np.ones((batch_size, height - 8, width - 8, 1)),
        not_occluded_masks[(1, 0, 0)][0][:, 4:-4, 4:-4, :]).all()
    self.assertTrue(is_ones_01)
    self.assertTrue(is_ones_10)

  def test_fb_consistency_with_occlusion(self):
    batch_size = 4
    height = 64
    width = 64
    # flows points right and up by 4
    flow_01 = np.ones((batch_size, height, width, 2)) * 4.
    # flow points left and down by 2
    imperfect_flow_10 = -flow_01 * .5
    flow_01 = tf.convert_to_tensor(value=flow_01.astype(np.float32))
    flow_01_level1 = tf.image.resize(flow_01, (height // 2, width // 2)) / 2.
    imperfect_flow_10 = tf.convert_to_tensor(
        value=imperfect_flow_10.astype(np.float32))
    imperfect_flow_10_level1 = -flow_01_level1 * .5
    flows = {}
    flows[(0, 1, 0)] = [flow_01, flow_01_level1]
    flows[(1, 0, 0)] = [imperfect_flow_10, imperfect_flow_10_level1]
    _, _, _, not_occluded_masks, _, _ = \
        uflow_utils.compute_warps_and_occlusion(
            flows, occlusion_estimation='brox')
    # assert that everything is occluded
    is_zeros_01 = np.equal(
        np.zeros((batch_size, height - 8, width - 8, 1)),
        not_occluded_masks[(0, 1, 0)][0][:, 4:-4, 4:-4, :]).all()
    is_zeros_10 = np.equal(
        np.zeros((batch_size, height - 8, width - 8, 1)),
        not_occluded_masks[(1, 0, 0)][0][:, 4:-4, 4:-4, :]).all()
    self.assertTrue(is_zeros_01)
    self.assertTrue(is_zeros_10)

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
    flow_resized, mask_resized = uflow_utils.resize(
        flow, 2, 2, is_flow=True, mask=mask)
    flow_okay = tf.reduce_all(tf.math.equal(flow_resized, flow_result)).numpy()
    mask_okay = tf.reduce_all(tf.math.equal(mask_resized, mask_result)).numpy()
    self.assertTrue(flow_okay)
    self.assertTrue(mask_okay)


if __name__ == '__main__':
  absltest.main()
