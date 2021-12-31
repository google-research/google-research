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

"""Tests for the tiny model used for the SMURF multi-frame self-supervision."""
from typing import Tuple

from absl.testing import absltest
import numpy as np
import tensorflow as tf

from smurf.multiframe_training import tiny_model

_IMAGE_HEIGHT = 10
_IMAGE_WIDTH = 20


class TinyModelTest(tf.test.TestCase):

  def _create_flows_and_masks(
      self):
    # Creates constant flow fields with the following mapping:
    # (-2, -2) -> (1, 1).
    flow_forward = tf.ones((1, _IMAGE_HEIGHT, _IMAGE_WIDTH, 2),
                           dtype=tf.float32)
    flow_backward = tf.ones(
        (1, _IMAGE_HEIGHT, _IMAGE_WIDTH, 2), dtype=tf.float32) * -2.0
    mask_forward = tf.ones((1, _IMAGE_HEIGHT, _IMAGE_WIDTH, 1),
                           dtype=tf.float32)
    mask_backward = tf.ones((1, _IMAGE_HEIGHT, _IMAGE_WIDTH, 1),
                            dtype=tf.float32)
    return flow_forward, flow_backward, mask_forward, mask_backward

  def _create_mask_with_some_invalid_location(self):
    mask = np.ones((1, _IMAGE_HEIGHT, _IMAGE_WIDTH, 1))
    mask[0, 3:5, 3:5, 0] = 0
    return tf.convert_to_tensor(mask, dtype=tf.float32)

  def test_train_and_run_tiny_model_all_valid_forward(self):
    (flow_forward, flow_backward, mask_forward,
     mask_backward) = self._create_flows_and_masks()

    fused_flow, fused_mask = tiny_model.train_and_run_tiny_model(
        flow_forward, flow_backward, mask_forward, mask_backward)

    # Regardless of the trained model the result should be exactly the forward
    # flow because no masked regions are filled in.
    self.assertAllEqual(fused_flow, flow_forward)
    self.assertAllEqual(fused_mask, tf.ones_like(fused_mask))

  def test_train_and_run_tiny_model_some_valid_forward_with_valid_backward(
      self):
    (flow_forward, flow_backward, _,
     mask_backward) = self._create_flows_and_masks()
    mask_forward = self._create_mask_with_some_invalid_location()

    fused_flow, fused_mask = tiny_model.train_and_run_tiny_model(
        flow_forward,
        flow_backward,
        mask_forward,
        mask_backward,
        iterations=10000)

    # There is a small patch with no valid forward flow, but valid backward
    # flow. This should allow the model to fill in the missing information.
    self.assertAllClose(fused_flow, tf.ones_like(flow_forward), atol=1e-3)
    self.assertAllEqual(fused_mask, tf.ones_like(mask_forward))

  def test_train_and_run_tiny_model_some_valid_forward_without_valid_backward(
      self):
    flow_forward, flow_backward, _, _ = self._create_flows_and_masks()
    mask_forward = self._create_mask_with_some_invalid_location()
    mask_backward = mask_forward

    fused_flow, fused_mask = tiny_model.train_and_run_tiny_model(
        flow_forward, flow_backward, mask_forward, mask_backward)

    # There is a small patch with neither valid forward nor backward flow.
    self.assertAllEqual(fused_flow, flow_forward * mask_forward)
    self.assertAllEqual(fused_mask, mask_forward)


if __name__ == '__main__':
  absltest.main()
