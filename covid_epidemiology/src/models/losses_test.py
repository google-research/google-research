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

"""Unit tests for losses.

Use python3 -m models.losses_test to run the tests.
"""
import unittest

import tensorflow as tf

from covid_epidemiology.src.models import losses


class MyTestCase(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.pred_states = tf.constant([[1., 2.], [0., 3.], [1., 2.], [2., 3.]],
                                   dtype=tf.float32)
    self.gt_list = tf.constant([[1., 2.], [0., 3.], [0., 0.], [0., 1.]],
                               dtype=tf.float32)
    self.gt_indicator = tf.constant([[1., 1.], [1., 1.], [0., 0.], [1., 1.]],
                                    dtype=tf.float32)

  def test_state_estimation_loss(self):
    output_loss = losses.state_estimation_loss(
        self.pred_states,
        self.gt_list,
        self.gt_indicator,
        begin_timestep=tf.identity(0),
        end_timestep=tf.identity(3))
    self.assertEqual(output_loss, 0)

  def test_interval_loss(self):
    output_loss = losses.interval_loss(
        self.pred_states,
        self.pred_states,
        self.gt_list,
        self.gt_indicator,
        begin_timestep=0,
        end_timestep=3)
    self.assertEqual(output_loss, 0)

    output_loss = losses.interval_loss(
        self.pred_states,
        self.pred_states,
        self.gt_list,
        self.gt_indicator,
        begin_timestep=0,
        end_timestep=3,
        tau=1)
    self.assertEqual(output_loss, 0)

    output_loss = losses.interval_loss(
        self.pred_states,
        self.pred_states,
        self.gt_list,
        self.gt_indicator,
        begin_timestep=0,
        end_timestep=3,
        tau=0.5)
    self.assertEqual(output_loss, 0)

  def test_weighted_interval_loss(self):
    quantiles = tf.convert_to_tensor([0.1, 0.5, 0.9], dtype=tf.float32)
    quantile_pred_states = tf.stack(
        [self.pred_states, self.pred_states, self.pred_states], axis=-1)
    output_loss = losses.weighted_interval_loss(
        quantile_pred_states,
        quantiles,
        self.gt_list,
        self.gt_indicator,
        begin_timestep=tf.identity(0),
        end_timestep=tf.identity(2))
    self.assertEqual(output_loss, 0)

  def test_boundary_loss_term_states(self):
    propagated_states = tf.convert_to_tensor([[[1, 2], [0, 3], [1, 2]]],
                                             dtype=tf.float32)
    lower_bound = 0
    upper_bound = 2

    boundary_loss = losses.boundary_loss_term_states(
        propagated_states,
        lower_bound,
        upper_bound,
        begin_timestep=0,
        end_timestep=2)
    self.assertEqual(boundary_loss, 1 / 6)

  def boundary_loss_term_coefs(self):
    coefficients = tf.convert_to_tensor([2, 3, 1], dtype=tf.float32)
    lower_bound = 0
    upper_bound = 2

    boundary_loss = losses.boundary_loss_term_coefs(coefficients, lower_bound,
                                                    upper_bound)
    self.assertEqual(boundary_loss, 1 / 3)


if __name__ == "__main__":
  unittest.main()
