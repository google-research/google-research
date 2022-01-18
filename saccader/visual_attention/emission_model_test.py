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

"""Unit test for emission model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf

from saccader.visual_attention import dram_config
from saccader.visual_attention import emission_model


class EmissionModelTest(tf.test.TestCase, parameterized.TestCase):

  def test_import(self):
    self.assertIsNotNone(emission_model)

  @parameterized.named_parameters(
      ("test_case_0", False, "learned", False),
      ("test_case_1", False, "center", True),
      ("test_case_2", False, "random", False),
      ("test_case_3", False, "learned", True),
      ("test_case_4", True, "center", False),
      ("test_case_5", True, "random", True),
      ("test_case_6", True, "learned", False),
      ("test_case_7", True, "center", True),
  )
  def test_build(self, use_prev_locations, policy, is_training):
    config = dram_config.get_config()
    state_dim = 64
    batch_size = 10
    location_dims = 2
    state = tf.placeholder(shape=(batch_size, state_dim), dtype=tf.float32)

    model = emission_model.EmissionNetwork(config.emission_model_config)
    if use_prev_locations:
      prev_locations = tf.convert_to_tensor(
          np.random.rand(batch_size, location_dims), dtype=tf.float32)
    else:
      prev_locations = None

    locations, _ = model(state, location_scale=1, prev_locations=prev_locations,
                         policy=policy,
                         is_training=is_training)
    init_op = model.init_op

    with self.test_session() as sess:
      sess.run(init_op)
      self.assertEqual((batch_size, location_dims),
                       sess.run(
                           locations,
                           feed_dict={
                               state: np.random.rand(batch_size, state_dim)
                           }).shape)


if __name__ == "__main__":
  tf.test.main()
