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

"""Tests for model_utils."""

import numpy as np
import tensorflow as tf
from multiple_user_representations.models import model_utils


class ModelUtilsTest(tf.test.TestCase):

  def test_positional_encoding(self):
    position_embedding = model_utils.positional_encoding(
        max_seq_size=4, embedding_dim=2).numpy()

    expected_output = np.array([[0., 1.],
                                [0.84147096, 0.5403023],
                                [0.9092974, -0.41614684],
                                [0.14112, -0.9899925]])

    self.assertSequenceEqual(position_embedding.shape, [1, 4, 2])
    np.testing.assert_array_almost_equal(position_embedding[0],
                                         expected_output)


if __name__ == '__main__':
  tf.test.main()
