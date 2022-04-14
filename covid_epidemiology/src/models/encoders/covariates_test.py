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


import numpy as np
from parameterized import parameterized
import tensorflow as tf
from covid_epidemiology.src.models.encoders import covariates

INPUT_COVARIATE_WEIGHTS = np.ones((5, 1), dtype="float32")


class CovariatesTest(tf.test.TestCase):

  @parameterized.expand([
      # Inference Tests, prior to observation window.
      (0, False, 0, 3, 1, False, [1, 0, 0, 0, 0]),  # Test simple
      (3, False, 2, 3, 1, False, [0, 0, 1, 0, 0]),  # Test covariate__offset
      (3, False, 2, 3, 2, False, [0, 0, 1, 1, 0]),  # Test larger active window
      (1, False, 2, 3, 1, False, [0, 0, 0, 0,
                                  0]),  # Test timestep before covariate_offset
      (1, False, 2, 3, 2, False, [0, 0, 0, 0,
                                  0]),  # Test too_large_active_window
      # Inference Tests, after observation window.
      (4, False, 0, 3, 1, False, [0, 0, 1, 0, 0]),  # Test simple
      (4, False, 3, 3, 1, False, [0, 0, 0, 1,
                                  0]),  # Test covariate_feature_time_offset
      (4, False, 3, 3, 2, False, [0, 0, 0, 1, 1]),  # Test larger active window
      (2, False, 3, 3, 1, False, [0, 0, 0, 0,
                                  0]),  # Test covariate_feature_time_offset
      # Training Tests, prior to observation_window.
      (1, True, 0, 3, 1, False, [1, 0, 0, 0, 0]
      ),  # Test simple - should sample window between (0, 1)
      (1, True, 1, 3, 1, False,
       [0, 1, 0, 0, 0]),  # Test covariate_offset - sample between (0, 0)
      (2, True, 1, 3, 2, False,
       [0, 1, 1, 0, 0]),  # Test covariate_offset - sample between (0, 0)
      (1, True, 2, 3, 1, False, [0, 0, 0, 0, 0
                                ]),  # Test covariate_offset - should return -1
      (1, True, 1, 3, 2, False, [0, 0, 0, 0, 0
                                ]),  # Test covariate_offset - too large window
      # Training Tests, post observation window
      (3, True, 0, 3, 1, False, [0, 1, 0, 0,
                                 0]),  # Test simple - sample between (1, 3)
      (3, True, 3, 3, 1, False,
       [0, 0, 0, 1, 0]),  # Test covariate_feature_time_offset -> (3, 3)
      (2, True, 3, 3, 1, False, [0, 0, 0, 0,
                                 0]),  # Test covariate_feature_time_offset
      (3, True, 1, 3, 2, True, [0, 1, 1, 0, 0]),  # Test fixed mask
  ])
  def test_mask_covariate_weights_for_timestep(
      self, timestep, is_training, covariate_feature_time_offset,
      num_known_steps, active_window_size, use_fixed_covariate_mask, expected):

    expected = tf.expand_dims(tf.constant(expected), axis=1)

    actual = covariates.mask_covariate_weights_for_timestep(
        INPUT_COVARIATE_WEIGHTS,
        timestep,
        num_known_steps=num_known_steps,
        is_training=is_training,
        covariate_feature_time_offset=covariate_feature_time_offset,
        active_window_size=active_window_size,
        use_fixed_covariate_mask=use_fixed_covariate_mask,
        seed=None)

    self.assertAllClose(actual, expected)


if __name__ == "__main__":
  tf.test.main()
