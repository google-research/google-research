# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Tests for half_sample."""

import unittest

import numpy as np
from sklearn import linear_model

from al_for_fep.models import half_sample


class HalfSampleTest(unittest.TestCase):

  def test_fit_and_predict_success(self):
    subestimator = linear_model.LinearRegression()
    learner = half_sample.HalfSampleRegressor(
        subestimator=subestimator, shards_log2=2, add_estimators=False)

    np.random.seed(12345)
    train_x = np.random.rand(10, 5)
    train_y = np.random.rand(10)

    learner.fit(train_x, train_y)

    predict_features = np.random.rand(5, 5)
    predictions = learner.predict(predict_features)

    np.testing.assert_array_almost_equal(
        predictions,
        np.array([[
            0.63102049, 0.63513979, 0.47315767, 1.29661048, 0.99107012,
            0.57297124
        ],
                  [
                      0.62262942, 0.41642359, 0.83925112, -0.01737988,
                      0.2764222, 0.74133841
                  ],
                  [
                      0.55203966, 0.51499607, 0.77481952, 0.14148882,
                      0.65506182, 0.64192521
                  ],
                  [
                      1.42425812, 0.07270757, 0.25163252, 1.03244175,
                      0.74677908, 0.76352765
                  ],
                  [
                      0.36649924, 0.60866013, 0.82320282, 0.08926722,
                      0.70117256, 0.58206726
                  ]]).T,
    )


if __name__ == '__main__':
  unittest.main()
