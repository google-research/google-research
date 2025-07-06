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

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from cisc.src.post_processing import calibration_util


class CalibrationUtilTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="prefect_calibration",
          confidence=[0, 0, 0, 1, 1, 1],
          is_correct=[False, False, False, True, True, True],
          expected_ece=0,
      ),
      dict(
          testcase_name="almost_perfect_calibration",
          confidence=[0.5] + [1] * 1000,
          is_correct=[False] + [True] * 1000,
          expected_ece=0,
      ),
      dict(
          testcase_name="half",
          confidence=[0.5] * 4,
          is_correct=[True] * 4,
          expected_ece=0.5,
      ),
  )
  def test_compute_ece_prefect_calibration(
      self, confidence, is_correct, expected_ece
  ):
    ece, _ = calibration_util.compute_ece(is_correct, confidence)
    self.assertAlmostEqual(ece, expected_ece, places=2)

  @parameterized.named_parameters(
      dict(
          testcase_name="basic",
          confidence=[0, 1, 1, 0],
          is_correct=[0.1, 0.9, 0.8, 0.3],
          expected_brier=0.037,
      ),
      dict(
          testcase_name="prefect_calibration",
          confidence=[0, 0, 0, 1, 1, 1],
          is_correct=[False, False, False, True, True, True],
          expected_brier=0,
      ),
      dict(
          testcase_name="almost_perfect_calibration",
          confidence=[0.5] + [1] * 1000,
          is_correct=[False] + [True] * 1000,
          expected_brier=0,
      ),
      dict(
          testcase_name="half",
          confidence=[0.5] * 4,
          is_correct=[True] * 4,
          expected_brier=0.25,
      ),
  )
  def test_compute_brier(self, confidence, is_correct, expected_brier):
    brier, _ = calibration_util.compute_brier(is_correct, confidence)
    self.assertAlmostEqual(brier, expected_brier, places=2)

  def test_temperature_scaling(self):
    # Always conffident that the answer is correct, but only correct half the
    # time. Calibration should calibrate all confidences to 0.5.
    confidences = np.array([0.9, 0.9, 0.9, 0.9])
    labels = np.array([0, 0, 1, 1])

    scalar = calibration_util.TemperatureScaling()
    scalar.fit(confidences, labels)
    self.assertAlmostEqual(scalar.temperature, 0.5 / 0.9, places=2)
    self.assertSequenceAlmostEqual(
        scalar.predict(confidences), [0.5, 0.5, 0.5, 0.5], places=2
    )


if __name__ == "__main__":
  absltest.main()
