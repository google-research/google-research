# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Utility functions.

- metrics: absolute weight differences
"""

# Necessary functions and packages call
import numpy as np
from sklearn.metrics import mean_absolute_error


def metrics(test_y_hat, test_y_fit, test_coef, test_c, test_idx):
  """performance metrics in terms of fidelity and awd.

  Args:
    test_y_hat: black-box model predictions
    test_y_fit: locally interpretable model predictions
    test_coef: local dynamics discovered by locally interpretable models
    test_c: ground truth
    test_idx: sorted testing sample index based on the distance from boundary

  Returns:
    fidelity: mean absolute error between test_y_hat and test_y_fit
    awd: absolute weight differences
  """

  division = 10

  # Outputs initialization
  fidelity = np.zeros([division, 2])
  awd = np.zeros([division,])

  thresh = (1.0/division)
  test_no = len(test_idx)

  # For each division (distance from the decision boundary)
  for i in range(division):

    # Samples in each division
    temp_idx = test_idx[int(test_no*thresh*i):int(test_no*thresh*(i+1))]

    # Fidelity
    fidelity[i] = mean_absolute_error(test_y_hat[temp_idx],
                                      test_y_fit[temp_idx])

    # awd computation only on the non-zero coefficient
    test_c_nonzero = 1*(test_c[temp_idx, :6] > 0)

    awd_sum = np.sum(np.abs((test_c[temp_idx, :6] * test_c_nonzero) - \
                            (test_coef[temp_idx, 1:7] * test_c_nonzero)))

    awd[i] = awd_sum / np.sum(test_c_nonzero)

  return fidelity, awd
