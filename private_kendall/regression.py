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

"""Linear regression (without differential privacy).

This code is adapted from
https://github.com/google-research/google-research/blob/master/dp_regression/baselines.py.
"""


import numpy as np
import sklearn.linear_model


def nondp(use_lasso, features, labels, max_iter=1e3):
  """Returns model computed using non-DP linear regression.

  Args:
    use_lasso: Whether or not to use Lasso regression.
    features: Matrix of feature vectors where each row is an example. Assumed to
      have intercept feature in the last column.
    labels: Vector of labels.
    max_iter: Maximum number of iterations to use for Lasso.

  Returns:
    Vector of regression coefficients.
  """
  _, d = features.shape
  if use_lasso:
    model = sklearn.linear_model.Lasso(
        fit_intercept=False, max_iter=max_iter
    ).fit(features, labels)
  else:
    model = sklearn.linear_model.LinearRegression(fit_intercept=False).fit(
        features, labels
    )
  model_v = np.zeros((d, 1))
  model_v[:, 0] = model.coef_
  return model_v
