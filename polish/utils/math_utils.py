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

"""Helper math functions for understanding data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np


def explained_variance(y_pred, y):
  """Computes fraction of variance that ypred explains about y.

  https://github.com/openai/baselines/blob/master/baselines/common/math_util.py

  Both `y_pred` and `y` must have an array dimension of one.

  Args:
    y_pred: predicted tensor.
    y: label tensor.

  Returns:
    1 - var[y - y_pred] / var[y] or np.nan if var[y] == 0
    interpretation:
      ev=0  =>  might as well have predicted zero.
      ev=1  =>  perfect prediction.
      ev<0  =>  worse than just predicting zero.
  """
  if y.ndim != 1 or y_pred.ndim != 1:
    raise ValueError('The number of array dimensions for both `y` and `y_pred`'
                     ' must be one.')
  var_y = np.var(y)
  return np.nan if var_y == 0 else 1 - np.var(y - y_pred) / var_y
