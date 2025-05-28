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

"""Ellipse expected squared l_2 norm comparison functions."""

import numpy as np


def compare_count_norms(d, k):
  """Computes ratio of expected squared l_2 norms, count ellipse / count ball.

  Args:
    d: Integer dimension.
    k: Integer l_0 bound. Requires k <= d/2, as the count ellipse is not valid
    for larger k.

  Returns:
    Ratio (expected squared l_2 norm of sample from count ellipse ) /
    (expected squared l_2 norm of sample from count ball), as computed from
    Lemma 4.7 and Theorem 8.9  in the paper.

  Raises:
    RuntimeError: k > d/2.
  """
  if k > d / 2:
    raise RuntimeError(
        "Input k and d must have k <= d/2, instead got k = "
        + str(k)
        + " > "
        + str(d / 2)
        + " = d / 2."
    )
  # When d = 1, spherical and ellipse noise are the same.
  if d == 1:
    return 1
  v1k_norm = k / np.sqrt(d)
  v2k_norm = np.sqrt(k * (d - k) / d)
  lam = (v1k_norm + v2k_norm * np.sqrt(d - 1)) ** 2
  a1_squared = np.sqrt(lam * v1k_norm**2)
  a2_squared = np.sqrt(lam * v2k_norm**2 / (d - 1))
  ellipse_expected = (a1_squared + (d - 1) * a2_squared) / (d + 2)
  # The minimum l_2 ball containing the count ball has radius sqrt(k)
  ball_expected = k * (d / (d + 2))
  return ellipse_expected / ball_expected


def compare_vote_norms(d):
  """Computes ratio of expected squared l_2 norms, vote ellipse / vote ball.

  Args:
    d: Integer dimension.

  Returns:
    Ratio (expected squared l_2 norm of sample from vote ellipse ) /
    (expected squared l_2 norm of sample from vote ball), as computed from
    Lemma 4.7 and Theorem 9.10 in the paper.
  """
  # When d = 1, spherical and ellipse noise are the same.
  if d == 1:
    return 1
  w1_norm = np.sqrt(d) * (d - 1) / 2
  w2_norm = np.sqrt(d * (d**2 - 1) / 12)
  lam = (w1_norm + w2_norm * np.sqrt(d - 1)) ** 2
  a1_squared = np.sqrt(lam * w1_norm**2)
  a2_squared = np.sqrt(lam * w2_norm**2 / (d - 1))
  ellipse_expected = (a1_squared + (d - 1) * a2_squared) / (d + 2)
  # The minimum l_2 ball containing the vote ball has radius
  # sqrt(sum_i=0^{d-1} i^2) = sqrt((d-1) * d * (2d-1) / 6)
  ball_expected = ((d - 1) * d * (2 * d - 1) / 6) * (d / (d + 2))
  return ellipse_expected / ball_expected
