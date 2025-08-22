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

"""Estimate rigid transform between two point clouds."""
# pylint: disable=missing-function-docstring
# pylint: disable=unused-variable
# pylint: disable=missing-class-docstring
# pylint: disable=g-builtin-op
# pylint: disable=invalid-name
# pylint: disable=g-doc-args
# pylint: disable=g-doc-return-or-yield
# pylint: disable=dangerous-default-value
# pylint: disable=unused-import
# pylint: disable=g-multiple-import
import numpy as np
import scipy


def estimate_rigid_transform(points_A, points_B):
  """Estimate Transform.

   points_A : N x 3
   points_B : N x 3
  """
  centroid_A = np.mean(points_A, axis=0)
  centroid_B = np.mean(points_B, axis=0)

  Am = points_A - centroid_A[None, :]
  Bm = points_B - centroid_B[None, :]
  R = estimate_rotation_transform(Am, Bm)
  t = -1 * np.matmul(R, centroid_A) + centroid_B
  RT = np.eye(4)
  RT[:3, :3] = R.T
  RT[:3, 3] = -1 * np.matmul(R.T, centroid_A) + centroid_B

  # pts = np.concatenate([points_A, points_A[:,2:]*0 + 1], axis=1)
  # temp = np.matmul(RT, pts.T) [0:3, :]
  # error1 =  np.abs(np.matmul(Am, R) - Bm).sum()
  # error = np.abs(temp.T - points_B).sum()
  # breakpoint()
  return RT


def estimate_rotation_transform(points_A, points_B):
  """Estimate rotation.

  Assumes the point sets are zero centered.

  points_A : N x 3
  points_B:  N x 3
  """
  R, sca = scipy.linalg.orthogonal_procrustes(
      points_A, points_B, check_finite=True)
  return R
