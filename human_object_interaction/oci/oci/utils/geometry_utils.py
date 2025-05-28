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

"""Geometry functions for 3D."""

# pylint: disable=missing-function-docstring
# pylint: disable=unused-variable
# pylint: disable=missing-class-docstring
# pylint: disable=g-builtin-op
# pylint: disable=invalid-name
# pylint: disable=g-doc-args
# pylint: disable=g-doc-return-or-yield
import numpy as np

from oci.utils import transformations


def transform_points(RT, points, return_homo=False):
  """Transform points,   RT : 4 x 4,points : N x 3."""
  points_homo = np.concatenate([points, points[:, 2:] * 0 + 1], axis=1)
  points_homo = points_homo.transpose()
  points_homo = np.matmul(RT, points_homo).transpose()
  if not return_homo:
    points_homo = points_homo[:, 0:3]

  return points_homo


def get_RT_from(euler_angles, translation):
  objCan2WorldRT = transformations.compose_matrix(
      angles=euler_angles, translate=translation)
  return objCan2WorldRT


def get_T_from(translation):
  tMatrix = transformations.translation_matrix(translation)
  return tMatrix


def get_R_from(angles):
  rMatrix = transformations.euler_matrix(angles[0], angles[1], angles[2])
  return rMatrix
