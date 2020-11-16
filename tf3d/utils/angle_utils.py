# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Utility functions for angles."""

import math
import tensorflow as tf


def absolute_angular_distance(radians_a, radians_b):
  """Absolute angular distance between two angle tensors.

  This function computes elementwise smallest L1 distance between angle
  tensors.

  Args:
    radians_a: A tf.Tensor containing angles as radians.
    radians_b: A tf.Tensor containing angles as radians.

  Returns:
    A tf.Tensor containing angular distances in [0, pi].
  """
  return tf.math.abs(signed_angular_distance(radians_a, radians_b))


def signed_angular_distance(radians_a, radians_b):
  """Signed angular distance between two angle tensors.

  This function computes elementwise smallest signed distance between angle
  tensors.

  Args:
    radians_a: A tf.Tensor containing angles as radians.
    radians_b: A tf.Tensor containing angles as radians.

  Returns:
    A tf.Tensor containing angular distances in [-pi, pi).
  """
  return tf.math.floormod(radians_a - radians_b + math.pi,
                          2 * math.pi) - math.pi


def wrap_to_pi(radians):
  """Wraps an angle tensor (in radians) to [-pi, pi).

  Args:
    radians: A tf.Tensor containing angles as radians.

  Returns:
    A tf.Tensor containing angles wrapped to [-pi, pi).
  """
  # wrap to [0..2*pi)
  wrapped = tf.math.floormod(radians, 2 * math.pi)
  # wrap to [-pi..pi)
  return tf.where(wrapped >= math.pi, wrapped - 2 * math.pi, wrapped)


def degrees_to_radians(degrees):
  """Converts angles from degrees to radians.

  Args:
    degrees: A tf.Tensor containing angles in degrees.

  Returns:
    A tf.Tensor containing angles in radians.
  """
  return degrees * math.pi / 180.0


def radians_to_degrees(radians):
  """Converts angles from radians to degrees.

  Args:
    radians: A tf.Tensor containing angles in radians.

  Returns:
    A tf.Tensor containing angles in degrees.
  """
  return radians * 180.0 / math.pi
