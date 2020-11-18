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

"""Camera configs."""

import numpy as np
import pybullet as p


class RealSenseD415():
  """Default configuration with 3 RealSense RGB-D cameras."""

  # Mimic RealSense D415 RGB-D camera parameters.
  image_size = (480, 640)
  intrinsics = (450., 0, 320., 0, 450., 240., 0, 0, 1)

  # Set default camera poses.
  front_position = (1., 0, 0.75)
  front_rotation = (np.pi / 4, np.pi, -np.pi / 2)
  front_rotation = p.getQuaternionFromEuler(front_rotation)
  left_position = (0, 0.5, 0.75)
  left_rotation = (np.pi / 4.5, np.pi, np.pi / 4)
  left_rotation = p.getQuaternionFromEuler(left_rotation)
  right_position = (0, -0.5, 0.75)
  right_rotation = (np.pi / 4.5, np.pi, 3 * np.pi / 4)
  right_rotation = p.getQuaternionFromEuler(right_rotation)

  # Default camera configs.
  CONFIG = [{
      'image_size': image_size,
      'intrinsics': intrinsics,
      'position': front_position,
      'rotation': front_rotation,
      'zrange': (0.01, 10.),
      'noise': False
  }, {
      'image_size': image_size,
      'intrinsics': intrinsics,
      'position': left_position,
      'rotation': left_rotation,
      'zrange': (0.01, 10.),
      'noise': False
  }, {
      'image_size': image_size,
      'intrinsics': intrinsics,
      'position': right_position,
      'rotation': right_rotation,
      'zrange': (0.01, 10.),
      'noise': False
  }]


class Oracle():
  """Top-down noiseless image used only by the oracle demonstrator."""

  # Near-orthographic projection.
  image_size = (480, 640)
  intrinsics = (63e4, 0, 320., 0, 63e4, 240., 0, 0, 1)
  position = (0.5, 0, 1000.)
  rotation = p.getQuaternionFromEuler((0, np.pi, -np.pi / 2))

  # Camera config.
  CONFIG = [{
      'image_size': image_size,
      'intrinsics': intrinsics,
      'position': position,
      'rotation': rotation,
      'zrange': (999.7, 1001.),
      'noise': False
  }]
