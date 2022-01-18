# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Utility functions for light."""
import tensorflow as tf
from osf import run_nerf_helpers


def compute_indirect_radiance(intersect, results, light_rgb, white_bkgd):
  """Computes per-primary sample radiance from per-object secondary rays.

  Args:
    intersect:
    results: [RS, SO, K]
    light_rgb:
    white_bkgd:

  Returns:
    indirect_radiance: [R, S, 3] tf.float32. Radiance along each indirect ray.
  """
  num_rays = tf.shape(intersect['normalized_rgb'])[0]

  # Compose the combined object outputs into the final rendered result.
  outputs = run_nerf_helpers.compose_outputs(  # [RS, K]
      results=results,  # [RS, SO, K]
      light_rgb=light_rgb,
      white_bkgd=white_bkgd)

  # Reshape the rgb values.
  indirect_radiance = outputs['rgb_map']  # [RS, 3]
  indirect_radiance = tf.reshape(  # [R, S, 3]
      indirect_radiance, [num_rays, -1, 3])
  return indirect_radiance
