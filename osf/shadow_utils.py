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

"""Utility functions for shadows."""
import tensorflow as tf
from osf import ray_utils
from osf import run_nerf_helpers


def create_shadow_ray_batch(ray_batch, pts, scene_info, light_pos):
  """Create batch for shadow rays.

  Args:
    ray_batch: [?, M] tf.float32. Primary ray batch.
    pts: [?, S, 3] tf.float32. Primary points.
    scene_info: Dict. Scene information.
    light_pos:

  Returns:
    shadow_ray_batch: [?S, M]
  """
  # num_rays = tf.shape(pts)[0]  # ?
  num_primary_samples = pts.shape[1]  # S

  pts = tf.reshape(pts, [-1, 3])  # [?S, 3]

  # Prepare light positions.
  light_positions = ray_utils.extract_light_positions_for_rays(
      ray_batch=ray_batch, scene_info=scene_info, light_pos=light_pos)

  # Get ray scene IDs.
  rays_sid = ray_utils.extract_slice_from_ray_batch(
      ray_batch=ray_batch, key='metadata')

  # Repeat ray-level information by the number of primary samples per ray.
  light_positions = tf.tile(
      light_positions[:, None, :],  # [?, S, 3]
      [1, num_primary_samples, 1])
  rays_sid = tf.tile(
      rays_sid[:, None, :],  # [?, S, 1]
      [1, num_primary_samples, 1])
  light_positions = tf.reshape(light_positions, [-1, 3])  # [?S, 3]
  rays_sid = tf.reshape(rays_sid, [-1, 1])  # [?S, 1]

  # Create the ray batch.
  shadow_ray_batch = ray_utils.create_ray_batch(
      rays_o=light_positions, rays_dst=pts, rays_sid=rays_sid)
  return shadow_ray_batch


def compute_shadow_transmittance(intersect, shadow_results):
  """Applies shadows to outputs.

  Args:
    intersect: Dict. Intersect values from primary rays.
    shadow_results: Dict. Shadow results from secondary rays.

  Returns:
    trans: [R, S, 1] tf.float32. Shadow transmittance per primary sample.
  """
  num_rays = tf.shape(intersect['normalized_rgb'])[0]

  shadow_alpha = shadow_results['normalized_alpha']  # [RS, S, 1]
  all_trans = run_nerf_helpers.compute_transmittance(alpha=shadow_alpha[Ellipsis, 0])

  # Transmittance is computed in the direction origin -> end, so we grab the
  # last transmittance.
  trans = all_trans[:, -1]  # [RS,]
  trans = tf.reshape(trans, [num_rays, -1, 1])  # [R, S, 1]
  # intersect['normalized_rgb'] *= trans  # [R, S, 3]
  return trans
