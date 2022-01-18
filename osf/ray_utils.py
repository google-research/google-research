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

"""Utility functions for ray computation."""
import tensorflow as tf
from osf import scene_utils


def compute_rays_length(rays_d):
  """Compute ray length.

  Args:
    rays_d: [R, 3] tf.float32. Ray directions.

  Returns:
    rays_length: [R, 1] tf.float32. Ray lengths.
  """
  rays_length = tf.linalg.norm(rays_d, axis=-1, keepdims=True)  # [N_rays, 1]
  return rays_length


def normalize_rays(rays):
  """Normalize ray directions.

  Args:
    rays: [R, 3] tf.float32. Ray directions.

  Returns:
    normalized_rays: [R, 3] tf.float32. Normalized ray directions.
  """
  normalized_rays = rays / compute_rays_length(rays_d=rays)
  return normalized_rays


def create_ray_batch(rays_o, rays_dst, rays_sid, ray_length_epsilon=0.0):
  """Create a batch of rays.

  Args:
    rays_o: [R, 3] tf.float32. Ray origins.
    rays_dst: [R, 3] tf.float32. Ray destinations.
    rays_sid: [R, 1] tf.int32. Ray sids.
    ray_length_epsilon: scalar float32. A epislon value to apply to the far
      bounds of rays.

  Returns:
    ray_batch: [R, M] tf.float32. Ray batch.
  """
  num_rays = tf.shape(rays_o)[0]
  # num_rays = rays_o.shape[0]

  # The ray directions are the difference between the ray destinations and the
  # ray origins.
  rays_d = rays_dst - rays_o  # [R, 3]  # Direction out of light source

  # Compute the length of the rays.
  rays_length = compute_rays_length(rays_d=rays_d)

  # Normalize the ray directions.
  rays_d = rays_d / rays_length  # [R, 3]  # Normalize direction

  # Not used atm so we just put rays_d for now.
  rays_viewdirs = rays_d

  # The ray integration times begin at time 0, at the light source.
  # tf.print('[create_ray_batch] num_rays: ', num_rays)
  # tf.print('[create_ray_batch] rays_o: ', tf.shape(rays_o))
  # logging.info('[create_ray_batch] num_rays: %s', num_rays)
  # logging.info('[create_ray_batch] rays_o: %s', rays_o.shape)
  # logging.info('[create_ray_batch] rays_o: %s', tf.shape(rays_o))
  rays_near = tf.zeros((num_rays, 1), dtype=tf.float32)  # [R, 1]

  # The integration times end at the surfaces, which is the length of the rays.
  # We subtract some epsilon amount so that the ray doesn't hit the surface
  # exactly.
  rays_far = rays_length - ray_length_epsilon  # [R, 1]

  # Unused. Placeholder zeros for now.
  rays_data = tf.zeros((num_rays, 1), dtype=tf.float32)  # [R, 1]

  # Combine everything into a ray batch.
  ray_batch = tf.concat(  # [R, N_dims]
      [
          rays_o, rays_d, rays_near, rays_far, rays_viewdirs, rays_data,
          tf.cast(rays_sid, dtype=tf.float32)
      ],
      axis=-1)
  return ray_batch


def extract_slice_from_ray_batch(ray_batch, key, use_viewdirs=True, oid=None):
  """Extract a slice of data from a ray batch.

  Args:
    ray_batch: [R, M] tf.float32.
    key: str.
    use_viewdirs: bool.
    oid: int.

  Returns:
    x: [R, K] tf.float32 or tf.int32.
  """
  if key == 'origin':
    start = 0
    end = 3
  elif key == 'direction':
    start = 3
    end = 6
  elif key == 'far':
    start = 7
    end = 8
  else:
    if key == 'example_id':
      start = 11
      end = 12
    elif key == 'metadata':
      start = 12
      end = 13
    elif key == 'object_bounds':
      start = 13 + 2 * oid
      end = start + 2
    else:
      raise ValueError(f'Invalid key: {key}')
    # Adjust index based on whether we are using viewdirs.
    # TODO(guom): Get rid of use_viewdirs dependency everywhere.
    if not use_viewdirs:
      start -= 3
      end -= 3
  x = ray_batch[:, start:end]

  if key == 'metadata':
    x = tf.cast(x, dtype=tf.int32)
  return x  # [R, ?]


def update_ray_batch_slice(ray_batch, x, start, end):
  # https://stackoverflow.com/questions/37697747/typeerror-tensor-object-does-not-support-item-assignment-in-tensorflow
  left = ray_batch[:, :start]  # [R, ?]
  right = ray_batch[:, end:]  # [R, ?]
  updated_ray_batch = tf.concat([left, x, right], axis=-1)
  return updated_ray_batch


def update_ray_batch_near_and_far(ray_batch, near, far):
  ray_batch = update_ray_batch_slice(
      ray_batch=ray_batch, x=near[:, None], start=6, end=7)
  ray_batch = update_ray_batch_slice(
      ray_batch=ray_batch, x=far[:, None], start=7, end=8)
  return ray_batch


def update_ray_batch_bounds(ray_batch, bounds):
  updated_ray_batch = update_ray_batch_slice(
      ray_batch=ray_batch, x=bounds, start=6, end=8)
  return updated_ray_batch


def compute_scene_indices_for_rays(ray_batch, sid, use_viewdirs):
  del use_viewdirs
  metadata = extract_slice_from_ray_batch(  # [R,]
      ray_batch=ray_batch, key='metadata')
  scene_mask = tf.equal(metadata[:, 0], sid)
  scene_indices = tf.where(scene_mask)[:, 0]  # [R,]
  return scene_indices


def extract_light_positions_for_rays(ray_batch, scene_info, light_pos):
  """Extract light positions for a batch of rays.

  Args:
    ray_batch: [R, M] tf.float32.
    scene_info: Dict.
    light_pos: Light position.

  Returns:
    light_positions: [R, 3] tf.float32.
  """
  ray_sids = extract_slice_from_ray_batch(  # [R,]
      ray_batch=ray_batch, key='metadata')
  light_positions = scene_utils.extract_light_positions_for_sids(
      sids=ray_sids,  # [R, 3]
      scene_info=scene_info,
      light_pos=light_pos)
  return light_positions
