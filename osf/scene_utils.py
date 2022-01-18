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

"""Utility functions for scene computation."""
import numpy as np
import tensorflow as tf
from osf import geo_utils


OID2NAME = {
    0: 'couch',
    1: 'chair',
    2: 'table',
}
NAME2OID = {v: k for k, v in OID2NAME.items()}
BKGD_ID = len(OID2NAME)  # Background is the last index.


OBJECT2SHIFT_Z_CENTER = [
    'antique_couch',
    'armchair',
    'cash_register',
    'commode',
    'drill',
    'ottoman',
    'rocking_chair',
    'side_table',
    'sign',
    'ukulele',
    'wooden_table',
]


def extract_box_for_scene_object(scene_info, sid, name, padding=0.0,
                                 swap_yz=False, box_delta_t=None,
                                 convert_to_tf=False):
  """Extracts the bounding box for a scene's object.

  Note: If the object does not exist in the scene, we return a dummy bounding
  box such that rays will never intersect with this box.

  Args:
    scene_info: Dict.
    sid: int.
    name: str.
    padding: float.
    swap_yz: bool.
    box_delta_t: List of floats.
    convert_to_tf: bool. Whether to convert the resulting box tensors into tf
      tensors.

  Returns:
    box_dims: [3,] float32, either numpy or tf depending on `convert_to_tf`.
    box_center: [3,] float32, either numpy or tf depending on `convert_to_tf`.
    box_rotation: [3, 3] float32, either numpy or tf depending on
      `convert_to_tf`.
  """
  if name in scene_info[sid]['objects']:
    object_info = scene_info[sid]['objects'][name]
  else:
    # Create a dummy box such that rays will never intersect it.
    object_info = {
        'R': np.eye(3),
        'T': [np.inf, np.inf, np.inf],
        'dims': [0.0, 0.0, 0.0],
        'scale': [0.0, 0.0, 0.0]
    }
  box_center = np.array(object_info['T'], dtype=np.float32)
  box_rotation = np.array(object_info['R'], dtype=np.float32)
  box_center = np.copy(box_center)

  # Temporarily add a z translation.
  if box_delta_t is not None:
    box_center = box_center + np.array(box_delta_t, dtype=np.float32)

  box_dims = np.array(object_info['dims']) * np.array(object_info['scale'])

  # Swap y and z if requested.
  if swap_yz:
    y, z = box_dims[1:]
    box_dims[1] = z
    box_dims[2] = y

  # The z transformation may be referring to the bottom of the object instead of
  # the center. If that is the case, we add z_dim / 2 to the z transformation.
  if name in OBJECT2SHIFT_Z_CENTER:
    box_center[2] += box_dims[2] / 2

  # Apply padding to the box.
  box_dims += padding

  if convert_to_tf:
    box_dims = tf.constant(box_dims, dtype=tf.float32)
    box_center = tf.constant(box_center, dtype=tf.float32)
    box_rotation = tf.constant(box_rotation, dtype=tf.float32)
  return box_dims, box_center, box_rotation


def extract_boxes_for_all_scenes(scene_info, name, padding, swap_yz,
                                 box_delta_t):
  """Extracts the bounding box for all scenes.

  Note: If the object does not exist in the scene, we return a dummy bounding
  box such that rays will never intersect with this box.

  Args:
    scene_info: Dict.
    name: str.
    padding: float.
    swap_yz: bool.
    box_delta_t: List of floats.

  Returns:
    all_box_dims: [N, 3,] tf.float32.
    all_box_center: [N, 3,] tf.float32.
    all_box_rotation: [N, 3, 3] tf.float32.

    where N is the number of scenes.
  """
  box_dims_list = []
  box_center_list = []
  box_rotation_list = []
  for sid in scene_info:
    box_dims, box_center, box_rotation = extract_box_for_scene_object(
        scene_info=scene_info, sid=sid, name=name, padding=padding,
        swap_yz=swap_yz, box_delta_t=box_delta_t, convert_to_tf=True)
    box_dims_list.append(box_dims)  # List of [3,]
    box_center_list.append(box_center)  # List of [3,]
    box_rotation_list.append(box_rotation)  # List of [3, 3]
  all_box_dims = tf.stack(box_dims_list, axis=0)  # [N, 3]
  all_box_center = tf.stack(box_center_list, axis=0)  # [N, 3]
  all_box_rotation = tf.stack(box_rotation_list, axis=0)  # [N, 3, 3]
  return all_box_dims, all_box_center, all_box_rotation


def extract_object_boxes_for_scenes(name, scene_info, sids, padding, swap_yz,
                                    box_delta_t):
  """Extracts object boxes given scene IDs.

  Args:
    name: The object name.
    scene_info: The scene information.
    sids: [R, 1] tf.int32. Scene IDs.
    padding: float32. The amount of padding to apply in all dimensions.
    swap_yz: bool. Whether to swap y and z box dimensions.
    box_delta_t: List of floats.

  Returns:
    sid_box_dims: [R, 3] tf.float32.
    sid_box_center: [R, 3] tf.float32.
    sid_box_rotation: [R, 3] tf.float32.
  """
  all_box_dims, all_box_center, all_box_rotation = extract_boxes_for_all_scenes(
      scene_info=scene_info, name=name, padding=padding, swap_yz=swap_yz,
      box_delta_t=box_delta_t)

  # Gather the corresponding boxes for the provided sids.
  sid_box_dims = tf.gather_nd(  # [R, 3]
      params=all_box_dims,  # [R, 3]
      indices=sids,  # [R, 1]
  )
  sid_box_center = tf.gather_nd(  # [R, 3]
      params=all_box_center,  # [R, 3]
      indices=sids,  # [R, 1]
  )
  sid_box_rotation = tf.gather_nd(  # [R, 3, 3]
      params=all_box_rotation,  # [R, 3, 3]
      indices=sids,  # [R, 1]
  )
  return sid_box_dims, sid_box_center, sid_box_rotation


def extract_w2o_transformations_per_scene(name, scene_info, box_delta_t):
  """Extract world-to-object transformations for each scene.

  Args:
    name: str. Object name.
    scene_info: dict.
    box_delta_t: List of floats.

  Returns:
    w2o_rt_per_scene: [N_scenes, 4, 4] tf.float32.
    w2o_r_per_scene: [N_scenes, 4, 4] tf.float32.
  """
  w2o_rt_per_scene = []
  w2o_r_per_scene = []
  for sid, info in scene_info.items():
    if name not in info['objects']:
      # The object does not exist in the scene. We will not end up selecting
      # this scene in the parent function, `create_w2o_transformations_tensors`
      # anyway.
      w2o_rt = geo_utils.construct_rt(r=None, t=None)
      w2o_r = geo_utils.construct_rt(r=None, t=None)
    else:
      _, box_center, box_rotation = extract_box_for_scene_object(
          scene_info=scene_info, sid=sid, name=name, box_delta_t=box_delta_t)
      w2o_rt = geo_utils.construct_rt(r=box_rotation, t=box_center,
                                      inverse=True)
      w2o_r = geo_utils.construct_rt(r=box_rotation, t=None, inverse=True)
    w2o_rt_per_scene.append(w2o_rt)
    w2o_r_per_scene.append(w2o_r)
  w2o_rt_per_scene = tf.constant(
      np.array(w2o_rt_per_scene), dtype=tf.float32)  # [N_scenes, 4, 4]
  w2o_r_per_scene = tf.constant(
      np.array(w2o_r_per_scene), dtype=tf.float32)  # [N_scenes, 4, 4]
  return w2o_rt_per_scene, w2o_r_per_scene


def extract_light_positions_for_all_scenes(scene_info, light_pos=None):
  """Extracts light positions for all scenes.

  Args:
    scene_info: Dict.
    light_pos: Hardcoded light pos to override with.

  Returns:
    light_positions: [N_scenes, 3] tf.float32.
  """
  light_positions = []
  for sid in scene_info:
    if light_pos is None:
      light_positions.append(scene_info[sid]['light_pos'])
    else:
      light_positions.append(light_pos)
  light_positions = tf.constant(light_positions, dtype=tf.float32)  # [N, 3]
  return light_positions


def extract_light_positions_for_sids(sids, scene_info, light_pos):
  """Extracts light positions given scene IDs.

  Args:
    sids: [N, 1] tf.int32.
    scene_info: Dict.
    light_pos: Light position.

  Returns:
    light_positions: [N, 3] tf.float32.
  """
  all_light_positions = extract_light_positions_for_all_scenes(
      scene_info=scene_info, light_pos=light_pos)  # [S, 3]
  light_positions = tf.gather_nd(  # [N, 3]
      params=all_light_positions,  # [S, 3]
      indices=sids,  # [N, 1]
  )
  return light_positions
