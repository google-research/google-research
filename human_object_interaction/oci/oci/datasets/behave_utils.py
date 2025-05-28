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

"""Behave data utils."""
import os.path as osp

import numpy as np
from oci.utils import geometry_utils
from psbody.mesh import Mesh
from scipy.spatial.transform import Rotation
import torch

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
# pylint: disable=g-complex-comprehension
# pylint: disable=using-constant-test
# pylint: disable=g-explicit-length-test
# pylint: disable=undefined-variable

simplified_mesh = {
    "backpack": "backpack/backpack_f1000.ply",
    "basketball": "basketball/basketball_f1000.ply",
    "boxlarge": "boxlarge/boxlarge_f1000.ply",
    "boxtiny": "boxtiny/boxtiny_f1000.ply",
    "boxlong": "boxlong/boxlong_f1000.ply",
    "boxsmall": "boxsmall/boxsmall_f1000.ply",
    "boxmedium": "boxmedium/boxmedium_f1000.ply",
    "chairblack": "chairblack/chairblack_f2500.ply",
    "chairwood": "chairwood/chairwood_f2500.ply",
    "monitor": "monitor/monitor_closed_f1000.ply",
    "keyboard": "keyboard/keyboard_f1000.ply",
    "plasticcontainer": "plasticcontainer/plasticcontainer_f1000.ply",
    "stool": "stool/stool_f1000.ply",
    "tablesquare": "tablesquare/tablesquare_f2000.ply",
    "toolbox": "toolbox/toolbox_f1000.ply",
    "suitcase": "suitcase/suitcase_f1000.ply",
    "tablesmall": "tablesmall/tablesmall_f1000.ply",
    "yogamat": "yogamat/yogamat_f1000.ply",
    "yogaball": "yogaball/yogaball_f1000.ply",
    "trashbin": "trashbin/trashbin_f1000.ply",
}

watertight_mesh = {
    "chairwood": "chairwood/chairwood-watertight.ply",
}


def compute_transform_to_canonical(frame_data):
  angles = frame_data["obj_params"]["angle"]  ## this is encoded as rot_vec
  trans = frame_data["obj_params"]["translation"]
  rMatrix = np.eye(4)
  rMatrix[:3, :3] = Rotation.from_rotvec(angles).as_matrix()
  tMatrix = geometry_utils.get_T_from(trans)
  rtMatrix = np.matmul(tMatrix, rMatrix)
  return rtMatrix


def read_frame(frame_reader, frame_id, smpl_name, obj_name):
  smpl_fit = frame_reader.get_smplfit(frame_id, smpl_name)
  obj_fit = frame_reader.get_objfit(frame_id, obj_name)

  obj_angle, obj_trans = frame_reader.get_objfit_params(frame_id, obj_name)
  smpl_pose, smpl_beta, smpl_trans = frame_reader.get_smplfit_params(
      frame_id, smpl_name)
  frame_data = {}

  frame_data["smpl_mesh"] = smpl_fit  ## psbody.mesh
  frame_data["obj_mesh"] = obj_fit  ## psbody.mesh
  frame_data["obj_params"] = {"angle": obj_angle, "translation": obj_trans}

  frame_data["smpl_params"] = {
      "pose": torch.FloatTensor(smpl_pose),
      "beta": torch.FloatTensor(smpl_beta),
      "trans": torch.FloatTensor(smpl_trans),
  }
  return frame_data


def read_object_mesh(frame_reader, frame_id, obj_name, from_repository=True):

  if from_repository:
    obj_angle, obj_trans = frame_reader.get_objfit_params(frame_id, obj_name)
    frame_data = {}
    frame_data["obj_params"] = {"angle": obj_angle, "translation": obj_trans}
    cat = frame_reader.get_obj_category()
    obj_dir = frame_reader.get_obj_repo_dir()
    obj_smpl_path = osp.join(obj_dir, simplified_mesh[cat])
    obj_smplified = Mesh()
    obj_smplified.load_from_ply(obj_smpl_path)

    obj_watertight_path = osp.join(obj_dir, watertight_mesh[cat])
    obj_watertight = Mesh()
    obj_watertight.load_from_ply(obj_watertight_path)

    obj_can = obj_watertight
    # obj_can = frame_reader.repository_object_path()
    can2world = compute_object_transform_to_canonical(frame_data)
    # obj_watertight.write_ply('watertight_can.ply')
    # obj_smplified.write_ply('simple_cam.ply')
    center = np.mean(obj_smplified.v, axis=0)
    obj_vertices = (obj_can.v - center[None,])

    obj_vertices = geometry_utils.transform_points(
        RT=can2world, points=obj_vertices, return_homo=False)
    obj_world = Mesh(v=obj_vertices, f=obj_can.f * 1)
    return obj_can, obj_world
  else:
    raise NotImplementedError("can only read objects from repository")

  return


def compute_object_transform_to_canonical(frame_data):
  angles = frame_data["obj_params"]["angle"]  ## this is encoded as rot_vec
  trans = frame_data["obj_params"]["translation"]
  rMatrix = np.eye(4)
  rMatrix[:3, :3] = Rotation.from_rotvec(angles).as_matrix()
  tMatrix = geometry_utils.get_T_from(trans)
  rtMatrix = np.matmul(tMatrix, rMatrix)
  return rtMatrix
