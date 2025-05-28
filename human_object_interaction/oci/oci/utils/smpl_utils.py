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

"""SMPL Utils."""
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

from oci.vibe_lib import geometry
from psbody.mesh import Mesh
import smplx
from smplx import body_models
import torch
from torch import nn


def split_smplh_pose_params(pose):
  global_pose = pose[Ellipsis, 0:3]
  body_pose = pose[Ellipsis, 3:21 * 3 + 3]
  left_hand_pose = pose[Ellipsis, 22 * 3:(22 + 15) * 3]
  right_hand_pose = pose[Ellipsis, (22 + 15) * 3:]

  params = {}
  params["global_pose"] = global_pose
  params["body_pose"] = body_pose
  params["left_hand_pose"] = left_hand_pose
  params["right_hand_pose"] = right_hand_pose
  return params


def combine_smplh_pose_parms(gpose, body_pose, left_hand_pose, right_hand_pose):
  pose = torch.cat([gpose, body_pose, left_hand_pose, right_hand_pose], axis=-1)
  return pose


def create_smpl_model(gender, smpl_model_path, **kwargs):
  smpl_model = body_models.create(
      smpl_model_path, model_type="smplh", gender=gender, **kwargs)
  return smpl_model


def smplh_canonical(gender, smpl_model_path, **kwargs):
  smpl_model = create_smpl_model(
      gender, smpl_model_path=smpl_model_path, **kwargs)
  smpl_output = smpl_model.forward()
  vertices = smpl_output.vertices.data.numpy()[0]
  faces = smpl_model.faces
  smpl_mesh = Mesh(vertices, faces)
  return smpl_mesh, smpl_model


class SMPL2Mesh(nn.Module):

  def __init__(self, smpl_model_path, gender):
    super().__init__()
    self.smplh = smplx.body_models.create(
        smpl_model_path,
        model_type="smplh",
        gender=gender,
        flat_hand_mean=True,
        use_pca=False,
    )

  def forward(self, _pose, _transl, _beta):
    pose_params = split_smplh_pose_params(_pose)
    smplh_outputs = self.smplh.forward(
        betas=_beta,
        body_pose=pose_params["body_pose"],
        left_hand_pose=pose_params["left_hand_pose"],
        right_hand_pose=pose_params["right_hand_pose"],
        global_orient=pose_params["global_pose"],
        transl=_transl,
        pose2rot=True,
    )
    self.faces = self.smplh.faces
    smplh_outputs.faces = self.smplh.faces
    return smplh_outputs


def convert_smpl_seq2mesh(smpl_model_path,
                          _pose,
                          _transl,
                          _beta,
                          gender="female"):
  device = _pose.device
  smplh = SMPL2Mesh(smpl_model_path=smpl_model_path, gender=gender)
  smplh.to(device)
  smplh_model = smplh.smplh
  smplh_outputs = smplh.forward(_pose=_pose, _transl=_transl, _beta=_beta)

  faces = smplh_model.faces
  vertices = smplh_outputs.vertices.detach().to("cpu").numpy()
  meshes = []
  for verts in vertices:
    mesh = Mesh(v=verts, f=faces)
    meshes.append(mesh)
  return meshes


def _transform_smplh_parameters_helper(
    smpl_model,
    betas,
    body_pose,
    left_hand_pose,
    right_hand_pose,
    global_pose,
    transl,
    RT,
):
  R = torch.FloatTensor(RT[:3, :3])  ## this is not batched
  t = torch.FloatTensor(RT[:3, 3])
  global_rot_matrix = geometry.angle_axis_to_rotation_matrix(global_pose)
  # if len(global_rot_matrix.shape) == 3:
  #   global_rot_matrix = global_rot_matrix[0]

  new_gpose_rotmat = torch.bmm(
      torch.FloatTensor(R[None,]),
      global_rot_matrix,
  )
  new_gpose = geometry.rotation_matrix_to_angle_axis(new_gpose_rotmat)
  smplh_out = smpl_model.forward(
      betas=betas,
      body_pose=body_pose,
      transl=transl * 0,
      left_hand_pose=left_hand_pose,
      right_hand_pose=right_hand_pose,
      global_orient=global_pose * 0,
  )
  pelvis = smplh_out.joints[0, 0]
  new_transl = torch.matmul(R, (pelvis + transl[0])) + t - pelvis
  new_transl = new_transl[None,]
  return new_gpose, new_transl


def transform_smplh_parameters(smplh_model, beta, pose, transl, RT):
  pose_params = split_smplh_pose_params(pose)

  new_gpose, new_transl = _transform_smplh_parameters_helper(
      smplh_model,
      betas=beta,
      body_pose=pose_params["body_pose"],
      left_hand_pose=pose_params["left_hand_pose"],
      right_hand_pose=pose_params["right_hand_pose"],
      global_pose=pose_params["global_pose"],
      transl=transl,
      RT=RT,
  )

  new_pose = (
      combine_smplh_pose_parms(
          gpose=new_gpose,
          body_pose=pose_params["body_pose"],
          left_hand_pose=pose_params["left_hand_pose"],
          right_hand_pose=pose_params["right_hand_pose"],
      ) * 1)

  return beta, new_pose, new_transl
