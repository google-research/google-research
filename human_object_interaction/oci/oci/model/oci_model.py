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

"""Object Centric SMPL model."""

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
# pylint: disable=g-importing-member

from os import stat

from cv2 import reduce
from oci.model import resnetFC, vposer_model
import torch
from torch import nn


## Input configs
class OCIModel(nn.Module):

  def __init__(self, opts):
    super().__init__()
    smpl_pose_dim = 156 + 10
    self.smpl_pose_encoder = vposer_model.VPoser(
        in_features=smpl_pose_dim, opts=opts.pose_encoder)
    self.positional_encoder_3d = resnetFC.PositionalEncoding()
    self.positional_encoder_canonical = resnetFC.PositionalEncoding()

    if opts.mlp_activation == "leakyReLU":
      activation = nn.LeakyReLU(0.2)
    else:
      activation = None
    self.last_op = None
    self.feature_mlp = resnetFC.ResnetFC(
        d_in=self.positional_encoder_3d.d_out,
        d_out=opts.feature_backbone.d_out,
        last_op=self.last_op,
        activation=activation,
    )

    if opts.predictor_out_type == "3d":
      predicter_out = 3

    self.motion_predictor = resnetFC.ResnetFC(
        d_in=self.feature_mlp.d_out + self.positional_encoder_3d.d_out +
        self.positional_encoder_canonical.d_out + self.smpl_pose_encoder.d_out,
        d_out=predicter_out,
        last_op=None,
        activation=activation,
    )
    return

  def forward(self, inputs):
    smpl_pose_params = torch.cat([inputs["smpl_pose"], inputs["smpl_beta"]],
                                 axis=-1)
    self.smpl_feat = self.smpl_pose_encoder.encode(
        smpl_pose_params)  # batch_size x feat
    self.predictions = self.query(inputs)
    return self.predictions

  def query(self, inputs):
    point_location = inputs["points_input"]
    can_point_location = inputs["points_can"]
    batch_size, npoints, _ = point_location.shape
    point_emb = self.flatten_and_encode(point_location,
                                        self.positional_encoder_3d)
    point_feature = self.flatten_and_encode(point_emb, self.feature_mlp)
    can_point_emb = self.flatten_and_encode(can_point_location,
                                            self.positional_encoder_canonical)
    smpl_feat = self.smpl_feat[:, None, :].expand(-1, npoints, -1)
    predictor_feat = torch.cat(
        [point_feature, point_emb, can_point_emb, smpl_feat],
        dim=-1)  # batch_size x npoints x feat_dim
    predicted_future = self.flatten_and_encode(predictor_feat,
                                               self.motion_predictor)
    predictions = {}
    predictions["points_futr"] = predicted_future
    return predictions

  def compute_loss(self, targets):

    futur_loss = torch.nn.functional.l1_loss(
        self.predictions["points_futr"], targets["points_futr"], reduce=False)

    futur_loss = futur_loss.mean(-1).mean(-1)

    loss = {"points_3d": futur_loss}
    return loss

  @staticmethod
  def flatten_and_encode(points, encoder):
    """Flatten and Encode."""

    batch_size, npoints, _ = points.shape
    points = points.reshape(batch_size * npoints, _)
    point_feat = encoder.forward(points)
    point_feat = point_feat.reshape(batch_size, npoints, -1)
    return point_feat
