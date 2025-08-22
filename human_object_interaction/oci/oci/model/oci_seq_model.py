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


from cv2 import reduce
from oci.model import resnetFC, vposer_model
import torch
from torch import nn

## Copy ResNet blocks
## Create MLP layers

## Input configs


class SeqModel(nn.Module):

  def __init__(self, input_size, hidden_size, out_size, num_layers):
    super().__init__()
    self.seq_model = nn.GRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=True,
    )
    self.mlp_layer = nn.Linear(hidden_size, out_size)
    self.d_out = out_size

  def forward(self, x):
    output_state, hidden_state = self.seq_model.forward(x)
    x = output_state[:, -1, :]
    x = self.mlp_layer.forward(x)
    return x


class OCIModel(nn.Module):

  def __init__(self, opts):
    super().__init__()
    self.opts = opts
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

    self.pose_sequence_encoder = SeqModel(
        input_size=self.smpl_pose_encoder.d_out,
        hidden_size=opts.pose_seq_model.hidden_dim,
        out_size=self.smpl_pose_encoder.d_out,
        num_layers=opts.pose_seq_model.num_layers,
    )
    self.point_sequence_encoder = SeqModel(
        input_size=self.positional_encoder_3d.d_out,
        hidden_size=opts.point_seq_model.hidden_dim,
        out_size=self.feature_mlp.d_out,
        num_layers=opts.point_seq_model.num_layers,
    )
    self.feature_sequence_encoder = SeqModel(
        input_size=self.feature_mlp.d_out,
        hidden_size=opts.point_seq_model.hidden_dim,
        out_size=self.feature_mlp.d_out,
        num_layers=opts.point_seq_model.num_layers,
    )
    if opts.predictor_out_type == "3d":
      predicter_out = 3

    pose_dim = self.pose_sequence_encoder.d_out
    feature_dim = self.feature_sequence_encoder.d_out
    point_dim = self.point_sequence_encoder.d_out

    self.motion_predictor = resnetFC.ResnetFC(
        d_in=feature_dim + point_dim + self.positional_encoder_canonical.d_out +
        pose_dim,
        d_out=predicter_out,
        last_op=None,
        activation=activation,
    )
    return

  def forward(self, inputs):
    smpl_pose_params = torch.cat([inputs["smpl_pose"], inputs["smpl_beta"]],
                                 axis=-1)
    smpl_seq_feat = self.flatten_and_encode(smpl_pose_params,
                                            self.smpl_pose_encoder)
    self.smpl_feat = self.pose_sequence_encoder.forward(
        smpl_seq_feat)  ## batch_size x feat
    self.predictions = self.query(inputs)
    return self.predictions

  @staticmethod
  def encode_feature_sequence(features, feature_seq_encoder):
    """Encode features."""
    B, n_seq, n_points, feat_dim = features.shape
    features = features.permute(0, 2, 1, 3)
    features = features.reshape(B * n_points, n_seq, feat_dim)
    features = feature_seq_encoder.forward(features)
    features = features.reshape(B, n_points, -1)
    return features

  def query(self, inputs):
    opts = self.opts
    self.inputs = inputs
    point_location = inputs["points_input"]

    can_point_location = inputs["points_can"]
    _, seq_len, npoints, _ = point_location.shape

    if opts.relative_inputs:
      relative_points = point_location - point_location[:, -1, None, :, :]
    point_emb = self.flatten_and_encode(relative_points,
                                        self.positional_encoder_3d)
    point_feature = self.flatten_and_encode(point_emb, self.feature_mlp)
    can_point_emb = self.flatten_and_encode(can_point_location,
                                            self.positional_encoder_canonical)
    smpl_feat = self.smpl_feat[:, None, :].expand(-1, npoints, -1)

    point_emb = self.encode_feature_sequence(point_emb,
                                             self.point_sequence_encoder)
    point_feature = self.encode_feature_sequence(point_feature,
                                                 self.feature_sequence_encoder)
    predictor_feat = torch.cat(
        [point_feature, point_emb, can_point_emb, smpl_feat],
        dim=-1)  # batch_size x npoints x feat_dim
    predicted_future = self.flatten_and_encode(predictor_feat,
                                               self.motion_predictor)
    if opts.relative_targets:
      predicted_future += point_location[:, -1, :, :]
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
  def flatten_and_encode(inputs, encoder):
    input_dims = inputs.shape
    new_b = 1
    for k in input_dims[:-1]:
      new_b *= k
    # batch_size, npoints, _ = inputs.shape
    inputs = inputs.reshape(new_b, -1)
    point_feat = encoder.forward(inputs)
    input_dims = list(input_dims)
    input_dims[-1] = point_feat.shape[1]
    point_feat = point_feat.reshape(*input_dims)
    return point_feat

  @staticmethod
  def convert_relative_to_absolute(inputs_locations, relative_points):
    device = inputs_locations.device
    points_futr = inputs_locations + relative_points.to(device)
    return points_futr
