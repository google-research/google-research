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
# pylint: disable=redefined-builtin

from cv2 import reduce
from oci.model import resnetFC, vposer_model
from oci.vibe_lib import spin
import torch
from torch import nn
import torch.nn.functional as F

## Copy ResNet blocks
## Create MLP layers


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


class TemporalEncoder(nn.Module):

  def __init__(
      self,
      n_layers=1,
      input_size=128,
      hidden_size=2048,
      add_linear=False,
      bidirectional=False,
      use_residual=True,
  ):
    super().__init__()

    self.gru = nn.GRU(
        input_size=input_size,
        hidden_size=hidden_size,
        bidirectional=bidirectional,
        num_layers=n_layers,
    )

    self.linear = None
    if bidirectional:
      self.linear = nn.Linear(hidden_size * 2, 2048)
    elif add_linear:
      self.linear = nn.Linear(hidden_size, 2048)
    self.use_residual = use_residual

  def forward(self, x):
    n, t, f = x.shape
    x = x.permute(1, 0, 2)  # NTF -> TNF
    y, _ = self.gru(x)
    if self.linear:
      y = F.relu(y)
      y = self.linear(y.view(-1, y.size(-1)))
      y = y.view(t, n, f)
    if self.use_residual and y.shape[-1] == 2048:
      y = y + x
    y = y.permute(1, 0, 2)  # TNF -> NTF
    return y


class VIBEModel(nn.Module):

  def __init__(self, opts, input_size):
    super().__init__()
    self.opts = opts
    self.encoder = TemporalEncoder(
        input_size=input_size,
        n_layers=opts.tgru.n_layers,
        hidden_size=opts.tgru.hidden_size,
        bidirectional=opts.tgru.bidirectional,
        add_linear=opts.tgru.add_linear,
        use_residual=opts.tgru.use_residual,
    )
    self.regressor = spin.Regressor(
        smpl_model_path=opts.smpl_model_path,
        input_size=opts.tgru.hidden_size,
        relative_targets=opts.relative_targets,
    )
    return

  def forward(self, input, J_regressor=None):
    # input size NTF

    gender = input["gender"]
    smpl_input = input["smpl"]
    batch_size, seqlen = smpl_input.shape[:2]
    feature = self.encoder(smpl_input)
    # feature = feature.reshape(-1, feature.size(-1))
    feature = feature[:, -1, :]  ## choose the last prediction only

    base_outputs = input["base_poses"]

    smpl_output = self.regressor(
        feature, base_outputs, gender=gender, J_regressor=J_regressor)
    # for s in smpl_output:
    #   s['theta'] = s['theta'].reshape(batch_size, seqlen, -1)
    #   s['verts'] = s['verts'].reshape(batch_size, seqlen, -1, 3)
    #   s['kp_2d'] = s['kp_2d'].reshape(batch_size, seqlen, -1, 2)
    #   s['kp_3d'] = s['kp_3d'].reshape(batch_size, seqlen, -1, 3)
    #   s['rotmat'] = s['rotmat'].reshape(batch_size, seqlen, -1, 3, 3)

    return smpl_output


class OCISMPLModel(nn.Module):

  def __init__(self, opts):
    super().__init__()
    self.opts = opts
    smpl_pose_dim = 156 + 10 + 3
    self.smpl_pose_encoder = vposer_model.VPoser(
        in_features=smpl_pose_dim, opts=opts.pose_encoder)
    # self.pose_sequence_encoder = SeqModel(
    #   input_size=self.smpl_pose_encoder.d_out,
    #   hidden_size=opts.pose_seq_model.hidden_dim,
    #   out_size=self.smpl_pose_encoder.d_out,
    #   num_layers=opts.pose_seq_model.num_layers,
    # )
    self.generator = VIBEModel(
        opts.vibe_model, input_size=self.smpl_pose_encoder.d_out)

  def forward(self, inputs):
    smpl_pose_params = torch.cat(
        [inputs["smpl_pose"], inputs["smpl_beta"], inputs["smpl_transl"]],
        axis=-1)
    smpl_seq_feat = self.flatten_and_encode(smpl_pose_params,
                                            self.smpl_pose_encoder)
    gen_inputs = {}
    gen_inputs["smpl"] = smpl_seq_feat
    gen_inputs["gender"] = inputs["gender"]
    gen_inputs["base_poses"] = {
        "smpl_pose": inputs["smpl_pose"][:, -1, :],
        "smpl_beta": inputs["smpl_beta"][:, -1, :],
        "smpl_transl": inputs["smpl_transl"][:, -1, :],
    }
    self.predictions = self.generator.forward(gen_inputs)
    return self.predictions

  def compute_loss(self, targets):
    opts = self.opts

    if opts.loss_type == "smooth_l1_loss":
      loss_func = torch.nn.functional.smooth_l1_loss
    elif opts.loss_type == "mse":
      loss_func = torch.nn.functional.mse_loss
    elif opts.loss_type == "l1":
      loss_func = torch.nn.functional.l1_loss
    else:
      assert False, "incorrect loss type"

    pose_loss = loss_func(
        self.predictions["smpl_pose"], targets["futr_smpl_pose"], reduce=False)
    beta_loss = loss_func(
        self.predictions["smpl_beta"], targets["futr_smpl_beta"], reduce=False)

    trans_loss = loss_func(
        self.predictions["smpl_transl"],
        targets["futr_smpl_transl"],
        reduce=False)
    pose_loss = pose_loss.mean(-1).mean(-1)
    beta_loss = beta_loss.mean(-1).mean(-1)
    trans_loss = trans_loss.mean(-1).mean(-1)
    loss = {"pose": pose_loss, "beta": beta_loss, "trans": trans_loss}
    return loss

  @staticmethod
  def convert_relative_to_absolute(inputs_locations, relative_points):
    device = inputs_locations.device
    points_futr = inputs_locations + relative_points.to(device)
    return points_futr

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
