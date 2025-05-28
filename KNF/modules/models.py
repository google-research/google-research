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

"""Pytorch implementation of Koopman Neural Operator."""
import itertools

from modules.normalizer import RevIN
import numpy as np
import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
  """MLP module used in the main Koopman architecture.

  Attributes:
    input_dim: number of features
    output_dim: output dimension of encoder
    hidden_dim: hidden dimension of encoder
    num_layers: number of layers
    use_instancenorm: whether to use instance normalization
    dropout_rate: dropout rate
  """

  def __init__(
      self,
      input_dim,
      output_dim,
      hidden_dim,
      num_layers,
      use_instancenorm=False,
      dropout_rate=0
  ):
    super().__init__()

    model = [nn.Linear(input_dim, hidden_dim)]
    if use_instancenorm:
      model += [nn.InstanceNorm1d(hidden_dim)]
    model += [nn.ReLU(), nn.Dropout(dropout_rate)]

    for _ in range(num_layers - 2):
      model += [nn.Linear(hidden_dim, hidden_dim)]
      if use_instancenorm:
        model += [nn.InstanceNorm1d(hidden_dim)]
      model += [nn.ReLU(), nn.Dropout(dropout_rate)]
    model += [nn.Linear(hidden_dim, output_dim)]

    self.model = nn.Sequential(*model)

  def forward(self, inps):
    # expected input dims (batch size, sequence length, number of features)
    return self.model(inps)


class Koopman(nn.Module):
  """Koopman Neural Forecaster.

  Attributes:
    input_dim: number of steps of historical observations encoded at every step
    input_length: input length of ts
    output_dim: number of output features
    num_steps: number of prediction steps every forward pass
    encoder_hidden_dim: hidden dimension of encoder
    decoder_hidden_dim: hidden dimension of decoder
    encoder_num_layers: number of layers in the encoder
    decoder_num_layers: number of layers in the decoder
    latent_dim: dimension of finite koopman space num_feats=1: number of
      features
    add_global_operator: whether to use a global operator
    add_control: whether to use a feedback module
    control_num_layers: number of layers in the control module
    control_hidden_dim: hidden dim in the control module
    use_RevIN: whether to use reversible normalization
    use_instancenorm: whether to use instance normalization on hidden states
    regularize_rank: Whether to regularize rank of Koopman Operator.
    num_sins: number of pairs of sine and cosine measurement functions
    num_poly: the highest order of polynomial functions
    num_exp: number of exponential functions
    num_heads: Number of the head the transformer encoder
    transformer_dim: hidden dimension of tranformer encoder
    transformer_num_layers: number of layers in the transformer encoder
    dropout_rate: dropout rate of MLP modules
  """

  def __init__(self,
               input_dim,
               input_length,
               output_dim,
               num_steps,
               encoder_hidden_dim,
               decoder_hidden_dim,
               encoder_num_layers,
               decoder_num_layers,
               latent_dim,
               num_feats=1,
               add_global_operator=False,
               add_control=False,
               control_num_layers=None,
               control_hidden_dim=None,
               use_revin=True,
               use_instancenorm=False,
               regularize_rank=False,
               num_sins=-1,
               num_poly=-1,
               num_exp=-1,
               num_heads=1,
               transformer_dim=128,
               transformer_num_layers=3,
               dropout_rate=0):

    super().__init__()
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.input_length = input_length
    self.num_steps = num_steps
    self.latent_dim = latent_dim
    self.use_revin = use_revin
    self.regularize_rank = regularize_rank
    self.use_instancenorm = use_instancenorm
    self.add_control = add_control
    self.add_global_operator = add_global_operator
    self.num_feats = num_feats

    # num_poly/num_sins/num_exp = -1 means using default values
    if num_poly == -1:
      self.num_poly = 3
    else:
      self.num_poly = num_poly

    if num_sins == -1:
      self.num_sins = input_length // 2 - 1
    else:
      self.num_sins = num_sins

    if num_exp == -1:
      self.num_exp = 1
    else:
      self.num_exp = num_exp

    # we also use interation terms for multivariate time series
    # calculate the number of second-order interaction terms
    if self.num_feats > 1:
      self.len_interas = len(
          list(itertools.combinations(np.arange(0, self.num_feats), 2)))
    else:
      self.len_interas = 0

    # reversible instance normalization
    if use_revin:
      self.normalizer = RevIN(num_features=self.output_dim, axis=(1, 2))

    if regularize_rank:
      dynamics_params = nn.init.xavier_normal_(
          torch.zeros(latent_dim, latent_dim))
      self.dynamics_summary = nn.Parameter(dynamics_params)

    ### MLP Encoder: Learning coeffcients of measurement functions ###
    self.encoder = MLP(
        input_dim=input_dim * self.num_feats,
        output_dim=(latent_dim + self.num_sins * 2) * input_dim *
        self.num_feats,  # we learn both the freq and magn for sin/cos,
        hidden_dim=encoder_hidden_dim,
        num_layers=encoder_num_layers,
        use_instancenorm=self.use_instancenorm,
        dropout_rate=dropout_rate)

    ### Global Linear Koopman Operator: A finite matrix ###
    if self.add_global_operator:
      self.global_linear_transform = nn.Linear(
          latent_dim * self.num_feats + self.len_interas,
          latent_dim * self.num_feats + self.len_interas,
          bias=False)

    ### Transformer Encoder: learning Local Koopman Operator ###
    self.encoder_layer = nn.TransformerEncoderLayer(
        d_model=input_length // input_dim,
        nhead=num_heads,
        dim_feedforward=transformer_dim)
    self.transformer_encoder = nn.TransformerEncoder(
        self.encoder_layer, num_layers=transformer_num_layers)
    self.attention = nn.MultiheadAttention(
        embed_dim=input_length // input_dim,
        num_heads=num_heads,
        batch_first=True)

    ### MLP Control/Feedback Module ###
    if self.add_control:
      # learn the adjustment to the koopman operator
      # based on the prediction error on the look back window.
      self.control = MLP(
          input_dim=(input_length - input_dim) * self.num_feats,
          output_dim=latent_dim * self.num_feats + self.len_interas,
          hidden_dim=control_hidden_dim,
          num_layers=control_num_layers,
          use_instancenorm=self.use_instancenorm,
          dropout_rate=dropout_rate)

    ### MLP Decoder: Reconstruct Observations from Measuremnets ###
    self.decoder = MLP(
        input_dim=latent_dim * self.num_feats + self.len_interas,
        output_dim=output_dim * self.num_feats,
        hidden_dim=decoder_hidden_dim,
        num_layers=decoder_num_layers,
        use_instancenorm=self.use_instancenorm,
        dropout_rate=dropout_rate)

  def single_forward(
      self,
      inps,  # input ts tensor
      num_steps  # num of prediction steps
  ):

    ##################### Encoding ######################
    # the encoder learns the coefficients of basis functions
    encoder_outs = self.encoder(inps)

    # reshape inputs and encoder outputs for next step
    encoder_outs = encoder_outs.reshape(inps.shape[0], inps.shape[1],
                                        (self.latent_dim + self.num_sins * 2),
                                        self.input_dim * self.num_feats)
    encoder_outs = encoder_outs.reshape(inps.shape[0], inps.shape[1],
                                        (self.latent_dim + self.num_sins * 2),
                                        self.input_dim, self.num_feats)
    inps = inps.reshape(inps.shape[0], inps.shape[1], self.input_dim,
                        self.num_feats)

    # the input to the measurement functions are
    # the muliplication of coeffcients and original observations.
    coefs = torch.einsum("blkdf, bldf -> blfk", encoder_outs, inps)
    #####################################################

    ################ Calculate Meausurements ############
    embedding = torch.zeros(encoder_outs.shape[0], encoder_outs.shape[1],
                            self.num_feats, self.latent_dim).to(inps.device)
    for f in range(self.num_feats):
      # polynomials
      for i in range(self.num_poly):
        embedding[:, :, f, i] = coefs[:, :, f, i]**(i + 1)

      # exponential function
      for i in range(self.num_poly, self.num_poly + self.num_exp):
        embedding[:, :, f, i] = torch.exp(coefs[:, :, f, i])

      # sine/cos functions
      for i in range(self.num_poly + self.num_exp,
                     self.num_poly + self.num_exp + self.num_sins):
        embedding[:, :, f,
                  i] = coefs[:, :, f, self.num_sins * 2 + i] * torch.cos(
                      coefs[:, :, f, i])
        embedding[:, :, f, self.num_sins +
                  i] = coefs[:, :, f, self.num_sins * 3 + i] * torch.sin(
                      coefs[:, :, f, self.num_sins + i])

      # the remaining ouputs are purely data-driven measurement functions.
      embedding[:, :, f, self.num_poly + self.num_exp +
                self.num_sins * 2:] = coefs[:, :, f, self.num_poly +
                                            self.num_exp + self.num_sins * 4:]

    embedding = embedding.reshape(embedding.shape[0], embedding.shape[1], -1)

    # if there are multiple features,
    # second-order interaction terms should also be included
    if self.num_feats > 1:
      inter_lsts = list(itertools.combinations(np.arange(0, self.num_feats), 2))
      embedding_inter = torch.zeros(encoder_outs.shape[0],
                                    encoder_outs.shape[1],
                                    len(inter_lsts)).to(inps.device)
      for i, item in enumerate(inter_lsts):
        embedding_inter[Ellipsis, i] = (
            coefs[:, :, item[0], 0] * coefs[:, :, item[1], 0]
        )
      embedding = torch.cat([embedding, embedding_inter], dim=-1)

    # Reconstruction
    reconstructions = self.decoder(embedding)
    #####################################################

    ###### Generate Predictions on the Lookback Window ######
    # Generate Local Koopman Operator
    trans_out = self.transformer_encoder(embedding.transpose(1, 2))
    local_transform = self.attention(trans_out, trans_out, trans_out)[1]

    # Collect predicted measurements on the lookback window
    inp_embed_preds = []
    forw = embedding[:, :1]
    for i in range(inps.shape[1] - 1):
      if self.add_global_operator:
        forw = self.global_linear_transform(forw)
      forw = torch.einsum("bnl, blh -> bnh", forw, local_transform)
      inp_embed_preds.append(forw)
    embed_preds = torch.cat(inp_embed_preds, dim=1)

    # Reconstruction
    inp_preds = self.decoder(embed_preds)
    #########################################################

    ########## Generate Predictions on the Forecasting Window ##########
    # If the predictions on the lookback window deviates a lot from groud truth,
    # adjust the koopman operator with the control module.
    if self.add_control:
      pred_diff = inp_preds.reshape(
          inp_preds.shape[0], -1) - inps[:, 1:].reshape(inp_preds.shape[0], -1)
      linear_adj = self.control(pred_diff)
      linear_adj = torch.stack(
          [torch.diagflat(linear_adj[i]) for i in range(len(linear_adj))])
    forw_preds = []

    forward_iters = self.num_steps // self.input_dim
    if self.num_steps % self.input_dim > 0:
      forward_iters += 1

    # Forward predictions
    for i in range(forward_iters):
      if self.add_global_operator:
        forw = self.global_linear_transform(forw)
      if self.add_control:
        forw = torch.einsum("bnl, blh -> bnh", forw,
                            local_transform + linear_adj)
      else:
        forw = torch.einsum("bnl, blh -> bnh", forw, local_transform)
      forw_preds.append(forw)
    forw_preds = torch.cat(forw_preds, dim=1)

    # Reconstruction
    forw_preds = self.decoder(forw_preds)
    #####################################################################

    if self.regularize_rank:
      ##### Regularize the rank of koopman operator ######
      local_transform = torch.matmul(local_transform, self.dynamics_summary)

      dynamics_cov = torch.matmul(self.dynamics_summary.t() / self.latent_dim,
                                  self.dynamics_summary / self.latent_dim)
      spec_norm = torch.linalg.eigvalsh(dynamics_cov)
      # dynamics_cov is positive semi-definite, so all its complex parts are 0.
      evals = spec_norm**2
      # -1 to make the minimum possible value 0.
      rank_regularizer = evals.sum() - evals.max() - 1
      #####################################################
      return (reconstructions, inp_preds, forw_preds, embedding[:, 1:],
              embed_preds, rank_regularizer)
    else:
      return (reconstructions, inp_preds, forw_preds, embedding[:, 1:],
              embed_preds, 0)

  def forward(self, org_inps, tgts):
    # number of autoregressive step
    auto_steps = tgts.shape[1] // self.num_steps
    if tgts.shape[1] % self.num_steps > 0:
      auto_steps += 1

    if self.use_revin:
      denorm_outs = []
      norm_tgts = []
      norm_outs = []
      norm_inps = []
      norm_recons = []
      norm_inp_preds = []
      rank_regularizers = []
      enc_embeds = []
      pred_embeds = []

      for i in range(auto_steps):
        try:
          inps = org_inps.reshape(org_inps.shape[0], -1, self.input_dim,
                                  self.num_feats)
          inps = inps.reshape(org_inps.shape[0], -1,
                              self.input_dim * self.num_feats)
        except ValueError as valueerror:
          raise ValueError(
              "Input length is not divisible by input dim") from valueerror

        norm_inp = self.normalizer.forward(inps, mode="norm")
        norm_inps.append(norm_inp)

        single_forward_output = self.single_forward(norm_inp, self.num_steps)
        if self.regularize_rank:
          (reconstructions, inp_preds, forw_preds, enc_embedding,
           pred_embedding, rank_regularizer) = single_forward_output
          rank_regularizers.append(rank_regularizer)
        else:
          (reconstructions, inp_preds, forw_preds, enc_embedding,
           pred_embedding, rank_regularizer) = single_forward_output

        norm_recons.append(reconstructions)
        norm_inp_preds.append(inp_preds)
        enc_embeds.append(enc_embedding)
        pred_embeds.append(pred_embedding)

        forw_preds = forw_preds.reshape(forw_preds.shape[0], -1,
                                        self.num_feats)[:, :self.num_steps]
        norm_outs.append(forw_preds)

        # normalize tgts
        norm_tgts.append(
            self.normalizer.normalize(tgts[:, i * self.num_steps:(i + 1) *
                                           self.num_steps]))

        # denormalize prediction and add back to the input
        denorm_outs.append(self.normalizer.forward(forw_preds, mode="denorm"))
        # print(org_inps.shape, denorm_outs[-1].shape)
        org_inps = torch.cat([org_inps[:, self.num_steps:], denorm_outs[-1]],
                             dim=1)

      norm_outs = torch.cat(norm_outs, dim=1)
      norm_tgts = torch.cat(norm_tgts, dim=1)
      denorm_outs = torch.cat(denorm_outs, dim=1)

      norm_inps = torch.cat(norm_inps, dim=0)
      norm_inp_preds = torch.cat(norm_inp_preds, dim=0)
      norm_recons = torch.cat(norm_recons, dim=0)
      enc_embeds = torch.cat(enc_embeds, dim=0)
      pred_embeds = torch.cat(pred_embeds, dim=0)

      forward_output = [
          denorm_outs[:, :norm_tgts.shape[1]],
          [norm_outs[:, :norm_tgts.shape[1]], norm_tgts],
          [norm_recons, norm_inp_preds, norm_inps], [enc_embeds, pred_embeds]
      ]

      if rank_regularizers:
        forward_output += [torch.mean(torch.stack(rank_regularizers))]

      return forward_output

    else:
      outs = []
      true_inps = []
      recons = []
      inputs_preds = []
      enc_embeds = []
      pred_embeds = []
      rank_regularizers = []

      for i in range(auto_steps):
        try:
          inps = org_inps.reshape(org_inps.shape[0], -1, self.input_dim,
                                  self.num_feats)
          inps = inps.reshape(org_inps.shape[0], -1,
                              self.input_dim * self.num_feats)
        except ValueError as valueerror:
          raise ValueError(
              "Input length is not divisible by input dim") from valueerror

        true_inps.append(inps)
        single_forward_output = self.single_forward(inps, self.num_steps)
        if self.regularize_rank:
          (reconstructions, inp_preds, forw_preds, enc_embedding,
           pred_embedding, rank_regularizer) = single_forward_output
          rank_regularizers.append(rank_regularizer)
        else:
          (reconstructions, inp_preds, forw_preds, enc_embedding,
           pred_embedding, rank_regularizer) = single_forward_output

        recons.append(reconstructions)
        inputs_preds.append(inp_preds)
        enc_embeds.append(enc_embedding)
        pred_embeds.append(pred_embedding)

        forw_preds = forw_preds.reshape(forw_preds.shape[0], -1,
                                        self.num_feats)[:, :self.num_steps]
        outs.append(forw_preds)

        org_inps = torch.cat([org_inps[:, self.num_steps:], outs[-1]], dim=1)

      outs = torch.cat(outs, dim=1)
      true_inps = torch.cat(true_inps, dim=0)
      inputs_preds = torch.cat(inputs_preds, dim=0)
      recons = torch.cat(recons, dim=0)
      enc_embeds = torch.cat(enc_embeds, dim=0)
      pred_embeds = torch.cat(pred_embeds, dim=0)

      forward_output = [
          outs[:, :tgts.shape[1]], [outs[:, :tgts.shape[1]], tgts],
          [recons, inputs_preds, true_inps], [enc_embeds, pred_embeds]
      ]

      if rank_regularizers:
        forward_output += [torch.mean(torch.stack(rank_regularizers))]

      return forward_output
