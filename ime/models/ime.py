# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Class that implement ime."""
from models.ar_net import ARNet
from models.linear import Linear
from models.lstm import LSTMime
from models.mlp import MLPime
from models.sdt import SDT
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.tools import add_gaussian_noise


class IME(nn.Module):
  """An implementation  of Interpretable Mixture of Experts (IME).

    IME consists of a group of experts and an assignment module which puts
    weights on different expert.
    Assignment module can either be interpretable model like Linear model or a
    black box model like LSTM.
  """

  def __init__(
      self,
      num_experts=3,
      n_forecasts=1,
      n_lags=0,
      input_features=1,
      gate_type="Linear",
      expert_type="Linear",
      d_model=512,
      layers=3,
      dropout=0.0,
      device="cpu",
  ):
    """Initializes a IME instance.

    Args:
      num_experts: Number of experts
      n_forecasts: Number of time steps to forecast
      n_lags: Lags (past time steps) used to make forecast
      input_features:  Input features dimension
      gate_type: Assignment module type can be "Linear" or "LSTM"
      expert_type: Interpretable experts type can be "Linear" or " ARNet"
      d_model: Hidden layer dimension for LSTM
      layers: Number of LSTM layers.
      dropout: Fraction of neurons affected by Dropout used by LSTM.
      device: Device used  by the model
    Inputs:
      original_inputs: A tensor of shape `(batch_size, seqence_length,
        input_size)`
      true_output: Actual forecast this is used for teacher forcing during
        training.
      past_values: Expert errors in the last time step

    Returns:
      expert_outputs: A tensor containing forecast produced by each expert
      with size `(batch_size, num_experts, n_forecasts)`
      weights: weights assigned to each expert by the assignment module
      reg_out: Regression output the forecast a tensor of `(batch_size,
      out_len)`
    """
    super(IME, self).__init__()
    self.num_experts = num_experts
    self.device = device
    self.n_forecasts = n_forecasts
    self.n_lags = n_lags
    self.expert_type = expert_type
    self.gate_type = gate_type
    # Only Linear and  ARNet experts are supported for experts
    assert self.expert_type == "Linear" or self.expert_type == " ARNet"
    # Only Linear and LSTM experts are supported for assignment module
    assert self.gate_type == "Linear" or self.gate_type == "LSTM"

    if self.expert_type == "Linear":
      self.experts = nn.ModuleList([
          Linear(n_forecasts=n_forecasts, n_lags=n_lags)
          for i in range(self.num_experts)
      ])
    elif self.expert_type == " ARNet":
      self.experts = nn.ModuleList([
          ARNet(n_forecasts=n_forecasts, n_lags=n_lags, device=device)
          for i in range(self.num_experts)
      ])
    else:
      raise NotImplementedError
    # Gate networking takes the lags and the past values as inputs and gives
    # output forecast and prediction for each expert
    if self.gate_type == "Linear":
      self.gate_network = nn.Linear((self.num_experts) + n_lags,
                                    self.num_experts + self.n_forecasts)
    elif self.gate_type == "LSTM":
      self.gate_network = LSTMime(
          input_features + self.num_experts,
          self.num_experts,
          n_forecasts,
          d_model=d_model,
          layers=layers,
          dropout=dropout,
          device=device)
    else:
      raise NotImplementedError

  def forward(self,
              original_inputs,
              true_output,
              past_values=None,
              noise=False):

    if self.expert_type == "Linear":
      inputs = original_inputs
    else:
      inputs = original_inputs[:, :, 0].squeeze(-1)

    # If gate is Linear then 2D concatentation with past values
    if self.gate_type == "Linear":
      if past_values is None:
        past_values = torch.zeros(
            (inputs.shape[0], self.num_experts)).to(self.device)
      # Concatenate past values with the inputs to create input to gate network
      gate_inputs = torch.cat((inputs, past_values), dim=1)
      # Pass the Concatenated input to the gate network
      gated_ouput = self.gate_network(gate_inputs).unsqueeze(2)
      # The first self.num_experts outputs are prediction for each expert
      assignment_logits = gated_ouput[:, :-self.n_forecasts, :]
      # Last output is the forecast
      reg_out = gated_ouput[:, -self.n_forecasts:, :].squeeze(-1)
    # If gate is LSTM then 3D concatentation with past values
    elif self.gate_type == "LSTM":
      if past_values is None:
        past_values = torch.zeros((original_inputs.shape)).to(self.device)
      else:
        past_values = past_values.unsqueeze(1).repeat(1,
                                                      original_inputs.shape[1],
                                                      1)
      gate_inputs = torch.cat((original_inputs, past_values), dim=2)

      assignment_logits, reg_out = self.gate_network(gate_inputs)
    else:
      raise NotImplementedError

    # The weights for each output is retrieved by passing logits through softmax
    weights = F.softmax(assignment_logits, dim=1)
    # If noise flag add noise to the input
    if noise:
      inputs = add_gaussian_noise(inputs, device=self.device)
    # If the interpretable experts are linear pass the input to each expert
    if self.expert_type == "Linear":
      expert_outputs_list = [
          self.experts[i](inputs).unsqueeze(2) for i in range(self.num_experts)
      ]
    # If  ARNet is used then the orignal inputs and prediction  is passed to
    # experts since it is used for teacher forcing
    elif self.expert_type == " ARNet":
      expert_outputs_list = [
          self.experts[i](inputs, true_output).unsqueeze(2)
          for i in range(self.num_experts)
      ]
    else:
      raise NotImplementedError
    # Concatenate the outputs of different experts
    for i in range(len(expert_outputs_list)):
      if i == 0:
        expert_outputs = expert_outputs_list[i]
      else:
        expert_outputs = torch.cat((expert_outputs, expert_outputs_list[i]),
                                   dim=2)

    return expert_outputs, weights, reg_out

  def predict(self, original_inputs, past_values=None):
    """Function used during inference time to make predictions.

    Args:
      original_inputs: A tensor of shape `(batch_size, seqence_length,
        input_size)`
      past_values: Expert errors in the last time step

    Returns:
      output: Forecast a tensor of shape `(batch_size, n_forecasts)`
      expert_outputs: A tensor containing forecast produced by each expert
      with size `(batch_size, num_experts, n_forecasts)`
      argmax_weights: argmax ofweights assigned to each expert by the
      assignment module
    """
    # Interpetable model only take single feature as input
    if self.expert_type == "Linear":
      inputs = original_inputs
    else:
      inputs = original_inputs[:, :, 0].squeeze(-1)
    # If gate is Linear then 2D concatentation with past values
    if self.gate_type == "Linear":
      if past_values is None:
        past_values = torch.zeros(
            (inputs.shape[0], self.num_experts)).to(self.device)
      # Concatenate past values with the inputs to create input to gate network
      gate_inputs = torch.cat((inputs, past_values), dim=1)
      gated_ouput = self.gate_network(gate_inputs).unsqueeze(2)
      # The first self.num_experts outputs are prediction for each expert
      assignment_logits = gated_ouput[:, :-self.n_forecasts, :]

    # If gate is LSTM then 3D concatentation with past values
    elif self.gate_type == "LSTM":
      if past_values is None:
        past_values = torch.zeros((original_inputs.shape)).to(self.device)
      else:
        past_values = past_values.unsqueeze(1).repeat(1,
                                                      original_inputs.shape[1],
                                                      1)

      gate_inputs = torch.cat((original_inputs, past_values), dim=2)
      assignment_logits, _ = self.gate_network(gate_inputs)
    else:
      raise NotImplementedError
    # The weights for each output is retrieved by passing logits through softmax
    weights = F.softmax(assignment_logits, dim=1)
    # get index of the maximum weight
    max_index = torch.argmax(weights, dim=1).flatten()
    # this gives argmax of weights by setting max index to 1 and reset to 0
    argmax_weights = F.one_hot(
        max_index, num_classes=assignment_logits.shape[1]).float().to(
            self.device).unsqueeze(-1)

    # If the interpretable experts are linear use regular forward function
    if self.expert_type == "Linear":
      expert_outputs_list = [
          self.experts[i](inputs).unsqueeze(2) for i in range(self.num_experts)
      ]
    # If  ARNet then used predict function for prediction
    elif self.expert_type == " ARNet":
      expert_outputs_list = [
          self.experts[i].predict(inputs).unsqueeze(2)
          for i in range(self.num_experts)
      ]
    else:
      raise NotImplementedError
    # Concatenate the outputs of different experts
    for i in range(len(expert_outputs_list)):
      if i == 0:
        expert_outputs = expert_outputs_list[i]
      else:
        expert_outputs = torch.cat((expert_outputs, expert_outputs_list[i]),
                                   dim=2)

    # Final output the is matrix multipication of expert outputs and argmax of
    # the weight
    output = torch.matmul(expert_outputs, argmax_weights).squeeze(-1)

    return output, expert_outputs, argmax_weights


class IMETabular(nn.Module):
  """An implementation  of Interpretable Mixture of Experts (IME) for tabular data.

    IME consists of a group of experts and an assignment module which puts
    weights on different expert.
    Assignment module can either be interpretable model like Linear model or a
    black box model like MLP.
  """

  def __init__(
      self,
      num_experts=3,
      input_features=1,
      output_features=1,
      expert_type="Linear",
      gate_type="Linear",
      d_model=512,
      layers=3,
      depth=5,
      device="cpu",
  ):
    """Initializes a  IMETabular instance.

    Args:
      num_experts: Number of experts
      input_features:  Input features dimension
      output_features:  Output features dimension
      expert_type: Interpretable experts type can be "Linear" or "SDT"
      gate_type: Assignment module type can be "Linear" or "MLP"
      d_model: Hidden layer dimension for MLP
      layers: Number of MLP layers.
      depth: depth of decision tree/
      device: Device used  by the model
    """
    super(IMETabular, self).__init__()
    self.num_experts = num_experts
    self.device = device
    self.input_features = input_features
    self.output_features = output_features
    self.expert_type = expert_type
    self.gate_type = gate_type

    # Only Linear and SDT experts are supported for experts
    assert self.expert_type == "Linear" or self.expert_type == "SDT"
    # Only Linear and LSTM experts are supported for assignment module
    assert self.gate_type == "Linear" or self.gate_type == "MLP"

    if self.expert_type == "Linear":
      self.experts = nn.ModuleList([
          nn.Linear(input_features, output_features)
          for i in range(self.num_experts)
      ])
    elif self.expert_type == "SDT":
      self.experts = nn.ModuleList([
          SDT(input_features, output_features, depth=depth, device=device)
          for i in range(self.num_experts)
      ])
    else:
      raise NotImplementedError
    # Gate networking takes the lags and the past values as inputs and gives
    # output forecast and prediction for each expert
    if self.gate_type == "Linear":
      self.gate_network = nn.Linear((self.num_experts) + input_features,
                                    self.num_experts + self.output_features)
    elif self.gate_type == "MLP":
      self.gate_network = MLPime(
          input_features + self.num_experts,
          output_features,
          num_experts,
          d_model=d_model,
          n_layers=layers)
    else:
      raise NotImplementedError

  def forward(self, inputs, past_values=None, noise=False):
    """Forward pass for  IMETabular.

    Args:
      inputs: A tensor of shape `(batch_size,input_size)`
      past_values: Expert errors in the last time step
      noise: Boolean determines if noise should be added to input

    Returns:
      expert_outputs: A tensor containing forecast produced by each expert
      with size `(batch_size, num_experts,output_dim)`
      weights: weights assigned to each expert by the assignment module
      reg_out: Regression output the forecast a tensor of `(batch_size,
      output_features)`
    """
    if past_values is None:
      past_values = torch.zeros(
          (inputs.shape[0], self.num_experts)).to(self.device)
    # Concatenate past values with the inputs to create input to gate network
    gate_inputs = torch.cat((inputs, past_values), dim=1)

    if self.gate_type == "Linear":
      gated_ouput = self.gate_network(gate_inputs).unsqueeze(2)
      # The first self.num_experts outputs are prediction for each expert
      assignment_logits = gated_ouput[:, :-self.output_features, :]
      # Last output is the forecast
      reg_out = gated_ouput[:, -self.output_features:, :].squeeze(-1)
    # If gate is MLP
    elif self.gate_type == "MLP":
      # Pass the Concatenated input to the gate network
      assignment_logits, reg_out = self.gate_network(gate_inputs)
    else:
      raise NotImplementedError

    # The weights for each output is retrieved by passing logits through softmax
    weights = F.softmax(assignment_logits, dim=1)
    # If noise flag add noise to the input
    if noise:
      inputs = add_gaussian_noise(inputs, device=self.device)
    # If the interpretable experts are linear pass the input to each expert
    if self.expert_type == "Linear":
      expert_outputs_list = [
          self.experts[i](inputs).unsqueeze(2) for i in range(self.num_experts)
      ]
      # Concatenate the outputs of different experts
      for i in range(len(expert_outputs_list)):
        if i == 0:
          expert_outputs = expert_outputs_list[i]
        else:
          expert_outputs = torch.cat((expert_outputs, expert_outputs_list[i]),
                                     dim=2)
    elif self.expert_type == "SDT":
      panelties = 0
      for i in range(self.num_experts):
        expert_output, panelty = self.experts[i](inputs, is_training_data=True)
        panelties += panelty
        if i == 0:
          expert_outputs = expert_output.unsqueeze(2)
        else:
          expert_outputs = torch.cat(
              (expert_outputs, expert_output.unsqueeze(2)), dim=2)
    else:
      raise NotImplementedError
    if self.expert_type == "Linear":
      return expert_outputs, weights, reg_out
    else:
      return expert_outputs, weights, reg_out, panelties

  def predict(self, original_inputs, past_values=None):
    """Function used during inference time to make predictions.

    Args:
      original_inputs: A tensor of shape `(batch_size, input_size)`
      past_values: Expert errors in the last time step

    Returns:
       output: Forecast a tensor of shape `(batch_size, output_features)`
       expert_outputs: A tensor containing forecast produced by each expert
        with size `(batch_size, num_experts, output_features)`
        argmax_weights: argmax ofweights assigned to each expert by the
        assignment module
    """

    inputs = torch.cat((original_inputs, past_values), dim=1)

    if self.gate_type == "Linear":
      gated_ouput = self.gate_network(inputs).unsqueeze(2)
      assignment_logits = gated_ouput[:, :-self.output_features, :]
    elif self.gate_type == "MLP":
      assignment_logits, _ = self.gate_network(inputs)
    else:

      raise NotImplementedError
    weights = F.softmax(assignment_logits, dim=1)
    max_index = torch.argmax(weights, dim=1).flatten()
    argmax_weights = F.one_hot(
        max_index, num_classes=assignment_logits.shape[1]).float().to(
            self.device).unsqueeze(-1)
    expert_outputs_list = [
        self.experts[i](original_inputs).unsqueeze(2)
        for i in range(self.num_experts)
    ]
    for i in range(len(expert_outputs_list)):
      if i == 0:
        expert_outputs = expert_outputs_list[i]
      else:
        expert_outputs = torch.cat((expert_outputs, expert_outputs_list[i]),
                                   dim=2)

    output = torch.matmul(expert_outputs, argmax_weights).squeeze(-1)

    return output, expert_outputs, argmax_weights
