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

"""Optimizer classes."""

import torch
import transformers

AdamW = transformers.AdamW


def get_bert_optim(
    network, lr, weight_decay
):
  """Adamw optimizer for text models."""
  no_decay = ["bias", "LayerNorm.weight"]
  decay_params = []
  no_decay_params = []
  for n, p in network.named_parameters():
    if any(nd in n for nd in no_decay):
      decay_params.append(p)
    else:
      no_decay_params.append(p)

  optimizer_grouped_parameters = [
      {
          "params": decay_params,
          "weight_decay": weight_decay,
      },
      {
          "params": no_decay_params,
          "weight_decay": 0.0,
      },
  ]
  optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)
  return optimizer


def get_sgd_optim(
    network, lr, weight_decay
):
  """SGD optimizer."""
  return torch.optim.SGD(
      network.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9
  )

get_optimizers = {"sgd": get_sgd_optim, "adamw": get_bert_optim}
