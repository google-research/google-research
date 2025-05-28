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

"""Module to map structural information to memory embeddings."""
import torch


class StructMapper(torch.nn.Module):

  def __init__(self, structural_feature_dim, hidden_dim, memory_emb_dim):
    super().__init__()

    self.layer1 = torch.torch.nn.Linear(structural_feature_dim, hidden_dim)
    self.sigmoid = torch.torch.nn.Sigmoid()
    self.layer2 = torch.torch.nn.Linear(hidden_dim, memory_emb_dim)

  def forward(self, x):
    out = self.layer1(x)
    out = self.sigmoid(out)
    out = self.layer2(out)
    return out
