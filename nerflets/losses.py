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

# Copyright 2022 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from torch import nn


class ColorLoss(nn.Module):

  def __init__(self, coef=1.):
    super().__init__()
    self.coef = coef
    self.loss = nn.MSELoss(reduction='mean')

  def forward(self, inputs, targets):
    loss = self.loss(inputs['rgb_coarse'], targets)
    if 'rgb_fine' in inputs:
      loss += self.loss(inputs['rgb_fine'], targets)

    return self.coef * loss


class CoverageLoss(nn.Module):

  def __init__(self, coef=0.001):
    super().__init__()
    self.coef = coef

  def forward(self, inputs):
    loss = inputs['coverage_pen_coarse'].mean()
    if 'coverage_pen_fine' in inputs:
      loss += inputs['coverage_pen_fine'].mean()
    return self.coef * loss


class SemanticsLoss(nn.Module):

  def __init__(self, coef=1.):
    super().__init__()
    self.coef = coef
    self.loss = nn.CrossEntropyLoss(reduction='mean')

  def forward(self, inputs, targets):
    loss = self.loss(inputs['sem_logits_coarse'], targets)
    if 'sem_logits_fine' in inputs:
      loss += self.loss(inputs['sem_logits_fine'], targets)

    return self.coef * loss


loss_dict = {
    'color': ColorLoss,
    'coverage': CoverageLoss,
    'semantics': SemanticsLoss
}
