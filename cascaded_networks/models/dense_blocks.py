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

"""Densenet block components."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from cascaded_networks.models import tdl


class Bottleneck(nn.Module):
  """Bottleneck layer for DenseNet."""

  def __init__(self, in_planes, growth_rate, norm_layer, **kwargs):
    """Initialize bottleneck layer."""
    super().__init__()

    mid_plane = 4 * growth_rate
    self._cascaded = kwargs['cascaded']

    self.relu = nn.ReLU()

    self.bn1 = norm_layer(in_planes)
    self.conv1 = nn.Conv2d(in_planes, mid_plane, kernel_size=1,
                           stride=1, padding=0, bias=False)

    self.bn2 = norm_layer(mid_plane)
    self.conv2 = nn.Conv2d(mid_plane, growth_rate, kernel_size=3,
                           stride=1, padding=1, bias=False)

    if self._cascaded:
      tdl_mode = kwargs.get('tdl_mode', 'OSD')
      self.tdline = tdl.setup_TDL(tdl_mode, growth_rate, kwargs)

  def set_time(self, t):
    self.t = t
    if t == 0:
      self.tdline.reset()

  def forward(self, x):
    out = self.bn1(x) if not self._cascaded else self.bn1(x, self.t)
    out = self.conv1(self.relu(out))

    out = self.bn2(out) if not self._cascaded else self.bn2(out, self.t)
    out = self.conv2(self.relu(out))

    # Tapped Delay Line
    if self._cascaded:
      out = self.tdline(out)

    out = torch.cat([out, x], 1)
    return out


class Transition(nn.Module):
  """Transition layer for DenseNet."""

  def __init__(self, in_planes, out_planes, norm_layer, **kwargs):
    """Initialize transition layer."""
    super().__init__()

    self._cascaded = kwargs['cascaded']

    self.bn = norm_layer(in_planes)
    self.relu = nn.ReLU()
    self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1,
                          stride=1, padding=0, bias=False)

  def set_time(self, t):
    self.t = t

  def forward(self, x):
    out = self.bn(x) if not self._cascaded else self.bn(x, self.t)
    out = self.conv(self.relu(out))
    out = F.avg_pool2d(out, 2)
    return out
