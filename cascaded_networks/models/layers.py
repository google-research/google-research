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

"""Resnet block components."""
import torch.nn as nn
from cascaded_networks.models import custom_ops
from cascaded_networks.models import tdl


class HeadLayer(nn.Module):
  """Head layer of ResNet."""

  def __init__(self, planes, norm_layer, **kwargs):
    """Initialize head layer."""
    super(HeadLayer, self).__init__()
    self.cascaded = kwargs['cascaded']
    self.time_bn = kwargs.get('time_bn', kwargs['cascaded'])

    inplanes = 3

    if kwargs.get('imagenet', False):
      self.conv1 = nn.Conv2d(inplanes,
                             planes,
                             kernel_size=7,
                             stride=2,
                             padding=3,
                             bias=False)
    else:
      self.conv1 = nn.Conv2d(inplanes,
                             planes,
                             kernel_size=3,
                             stride=1,
                             padding=1,
                             bias=False)

    self.bn1 = norm_layer(planes)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    if self.cascaded:
      tdl_mode = kwargs.get('tdl_mode', 'OSD')
      self.tdline = tdl.setup_tdl_kernel(tdl_mode, kwargs)

  def set_time(self, t):
    self.t = t
    if t == 0:
      self.tdline.reset()

  def forward(self, x):
    out = self.conv1(x)
    out = self.bn1(out, self.t) if self.time_bn else self.bn1(out)
    out = self.relu(out)
    out = self.maxpool(out)

    if self.cascaded:
      # Add delay line
      out = self.tdline(out)

    return out


class BasicBlock(nn.Module):
  """Basic resnet block."""
  expansion = 1

  def __init__(self,
               inplanes,
               planes,
               stride=1,
               downsample=None,
               norm_layer=None,
               **kwargs):
    """Initialize basic block."""
    super(BasicBlock, self).__init__()

    self.cascaded = kwargs['cascaded']
    self.time_bn = kwargs.get('time_bn', kwargs['cascaded'])
    self.downsample = downsample
    self.stride = stride

    # Setup ops
    self.conv1 = custom_ops.conv3x3(inplanes, planes, stride)
    self.bn1 = norm_layer(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = custom_ops.conv3x3(planes, planes)
    self.bn2 = norm_layer(planes)

    # TDL
    if self.cascaded:
      tdl_mode = kwargs.get('tdl_mode', 'OSD')
      self.tdline = tdl.setup_tdl_kernel(tdl_mode, kwargs)

  def set_time(self, t):
    self.t = t
    if t == 0:
      self.tdline.reset()

  def _residual_block(self, x):
    # Conv1
    out = self.conv1(x)
    out = self.bn1(out, self.t) if self.time_bn else self.bn1(out)
    out = self.relu(out)

    # Conv2
    out = self.conv2(out)
    out = self.bn2(out, self.t) if self.time_bn else self.bn2(out)

    return out

  def forward(self, x):
    # Identity
    identity = x
    if self.downsample is not None:
      identity = self.downsample(x)

    # Residual
    residual = self._residual_block(x)

    # TDL if cascaded
    if self.cascaded:
      residual = self.tdline(residual)

    # Identity + Residual
    out = residual + identity

    # Nonlinear activation
    out = self.relu(out)

    return out


class Bottleneck(nn.Module):
  """Bottleneck Block."""
  expansion = 4

  def __init__(self,
               inplanes,
               planes,
               stride=1,
               downsample=None,
               norm_layer=None,
               **kwargs):
    """Initialize bottleneck block."""
    super(Bottleneck, self).__init__()
    base_width = 64
    width = int(planes * (base_width / 64.))

    self.downsample = downsample
    self.stride = stride
    self.cascaded = kwargs['cascaded']
    self.time_bn = kwargs.get('time_bn', kwargs['cascaded'])

    self.conv1 = custom_ops.conv1x1(inplanes, width)
    self.bn1 = norm_layer(width)
    self.conv2 = custom_ops.conv3x3(width, width, stride)
    self.bn2 = norm_layer(width)
    self.conv3 = custom_ops.conv1x1(width, planes * self.expansion)
    self.bn3 = norm_layer(planes * self.expansion)
    self.relu = nn.ReLU(inplace=True)

    if self.cascaded:
      tdl_mode = kwargs.get('tdl_mode', 'OSD')
      self.tdline = tdl.setup_tdl_kernel(tdl_mode, kwargs)

  def set_time(self, t):
    self.t = t
    if t == 0:
      self.tdline.reset()

  def _residual_block(self, x):
    # Conv 1
    out = self.conv1(x)
    out = self.bn1(out, self.t) if self.time_bn else self.bn1(out)
    out = self.relu(out)

    # Conv 2
    out = self.conv2(out)
    out = self.bn2(out, self.t) if self.time_bn else self.bn2(out)
    out = self.relu(out)

    # Conv 3
    out = self.conv3(out)
    out = self.bn3(out, self.t) if self.time_bn else self.bn3(out)

    return out

  def forward(self, x):
    # Identity
    identity = x
    if self.downsample is not None:
      identity = self.downsample(x)

    # Residual
    residual = self._residual_block(x)

    # TDL if cascaded
    if self.cascaded:
      residual = self.tdline(residual)

    # Identity + Residual
    out = residual + identity

    # Nonlinear activation
    out = self.relu(out)

    return out
