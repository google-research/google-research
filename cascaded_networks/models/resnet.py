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

"""ResNet handler.

  Adapted from
  https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

  Two primary changes from original ResNet code:
  1) Tapped delay line op is added to the output of every residual computation
    - See project.models.layers & project.models.tdl
  2) The timestep is set on the TDL in the forward pass
"""
import functools
import numpy as np
import torch
import torch.nn as nn
from cascaded_networks.models import custom_ops
from cascaded_networks.models import layers as res_layers
from cascaded_networks.models import model_utils


class ResNet(nn.Module):
  """Resnet base class."""

  def __init__(self, name, block, layers, num_classes, **kwargs):
    """Initialize resnet."""
    super(ResNet, self).__init__()
    self.name = name
    self._layers_arch = layers
    self._cascaded = kwargs.get('cascaded', False)
    self._time_bn = kwargs.get('time_bn', self._cascaded)

    # Set up batch norm operation
    self._norm_layer_op = self._setup_bn_op(**kwargs)

    # Head layer
    self.inplanes = 64
    self.layer0 = res_layers.HeadLayer(self.inplanes,
                                       self._norm_layer_op,
                                       **kwargs)

    # Residual Layers
    self.layer1 = self._make_layer(block, 64, layers[0], **kwargs)
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2, **kwargs)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2, **kwargs)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2, **kwargs)
    self.layers = [self.layer1, self.layer2, self.layer3, self.layer4]

    # Final layer
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    final_bias = not kwargs.get('final_bias_off', False)
    self.fc = nn.Linear(512 * block.expansion, num_classes, bias=final_bias)

    # Weight initialization
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, (self._norm_layer, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

  def _setup_bn_op(self, **kwargs):
    if self._cascaded:
      if self._time_bn:
        self._norm_layer = custom_ops.BatchNorm2d

        # Setup batchnorm opts
        self.bn_opts = kwargs['bn_opts']
        self.bn_opts['n_timesteps'] = self.timesteps
        norm_layer_op = functools.partial(self._norm_layer, self.bn_opts)
      else:
        self._norm_layer = nn.BatchNorm2d
        norm_layer_op = self._norm_layer
    else:
      self._norm_layer = nn.BatchNorm2d
      norm_layer_op = self._norm_layer

    return norm_layer_op

  def _make_layer(self, block, planes, blocks, stride=1, **kwargs):
    tdl_mode = kwargs.get('tdl_mode', 'OSD')
    tdl_alpha = kwargs.get('tdl_alpha', 0.0)
    noise_var = kwargs.get('noise_var', 0.0)

    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
          custom_ops.conv1x1(self.inplanes, planes * block.expansion, stride),
      )
    layers = []
    layers.append(
        block(self.inplanes,
              planes,
              stride,
              downsample,
              self._norm_layer_op,
              tdl_alpha=tdl_alpha,
              tdl_mode=tdl_mode,
              noise_var=noise_var,
              cascaded=self._cascaded,
              time_bn=self._time_bn))

    self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
      layers.append(
          block(self.inplanes,
                planes,
                norm_layer=self._norm_layer_op,
                tdl_alpha=tdl_alpha,
                tdl_mode=tdl_mode,
                noise_var=noise_var,
                cascaded=self._cascaded,
                time_bn=self._time_bn))

    return nn.Sequential(*layers)

  @property
  def timesteps(self):
    if self._cascaded:
      n_timesteps = np.sum(self._layers_arch) + 1
    else:
      n_timesteps = 1
    return n_timesteps

  def _set_time(self, t):
    self.layer0.set_time(t)
    for layer in self.layers:
      for block in layer:
        block.set_time(t)

  def forward(self, x, t):
    # Set time on all blocks
    if self._cascaded:
      self._set_time(t)

    # Head layer
    out = self.layer0(x)

    # Res Layers
    for layer in self.layers:
      out = layer(out)

    # Final layer
    out = self.avgpool(out)
    out = torch.flatten(out, 1)

    # Classification
    out = self.fc(out)

    return out


def make_resnet(arch, block, layers, pretrained, **kwargs):
  model = ResNet(arch, block, layers, **kwargs)
  if pretrained:
    model = model_utils.load_model(model, kwargs)
  return model


def resnet18(pretrained=False, **kwargs):
  return make_resnet('resnet18', res_layers.BasicBlock, [2, 2, 2, 2],
                     pretrained, **kwargs)


def resnet34(pretrained=False, **kwargs):
  return make_resnet('resnet34', res_layers.BasicBlock, [3, 4, 6, 3],
                     pretrained, **kwargs)


def resnet50(pretrained=False, **kwargs):
  return make_resnet('resnet50', res_layers.Bottleneck, [3, 4, 6, 3],
                     pretrained, **kwargs)


def resnet101(pretrained=False, **kwargs):
  return make_resnet('resnet101', res_layers.Bottleneck, [3, 4, 23, 3],
                     pretrained, **kwargs)


def resnet152(pretrained=False, **kwargs):
  return make_resnet('resnet152', res_layers.Bottleneck, [3, 8, 36, 3],
                     pretrained, **kwargs)
