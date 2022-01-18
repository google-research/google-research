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

"""Densnet handler.

  Adapted from
  https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
"""
import functools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from cascaded_networks.models import custom_ops
from cascaded_networks.models import dense_blocks
from cascaded_networks.models import model_utils


class DenseNet(nn.Module):
  """Densenet."""

  def __init__(self,
               name,
               block,
               block_arch,
               growth_rate=12,
               reduction=0.5,
               num_classes=10,
               **kwargs):
    """Initialize dense net."""
    super(DenseNet, self).__init__()

    self.name = name
    self.growth_rate = growth_rate
    self._cascaded = kwargs['cascaded']
    self.block_arch = block_arch

    self._norm_layer_op = self._setup_bn_op(**kwargs)

    self._build_net(block, block_arch, growth_rate,
                    reduction, num_classes, **kwargs)

  def _build_net(self,
                 block,
                 block_arch,
                 growth_rate,
                 reduction,
                 num_classes,
                 **kwargs):
    self.layers = []

    num_planes = 2 * growth_rate
    self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

    self.dense1 = self._make_dense_layers(block, num_planes,
                                          block_arch[0], **kwargs)
    num_planes += block_arch[0] * growth_rate
    out_planes = int(np.floor(num_planes * reduction))
    self.trans1 = dense_blocks.Transition(num_planes,
                                          out_planes,
                                          norm_layer=self._norm_layer_op,
                                          **kwargs)
    num_planes = out_planes

    self.dense2 = self._make_dense_layers(block, num_planes,
                                          block_arch[1], **kwargs)
    num_planes += block_arch[1] * growth_rate
    out_planes = int(np.floor(num_planes * reduction))
    self.trans2 = dense_blocks.Transition(num_planes,
                                          out_planes,
                                          norm_layer=self._norm_layer_op,
                                          **kwargs)
    num_planes = out_planes

    self.dense3 = self._make_dense_layers(block, num_planes,
                                          block_arch[2], **kwargs)
    num_planes += block_arch[2] * growth_rate
    out_planes = int(np.floor(num_planes * reduction))
    self.trans3 = dense_blocks.Transition(num_planes,
                                          out_planes,
                                          norm_layer=self._norm_layer_op,
                                          **kwargs)
    num_planes = out_planes

    self.dense4 = self._make_dense_layers(block, num_planes,
                                          block_arch[3], **kwargs)
    num_planes += block_arch[3] * growth_rate

    self.bn = self._norm_layer_op(num_planes)
    self.fc = nn.Linear(num_planes, num_classes)

    self.layers.append(self.trans1)
    self.layers.append(self.trans2)
    self.layers.append(self.trans3)

  def _make_dense_layers(self, block, in_planes, n_blocks, **kwargs):
    layers = []
    for _ in range(n_blocks):
      block_i = block(in_planes,
                      self.growth_rate,
                      norm_layer=self._norm_layer_op,
                      **kwargs)
      self.layers.append(block_i)
      layers.append(block_i)
      in_planes += self.growth_rate
    return nn.Sequential(*layers)

  @property
  def timesteps(self):
    return sum(self.block_arch) + 1

  def _setup_bn_op(self, **kwargs):

    if self._cascaded:
      self._norm_layer = custom_ops.BatchNorm2d

      # Setup batchnorm opts
      self.bn_opts = kwargs.get('bn_opts', {
          'affine': False,
          'standardize': False
      })
      self.bn_opts['n_timesteps'] = self.timesteps
      norm_layer_op = functools.partial(self._norm_layer, self.bn_opts)
    else:
      self._norm_layer = nn.BatchNorm2d
      norm_layer_op = self._norm_layer

    return norm_layer_op

  def _set_time(self, t):
    for block in self.layers:
      block.set_time(t)

  def forward(self, x, t=0):
    # Set time on all blocks
    if self._cascaded:
      self._set_time(t)

    # Feature extraction
    out = self.conv1(x)
    out = self.dense1(out)
    out = self.trans1(out)

    out = self.dense2(out)
    out = self.trans2(out)

    out = self.dense3(out)
    out = self.trans3(out)

    out = self.dense4(out)

    # Classifier
    out = self.bn(out) if not self._cascaded else self.bn(out, t)
    out = F.avg_pool2d(F.relu(out), 4)
    out = out.view(out.size(0), -1)
    out = self.fc(out)
    return out


def make_densenet(name, block, layers, pretrained, growth_rate, **kwargs):
  model = DenseNet(name, block, layers, growth_rate=growth_rate, **kwargs)

  if pretrained:
    kwargs['model_name'] = name
    model = model_utils.load_model(model, kwargs)

  return model


def densenet121(pretrained=False, **kwargs):
  return make_densenet('densenet121', dense_blocks.Bottleneck, [6, 12, 24, 16],
                       pretrained, growth_rate=32, **kwargs)


def densenet161(pretrained=False, **kwargs):
  return make_densenet('densenet161', dense_blocks.Bottleneck, [6, 12, 36, 24],
                       pretrained, growth_rate=48, **kwargs)


def densenet169(pretrained=False, **kwargs):
  return make_densenet('densenet169', dense_blocks.Bottleneck, [6, 12, 32, 32],
                       pretrained, growth_rate=32, **kwargs)


def densenet201(pretrained=False, **kwargs):
  return make_densenet('densenet201', dense_blocks.Bottleneck, [6, 12, 48, 32],
                       pretrained, growth_rate=32, **kwargs)


def densenet_cifar(pretrained=False, **kwargs):
  block_arch = [6, 12, 24, 16]
  growth_rate = 16
  return make_densenet('densenet121_cifar', dense_blocks.Bottleneck, block_arch,
                       pretrained, growth_rate=growth_rate, **kwargs)
