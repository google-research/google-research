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

"""Model classes for robustness algorithms."""

from typing import Any, Optional, Union
import torch
import torchvision.models


class Identity(torch.nn.Module):
  """Identity layer."""

  def forward(self, x):
    return x


def Conv3x3(
    in_planes,
    out_planes,
    stride = 1,
    groups = 1,
    dilation = 1,
):
  return torch.nn.Conv2d(
      in_planes,
      out_planes,
      kernel_size=3,
      stride=stride,
      padding=dilation,
      groups=groups,
      bias=False,
      dilation=dilation,
  )


class BasicBlock(torch.nn.Module):
  """Resnet Basicblock."""

  expansion = 1

  def __init__(
      self,
      inplanes,
      planes,
      stride = 1,
      downsample = None,
      norm_layer = torch.nn.BatchNorm2d,
      apply_skip = True,
  ):
    super().__init__()
    self.conv1 = Conv3x3(inplanes, planes, stride)
    self.bn1 = norm_layer(planes)
    self.relu = torch.nn.ReLU(inplace=True)
    self.conv2 = Conv3x3(planes, planes)
    self.bn2 = norm_layer(planes)
    self.downsample = downsample
    self.stride = stride
    self.apply_skip = apply_skip

  def forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample:
      identity = self.downsample(x)

    if self.apply_skip:
      out += identity
    out = self.relu(out)

    return out


class PretrainedImageModel(torch.nn.Module):
  """Base class for pretrained image models."""

  def forward(self, x):
    """Encode x into a feature vector of size n_outputs."""
    return self.dropout(self.network(x))

  def train(self, mode = True):
    """Override the default train() to freeze the BN parameters."""
    super().train(mode)
    self.freeze_bn()

  def freeze_bn(self):
    """Freeze the BN parameters."""
    for m in self.network.modules():
      if isinstance(m, torch.nn.BatchNorm2d):
        m.eval()


class ResNet(PretrainedImageModel):
  """ResNet."""

  def __init__(
      self,
      input_shape,
      hparams,
      pretrained = True,
      freeze_bn = False,
  ):
    super().__init__()
    if hparams.resnet18:
      self.network = torchvision.models.resnet18(pretrained=pretrained)
      self.n_outputs = 512
    else:
      self.network = torchvision.models.resnet50(pretrained=pretrained)
      self.n_outputs = 2048

    # adapt number of channels
    nc = input_shape[0]
    if nc != 3:
      tmp = self.network.conv1.weight.data.clone()

      self.network.conv1 = torch.nn.Conv2d(
          nc, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
      )
      for i in range(nc):
        self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

    # save memory
    del self.network.fc
    self.network.fc = Identity()
    self.hparams = hparams
    self.dropout = torch.nn.Dropout(hparams.last_layer_dropout)

    if freeze_bn:
      self.freeze_bn()
    else:
      assert hparams.last_layer_dropout == 0.0

  def get_features(self, x):
    aux = [None]
    x = self.network.conv1(x)
    x = self.network.bn1(x)
    x = self.network.relu(x)
    x = self.network.maxpool(x)

    x = self.network.layer1(x)
    aux.append(x)
    x = self.network.layer2(x)
    aux.append(x)
    x = self.network.layer3(x)
    aux.append(x)
    x = self.network.layer4(x)
    aux.append(x)
    x = self.network.avgpool(x)
    return aux, x


class MetaNet(PretrainedImageModel):
  """MetaNet for ReVaR."""

  def __init__(
      self,
      input_shape,
      hparams,
      pretrained = True,
      n_outputs = 1,
  ):
    super().__init__()
    self.network = torchvision.models.resnet18(pretrained=pretrained)
    self.n_outputs = 512

    # adapt number of channels
    nc = input_shape[0]
    if nc != 3:
      tmp = self.network.conv1.weight.data.clone()

      self.network.conv1 = torch.nn.Conv2d(
          nc, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
      )
      for i in range(nc):
        self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

    # save memory
    del self.network.fc
    self.network.fc = torch.nn.Linear(512, n_outputs)
    self.hparams = hparams

  def forward(self, x):
    """Encode x into a feature vector of size n_outputs."""
    return torch.sigmoid(self.network(x))


class ConvAux(torch.nn.Module):
  """Convolutional auxiliary layer for SiFeR."""

  def __init__(
      self, num_classes, hparams, block = BasicBlock
  ):
    super().__init__()
    if hparams['resnet18']:
      channel_list = [64, 128, 256, 512]
    else:
      channel_list = [64 * 4, 128 * 4, 256 * 4, 512 * 4]
    self.norm_layer = torch.nn.BatchNorm2d
    self.depth = hparams.aux_depth
    self.width = hparams.aux_width
    if self.depth == 0:
      self.layers = None
      self.fc = torch.nn.Linear(channel_list[hparams.aux_pos - 1], num_classes)
    else:
      self.layers = self._make_layer(
          block, channel_list[hparams.aux_pos - 1], self.width, self.depth
      )
      self.fc = torch.nn.Linear(self.width * block.expansion, num_classes)
    self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

  def _make_layer(
      self, block, in_channels, width, blocks
  ):
    layers = []
    layers.append(
        block(in_channels, width, norm_layer=self.norm_layer, apply_skip=False)
    )
    self.inplanes = width * block.expansion
    for _ in range(1, blocks):
      layers.append(block(self.inplanes, width, norm_layer=self.norm_layer))
    return torch.nn.Sequential(*layers)

  def forward(self, x):
    if self.depth:
      x = self.layers(x)
    out = self.avgpool(x)
    out = self.fc(out.squeeze())
    return out


class AuxFeaturizer(torch.nn.Module):
  """Applies auxiliary layer to a featurizer."""

  def __init__(
      self,
      featurizer,
      classifier,
      num_classes,
      hparams,
      aux_pos = 0,
      last_block_dropout = 0,
  ):
    super().__init__()
    self.featurizer = featurizer
    self.classifier = classifier
    self.aux_pos = aux_pos
    if self.aux_pos:
      self.aux_layer = ConvAux(num_classes, hparams)
    else:
      self.aux_layer = None
    self.hparams = hparams
    self.dropout = torch.nn.Dropout(hparams.last_layer_dropout)

  def forward(
      self, x
  ):
    aux = None
    aux_list, x = self.featurizer.network.get_features(x)
    if self.aux_pos:
      assert self.aux_pos <= len(aux_list)
      aux = aux_list[self.aux_pos]
    x = self.dropout(torch.flatten(x, 1))
    x = self.classifier(x)

    if aux is not None:
      aux = self.aux_layer(aux)
      return aux, x
    return x


def Featurizer(
    data_type, input_shape, hparams
):
  """Auto-select an appropriate featurizer for the given data type & input shape."""
  if data_type == 'images' and input_shape[1:3] == (224, 224):
    if hparams.image_arch == 'resnet_sup_in1k':
      return ResNet(input_shape, hparams, hparams.pretrained)
    else:
      raise NotImplementedError
  else:
    raise NotImplementedError


def Classifier(
    in_features, out_features, is_nonlinear = False
):
  """Classification layer."""
  if is_nonlinear:
    return torch.nn.Sequential(
        torch.nn.Linear(in_features, in_features // 2),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features // 2, in_features // 4),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features // 4, out_features),
    )
  else:
    return torch.nn.Linear(in_features, out_features)
