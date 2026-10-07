# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

# This module contains the models we will use for influence functions.
#

from collections.abc import Sequence
import numpy as np
import torch
from torch import nn

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class LogReg(nn.Module):

  def __init__(self, input_dim, output_dim = 1):
    super(LogReg, self).__init__()
    self.fc = nn.Linear(input_dim, output_dim)

  def forward(self, x):
    x = self.fc(x)
    return x


class DNN(nn.Module):

  def __init__(
      self,
      input_dim,
      m = (8, 8),
      output_dim = 10,
  ):
    super(DNN, self).__init__()
    self.layers = nn.Sequential(
        nn.Linear(input_dim, m[0]),
        nn.ReLU(),
        nn.Linear(m[0], m[1]),
        nn.ReLU(),
        nn.Linear(m[1], output_dim),
    )

  def forward(self, x):
    x = self.layers(x)
    return x


def _reshape_input_to_2d_image(
    x, in_channels, h, w
):
  """Reshapes input tensor to (batch, in_channels, h, w).

  Handles 2D flattened inputs (accounting for HWC flattening in multi-channel
  datasets like CIFAR) as well as 4D HWC/CHW and 3D tensors.
  """
  if x.ndim == 2:
    if in_channels > 1:
      x = x.view(-1, h, w, in_channels).permute(0, 3, 1, 2).contiguous()
    else:
      x = x.view(-1, in_channels, h, w)
  elif x.ndim == 4 and x.shape[1] != in_channels and x.shape[-1] == in_channels:
    x = x.permute(0, 3, 1, 2).contiguous()
  elif x.ndim == 3 and in_channels == 1:
    x = x.unsqueeze(1)
  return x


class CNN(nn.Module):
  """Convolutional Neural Network for 2D image classification."""

  def __init__(
      self,
      input_dim = 784,
      in_channels = None,
      output_dim = 10,
      hidden_channels = 32,
  ):
    super().__init__()
    self.input_dim = input_dim
    if in_channels is None:
      if input_dim == 3072:
        self.in_channels = 3
        self.h, self.w = 32, 32
      else:
        self.in_channels = 1
        side = int(np.sqrt(input_dim))
        self.h = side if side * side == input_dim else 28
        self.w = side if side * side == input_dim else 28
    else:
      self.in_channels = in_channels
      side = int(np.sqrt(input_dim // in_channels))
      self.h, self.w = side, side

    self.conv_layers = nn.Sequential(
        nn.Conv2d(self.in_channels, hidden_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(
            hidden_channels, hidden_channels * 2, kernel_size=3, padding=1
        ),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
    )
    fc_in = (hidden_channels * 2) * (self.h // 4) * (self.w // 4)
    self.fc = nn.Sequential(
        nn.Linear(fc_in, 64),
        nn.ReLU(),
        nn.Linear(64, output_dim),
    )

  def forward(self, x):
    x = _reshape_input_to_2d_image(x, self.in_channels, self.h, self.w)
    out = self.conv_layers(x)
    out = out.view(out.size(0), -1)
    return self.fc(out)


class VisionTransformer(nn.Module):
  """Lightweight Vision Transformer (ViT) for image classification."""

  def __init__(
      self,
      input_dim = 784,
      in_channels = None,
      patch_size = 4,
      embed_dim = 64,
      depth = 2,
      num_heads = 4,
      output_dim = 10,
  ):
    super().__init__()
    self.input_dim = input_dim
    if in_channels is None:
      if input_dim == 3072:
        self.in_channels = 3
        self.h, self.w = 32, 32
      else:
        self.in_channels = 1
        side = int(np.sqrt(input_dim))
        self.h = side if side * side == input_dim else 28
        self.w = side if side * side == input_dim else 28
    else:
      self.in_channels = in_channels
      side = int(np.sqrt(input_dim // in_channels))
      self.h, self.w = side, side

    self.patch_size = patch_size
    num_patches = (self.h // patch_size) * (self.w // patch_size)
    patch_dim = self.in_channels * patch_size * patch_size

    self.patch_embed = nn.Linear(patch_dim, embed_dim)
    self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
    self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

    encoder_layer = nn.TransformerEncoderLayer(
        d_model=embed_dim,
        nhead=num_heads,
        dim_feedforward=embed_dim * 2,
        activation='gelu',
        batch_first=True,
    )
    self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
    self.head = nn.Linear(embed_dim, output_dim)

  def forward(self, x):
    x = _reshape_input_to_2d_image(x, self.in_channels, self.h, self.w)

    b = x.size(0)
    c = self.in_channels
    p = self.patch_size
    patches = x.unfold(2, p, p).unfold(3, p, p)
    patches = patches.contiguous().view(b, c, -1, p, p).permute(0, 2, 1, 3, 4)
    patches = patches.contiguous().view(b, patches.shape[1], -1)

    tokens = self.patch_embed(patches)
    cls_tokens = self.cls_token.expand(b, -1, -1)
    tokens = torch.cat((cls_tokens, tokens), dim=1)
    tokens = tokens + self.pos_embed

    encoded = self.transformer(tokens)
    cls_rep = encoded[:, 0]
    return self.head(cls_rep)


def _make_norm(planes, norm_type = 'batch_norm'):
  """Builds normalization layer based on norm_type for ResNet."""
  norm_type = norm_type.lower()
  if norm_type in ('batch_norm', 'bn', 'batchnorm'):
    return nn.BatchNorm2d(planes)
  elif norm_type in ('batch_norm_no_stats', 'bn_no_stats', 'static_bn'):
    return nn.BatchNorm2d(planes, track_running_stats=False)
  elif norm_type in ('group_norm', 'gn', 'groupnorm'):
    num_groups = min(32, planes)
    while planes % num_groups != 0 and num_groups > 1:
      num_groups -= 1
    return nn.GroupNorm(num_groups, planes)
  elif norm_type in ('layer_norm', 'ln', 'layernorm'):
    return nn.GroupNorm(1, planes)
  elif norm_type in ('none', 'identity'):
    return nn.Identity()
  else:
    raise ValueError(
        f'Unsupported norm_type: {norm_type}. Expected one of ["batch_norm",'
        ' "batch_norm_no_stats", "group_norm", "layer_norm", "none"].'
    )


class BasicBlock(nn.Module):
  """Basic Block for ResNet."""

  expansion: int = 1

  def __init__(
      self,
      in_planes,
      planes,
      stride = 1,
      norm_type = 'batch_norm',
  ):
    super().__init__()
    self.conv1 = nn.Conv2d(
        in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
    )
    self.bn1 = _make_norm(planes, norm_type=norm_type)
    self.relu = nn.ReLU()
    self.conv2 = nn.Conv2d(
        planes, planes, kernel_size=3, stride=1, padding=1, bias=False
    )
    self.bn2 = _make_norm(planes, norm_type=norm_type)

    self.shortcut = nn.Sequential()
    if stride != 1 or in_planes != self.expansion * planes:
      self.shortcut = nn.Sequential(
          nn.Conv2d(
              in_planes,
              self.expansion * planes,
              kernel_size=1,
              stride=stride,
              bias=False,
          ),
          _make_norm(self.expansion * planes, norm_type=norm_type),
      )

  def forward(self, x):
    out = self.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    out = out + self.shortcut(x)
    out = self.relu(out)
    return out


class ResNet(nn.Module):
  """ResNet architecture for 2D image classification."""

  def __init__(
      self,
      block = BasicBlock,
      num_blocks = (2, 2, 2, 2),
      input_dim = 784,
      in_channels = None,
      output_dim = 10,
      norm_type = 'batch_norm',
  ):
    super().__init__()
    self.norm_type = norm_type
    self.input_dim = input_dim
    if in_channels is None:
      if input_dim == 3072:
        self.in_channels = 3
        self.h, self.w = 32, 32
      else:
        self.in_channels = 1
        side = int(np.sqrt(input_dim))
        self.h = side if side * side == input_dim else 28
        self.w = side if side * side == input_dim else 28
    else:
      self.in_channels = in_channels
      side = int(np.sqrt(input_dim // in_channels))
      self.h, self.w = side, side

    self.in_planes = 64
    self.conv1 = nn.Conv2d(
        self.in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    self.bn1 = _make_norm(64, norm_type=norm_type)
    self.relu = nn.ReLU()
    self.layer1 = self._make_layer(block, 64, int(num_blocks[0]), stride=1)
    self.layer2 = self._make_layer(block, 128, int(num_blocks[1]), stride=2)
    self.layer3 = self._make_layer(block, 256, int(num_blocks[2]), stride=2)
    self.layer4 = self._make_layer(block, 512, int(num_blocks[3]), stride=2)
    self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    self.linear = nn.Linear(512 * block.expansion, output_dim)

  def _make_layer(
      self, block, planes, num_blocks, stride
  ):
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for s in strides:
      layers.append(
          block(self.in_planes, planes, s, norm_type=self.norm_type)
      )
      self.in_planes = planes * block.expansion
    return nn.Sequential(*layers)

  def forward(self, x):
    x = _reshape_input_to_2d_image(x, self.in_channels, self.h, self.w)
    out = self.relu(self.bn1(self.conv1(x)))
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = self.avg_pool(out)
    out = out.view(out.size(0), -1)
    out = self.linear(out)
    return out


class ResNet18(ResNet):
  """ResNet-18 model for 2D image classification."""

  def __init__(
      self,
      input_dim = 784,
      in_channels = None,
      output_dim = 10,
      norm_type = 'batch_norm',
  ):
    super().__init__(
        block=BasicBlock,
        num_blocks=(2, 2, 2, 2),
        input_dim=input_dim,
        in_channels=in_channels,
        output_dim=output_dim,
        norm_type=norm_type,
    )


def get_model(
    model_type,
    input_dim,
    output_dim = 10,
    norm_type = 'batch_norm',
):
  """Factory function to instantiate neural network models by name."""
  model_type = model_type.lower()
  if model_type in ('linear', 'logreg'):
    return LogReg(input_dim=input_dim, output_dim=output_dim)
  elif model_type == 'dnn':
    return DNN(input_dim=input_dim, m=[64, 64], output_dim=output_dim)
  elif model_type == 'huge_dnn':
    return DNN(input_dim=input_dim, m=[512, 512], output_dim=output_dim)
  elif model_type == 'small_dnn':
    return DNN(input_dim=input_dim, m=[8, 8], output_dim=output_dim)
  elif model_type == 'cnn':
    return CNN(input_dim=input_dim, output_dim=output_dim)
  elif model_type in ('vit', 'vision_transformer'):
    return VisionTransformer(input_dim=input_dim, output_dim=output_dim)
  elif model_type in ('resnet', 'resnet18', 'resnet-18'):
    return ResNet18(
        input_dim=input_dim, output_dim=output_dim, norm_type=norm_type
    )
  else:
    raise ValueError(
        f'Unsupported model_type: {model_type}. Expected one of ["linear",'
        ' "small_dnn", "dnn", "cnn", "vit", "resnet", "resnet18"].'
    )
