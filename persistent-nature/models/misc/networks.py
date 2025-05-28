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

"""Miscellaneous network components."""
from external.gsn.models.layers import EqualConv2d
import torch
from torch import nn
from torch_utils import persistence

# -------------------------------------------------------------------------------------
# additional network components


@persistence.persistent_class
class ToRGBTexture(nn.Module):
  """RGB projection layer."""

  def __init__(self, in_channel, **kwargs):
    super().__init__()
    self.toRGB = EqualConv2d(  # pylint: disable=invalid-name
        in_channel, 3, kernel_size=1, stride=1, padding=0, bias=True
    )

  def forward(self, x):
    rgb = self.toRGB(x)
    # rgb = torch.sigmoid(rgb)
    rgb = torch.tanh(rgb / 2)
    return rgb


@persistence.persistent_class
class WrappedDiscriminator(torch.nn.Module):
  """Module to wrap two discriminators."""

  def __init__(self, D_img, D_acc):  # pylint: disable=invalid-name
    super().__init__()
    self.D_img = D_img  # pylint: disable=invalid-name
    self.D_acc = D_acc  # pylint: disable=invalid-name
    self.img_resolution = self.D_img.img_resolution
    self.img_channels = self.D_img.img_channels + self.D_acc.img_channels
    self.recon = self.D_img.recon
    assert self.recon  # should use reconstruction D

  def forward(self, img, c, **kwargs):
    rgbd = img[:, :-1]
    acc = img[:, [-1]]
    x, recon = self.D_img(rgbd, c, **kwargs)
    y, recon_acc = self.D_acc(acc, c, **kwargs)
    return (x + y) / 2, torch.cat([recon, recon_acc], dim=1)
