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

# pylint: skip-file
from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
from torch import Tensor
import torch.nn as nn


def conv3x3(
    in_planes,
    out_planes,
    stride = 1,
    groups = 1,
    dilation = 1,
):
  """3x3 convolution with padding"""
  return nn.Conv2d(
      in_planes,
      out_planes,
      kernel_size=3,
      stride=stride,
      padding=dilation,
      groups=groups,
      bias=False,
      dilation=dilation,
  )


def conv1x1(in_planes, out_planes, stride = 1):
  """1x1 convolution"""
  return nn.Conv2d(
      in_planes, out_planes, kernel_size=1, stride=stride, bias=False
  )


class BasicBlock(nn.Module):
  expansion: int = 1

  def __init__(
      self,
      inplanes,
      planes,
      stride = 1,
      downsample = None,
      groups = 1,
      base_width = 64,
      dilation = 1,
      norm_layer = None,
  ):
    super().__init__()
    if norm_layer is None:
      norm_layer = nn.BatchNorm2d
    if groups != 1 or base_width != 64:
      raise ValueError("BasicBlock only supports groups=1 and base_width=64")
    if dilation > 1:
      raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
    # Both self.conv1 and self.downsample layers downsample the input when stride != 1
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = norm_layer(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = norm_layer(planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      identity = self.downsample(x)

    out += identity
    out = self.relu(out)

    return out

class Triplane_SimpleConv(nn.Module):

  def __init__(
      self,
      resolution,
      inplanes,
      midplanes,
      outplanes,
      voxel_size,
      channel_stack=True,
      dense_type=None,
  ):
    super().__init__()
    self.resolution_h, self.resolution_w = resolution
    self.inplanes = inplanes
    self.midplanes = midplanes
    self.outplanes = outplanes
    self.voxel_size = voxel_size
    self.dense_type = dense_type
    self.channel_stack = channel_stack

    norm_layer = nn.BatchNorm2d
    downsample = nn.Sequential(
        conv1x1(midplanes, outplanes, stride=2),
        norm_layer(outplanes),
    )
    self.inputconv = nn.Conv2d(
        inplanes, midplanes, kernel_size=3, stride=2, padding=1, bias=False
    )
    self.inputbn = norm_layer(midplanes)
    self.inputrelu = nn.ReLU(inplace=True)
    self.conv1 = BasicBlock(midplanes, midplanes)
    self.conv2 = BasicBlock(
        midplanes, outplanes, stride=2, downsample=downsample
    )
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
      elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

  def forward(self, points, num_points, voxel_coors):
    image = self.points2depth(
        points, image_height=self.resolution_h, image_width=self.resolution_w
    )

    ### extract image feature, (N, outplane)
    x = self.inputrelu(self.inputbn(self.inputconv(image)))
    image_feat = self.avgpool(self.conv2(self.conv1(x)))
    if not self.channel_stack:
      image_feat = image_feat.squeeze().view(
          points.shape[0], 3, image_feat.shape[1]
      )
      image_feat = image_feat.mean(dim=1)
    else:
      image_feat = image_feat.squeeze()
    return image_feat

  def points2depth(self, points, image_height, image_width):
    # normalize point coords to the voxel-wise axis
    voxel_size = torch.Tensor(self.voxel_size).view(1, 1, 3).to(points)
    points_coors_norm = points % voxel_size
    # points_coors_norm = torch.remainder(points, voxel_size)

    # to avoid the bug
    eps = 1e-4
    points_coors_norm = torch.clamp(points_coors_norm, max=(voxel_size - eps))

    def draw_image(
        points_coors_norm,
        idx_select=[0, 1],
        image_x=None,
        image_y=None,
        idx_rest=None,
    ):
      bs = points_coors_norm.shape[0]
      target_img_size = (
          torch.Tensor([image_x, image_y]).view(1, 1, 2).to(points)
      )
      points_coors_m = points_coors_norm[:, :, idx_select]
      points_coors_m = (
          points_coors_m / voxel_size[:, :, idx_select] * target_img_size
      )
      points_coors_m = points_coors_m.long()  ## the coordinates in the matrix

      points_coors_m = (
          points_coors_m[:, :, 0] * image_x + points_coors_m[:, :, 1]
      )  # change to sequence [bs, 64]
      target_matrix = (
          torch.zeros((bs, image_x, image_y)).view(bs, -1).to(points)
      )
      if self.dense_type is None:
        target_matrix.scatter_(1, points_coors_m.long(), 1)
      elif self.dense_type == "direct":
        target_matrix.scatter_(
            1, points_coors_m.long(), points_coors_norm[:, :, idx_rest]
        )
      return target_matrix.view(bs, image_x, image_y)

    # for x-view (x as depth)
    points_coors_x = draw_image(
        points_coors_norm,
        [1, 2],
        image_x=image_width,
        image_y=image_height,
        idx_rest=0,
    )
    points_coors_y = draw_image(
        points_coors_norm,
        [0, 2],
        image_x=image_width,
        image_y=image_height,
        idx_rest=1,
    )
    points_coors_z = draw_image(
        points_coors_norm,
        [0, 1],
        image_x=image_width,
        image_y=image_height,
        idx_rest=2,
    )
    if self.channel_stack:
      points_coors_img = torch.stack(
          [points_coors_x, points_coors_y, points_coors_z], dim=1
      )
    else:
      voxel_num = points_coors_x.shape[0]
      points_coors_img = (
          torch.stack([points_coors_x, points_coors_y, points_coors_z], dim=1)
          .view(3 * voxel_num, image_width, image_height)
          .unsqueeze(dim=1)
      )
    return points_coors_img


class Triplane_DoubleConv(nn.Module):

  def __init__(
      self,
      resolution,
      inplanes,
      midplanes,
      outplanes,
      voxel_size,
      channel_stack=True,
      dense_type=None,
  ):
    super().__init__()
    self.resolution_h, self.resolution_w = resolution
    self.inplanes = inplanes
    self.midplanes = midplanes
    self.outplanes = outplanes
    self.voxel_size = voxel_size
    self.dense_type = dense_type
    self.channel_stack = channel_stack

    norm_layer = nn.BatchNorm2d
    downsample = nn.Sequential(
        conv1x1(midplanes, outplanes, stride=2),
        norm_layer(outplanes),
    )
    self.inputconv1 = nn.Conv2d(
        inplanes, midplanes, kernel_size=3, stride=2, padding=1, bias=False
    )
    self.inputbn1 = norm_layer(midplanes)
    self.inputconv2 = nn.Conv2d(
        inplanes, midplanes, kernel_size=3, stride=2, padding=1, bias=False
    )
    self.inputbn2 = norm_layer(midplanes)
    self.inputrelu = nn.ReLU(inplace=True)
    self.conv1 = BasicBlock(midplanes, midplanes)
    self.conv2 = BasicBlock(
        midplanes, outplanes, stride=2, downsample=downsample
    )
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
      elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

  def forward(self, points, num_points, voxel_coors):
    image = self.points2depth(
        points, image_height=self.resolution_h, image_width=self.resolution_w
    )

    voxel_num = image.shape[0]
    image_xy = image[:, 2, :].unsqueeze(1)
    image_yz_xz = image[:, :2, :].reshape(voxel_num * 2, 1, *image.shape[-2:])
    ### extract image feature, (N, outplane)
    x_xy = self.inputrelu(self.inputbn1(self.inputconv1(image_xy))).unsqueeze(1)
    x_yz_xz = self.inputrelu(self.inputbn2(self.inputconv2(image_yz_xz))).view(
        voxel_num, 2, *x_xy.shape[-3:]
    )
    x = torch.cat([x_xy, x_yz_xz], dim=1)
    x = x.view(voxel_num * 3, *x.shape[2:])

    image_feat = self.avgpool(self.conv2(self.conv1(x)))
    if not self.channel_stack:
      image_feat = image_feat.squeeze().view(
          points.shape[0], 3, image_feat.shape[1]
      )
      image_feat = image_feat.mean(dim=1)
    else:
      image_feat = image_feat.squeeze()
    return image_feat

  def points2depth(self, points, image_height, image_width):
    # normalize point coords to the voxel-wise axis
    voxel_size = torch.Tensor(self.voxel_size).view(1, 1, 3).to(points)
    points_coors_norm = points % voxel_size
    # points_coors_norm = torch.remainder(points, voxel_size)

    # to avoid the bug
    eps = 1e-4
    points_coors_norm = torch.clamp(points_coors_norm, max=(voxel_size - eps))

    def draw_image(
        points_coors_norm,
        idx_select=[0, 1],
        image_x=None,
        image_y=None,
        idx_rest=None,
    ):
      bs = points_coors_norm.shape[0]
      target_img_size = (
          torch.Tensor([image_x, image_y]).view(1, 1, 2).to(points)
      )
      points_coors_m = points_coors_norm[:, :, idx_select]
      points_coors_m = (
          points_coors_m / voxel_size[:, :, idx_select] * target_img_size
      )
      points_coors_m = points_coors_m.long()  ## the coordinates in the matrix

      points_coors_m = (
          points_coors_m[:, :, 0] * image_x + points_coors_m[:, :, 1]
      )  # change to sequence [bs, 64]
      target_matrix = (
          torch.zeros((bs, image_x, image_y)).view(bs, -1).to(points)
      )
      if self.dense_type is None:
        target_matrix.scatter_(1, points_coors_m.long(), 1)
      elif self.dense_type == "direct":
        target_matrix.scatter_(
            1, points_coors_m.long(), points_coors_norm[:, :, idx_rest]
        )
      return target_matrix.view(bs, image_x, image_y)

    # for x-view (x as depth)
    points_coors_x = draw_image(
        points_coors_norm,
        [1, 2],
        image_x=image_width,
        image_y=image_height,
        idx_rest=0,
    )
    points_coors_y = draw_image(
        points_coors_norm,
        [0, 2],
        image_x=image_width,
        image_y=image_height,
        idx_rest=1,
    )
    points_coors_z = draw_image(
        points_coors_norm,
        [0, 1],
        image_x=image_width,
        image_y=image_height,
        idx_rest=2,
    )
    if self.channel_stack:
      points_coors_img = torch.stack(
          [points_coors_x, points_coors_y, points_coors_z], dim=1
      )
    else:
      voxel_num = points_coors_x.shape[0]
      points_coors_img = torch.stack(
          [points_coors_x, points_coors_y, points_coors_z], dim=1
      )
    return points_coors_img


class Triplane_DoubleBlock(nn.Module):

  def __init__(
      self,
      resolution,
      inplanes,
      midplanes,
      outplanes,
      voxel_size,
      channel_stack=True,
      dense_type=None,
  ):
    super().__init__()
    self.resolution_h, self.resolution_w = resolution
    self.inplanes = inplanes
    self.midplanes = midplanes
    self.outplanes = outplanes
    self.voxel_size = voxel_size
    self.dense_type = dense_type
    self.channel_stack = channel_stack

    norm_layer = nn.BatchNorm2d
    downsample1 = nn.Sequential(
        conv1x1(midplanes, outplanes, stride=2),
        norm_layer(outplanes),
    )
    downsample2 = nn.Sequential(
        conv1x1(midplanes, outplanes, stride=2),
        norm_layer(outplanes),
    )
    self.inputconv1 = nn.Conv2d(
        inplanes, midplanes, kernel_size=3, stride=2, padding=1, bias=False
    )
    self.inputbn1 = norm_layer(midplanes)
    self.inputconv2 = nn.Conv2d(
        inplanes, midplanes, kernel_size=3, stride=2, padding=1, bias=False
    )
    self.inputbn2 = norm_layer(midplanes)
    self.inputrelu = nn.ReLU(inplace=True)
    self.conv11 = BasicBlock(midplanes, midplanes)
    self.conv12 = BasicBlock(
        midplanes, outplanes, stride=2, downsample=downsample1
    )
    self.conv21 = BasicBlock(midplanes, midplanes)
    self.conv22 = BasicBlock(
        midplanes, outplanes, stride=2, downsample=downsample2
    )
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
      elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

  def forward(self, points, num_points, voxel_coors):
    image = self.points2depth(
        points, image_height=self.resolution_h, image_width=self.resolution_w
    )

    voxel_num = image.shape[0]
    image_xy = image[:, 2, :].unsqueeze(1)
    image_yz_xz = image[:, :2, :].reshape(voxel_num * 2, 1, *image.shape[-2:])
    ### extract image feature, (N, outplane)
    x_xy = self.inputrelu(self.inputbn1(self.inputconv1(image_xy)))
    x_xy = self.avgpool(self.conv12(self.conv11(x_xy))).unsqueeze(1)

    x_yz_xz = self.inputrelu(self.inputbn2(self.inputconv2(image_yz_xz)))
    x_yz_xz = self.avgpool(self.conv22(self.conv21(x_yz_xz))).view(
        voxel_num, 2, *x_xy.shape[-3:]
    )

    x = torch.cat([x_xy, x_yz_xz], dim=1)
    image_feat = x.view(voxel_num * 3, *x.shape[2:])

    if not self.channel_stack:
      image_feat = image_feat.squeeze().view(
          points.shape[0], 3, image_feat.shape[1]
      )
      image_feat = image_feat.mean(dim=1)
    else:
      image_feat = image_feat.squeeze()
    return image_feat

  def points2depth(self, points, image_height, image_width):
    # normalize point coords to the voxel-wise axis
    voxel_size = torch.Tensor(self.voxel_size).view(1, 1, 3).to(points)
    points_coors_norm = points % voxel_size
    # points_coors_norm = torch.remainder(points, voxel_size)

    # to avoid the bug
    eps = 1e-4
    points_coors_norm = torch.clamp(points_coors_norm, max=(voxel_size - eps))

    def draw_image(
        points_coors_norm,
        idx_select=[0, 1],
        image_x=None,
        image_y=None,
        idx_rest=None,
    ):
      bs = points_coors_norm.shape[0]
      target_img_size = (
          torch.Tensor([image_x, image_y]).view(1, 1, 2).to(points)
      )
      points_coors_m = points_coors_norm[:, :, idx_select]
      points_coors_m = (
          points_coors_m / voxel_size[:, :, idx_select] * target_img_size
      )
      points_coors_m = points_coors_m.long()  ## the coordinates in the matrix

      points_coors_m = (
          points_coors_m[:, :, 0] * image_x + points_coors_m[:, :, 1]
      )  # change to sequence [bs, 64]
      target_matrix = (
          torch.zeros((bs, image_x, image_y)).view(bs, -1).to(points)
      )
      if self.dense_type is None:
        target_matrix.scatter_(1, points_coors_m.long(), 1)
      elif self.dense_type == "direct":
        target_matrix.scatter_(
            1, points_coors_m.long(), points_coors_norm[:, :, idx_rest]
        )
      return target_matrix.view(bs, image_x, image_y)

    # for x-view (x as depth)
    points_coors_x = draw_image(
        points_coors_norm,
        [1, 2],
        image_x=image_width,
        image_y=image_height,
        idx_rest=0,
    )
    points_coors_y = draw_image(
        points_coors_norm,
        [0, 2],
        image_x=image_width,
        image_y=image_height,
        idx_rest=1,
    )
    points_coors_z = draw_image(
        points_coors_norm,
        [0, 1],
        image_x=image_width,
        image_y=image_height,
        idx_rest=2,
    )
    if self.channel_stack:
      points_coors_img = torch.stack(
          [points_coors_x, points_coors_y, points_coors_z], dim=1
      )
    else:
      voxel_num = points_coors_x.shape[0]
      points_coors_img = torch.stack(
          [points_coors_x, points_coors_y, points_coors_z], dim=1
      )
    return points_coors_img

class Triplane_SimpleConv_TPN(nn.Module):

  def __init__(
      self,
      resolution,
      inplanes,
      midplanes,
      outplanes,
      voxel_size,
      channel_stack=True,
      dense_type=None,
  ):
    super().__init__()
    self.resolution_h, self.resolution_w = resolution
    self.inplanes = inplanes
    self.midplanes = midplanes
    self.outplanes = outplanes
    self.voxel_size = voxel_size
    self.dense_type = dense_type
    self.channel_stack = channel_stack

    norm_layer = nn.BatchNorm2d
    downsample = nn.Sequential(
        conv1x1(midplanes, outplanes, stride=2),
        norm_layer(outplanes),
    )
    self.inputconv = nn.Conv2d(
        inplanes, midplanes, kernel_size=3, stride=2, padding=1, bias=False
    )
    self.inputbn = norm_layer(midplanes)
    self.inputrelu = nn.ReLU(inplace=True)
    self.conv1 = BasicBlock(midplanes, midplanes)
    self.conv2 = BasicBlock(
        midplanes, outplanes, stride=2, downsample=downsample
    )
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
      elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

  def forward(self, points, num_points, voxel_coors):
    image, points_norm = self.points2depth(
        points, image_height=self.resolution_h, image_width=self.resolution_w
    )

    ### extract image feature, (N, outplane)
    x1 = self.inputrelu(self.inputbn(self.inputconv(image)))
    x2 = self.conv1(x1)
    x3 = self.conv2(x2)
    image_feat = self.avgpool(x3)
    if not self.channel_stack:
      image_feat = image_feat.squeeze().view(
          points.shape[0], 3, image_feat.shape[1]
      )
      image_feat = image_feat.mean(dim=1)
    else:
      image_feat = image_feat.squeeze()
    return [x1, x2, x3, image_feat], points_norm

  def points2depth(self, points, image_height, image_width):
    # normalize point coords to the voxel-wise axis
    voxel_size = torch.Tensor(self.voxel_size).view(1, 1, 3).to(points)
    points_coors_norm = points % voxel_size
    # points_coors_norm = torch.remainder(points, voxel_size)

    # to avoid the bug
    eps = 1e-4
    points_coors_norm = torch.clamp(points_coors_norm, max=(voxel_size - eps))

    def draw_image(
        points_coors_norm,
        idx_select=[0, 1],
        image_x=None,
        image_y=None,
        idx_rest=None,
    ):
      bs = points_coors_norm.shape[0]
      target_img_size = (
          torch.Tensor([image_x, image_y]).view(1, 1, 2).to(points)
      )
      points_coors_m = points_coors_norm[:, :, idx_select]
      points_coors_m = (
          points_coors_m / voxel_size[:, :, idx_select] * target_img_size
      )
      points_coors_m = points_coors_m.long()  ## the coordinates in the matrix

      points_coors_m = (
          points_coors_m[:, :, 0] * image_x + points_coors_m[:, :, 1]
      )  # change to sequence [bs, 64]
      target_matrix = (
          torch.zeros((bs, image_x, image_y)).view(bs, -1).to(points)
      )
      if self.dense_type is None:
        target_matrix.scatter_(1, points_coors_m.long(), 1)
      elif self.dense_type == "direct":
        target_matrix.scatter_(
            1, points_coors_m.long(), points_coors_norm[:, :, idx_rest]
        )
      return target_matrix.view(bs, image_x, image_y)

    # for x-view (x as depth)
    points_coors_x = draw_image(
        points_coors_norm,
        [1, 2],
        image_x=image_width,
        image_y=image_height,
        idx_rest=0,
    )
    points_coors_y = draw_image(
        points_coors_norm,
        [0, 2],
        image_x=image_width,
        image_y=image_height,
        idx_rest=1,
    )
    points_coors_z = draw_image(
        points_coors_norm,
        [0, 1],
        image_x=image_width,
        image_y=image_height,
        idx_rest=2,
    )
    if self.channel_stack:
      points_coors_img = torch.stack(
          [points_coors_x, points_coors_y, points_coors_z], dim=1
      )
    else:
      voxel_num = points_coors_x.shape[0]
      points_coors_img = (
          torch.stack([points_coors_x, points_coors_y, points_coors_z], dim=1)
          .view(3 * voxel_num, image_width, image_height)
          .unsqueeze(dim=1)
      )
    return points_coors_img, points_coors_norm / voxel_size
