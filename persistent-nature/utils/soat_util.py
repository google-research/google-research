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

# pylint: disable=invalid-name
"""utilities for SOAT stitching."""
import math

import torch
from torch_utils.ops import upfirdn2d

G_soat = None


def init_soat_model(layout_model):
  global G_soat
  G_soat = layout_model.layout_generator
  return G_soat


# map z -> w style code
def prepare_ws(zs, c=None, truncation_psi=1, truncation_cutoff=None):
  return G_soat.mapping(
      zs,
      c=c,
      truncation_psi=truncation_psi,
      truncation_cutoff=truncation_cutoff,
  )


# split w style code into groups
def split_ws(ws):
  ws = ws.to(torch.float32)
  w_idx = 0
  block_ws = []
  for res in G_soat.synthesis.block_resolutions:
    block = getattr(G_soat.synthesis, f'b{res}')
    block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
    w_idx += block.num_conv
  return block_ws


# 1D feature interpolation
def interpolate_features(feature1, feature2):
  alpha = torch.zeros([feature1.size(3)]).to(feature1.device)
  pad = feature1.size(3) // 4
  alpha[-pad:] = 1
  xs = torch.linspace(0, 1, alpha.size(0) - 2 * pad)
  # weights = xs
  weights = torch.cos(xs * math.pi + math.pi) * 0.5 + 0.5
  alpha[pad:-pad] = weights
  alpha = alpha.view(1, 1, 1, -1).expand_as(feature1).to(feature1.device)
  feature = (1 - alpha) * feature1 + alpha * feature2
  return feature


# 2D feature interpolation
def bilinear_interpolate_features(feature00, feature01, feature10, feature11):
  """bilinear interpolation on 4 HxW feature grids."""
  padh = feature00.shape[2] // 4
  padw = feature00.shape[3] // 4

  weights_h_nopad = torch.linspace(0, 1, feature00.shape[2] - 2 * padh)
  weights_w_nopad = torch.linspace(0, 1, feature00.shape[3] - 2 * padw)
  weights_h = torch.zeros(feature00.shape[2])
  weights_w = torch.zeros(feature00.shape[3])
  weights_h[padh:-padh] = weights_h_nopad
  weights_h[-padh:] = 1
  weights_w[padw:-padw] = weights_w_nopad
  weights_w[-padw:] = 1
  weights_h = weights_h[None].view(-1, 1)
  weights_w = weights_w[None]

  alpha00 = ((1 - weights_h) * (1 - weights_w)).to(feature00.device)
  alpha01 = ((1 - weights_h) * weights_w).to(feature01.device)
  alpha10 = (weights_h * (1 - weights_w)).to(feature10.device)
  alpha11 = (weights_h * weights_w).to(feature11.device)

  assert torch.mean(alpha00 + alpha01 + alpha10 + alpha11) == 1
  feature = (
      feature00 * alpha00[None][None]
      + feature01 * alpha01[None][None]
      + feature10 * alpha10[None][None]
      + feature11 * alpha11[None][None]
  )
  return feature


def concat_overlapping(imgs, dim=3, device='cuda'):
  """blend outputs in a 50% overlapping manner."""
  b, c, h, w = imgs[0].shape
  num_imgs = len(imgs)
  if dim == 3:
    pad = w // 4
    full_im = torch.zeros(b, c, h, (num_imgs + 1) * (w // 2), device=device)
    coord = w
    full_im[Ellipsis, :coord] = imgs[0]
    decay = torch.linspace(0, 1, 2 * pad).view(1, 1, 1, -1).to(device)
    for img in imgs[1:]:
      full_im[Ellipsis, coord - 2 * pad : coord] *= 1 - decay
      img_decay = img.clone().to(full_im.device)
      img_decay[Ellipsis, : 2 * pad] *= decay
      full_im[Ellipsis, coord - 2 * pad : coord + 2 * pad] += img_decay
      coord += 2 * pad
  elif dim == 2:
    pad = h // 4
    full_im = torch.zeros(b, c, (num_imgs + 1) * (h // 2), w, device=device)
    coord = h
    full_im[Ellipsis, :coord, :] = imgs[0]
    decay = torch.linspace(0, 1, 2 * pad).view(1, 1, -1, 1).to(device)
    for img in imgs[1:]:
      full_im[Ellipsis, coord - 2 * pad : coord, :] *= 1 - decay
      img_decay = img.clone().to(full_im.device)
      img_decay[Ellipsis, : 2 * pad, :] *= decay
      full_im[Ellipsis, coord - 2 * pad : coord + 2 * pad, :] += img_decay
      coord += 2 * pad
  else:
    assert False
  return full_im


def sample_layout_zs(seed, grid_h=4, grid_w=4, device='cuda'):
  if seed is not None:
    torch.manual_seed(seed)
  zs = torch.randn(grid_h, grid_w, 1, G_soat.z_dim, device=device)
  return zs


# adapted from: https://github.com/mchong6/SOAT/blob/main/model.py#L773
def generate_layout(
    seed,
    grid_h=4,
    grid_w=4,
    device='cuda',
    truncation_psi=1,
    truncation_cutoff=None,
):
  """2D implementation of SOAT for stylegan2."""
  zs = sample_layout_zs(seed, grid_h, grid_w, device)
  frames = []
  mapping_kwargs = dict(
      truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff
  )
  force_fp32 = False
  for h in range(1, grid_h):
    row = []
    for w in range(1, grid_w):
      ws00 = split_ws(prepare_ws(zs[h - 1, w - 1], **mapping_kwargs))
      ws01 = split_ws(prepare_ws(zs[h - 1, w], **mapping_kwargs))
      ws10 = split_ws(prepare_ws(zs[h, w - 1], **mapping_kwargs))
      ws11 = split_ws(prepare_ws(zs[h, w], **mapping_kwargs))
      img = None
      for i, res in enumerate(G_soat.synthesis.block_resolutions):
        block = getattr(G_soat.synthesis, f'b{res}')
        ws00_iter = ws00[i].unbind(dim=1)
        ws01_iter = ws01[i].unbind(dim=1)
        ws10_iter = ws10[i].unbind(dim=1)
        ws11_iter = ws11[i].unbind(dim=1)
        dtype = (
            torch.float16
            if block.use_fp16 and not force_fp32
            else torch.float32
        )
        memory_format = (
            torch.channels_last
            if block.channels_last and not force_fp32
            else torch.contiguous_format
        )
        if block.in_channels == 0:
          out00 = (
              block.const.to(dtype=dtype, memory_format=memory_format)
              .unsqueeze(0)
              .repeat(ws00[i].shape[0], 1, 1, 1)
          )
          out01 = (
              block.const.to(dtype=dtype, memory_format=memory_format)
              .unsqueeze(0)
              .repeat(ws01[i].shape[0], 1, 1, 1)
          )
          out10 = (
              block.const.to(dtype=dtype, memory_format=memory_format)
              .unsqueeze(0)
              .repeat(ws10[i].shape[0], 1, 1, 1)
          )
          out11 = (
              block.const.to(dtype=dtype, memory_format=memory_format)
              .unsqueeze(0)
              .repeat(ws11[i].shape[0], 1, 1, 1)
          )

          out00 = block.conv1(out00, ws00_iter[0])
          out01 = block.conv1(out01, ws01_iter[0])
          out10 = block.conv1(out10, ws10_iter[0])
          out11 = block.conv1(out11, ws11_iter[0])

          # feature
          out0 = torch.cat([out00, out01], dim=3)
          out1 = torch.cat([out10, out11], dim=3)
          out = torch.cat([out0, out1], dim=2)

          # rgb skip connection
          skip00 = block.torgb(out, ws00_iter[1])
          skip01 = block.torgb(out, ws01_iter[1])
          skip10 = block.torgb(out, ws10_iter[1])
          skip11 = block.torgb(out, ws11_iter[1])
          skip = bilinear_interpolate_features(skip00, skip01, skip10, skip11)
          img = img.add_(skip) if img is not None else skip
        else:
          out = out.to(dtype=dtype, memory_format=memory_format)

          out00 = block.conv0(out, ws00_iter[0])
          out01 = block.conv0(out, ws01_iter[0])
          out10 = block.conv0(out, ws10_iter[0])
          out11 = block.conv0(out, ws11_iter[0])
          out = bilinear_interpolate_features(out00, out01, out10, out11)

          out00 = block.conv1(out, ws00_iter[1])
          out01 = block.conv1(out, ws01_iter[1])
          out10 = block.conv1(out, ws10_iter[1])
          out11 = block.conv1(out, ws11_iter[1])
          out = bilinear_interpolate_features(out00, out01, out10, out11)

          skip00 = block.torgb(out, ws00_iter[2])
          skip01 = block.torgb(out, ws01_iter[2])
          skip10 = block.torgb(out, ws10_iter[2])
          skip11 = block.torgb(out, ws11_iter[2])
          skip = bilinear_interpolate_features(skip00, skip01, skip10, skip11)

          if img is not None:
            img = upfirdn2d.upsample2d(img, block.resample_filter)
          img = (
              img.add_(skip.to(dtype=torch.float32))
              if img is not None
              else skip
          )
      row.append(img)
    frames.append(row)
  rows_concat = [concat_overlapping(row, dim=3) for row in frames]
  concat = concat_overlapping(rows_concat, dim=2)
  return concat.to(device)
