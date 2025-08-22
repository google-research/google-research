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
# -*- coding: utf-8 -*-
"""CoMoDGAN class definition.
"""
import collections
import random

from models.networks.model import ConvLayer
from models.networks.model import EqualLinear
from models.networks.model import PixelNorm
from models.networks.model import StyledConv
from models.networks.model import ToRGB

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class MappingNet(nn.Module):
  """Mapping Network."""

  def __init__(self,
               latent=512,
               num_latent=512,
               dlatent_broadcast=None,
               num_layers=8,
               num_features=512,
               lr_mul=0.01,
               normalize_latent=True,
               **kwargs):
    super().__init__()
    layers = []

    if normalize_latent:
      layers.append(
          ('Normalize', PixelNorm()))

    dim_in = latent
    for layer_idx in range(num_layers):
      fmaps = num_latent if layer_idx == num_layers - 1 else num_features
      layers.append(('Dense%d' % layer_idx,
                     EqualLinear(
                         dim_in,
                         fmaps,
                         lr_mul=lr_mul,
                         activation='fused_lrelu')))
      dim_in = fmaps

    self.G_mapping = nn.Sequential(collections.OrderedDict(layers))

  def forward(
      self,
      latents_input):
    code = self.G_mapping(latents_input)
    return code


def compute_features(f_base, f_decay, stage, f_min, f_max):
  return np.clip(
      int(f_base / (2.0**(stage * f_decay))), f_min, f_max)


class StyleEncoder(nn.Module):
  """Style Encoder."""

  def __init__(self,
               resolution=128,
               num_channels=4,
               f_base=4096,
               f_decay=1.0,
               f_min=1,
               f_max=512,
               dp_rate=0.5):
    super().__init__()

    class InputEncoder(nn.Module):
      """Encoder from RGB input."""

      def __init__(self, res):
        super().__init__()
        self.FromRGB = ConvLayer(
            num_channels,
            compute_features(f_base, f_decay, res-1, f_min, f_max),
            1,
            blur_kernel=[1, 3, 3, 1],
            activate=True)

      def forward(self, x):
        t = self.FromRGB(x)
        return t

    class EncoderBlock(nn.Module):
      """Encoder block."""

      def __init__(self, res):
        super().__init__()
        self.Conv0 = ConvLayer(
            compute_features(f_base, f_decay, res-1, f_min, f_max),
            compute_features(f_base, f_decay, res-1, f_min, f_max),
            kernel_size=3,
            activate=True)
        self.Conv1_down = ConvLayer(
            compute_features(f_base, f_decay, res-1, f_min, f_max),
            compute_features(f_base, f_decay, res-2, f_min, f_max),
            kernel_size=3,
            downsample=True,
            blur_kernel=[1, 3, 3, 1],
            activate=True)
        self.res = res

      def forward(self, x):
        y = self.Conv0(x)
        y = self.Conv1_down(y)
        return y

    class EncoderFinal(nn.Module):
      """Encoder block final."""

      def __init__(self):
        super().__init__()
        self.Conv = ConvLayer(
            compute_features(f_base, f_decay, 2, f_min, f_max),
            compute_features(f_base, f_decay, 1, f_min, f_max),
            kernel_size=3,
            activate=True)
        self.Dense0 = EqualLinear(
            compute_features(f_base, f_decay, 1, f_min, f_max) * 4 * 4,
            compute_features(f_base, f_decay, 1, f_min, f_max),
            activation='fused_lrelu')
        self.dropout = nn.Dropout(dp_rate)

      def forward(self, x):
        x = self.Conv(x)
        bsize = x.size(0)
        x = x.view(bsize, -1)
        x = self.Dense0(x)
        x = self.dropout(x)
        return x

    resolution_log2 = int(np.log2(resolution))

    Es = []
    for res in range(resolution_log2, 2, -1):
      if res == resolution_log2:
        Es.append(('%dx%d_0' % (2**res, 2**res), InputEncoder(res)))
      Es.append(('%dx%d' % (2**res, 2**res), EncoderBlock(res)))
    Es.append(('4x4', EncoderFinal()))
    self.E = nn.Sequential(collections.OrderedDict(Es))

  def forward(self, x):
    # from [0, 1] -> [-1, 1]
    x_global = self.E(x)
    return x_global


class FeatureEncoder(nn.Module):
  """Feature encoder."""

  def __init__(self,
               resolution=128,
               num_channels=4,
               f_base=8192,
               f_decay=1.0,
               f_min=1,
               f_max=512,
               dp_rate=0.5):
    super().__init__()

    class InputEncoder(nn.Module):
      """Encoder from RGB."""

      def __init__(self, res):
        super().__init__()
        self.FromRGB = ConvLayer(
            num_channels + 1,  # RGBDA
            compute_features(f_base, f_decay, res - 1, f_min, f_max),
            1,
            blur_kernel=[1, 3, 3, 1],
            activate=True)

      def forward(self, x):
        y, encoder_feature = x
        t = self.FromRGB(y)
        return t, encoder_feature

    class EncoderBlock(nn.Module):
      """Encoder block."""

      def __init__(self, res):
        super().__init__()

        self.Conv0 = ConvLayer(
            compute_features(f_base, f_decay, res-1, f_min, f_max),
            compute_features(f_base, f_decay, res-1, f_min, f_max),
            kernel_size=3,
            activate=True)
        self.Conv1_down = ConvLayer(
            compute_features(f_base, f_decay, res-1, f_min, f_max),
            compute_features(f_base, f_decay, res-2, f_min, f_max),
            kernel_size=3,
            downsample=True,
            blur_kernel=[1, 3, 3, 1],
            activate=True)
        self.res = res

      def forward(self, input_data):
        x, encoder_feature = input_data
        x = self.Conv0(x)
        encoder_feature[self.res] = x
        x = self.Conv1_down(x)
        return x, encoder_feature

    class EncoderFinal(nn.Module):
      """Encoder block final."""

      def __init__(self):
        super().__init__()
        self.Conv = ConvLayer(
            compute_features(f_base, f_decay, 2, f_min, f_max),
            compute_features(f_base, f_decay, 1, f_min, f_max),
            kernel_size=3,
            activate=True)
        self.dropout = nn.Dropout(dp_rate)

      def forward(self, data):
        x, encoder_feature = data
        x = self.Conv(x)
        encoder_feature[2] = x
        return encoder_feature

    resolution_log2 = int(np.log2(resolution))

    # Main layers.
    Es = []
    for res in range(resolution_log2, 2, -1):
      if res == resolution_log2:
        Es.append(('%dx%d_0' % (2**res, 2**res), InputEncoder(res)))
      Es.append(('%dx%d' % (2**res, 2**res), EncoderBlock(res)))
    # Final layers.
    Es.append(('4x4', EncoderFinal()))
    self.E = nn.Sequential(collections.OrderedDict(Es))

  def forward(self, x):
    # from [0, 1] -> [-1, 1]
    encoder_feature = self.E(x)
    return encoder_feature


def get_modulation_vector(latent, x_global):
  mod_vector = []
  mod_vector.append(latent)
  mod_vector.append(x_global)
  mod_vector = torch.cat(mod_vector, 1)
  return mod_vector


class SynthesisNetwork(nn.Module):
  """Synthesis Network."""

  def __init__(self,
               num_latent=512,
               num_channels=5,
               resolution=128,
               f_base=8 << 10,
               f_decay=1.0,
               f_min=1,
               f_max=512,
               randomize_noise=True,
               architecture='skip',
               nonlinearity='lrelu',
               fused_modconv=True,
               dp_rate=0.5,
               noise_injection=True,
               **kwargs):
    super().__init__()

    resolution_log2 = int(np.log2(resolution))
    self.num_layers = resolution_log2 * 2 - 2
    self.resolution_log2 = resolution_log2

    mod_size = 0
    mod_size += num_latent
    mod_size += compute_features(f_base, f_decay, 1, f_min, f_max)

    class BlockMiddle(nn.Module):
      """Middle block."""

      def __init__(self, res):
        super().__init__()
        self.res = res
        self.Conv0_up = StyledConv(
            compute_features(f_base, f_decay, res-2, f_min, f_max),
            compute_features(f_base, f_decay, res-1, f_min, f_max),
            kernel_size=3,
            style_dim=mod_size,
            upsample=True,
            blur_kernel=[1, 3, 3, 1])
        self.Conv1 = StyledConv(
            compute_features(f_base, f_decay, res-1, f_min, f_max),
            compute_features(f_base, f_decay, res-1, f_min, f_max),
            kernel_size=3,
            style_dim=mod_size,
            upsample=False)
        self.ToRGB = ToRGB(
            compute_features(f_base, f_decay, res-1, f_min, f_max),
            mod_size,
            output_channel=4)

      def forward(self, x, y, dlatents_in, x_global, encoder_feature):
        x_skip = encoder_feature[self.res]
        mod_vector = get_modulation_vector(dlatents_in[:, self.res * 2 - 5],
                                           x_global)
        x = self.Conv0_up(x, mod_vector)
        x = x + x_skip
        mod_vector = get_modulation_vector(dlatents_in[:, self.res * 2 - 4],
                                           x_global)
        x = self.Conv1(x, mod_vector)
        mod_vector = get_modulation_vector(dlatents_in[:, self.res * 2 - 3],
                                           x_global)
        y = self.ToRGB(x, mod_vector, skip=y)
        return x, y

    class BlockFirst(nn.Module):
      """Block 0."""

      def __init__(self):
        super().__init__()

        self.Conv = StyledConv(
            compute_features(f_base, f_decay, 1, f_min, f_max),
            compute_features(f_base, f_decay, 1, f_min, f_max),
            kernel_size=3,
            style_dim=mod_size)
        self.ToRGB = ToRGB(
            compute_features(f_base, f_decay, 1, f_min, f_max),
            style_dim=mod_size,
            output_channel=4,
            upsample=False)

      def forward(self, x, dlatents_in, x_global):
        mod_vector = get_modulation_vector(dlatents_in[:, 0], x_global)
        x = self.Conv(x, mod_vector)
        mod_vector = get_modulation_vector(dlatents_in[:, 1], x_global)
        y = self.ToRGB(x, mod_vector)
        return x, y

    self.G_4x4 = BlockFirst()
    for res in range(3, resolution_log2 + 1):
      setattr(self, 'G_%dx%d' % (2**res, 2**res), BlockMiddle(res))

  def forward(self, dlatents_in, x_global, encoder_feature):
    x = encoder_feature[2]

    x, y = self.G_4x4(x, dlatents_in, x_global)

    for res in range(3, self.resolution_log2 + 1):
      block = getattr(self, 'G_%dx%d' % (2**res, 2**res))
      x, y = block(x, y, dlatents_in, x_global, encoder_feature)

    # convert [-1, 1] back [0, 1]
    images_out = (F.tanh(y) + 1.) / 2.
    return images_out


class Generator(nn.Module):
  """CoModGAN Generator."""

  def __init__(
      self,
      **kwargs):
    super().__init__()
    self.G_mapping = MappingNet(**kwargs)
    self.style_encoder = StyleEncoder(**kwargs)
    self.feature_encoder = FeatureEncoder(**kwargs)
    self.G_synthesis = SynthesisNetwork(**kwargs)
    self.num_layers = self.G_synthesis.num_layers

  def mean_latent(self, n_latent, device):
    latent_in = torch.randn(n_latent, 512, device=device)
    latent = self.G_mapping(latent_in).mean(0, keepdim=True)
    return latent

  def forward(
      self,
      encoder_feature,
      global_code,
      latents_in,
      inject_index=None,
      truncation=1,
      truncation_mean=None,
      ):

    assert isinstance(latents_in, list)

    dlatents_in = [self.G_mapping(s) for s in latents_in]

    if truncation is not None and truncation < 0.999:
      dlatents_t = []
      for style in dlatents_in:
        dlatents_t.append(truncation_mean + truncation *
                          (style - truncation_mean))
      dlatents_in = dlatents_t

    self.dlatents_in = dlatents_in

    if len(dlatents_in) < 2:
      inject_index = self.num_layers
      if dlatents_in[0].ndim < 3:
        num_latent = dlatents_in[0].unsqueeze(1).repeat(1, inject_index, 1)
      else:
        num_latent = dlatents_in[0]
    else:
      if inject_index is None:
        inject_index = random.randint(1, self.num_layers - 1)
      num_latent = dlatents_in[0].unsqueeze(1).repeat(1, inject_index, 1)
      dlatent2 = dlatents_in[1].unsqueeze(1).repeat(
          1, self.num_layers - inject_index, 1)
      num_latent = torch.cat([num_latent, dlatent2], 1)

    output = self.G_synthesis(num_latent, global_code, encoder_feature)
    return output

