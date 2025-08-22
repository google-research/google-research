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

# Copyright 2022 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
from torch import nn

from utils.ops_test import StackedFcDense
from utils.ops_test import StackedFcSlow


class ShiftedSoftplus(nn.Module):
  """Shifted Softplus.

  Attributes:
    shift:
    softplus:
  """

  def __init__(self, shift=-5., beta=1., threshold=20.):
    super().__init__()
    self.shift = shift
    self.softplus = nn.Softplus(beta=beta, threshold=threshold)

  def forward(self, x):
    return self.softplus(x + self.shift)


class Embedding(nn.Module):
  """Freq Embedding.

  Attributes:
    N_freqs:
    funcs:
    freq_bands:
  """

  def __init__(self, N_freqs, logscale=True):
    """Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)

        in_channels: number of input channels (3 for both xyz and direction)
    """
    super().__init__()
    self.N_freqs = N_freqs
    self.funcs = [torch.sin, torch.cos]

    if logscale:
      self.freq_bands = 2**torch.linspace(0, n_freqs - 1, N_freqs)
    else:
      self.freq_bands = torch.linspace(1, 2**(n_freqs - 1), N_freqs)

  def forward(self, x):
    """Embeds x to (x, sin(2^k x), cos(2^k x), ...)

        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

    Args:
        x: (B, f)

    Returns:
        out: (B, 2*f*N_freqs+f)
    """
    out = [x]
    for freq in self.freq_bands:
      for func in self.funcs:
        out += [func(freq * x)]

    return torch.cat(out, -1)


class Nerflets(nn.Module):
  """Nerflets."""

  def __init__(self,
               n,
               D,
               W,
               in_channels_xyz=63,
               in_channels_dir=27,
               skips=[4],
               with_semantics=False,
               n_classes=6,
               topk=0):
    """n: number of nerflets

        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by
        default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by
        default)
        skips: add skip connection in the Dth layer
    """
    super().__init__()
    self.n = n
    self.D = D
    self.W = W
    self.in_channels_xyz = in_channels_xyz
    self.in_channels_dir = in_channels_dir
    self.skips = skips

    if topk > 0:
      StackedFcLayers = StackedFcSlow
    else:
      StackedFcLayers = StackedFcDense

    # xyz encoding layers
    for i in range(D):
      if i == 0:
        layer = StackedFcLayers(n, topk, in_channels_xyz, W, 'relu')
      elif i in skips:
        layer = StackedFcLayers(n, topk, W + in_channels_xyz, W, 'relu')
      else:
        layer = StackedFcLayers(n, topk, W, W, 'relu')
      setattr(self, f'xyz_encoding_{i+1}', layer)
    self.xyz_encoding_final = StackedFcLayers(n, topk, W, W, 'none')

    # direction encoding layers
    self.dir_encoding = StackedFcLayers(n, topk, W + in_channels_dir, W // 2,
                                        'relu')

    # output layers
    self.sigma = StackedFcLayers(n, topk, W, 1, 'ssoftplus')
    self.rgb = StackedFcLayers(n, topk, W // 2, 3, 'sigmoid')

    # semantics
    self.with_semantics = with_semantics
    self.n_classes = n_classes
    if with_semantics:
      self.sem_logits = nn.Parameter(torch.zeros(n, n_classes))

    self.topk = topk

  def forward(self, x, sigma_only=False, rbfs=None):
    """Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).

        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)
            rbfs [N_n, B, 1]

        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
    """
    if self.topk > 0:
      rbfs = rbfs.squeeze(-1).transpose(0, 1)
      idx = torch.topk(rbfs, self.topk).indices

      # print(rbfs.shape, x.shape, idx.shape)

    else:
      idx = None
      x = x[None, Ellipsis].expand(self.n, -1, -1)

    if not sigma_only:
      input_xyz, input_dir = torch.split(
          x, [self.in_channels_xyz, self.in_channels_dir], dim=-1)
    else:
      input_xyz = x

    xyz_ = input_xyz
    for i in range(self.D):
      if i in self.skips:
        xyz_ = torch.cat([input_xyz, xyz_], -1)
      xyz_ = getattr(self, f'xyz_encoding_{i+1}')(xyz_, idx=idx)

    sigma = self.sigma(xyz_, idx=idx)
    if sigma_only:
      return sigma if self.topk == 0 else sigma.transpose(0, 1)

    xyz_encoding_final = self.xyz_encoding_final(xyz_, idx=idx)

    if self.topk > 0:
      input_dir = input_dir.unsqueeze(1).expand(-1, self.topk, -1)
    dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
    dir_encoding = self.dir_encoding(dir_encoding_input, idx=idx)
    rgb = self.rgb(dir_encoding, idx=idx)

    out = torch.cat([rgb, sigma], -1)

    if self.topk > 0:
      out = out.transpose(0, 1)

    return out  # [n / k, B, 4]


class BgNeRF(nn.Module):

  def __init__(self,
               D,
               W,
               in_channels_dir=27,
               with_semantics=False,
               n_classes=6):
    """D: number of layers

        W: number of hidden units in each layer
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by
        default)
        skips: add skip connection in the Dth layer
    """
    super().__init__()
    self.D = D
    self.W = W
    self.in_channels_dir = in_channels_dir

    # xyz encoding layers
    for i in range(D):
      if i == 0:
        layer = nn.Sequential(nn.Linear(in_channels_dir, W), nn.ReLU())
      else:
        layer = nn.Sequential(nn.Linear(W, W), nn.ReLU())
      setattr(self, f'dir_encoding_{i+1}', layer)

    # output layers
    self.rgb = nn.Sequential(nn.Linear(W, 3, 'sigmoid'), nn.Sigmoid())

    # semantics
    self.with_semantics = with_semantics
    self.n_classes = n_classes
    if with_semantics:
      self.sem = nn.Linear(W, n_classes)

  def forward(self, x):
    """Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).

        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_dir)
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)

        Outputs:
            if with_sem:
                sigma: (B, 3+C) sem and rgb
            else:
                out: (B, 3), rgb
    """
    for i in range(self.D):
      x = getattr(self, f'dir_encoding_{i+1}')(x)

    out = self.rgb(x)

    if self.with_semantics:
      sem = self.sem(x)
      out = torch.cat([out, sem], -1)

    return out  # [B, 3(+C)]


if __name__ == '__main__':
  nerflets = Nerflets(64, 2, 128)
  a = torch.randn((10, 90))
  print(nerflets(a).shape)

  nerflets = Nerflets(64, 2, 128).cuda()
  a = torch.randn((4096, 90)).cuda()
  b = nerflets(a)
  print(b)
  print(b.shape)
  b.sum().backward()

  nerflets = Nerflets(64, 2, 128, 63, 27).cuda()
  a = torch.randn((4096, 63)).cuda()
  b = nerflets(a, sigma_only=True)
  print(b)
  print(b.shape)
  b.sum().backward()

  nerflets = Nerflets(256, 5, 128, 63, 27).cuda()
  a = torch.randn((4096, 90)).cuda()
  b = nerflets(a, sigma_only=False)
  print(b)
  print(b.shape)
  b.sum().backward()

  import IPython
  IPython.embed()
