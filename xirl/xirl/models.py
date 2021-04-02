# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Self supervised models."""

import abc
import math
from typing import List, Union

import dataclasses
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import ResNet
from torchvision.models.utils import load_state_dict_from_url


@dataclasses.dataclass
class SelfSupervisedOutput:
  """The output of a self-supervised model."""

  frames: Union[np.ndarray, torch.FloatTensor]
  feats: Union[np.ndarray, torch.FloatTensor]
  embs: Union[np.ndarray, torch.FloatTensor]

  def squeeze(self, dim):
    kwargs = {}
    for k, v in dataclasses.asdict(self).items():
      kwargs[k] = v.squeeze(dim)
    return self.__class__(**kwargs)

  def cpu(self):
    kwargs = {}
    for k, v in dataclasses.asdict(self).items():
      kwargs[k] = v.cpu()
    return self.__class__(**kwargs)

  def numpy(self):
    kwargs = {}
    for k, v in dataclasses.asdict(self).items():
      if k != "frames":
        kwargs[k] = v.cpu().detach().numpy()
    kwargs["frames"] = self.frames.permute(0, 2, 3, 1).cpu().detach().numpy()
    return self.__class__(**kwargs)

  @classmethod
  def merge(
      cls, output_list):
    kwargs = {}
    for k in dataclasses.asdict(output_list[0]).keys():
      kwargs[k] = torch.cat([getattr(o, k) for o in output_list], dim=1)
    return cls(**kwargs)


class SelfSupervisedModel(abc.ABC, nn.Module):
  """A self-supervised model trained on video data."""

  @abc.abstractmethod
  def __init__(
      self,
      num_ctx_frames,
      normalize_embeddings,
      learnable_temp,
  ):
    super().__init__()

    self.num_ctx_frames = num_ctx_frames
    self.normalize_embeddings = normalize_embeddings
    self.learnable_temp = learnable_temp

    # Log-parameterized multiplicative softmax temperature param.
    if learnable_temp:
      self.logit_scale = nn.Parameter(torch.ones([]))

  def forward(self, x):
    """Forward the video frames through the network.

    Args:
      x: The video frames of shape (B, T, C, H, W). If there are S video frames
        and we are using X context frames, then T = S * X.

    Returns:
      An instance of SelfSupervisedOutput.
    """
    batch_size, t, c, h, w = x.shape
    x_flat = x.view((batch_size * t, c, h, w))
    feats = self.backbone(x_flat)
    feats_flat = torch.flatten(feats, 1)
    embs = self.encoder(feats_flat)
    if self.normalize_embeddings:
      embs = embs / (embs.norm(dim=-1, keepdim=True) + 1e-7)
    if self.learnable_temp:
      logit_scale = self.logit_scale.exp()
      embs = logit_scale * embs
    embs = embs.view((batch_size, t, -1))
    feats = feats.view((batch_size, t, -1))
    return SelfSupervisedOutput(frames=x, feats=feats, embs=embs)

  @torch.no_grad()
  def infer(
      self,
      x,
      max_batch_size = 128,
  ):
    """Forward at inference with possible very large batch sizes."""
    # Figure out a max batch size that's a multiple of the number of context
    # frames. This is so we can support large videos with many frames.
    lcm = self.num_ctx_frames
    effective_bs = math.floor(max_batch_size / lcm) * lcm
    if x.shape[1] > effective_bs:
      out = []
      for i in range(math.ceil(x.shape[1] / effective_bs)):
        sub_frames = x[:, i * effective_bs:(i + 1) * effective_bs]
        out.append(self.forward(sub_frames).cpu())
      out = SelfSupervisedOutput.merge(out)
    else:
      out = self.forward(x).cpu()
    return out.squeeze(0)


class Resnet18LinearEncoderNet(SelfSupervisedModel):
  """A resnet18 backbone with a linear encoder head."""

  def __init__(self, embedding_size, *args, **kwargs):
    super().__init__(*args, **kwargs)

    # Visual backbone.
    resnet = models.resnet18(pretrained=True)
    num_ftrs = resnet.fc.in_features
    layers_ = list(resnet.children())[:-1]
    self.backbone = nn.Sequential(*layers_)

    # Encoder.
    self.encoder = nn.Linear(num_ftrs, embedding_size)


class GoalClassifier(SelfSupervisedModel):
  """A resnet18 backbone with a binary classification head."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    # Visual backbone.
    resnet = models.resnet18(pretrained=True)
    num_ftrs = resnet.fc.in_features
    layers_ = list(resnet.children())[:-1]
    self.backbone = nn.Sequential(*layers_)

    # Classification head.
    self.encoder = nn.Linear(num_ftrs, 1)


class Resnet18RawImageNetFeaturesNet(SelfSupervisedModel):
  """A resnet18 backbone with an identity encoder head."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    # Visual backbone.
    resnet = models.resnet18(pretrained=True)
    layers_ = list(resnet.children())[:-1]
    self.backbone = nn.Sequential(*layers_)

    # Identity encoder.
    self.encoder = nn.Identity()


class Upsampling(nn.Module):
  """Unet upsampling adapted from [1].

  References:
    [1]: https://github.com/milesial/Pytorch-UNet
  """

  def __init__(self, in_channels, out_channels):
    super().__init__()

    self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
        nn.BatchNorm2d(in_channels // 2),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels // 2, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

  def forward(self, x1, x2):
    x1 = self.up(x1)
    diffy = x2.size()[2] - x1.size()[2]
    diffx = x2.size()[3] - x1.size()[3]
    x1 = F.pad(x1,
               [diffx // 2, diffx - diffx // 2, diffy // 2, diffy - diffy // 2])
    x = torch.cat([x2, x1], dim=1)
    return self.conv(x)


@dataclasses.dataclass
class SelfSupervisedReconOutput(SelfSupervisedOutput):
  """Self-supervised output with a reconstruction tensor."""

  reconstruction: Union[np.ndarray, torch.FloatTensor]

  def numpy(self):
    kwargs = {}
    for k, v in dataclasses.asdict(self).items():
      if k != "frames" or k != "reconstruction":
        kwargs[k] = v.cpu().detach().numpy()
    kwargs["frames"] = self.frames.permute(0, 2, 3, 1).cpu().detach().numpy()
    kwargs["reconstruction"] = self.reconstruction.permute(
        0, 2, 3, 1).cpu().detach().numpy()
    return self.__class__(**kwargs)


class Resnet18LinearEncoderAutoEncoderNet(ResNet):
  """Resnet18LinearEncoder with an auxiliary autoencoding path."""

  def __init__(
      self,
      embedding_size,
      num_ctx_frames,
      normalize_embeddings,
      learnable_temp,
  ):
    super().__init__(BasicBlock, [2, 2, 2, 2])

    self.num_ctx_frames = num_ctx_frames
    self.normalize_embeddings = normalize_embeddings
    self.learnable_temp = learnable_temp

    # Load pretrained weights.
    state_dict = load_state_dict_from_url(
        "https://download.pytorch.org/models/resnet18-5c106cde.pth",
        progress=True,
    )
    self.load_state_dict(state_dict)

    # Embedding head.
    self.fc = nn.Linear(self.fc.in_features, embedding_size)

    # Upsampling path.
    self.up1 = Upsampling(1024, 512 // 2)
    self.up2 = Upsampling(512, 256 // 2)
    self.up3 = Upsampling(256, 128 // 2)
    self.up4 = Upsampling(128, 64)
    self.out_conv = nn.Conv2d(64, 3, kernel_size=1)

    # Log-parameterized multiplicative softmax temperature param.
    if learnable_temp:
      self.logit_scale = nn.Parameter(torch.ones([]))

  def encode(self, x):
    # Compute embeddings.
    batch_size, t, c, h, w = x.shape
    x = x.view((batch_size * t, c, h, w))

    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x1 = self.layer1(x)  # B, 64, 56, 56
    x2 = self.layer2(x1)  # B, 128, 28, 28
    x3 = self.layer3(x2)  # B, 256, 14, 14
    x4 = self.layer4(x3)  # B, 512, 7, 7

    # Compute embeddings.
    feats = self.avgpool(x4)  # B, 512, 1, 1
    flat_feats = torch.flatten(feats, 1)
    embs = self.fc(flat_feats)
    if self.normalize_embeddings:
      embs = embs / (embs.norm(dim=-1, keepdim=True) + 1e-7)
    if self.learnable_temp:
      logit_scale = self.logit_scale.exp()
      embs = logit_scale * embs
    embs = embs.view((batch_size, t, -1))

    return embs, [x1, x2, x3, x4, feats]

  def decode_all_res(self, feature_maps):
    """Decode using all spatial resolutions, a la u-net."""
    x1, x2, x3, x4, feats = feature_maps
    x = self.up1(feats, x4)
    x = self.up2(x, x3)
    x = self.up3(x, x2)
    x = self.up4(x, x1)
    recon = self.out_conv(x)
    return recon

  def decode_lowest_res(self, feature_maps):
    _, _, _, x, _ = feature_maps
    for up_conv in self.up_convs:
      x = F.relu(up_conv(x))
      x = F.interpolate(
          x,
          scale_factor=2,
          mode="bilinear",
          recompute_scale_factor=False,
          align_corners=True,
      )
    x = self.out_conv(x)
    return x

  def forward(self, x):
    embs, feature_maps = self.encode(x)
    recon = self.decode_all_res(feature_maps)
    feats = feature_maps[-1]
    feats = feats.view((embs.shape[0], embs.shape[1], *feats.shape[1:]))
    recon = recon.view((embs.shape[0], embs.shape[1], *recon.shape[1:]))
    return SelfSupervisedReconOutput(
        frames=x,
        feats=feats,
        embs=embs,
        reconstruction=recon,
    )

  @torch.no_grad()
  def infer(
      self,
      x,
      max_batch_size = 128,
  ):
    """Forward at inference with possible very large batch sizes."""
    # Figure out a max batch size that's a multiple of the number of context
    # frames. This is so we can support large videos with many frames.
    lcm = self.num_ctx_frames
    effective_bs = math.floor(max_batch_size / lcm) * lcm
    if x.shape[1] > effective_bs:
      out = []
      for i in range(math.ceil(x.shape[1] / effective_bs)):
        sub_frames = x[:, i * effective_bs:(i + 1) * effective_bs]
        out.append(self.forward(sub_frames).cpu())
      out = SelfSupervisedReconOutput.merge(out)
    else:
      out = self.forward(x).cpu()
    return out.squeeze(0)
