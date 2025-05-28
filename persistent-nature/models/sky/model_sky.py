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

"""Module for sky model."""
from baukit import renormalize
import clip
from PIL import Image
import torch
from torch_utils import persistence
from torchvision import transforms


@persistence.persistent_class
class ModelSky(torch.nn.Module):
  """wrap sky generator model with a clip feature encoder."""

  def __init__(self, sky_generator):
    """Initialize sky model, consisting of encoder and sky generator.

    Args:
        sky_generator: generator model for the sky image
    """
    super().__init__()
    self.G = G = sky_generator  # pylint: disable=invalid-name
    self.z_dim = G.z_dim
    self.c_dim = G.c_dim
    self.img_channels = G.img_channels
    self.img_resolution = G.img_resolution

    model, _ = clip.load('ViT-B/32')
    self.encoder = model

  def mapping(
      self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False
  ):
    return self.G.mapping(
        z,
        c,
        truncation_psi=truncation_psi,
        truncation_cutoff=truncation_cutoff,
        update_emas=update_emas,
    )

  def synthesis(self, ws, img, acc, multiply=True, **layer_kwargs):
    # img = img * acc # zero out sky for encoder
    features = self.encode(img)
    _, num_layers, _ = ws.shape
    features = features[:, None].repeat(1, num_layers, 1)
    ws = torch.cat([ws, features], dim=2)
    out = self.G.synthesis(ws, **layer_kwargs)
    if not multiply:
      return out
    return out * (1 - acc) + img * acc

  def encode(self, img):
    with torch.no_grad():
      self.encoder.eval()
      img_for_encoder = img.clone()
      # resize and renormalize image [-1, 1]
      # to expected clip input normalization
      img_for_encoder = transforms.functional.resize(
          img_for_encoder, 224, Image.BILINEAR
      )
      img_for_encoder = renormalize.as_tensor(
          img_for_encoder, source='zc', target='pt'
      )
      img_for_encoder = transforms.functional.normalize(
          img_for_encoder,
          mean=(0.48145466, 0.4578275, 0.40821073),
          std=(0.26862954, 0.26130258, 0.27577711),
      )
      features = self.encoder.encode_image(img_for_encoder)
    return features

  def forward(
      self,
      z,
      c,
      img,
      acc,
      truncation_psi=1,
      truncation_cutoff=None,
      update_emas=False,
      **synthesis_kwargs
  ):
    """Generates the sky image.

    Based on sampled latent code, terrain image and sky mask image.

    Args:
        z: torch.Tensor of shape [batch_size, z_dim]
        c: torch.Tensor of shape [batch_size, c_dim] or None, the class
          conditioning input of StyleGAN
        img: torch.Tensor of shape [batch_size, 3, height, width] of the terrain
          image, with the sky pixels set to zero value (gray) when using [-1, 1]
          normalization
        acc: torch.Tensor of shape [batch_size, 1, height, width] meant to
          represent the sky mask (zero for sky, and one for terrain)
        truncation_psi: float, StyleGAN truncation value
        truncation_cutoff: int or None, StyleGAN truncation cutoff
        update_emas: bool, whether layer ema values should be updated, from
          StyleGAN training framework
        **synthesis_kwargs: dictionary of additional arguments to StyleGAN
          layers

    Returns:
        img: torch.Tensor of shape [batch_size, 3, height, width] of the
        sky image predicted to match with the terrain input
    """
    ws = self.mapping(
        z,
        c,
        truncation_psi=truncation_psi,
        truncation_cutoff=truncation_cutoff,
        update_emas=update_emas,
    )
    img = self.synthesis(
        ws, img, acc, update_emas=update_emas, **synthesis_kwargs
    )
    return img
