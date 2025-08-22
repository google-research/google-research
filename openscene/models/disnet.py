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
"""Distillation model."""

from models.unet_3d import mink_unet as model3D
from torch import nn


def constructor3d(**kwargs):
  """Construct the 3D model."""
  return model3D(**kwargs)


class DisNet(nn.Module):
  """Define the 3D point encoder."""

  def __init__(self, cfg=None):
    super().__init__()
    if not hasattr(cfg, 'feat_2d'):
      cfg.feat_2d = 'lseg'
    feat_type = cfg.feat_2d

    if 'lseg' in feat_type:
      last_dim = 512
    elif 'openseg' in feat_type:
      last_dim = 640
    elif 'osegclip' in feat_type:
      last_dim = 768
    elif 'mseg' in feat_type:
      last_dim = 0
    else:
      raise NotImplementedError

    # MinkowskiNet for 3D point clouds
    net3d = constructor3d(in_channels=3, out_channels=last_dim, d=3)
    self.net3d = net3d

  def forward(self, sparse_3d):
    """Forward pass."""
    return self.net3d(sparse_3d)
