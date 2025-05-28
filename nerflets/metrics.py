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
from kornia.losses import ssim as dssim


def mse(image_pred, image_gt, valid_mask=None, reduction='mean'):
  value = (image_pred - image_gt)**2
  if valid_mask is not None:
    value = value[valid_mask]
  if reduction == 'mean':
    return torch.mean(value)
  return value


def psnr(image_pred, image_gt, valid_mask=None, reduction='mean'):
  return -10 * torch.log10(mse(image_pred, image_gt, valid_mask, reduction))


def ssim(image_pred, image_gt, reduction='mean'):
  """image_pred and image_gt: (1, 3, H, W)"""
  dssim_ = dssim(image_pred, image_gt, 3, reduction)  # dissimilarity in [0, 1]
  return 1 - 2 * dssim_  # in [-1, 1]
