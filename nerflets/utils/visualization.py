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
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image
import matplotlib
import matplotlib.cm
import torch


def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
  """depth: (H, W)"""
  x = depth.cpu().numpy()
  x = np.nan_to_num(x)  # change nan to 0
  mi = np.min(x)  # get minimum depth
  ma = np.max(x)
  x = (x - mi) / max(ma - mi, 1e-8)  # normalize to 0~1
  x = (255 * x).astype(np.uint8)
  x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
  x_ = T.ToTensor()(x_)  # (3, H, W)
  return x_


def make_palette(num_classes):
  """Maps classes to colors in the style of PASCAL VOC.

    Close values are mapped to far colors for segmentation visualization.
    See http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
    Takes:
        num_classes: the number of classes
    Gives:
        palette: the colormap as a k x 3 array of RGB colors
  """
  palette = torch.zeros((num_classes, 3), dtype=torch.uint8)
  for k in range(0, num_classes):
    label = k
    i = 0
    while label:
      palette[k, 0] |= (((label >> 0) & 1) << (7 - i))
      palette[k, 1] |= (((label >> 1) & 1) << (7 - i))
      palette[k, 2] |= (((label >> 2) & 1) << (7 - i))
      label >>= 3
      i += 1
  return palette
