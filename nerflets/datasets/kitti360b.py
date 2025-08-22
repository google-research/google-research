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
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T

from .ray_utils import *

KITTI360B_TESTIDX = [2, 23, 42, 61]


class Kitti360BDataset(Dataset):

  def __init__(self, root_dir, split='train', img_wh=(1408, 376)):
    self.root_dir = root_dir
    self.split = split
    self.img_wh = img_wh
    self.define_transforms()

    self.read_meta()
    self.white_back = False

  def read_meta(self):
    with open(os.path.join(self.root_dir, f'scene.json'), 'r') as f:
      self.meta = json.load(f)

    w, h = self.img_wh
    self.focal = 0.5 * 1408 / np.tan(0.5 * float(self.meta['camera_angle_x']))
    self.focal *= w / 1408
    # assert w == 1408 and h == 376

    if 'scene_box' in self.meta:
      self.scene_box = np.array(self.meta['scene_box'])

    # bounds, common for all scenes
    self.near = 2.0
    self.far = 122.0
    self.bounds = np.array([self.near, self.far])

    # ray directions for all pixels, same for all images (same H, W, focal)
    self.directions = \
        get_ray_directions(h, w, self.focal) # (h, w, 3)

    if self.split == 'train':  # create buffer of all rays and rgb data
      self.image_paths = []
      self.poses = []
      self.all_rays = []
      self.all_rgbs = []
      for idx, frame in enumerate(self.meta['frames']):
        if idx in KITTI360B_TESTIDX:
          continue
        pose = np.array(frame['transform_matrix'])[:3, :4]
        self.poses += [pose]
        c2w = torch.FloatTensor(pose)

        image_path = os.path.join(self.root_dir, f"{frame['file_path']}.jpg")
        self.image_paths += [image_path]
        img = Image.open(image_path)
        img = img.resize(self.img_wh, Image.Resampling.LANCZOS)
        img = self.transform(img)  # (3, h, w)
        img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGB
        self.all_rgbs += [img]

        rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
        rays = torch.cat([
            rays_o, rays_d, self.near * torch.ones_like(rays_o[:, :1]),
            self.far * torch.ones_like(rays_o[:, :1])
        ], 1)

        self.all_rays += [rays]  # (h*w, 8)

      self.all_rays = torch.cat(self.all_rays,
                                0)  # (len(self.meta['frames])*h*w, 8)
      self.all_rgbs = torch.cat(self.all_rgbs,
                                0)  # (len(self.meta['frames])*h*w, 3)

  def define_transforms(self):
    self.transform = T.ToTensor()

  def __len__(self):
    if self.split == 'train':
      return len(self.all_rays)
    if self.split == 'val':
      return len(KITTI360B_TESTIDX)
    if self.split == 'test':
      return len(KITTI360B_TESTIDX)

  def __getitem__(self, idx):
    if self.split == 'train':  # use data in the buffers
      sample = {'rays': self.all_rays[idx], 'rgbs': self.all_rgbs[idx]}

    else:  # create data for each image separately
      frame = self.meta['frames'][KITTI360B_TESTIDX[idx]]
      c2w = torch.FloatTensor(frame['transform_matrix'])[:3, :4]

      img = Image.open(os.path.join(self.root_dir, f"{frame['file_path']}.jpg"))
      img = img.resize(self.img_wh, Image.Resampling.LANCZOS)
      img = self.transform(img)  # (3, H, W)
      img = img.view(3, -1).permute(1, 0)  # (H*W, 3) RGB
      valid_mask = torch.ones_like(img[:, 0])

      rays_o, rays_d = get_rays(self.directions, c2w)
      rays = torch.cat([
          rays_o, rays_d, self.near * torch.ones_like(rays_o[:, :1]),
          self.far * torch.ones_like(rays_o[:, :1])
      ], 1)  # (H*W, 8)

      sample = {
          'rays': rays,
          'rgbs': img,
          'c2w': c2w,
          'sems': [],
          'valid_mask': valid_mask
      }

    return sample
