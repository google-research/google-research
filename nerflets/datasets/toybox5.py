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


def blender_quat2rot(quaternion):
  """Convert quaternion to rotation matrix.

    Equivalent to, but support batched case:
    ```python
    rot3x3 = mathutils.Quaternion(quaternion).to_matrix()
    ```
    Args:
      quaternion:

    Returns:
      rotation matrix
  """
  # Note: Blender first cast to double values for numerical precision while
  # we're using float32.
  q = np.sqrt(2) * quaternion

  q0 = q[Ellipsis, 0]
  q1 = q[Ellipsis, 1]
  q2 = q[Ellipsis, 2]
  q3 = q[Ellipsis, 3]

  qda = q0 * q1
  qdb = q0 * q2
  qdc = q0 * q3
  qaa = q1 * q1
  qab = q1 * q2
  qac = q1 * q3
  qbb = q2 * q2
  qbc = q2 * q3
  qcc = q3 * q3

  # Note: idx are inverted as blender and numpy convensions do not
  # match (x, y) -> (y, x)
  rotation = np.empty((*quaternion.shape[:-1], 3, 3), dtype=np.float32)
  rotation[Ellipsis, 0, 0] = 1.0 - qbb - qcc
  rotation[Ellipsis, 1, 0] = qdc + qab
  rotation[Ellipsis, 2, 0] = -qdb + qac

  rotation[Ellipsis, 0, 1] = -qdc + qab
  rotation[Ellipsis, 1, 1] = 1.0 - qaa - qcc
  rotation[Ellipsis, 2, 1] = qda + qbc

  rotation[Ellipsis, 0, 2] = qdb + qac
  rotation[Ellipsis, 1, 2] = -qda + qbc
  rotation[Ellipsis, 2, 2] = 1.0 - qaa - qbb
  return rotation


class Toybox5Dataset(Dataset):

  def __init__(self,
               root_dir,
               split='train',
               img_wh=(256, 256),
               remove_bg=False,
               center_crop=True,
               center_crop_margin_ratio=0.15,
               load_depth=False,
               load_xyz=False):
    super().__init__()
    self.root_dir = root_dir
    self.split = split
    assert img_wh[0] == img_wh[1], 'image width must equal image height!'
    self.img_wh = img_wh
    self.define_transforms()

    self.load_depth = load_depth
    self.load_xyz = load_xyz
    if load_xyz:
      assert load_depth

    self.remove_bg = remove_bg
    # self.white_back = False
    self.white_back = remove_bg
    self.center_crop = center_crop
    self.center_crop_start = int(center_crop_margin_ratio *
                                 self.img_wh[0])  # TODO
    self.read_meta()

  def read_meta(self):
    with open(os.path.join(self.root_dir, f'metadata.json'), 'r') as f:
      self.metadata = json.load(f)

    w, h = self.img_wh
    self.focal = 0.5 * 256 / np.tan(
        0.5 * self.metadata['camera']['field_of_view'])
    self.focal *= self.img_wh[0] / 256

    # bounds, common for all scenes
    # TODO: check this or apply normalization
    if self.remove_bg:
      self.near, self.far = 3.0, 52.0
    else:
      self.near, self.far = 3.9, 20.0
    self.bounds = np.array([self.near, self.far])

    # ray directions for all pixels, same for all images (same H, W, focal)
    self.directions = \
        get_ray_directions(h, w, self.focal) # (h, w, 3)

    self.all_positions = self.metadata['camera']['positions']
    self.all_quaternions = self.metadata['camera']['quaternions']

    if self.split == 'train':  # create buffer of all rays and rgb data
      self.image_paths = []
      self.poses = []
      self.all_rays = []
      self.all_rgbs = []
      self.all_sems = []

      for frame_id in self.metadata['split_ids']['train']:
        pose = np.empty((3, 4))
        pose[:3, :3] = blender_quat2rot(
            np.array(self.all_quaternions[frame_id]))
        pose[:3, 3] = self.all_positions[frame_id]
        self.poses += [pose]
        c2w = torch.FloatTensor(pose)

        image_path = os.path.join(self.root_dir, f'rgba_{frame_id:05d}.png')
        self.image_paths += [image_path]
        img = Image.open(image_path)
        img = img.resize(self.img_wh, Image.Resampling.LANCZOS)
        img = self.transform(img)  # (4, h, w)
        if self.center_crop:
          img = img[:, self.center_crop_start:-self.center_crop_start,
                    self.center_crop_start:-self.center_crop_start]
        img = img.reshape(4, -1).permute(1, 0)  # (h*w, 4) RGBA
        img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
        self.all_rgbs += [img]

        semantic_path = os.path.join(self.root_dir,
                                     f'segmentation_{frame_id:05d}.png')
        semantic = Image.open(semantic_path)
        semantic = semantic.resize(self.img_wh, Image.Resampling.NEAREST)
        semantic = torch.tensor(np.array(semantic))  # (h, w)
        if self.center_crop:
          semantic = semantic[self.center_crop_start:-self.center_crop_start,
                              self.center_crop_start:-self.center_crop_start]
        semantic = semantic.reshape(-1)  # (h*w)
        self.all_sems += [semantic]

        rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
        if self.center_crop:
          rays_o = rays_o.view(*self.img_wh, 3)
          rays_d = rays_d.view(*self.img_wh, 3)

          rays_o = rays_o[self.center_crop_start:-self.center_crop_start,
                          self.center_crop_start:-self.center_crop_start, :]
          rays_d = rays_d[self.center_crop_start:-self.center_crop_start,
                          self.center_crop_start:-self.center_crop_start, :]

          rays_o = rays_o.reshape(-1, 3)
          rays_d = rays_d.reshape(-1, 3)

        self.all_rays += [
            torch.cat([
                rays_o, rays_d, self.near * torch.ones_like(rays_o[:, :1]),
                self.far * torch.ones_like(rays_o[:, :1])
            ], 1)
        ]  # (h*w, 8)

      self.all_rays = torch.cat(self.all_rays,
                                0)  # (len(self.meta['frames])*h*w, 8)
      self.all_rgbs = torch.cat(self.all_rgbs,
                                0)  # (len(self.meta['frames])*h*w, 3)
      self.all_sems = torch.cat(self.all_sems,
                                0)  # (len(self.meta['frames])*h*w,)

      if self.remove_bg:
        self.all_rgbs = torch.where((self.all_sems > 0)[Ellipsis, None],
                                    self.all_rgbs,
                                    torch.ones_like(self.all_rgbs)
                                    # torch.zeros_like(self.all_rgbs)
                                   )
    elif self.split == 'all':
      self.frame_ids = self.metadata['split_ids']['train'] \
                       + self.metadata['split_ids']['test']
    else:
      self.frame_ids = self.metadata['split_ids']['test']

  def define_transforms(self):
    self.transform = T.ToTensor()

  def __len__(self):
    if self.split == 'train':
      return len(self.all_rays)
    if self.split == 'val':
      return 8  # only validate 8 images (to support <=8 gpus)
    return len(self.frame_ids)

  def __getitem__(self, idx):
    if self.split == 'train':  # use data in the buffers
      sample = {
          'rays': self.all_rays[idx],
          'rgbs': self.all_rgbs[idx],
          'sems': self.all_sems[idx]
      }

    else:  # create data for each image separately
      frame_id = self.frame_ids[idx]

      pose = np.empty((3, 4))
      pose[:3, :3] = blender_quat2rot(np.array(self.all_quaternions[frame_id]))
      pose[:3, 3] = self.all_positions[frame_id]
      c2w = torch.FloatTensor(pose)

      image_path = os.path.join(self.root_dir, f'rgba_{frame_id:05d}.png')
      img = Image.open(image_path)
      img = img.resize(self.img_wh, Image.Resampling.LANCZOS)
      img = self.transform(img)  # (4, H, W)
      valid_mask = (img[-1] > 0).flatten()  # (H*W) valid color area
      img = img.view(4, -1).permute(1, 0)  # (H*W, 4) RGBA
      img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB

      semantic_path = os.path.join(self.root_dir,
                                   f'segmentation_{frame_id:05d}.png')
      semantic = Image.open(semantic_path)
      semantic = semantic.resize(self.img_wh, Image.Resampling.NEAREST)
      semantic = torch.tensor(np.array(semantic)).view(-1)  # (h*w)
      if self.remove_bg:
        valid_mask = (semantic > 0).flatten()
        img = torch.where((semantic > 0)[Ellipsis, None], img,
                          torch.ones_like(img)
                          # torch.zeros_like(img)
                         )

      rays_o, rays_d = get_rays(self.directions, c2w)  # (H*W, 3) x2
      rays = torch.cat([
          rays_o, rays_d, self.near * torch.ones_like(rays_o[:, :1]),
          self.far * torch.ones_like(rays_o[:, :1])
      ], 1)  # (H*W, 8)

      sample = {
          'rays': rays,
          'rgbs': img,
          'sems': semantic,
          'c2w': c2w,
          'valid_mask': valid_mask
      }

      if self.load_depth:
        depth_path = os.path.join(self.root_dir, f'depth_{frame_id:05d}.tiff')
        depth = Image.open(depth_path)
        depth = depth.resize(self.img_wh, Image.Resampling.NEAREST)
        depth = torch.tensor(np.array(depth)).view(-1)  # (h*w)
        sample['depths'] = depth

        if self.load_xyz:
          xyz = rays_o + rays_d * depth.unsqueeze(1)
          sample['xyz'] = xyz

    return sample
