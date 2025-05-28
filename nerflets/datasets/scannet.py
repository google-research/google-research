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

from kornia import create_meshgrid
from einops import rearrange
from tqdm import tqdm

from .ray_utils import *

SCANNET_FAR = 2.0


@torch.cuda.amp.autocast(dtype=torch.float32)
def get_ray_directions_scannet(H, W, K):
  """Get ray directions for all pixels in camera coordinate [right down front].

    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        H, W: image height and width
        K: (3, 3) camera intrinsics
  """
  grid = create_meshgrid(H, W, normalized_coordinates=False)[0]  # (H, W, 2)
  u, v = grid.unbind(-1)

  fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
  directions = \
      torch.stack([(u-cx+0.5)/fx, -(v-cy+0.5)/fy, -torch.ones_like(u)], -1)

  return directions


class ScanNetDataset(Dataset):

  def __init__(self,
               root_dir,
               split='train',
               img_wh=(1296, 968),
               ref_loc_file='',
               **kwargs):
    super().__init__()
    self.root_dir = root_dir
    self.split = split
    self.img_wh = img_wh
    self.define_transforms()

    self.load_depth = False
    self.load_xyz = False
    self.remove_bg = False
    self.white_back = False

    self.ref_loc_file = ref_loc_file
    self.read_meta()

  def read_meta(self):
    w, h = self.img_wh
    K = np.loadtxt(
        os.path.join(self.root_dir, './intrinsic/intrinsic_color.txt'))[:3, :3]

    K[0] /= (968 / h)
    K[1] /= (1296 / w)

    # bounds, common for all scenes
    self.near, self.far = 0.01, 1.5
    self.bounds = np.array([self.near, self.far])

    # ray directions for all pixels, same for all images (same H, W, focal)
    self.directions = \
        get_ray_directions_scannet(h, w, K)  # (h, w, 3)

    cam_bbox = np.loadtxt(os.path.join(self.root_dir, f'cam_bbox.txt'))
    self.sbbox_scale = (cam_bbox[1] - cam_bbox[0]).max() + 2 * SCANNET_FAR
    self.sbbox_shift = cam_bbox.mean(axis=0)

    if self.ref_loc_file:
      import trimesh
      points = np.array(trimesh.load(self.ref_loc_file).vertices)
      points -= self.sbbox_shift
      points /= self.sbbox_scale
      self.ref_points = points
    else:
      self.ref_points = None

    if self.split == 'train':  # create buffer of all rays and rgb data
      with open(os.path.join(self.root_dir, 'train.txt'), 'r') as f:
        # self.frames = f.read().strip().split()[:10]
        # self.frames = f.read().strip().split()[::5]
        self.frames = f.read().strip().split()[::3]
        # self.frames = f.read().strip().split()[::20]

      self.image_paths = []
      self.poses = []
      self.all_rays = []
      self.all_rgbs = []
      self.all_sems = []

      for frame_id in tqdm(self.frames):
        c2w = np.loadtxt(os.path.join(self.root_dir,
                                      f'pose/{frame_id}.txt'))[:3]
        c2w[0, 3] -= self.sbbox_shift[0]
        c2w[1, 3] -= self.sbbox_shift[1]
        c2w[2, 3] -= self.sbbox_shift[2]
        c2w[:, 3] /= self.sbbox_scale
        c2w[:, 1:3] *= -1

        self.poses += [c2w]
        c2w = torch.FloatTensor(c2w)

        image_path = os.path.join(self.root_dir, f'color/{frame_id}.jpg')
        self.image_paths += [image_path]
        img = Image.open(image_path)
        img = img.resize(self.img_wh, Image.Resampling.LANCZOS)
        img = self.transform(img)  # (3, h, w)
        img = img.reshape(3, -1).permute(1, 0)  # (h*w, 3) RGBA
        self.all_rgbs += [img]

        semantic_path = os.path.join(self.root_dir,
                                     f'instance-filt/{frame_id}.png')
        semantic = Image.open(semantic_path)
        semantic = semantic.resize(self.img_wh, Image.Resampling.NEAREST)
        semantic = torch.tensor(np.array(semantic))  # (h, w)
        semantic = semantic.reshape(-1)  # (h*w)
        self.all_sems += [semantic]

        rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)

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

    elif self.split in ['test', 'val']:
      with open(os.path.join(self.root_dir, f'{self.split}.txt'), 'r') as f:
        self.frames = f.read().strip().split()
    else:
      raise

  def define_transforms(self):
    self.transform = T.ToTensor()

  def __len__(self):
    if self.split == 'train':
      return len(self.all_rays)
    if self.split == 'val':
      return 8  # only validate 8 images (to support <=8 gpus)
    return len(self.frames)

  def __getitem__(self, idx):
    if self.split == 'train':  # use data in the buffers
      sample = {
          'rays': self.all_rays[idx],
          'rgbs': self.all_rgbs[idx],
          'sems': self.all_sems[idx]
      }

    else:  # create data for each image separately
      frame_id = self.frames[idx]

      c2w = np.loadtxt(os.path.join(self.root_dir, f'pose/{frame_id}.txt'))[:3]
      c2w[0, 3] -= self.sbbox_shift[0]
      c2w[1, 3] -= self.sbbox_shift[1]
      c2w[2, 3] -= self.sbbox_shift[2]
      c2w[:, 3] /= self.sbbox_scale
      c2w[:, 1:3] *= -1
      c2w = torch.FloatTensor(c2w)

      image_path = os.path.join(self.root_dir, f'color/{frame_id}.jpg')
      img = Image.open(image_path)
      img = img.resize(self.img_wh, Image.Resampling.LANCZOS)
      img = self.transform(img)  # (3, h, w)
      img = img.reshape(3, -1).permute(1, 0)  # (h*w, 3) RGBA

      semantic_path = os.path.join(self.root_dir,
                                   f'instance-filt/{frame_id}.png')
      semantic = Image.open(semantic_path)
      semantic = semantic.resize(self.img_wh, Image.Resampling.NEAREST)
      semantic = torch.tensor(np.array(semantic)).view(-1)  # (h*w)

      valid_mask = torch.ones_like(img[:, 0])

      rays_o, rays_d = get_rays(self.directions, c2w)  # (H*W, 3) x2
      rays = torch.cat([
          rays_o, rays_d, self.near * torch.ones_like(rays_o[:, :1]),
          self.far * torch.ones_like(rays_o[:, :1])
      ], 1)  # (H*W, 8)

      sample = {
          'rays': rays,
          'rgbs': img,
          'c2w': c2w,
          'sems': semantic,
          'valid_mask': valid_mask
      }

    return sample


if __name__ == '__main__':
  from torch.utils.data import DataLoader

  root_dir = '/zdata/datasets/scannet_expanded/scene0113_00'

  train_dataset = ScanNetDataset(root_dir, 'train')
  train_dataloader = DataLoader(
      train_dataset,
      shuffle=True,
      num_workers=4,
      batch_size=1024,
      pin_memory=True)
  val_dataset = ScanNetDataset(root_dir, 'val')
  val_dataloader = DataLoader(
      val_dataset, shuffle=False, num_workers=4, batch_size=1, pin_memory=True)
  import IPython

  IPython.embed()
