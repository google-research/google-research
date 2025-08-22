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

"""A test to validate the kernels work as intended."""
# Unfortunately we have to use pickle, as the inputs are given to us in that
# format and we need to parse them.
import pickle

import numpy as np
from PIL import Image
import torch
import torch.utils.cpp_extension

images2triplanes = torch.utils.cpp_extension.load(
    name='images2triplanes',
    sources='./images2triplanes.cu',
)
points2triplanes = torch.utils.cpp_extension.load(
    name='points2triplanes',
    sources='./points2triplanes.cu',
)


with open('./sample_rgb_point_info_forkyle.pkl', 'rb') as f:
  sample_info = pickle.load(f)

  ## rgb image list
  ## len: 6 (6 views)
  ## every view: H x W x 3
  images = torch.Tensor(
      np.stack(sample_info['rgb_img'])[Ellipsis, ::-1].copy()
  ).cuda()
  images = torch.clip(images, 0, 255.0).type(torch.uint8).contiguous()

  ## intrinsic matrix list
  ## len: 6 (6 views)
  ## every view: 3 x 3
  intrinsics = (
      torch.Tensor(np.stack(sample_info['intrinsic'])).cuda().contiguous()
  )
  print(f'Intrinsics shape: {intrinsics.shape}')

  ## extrinsic matrix list
  ## len: 6 (6 views)
  ## every view: 4 x 4
  # This is weird. It definitely seems to map camera space points
  # to lidar space, but it is named as if it maps lidar points to the camera
  # frame.
  cam2lidar = np.stack(sample_info['extrinsic_lidar2cam'])
  n_cams = 6
  assert cam2lidar.shape[0] == 6
  lidar2cam = []
  for i in range(6):
    lidar2cam.append(np.linalg.inv(cam2lidar[i, :, :]))
  lidar2cam = np.stack(lidar2cam)
  extrinsics = torch.Tensor(lidar2cam).cuda().contiguous()

  ## labeled points list
  ## len: 6 (6 views), actually the 6 labeled points are same
  ## every view: N x 5 (x,y,z)
  points = (
      torch.Tensor(np.stack(sample_info['points']))
      .cuda()[0, :, :3]
      .contiguous()
  )

if False:  # pylint: disable=using-constant-test
  grid_lower_corner = torch.tensor(
      [-8.0, -8.0, -2.0], dtype=torch.float32
  ).cuda()
  voxel_size = torch.tensor([16.0, 16.0, 8.0], dtype=torch.float32).cuda()
  grid_res = torch.tensor([2, 2, 1], dtype=torch.int32).cuda()
  triplane_res = torch.tensor([256, 256], dtype=torch.int32).cuda()

grid_lower_corner = torch.tensor([-8.0, -8.0, -3.0], dtype=torch.float32).cuda()
voxel_size = torch.tensor([1.0, 1.0, 2.0], dtype=torch.float32).cuda()
grid_res = torch.tensor([16, 16, 2], dtype=torch.int32).cuda()
triplane_res = torch.tensor([128, 128], dtype=torch.int32).cuda()

triplanes = images2triplanes.forward(
    points,
    images,
    extrinsics,
    intrinsics,
    grid_lower_corner,
    voxel_size,
    grid_res,
    triplane_res,
)

point_triplanes, _, _ = points2triplanes.forward(
    points, grid_lower_corner, voxel_size, grid_res, triplane_res
)

pi_lookup = ['xy', 'xz', 'yz']
for viz in range(triplanes.shape[0]):
  for viy in range(triplanes.shape[1]):
    for vix in range(triplanes.shape[2]):
      for pi in range(3):
        arr = np.array(triplanes[viz, viy, vix, pi, Ellipsis].cpu())
        points_im = np.array(point_triplanes[viz, viy, vix, pi, Ellipsis].cpu())
        write = False
        for ri in range(triplane_res[0]):
          for ci in range(triplane_res[1]):
            if points_im[ri, ci] > 0:
              arr[ri, ci, 0] = 255
              arr[ri, ci, 1] = 0
              arr[ri, ci, 2] = 0
              write = True
        path = f'./x_{vix}_y_{viy}_z_{viz}_plane_{pi_lookup[pi]}.png'
        if write:
          print(path, arr.shape)
          Image.fromarray(arr).save(path)
