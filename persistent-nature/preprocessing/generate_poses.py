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

"""Generate training cameras.

randomly sample camera origin
then compute valid rotation to overlap with layout
"""
import os

from external.gsn.models.nerf_utils import get_sample_points
import matplotlib.pyplot as plt
import numpy as np
import torch
from utils import camera_util

plane_width = 256 * 0.15  # 38.4
nerf_far = 16
ymax = 0.5
seed = 0
rng = np.random.RandomState(seed)

sampled_Rts = []
sampled_cameras = []
for i in range(1000):
  if i % 100 == 0:
    print(i)

  Tx = rng.rand() * plane_width - plane_width / 2
  Tz = rng.rand() * plane_width - plane_width / 2
  Ty = rng.randn() * ymax / 3

  valid_degrees = []
  # find the rotations that are valid
  for degree in range(360):
    # compute world2cam matrix
    camera = camera_util.Camera(Tx, Ty, Tz, degree, 0.0)
    Rt = camera_util.pose_from_camera(camera)[None]
    # convert to cam2world matrix
    # (used FOV=90, to reproduce previous pose distribution)
    xyz, viewdirs, zvals, rd, ro = get_sample_points(
        tform_cam2world=Rt.inverse(),
        F=(16, -16),
        H=1,
        W=32,
        samples_per_ray=2,
        near=0,
        far=nerf_far / 2,
        perturb=False,
        mask=None,
    )
    if np.all(np.abs(xyz).numpy() < plane_width / 2):
      valid_degrees.append(degree)
  # sample from the valid degrees
  degree = rng.choice(valid_degrees) + (rng.rand() - 0.5)
  camera = camera_util.Camera(Tx, Ty, Tz, degree, 0.0)
  Rt = camera_util.pose_from_camera(camera)[None]

  # store world2cam transformation
  sampled_Rts.append(Rt)
  sampled_cameras.append(camera)

f, ax = plt.subplots()
for i in np.random.choice(len(sampled_Rts), 500):
  Rt = sampled_Rts[i]
  xyz, viewdirs, zvals, rd, ro = get_sample_points(
      tform_cam2world=Rt.inverse(),
      F=(16, -16),
      H=1,
      W=1,
      samples_per_ray=64,
      near=0,
      far=8,
      perturb=False,
      mask=None,
  )
  ax.scatter(xyz[0, 0, 0, 0], xyz[0, 0, 0, 2])
  ax.arrow(
      xyz[0, 0, 0, 0],
      xyz[0, 0, 0, 2],
      xyz[0, 0, 20, 0] - xyz[0, 0, 0, 0],
      xyz[0, 0, 20, 2] - xyz[0, 0, 0, 2],
  )
ax.set_xlim([-plane_width / 2, plane_width / 2])
ax.set_ylim([-plane_width / 2, plane_width / 2])
ax.set_aspect('equal', adjustable='box')
f.savefig('preprocessing/poses.jpg')

# save with noisy camera heights
os.makedirs('poses', exist_ok=True)
torch.save(
    {'Rts': torch.stack(sampled_Rts), 'cameras': sampled_cameras},
    f'./poses/width{plane_width}_far{nerf_far}_noisy_height.pth',
)
