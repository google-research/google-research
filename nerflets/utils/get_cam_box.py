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
import os
import json
import numpy as np
import torch

from datasets.ray_utils import get_ray_directions
from datasets.ray_utils import get_rays


def get_kitti360_boxes(root_dir):
  H = 376
  W = 1408
  NEAR = 2.0
  FAR = 122.0

  with open(os.path.join(root_dir, f'scene.json'), 'r') as f:
    meta = json.load(f)
  focal = 0.5 * W / np.tan(0.5 * float(meta['camera_angle_x']))

  # ray directions for all pixels, same for all images (same H, W, focal)
  directions = get_ray_directions(H, W, focal)  # (h, w, 3)

  all_xyz_minmax = []
  all_camxyz = []

  for frame in meta['frames']:
    pose = np.array(frame['transform_matrix'])[:3, :4]
    c2w = torch.FloatTensor(pose)

    all_camxyz += [c2w[:, 3]]

    rays_o, rays_d = get_rays(directions, c2w)  # both (h*w, 3)

    xyz_minmax = torch.cat((rays_o + rays_d * NEAR, rays_o + rays_d * FAR),
                           0)  # (2*h*w, 3)
    all_xyz_minmax += [xyz_minmax]

  all_xyz_minmax = torch.cat(all_xyz_minmax, 0)
  box_min = all_xyz_minmax.min(dim=0)[0]
  box_max = all_xyz_minmax.max(dim=0)[0]

  all_camxyz = torch.stack(all_camxyz, 0)
  cam_min = all_camxyz.min(dim=0)[0]
  cam_max = all_camxyz.max(dim=0)[0]

  return torch.stack((box_min, box_max)), torch.stack((cam_min, cam_max)),
