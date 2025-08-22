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

# pylint: disable=redefined-outer-name
"""set up dataset for triplane model training."""
import json
import os

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

num_cameras = 50
output_dir = 'dataset/lhq_processed_for_triplane_cam%03d' % num_cameras
reference_dir = 'dataset/lhq_processed'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'img'), exist_ok=True)
pose_input = 'poses/width38.4_far16_noisy_height.pth'

# create mirrored dataset
# (not efficient, but to match the training input of the previous model)
# (I think this step is not required since the pose is randomized, and can use
#  --mirror=True during training instead)
for img_name in tqdm(os.listdir(os.path.join(reference_dir, 'img'))):
  img = Image.open(os.path.join(reference_dir, 'img', img_name))
  img.save(os.path.join(output_dir, 'img', img_name))
  img_mirror = img.transpose(Image.FLIP_LEFT_RIGHT)
  img_mirror.save(
      os.path.join(output_dir, 'img', img_name.replace('.png', '_mirror.png'))
  )

# link depth and segmentation (mirroring will be handled in dataset loader)
depth_reference = os.path.abspath(os.path.join(reference_dir, 'dpt_depth'))
seg_reference = os.path.abspath(os.path.join(reference_dir, 'dpt_sky'))
depth_output = os.path.join(output_dir, 'disp')
seg_output = os.path.join(output_dir, 'seg')
cmd = f'ln -s {depth_reference} {depth_output}'
os.system(cmd)
cmd = f'ln -s {seg_reference} {seg_output}'
os.system(cmd)


def create_dataset_json(output_dir, cam2world_poses, lhq_files, intrinsics):
  """create dataset.json file for training."""
  rng = np.random.RandomState(0)
  dataset_out = {'labels': []}
  # max_images = len(lhq_files)
  for _, filename in enumerate(lhq_files):
    if not filename.endswith('png'):
      continue
    pose_idx = rng.randint(len(cam2world_poses))
    pose = cam2world_poses[pose_idx].squeeze().numpy()
    label = np.concatenate([pose.reshape(-1), intrinsics.reshape(-1)]).tolist()
    dataset_out['labels'].append([os.path.join('img', filename), label])

  with open(os.path.join(output_dir, 'dataset.json'), 'w') as f:
    json.dump(dataset_out, f, indent=4)


# intrinsics matrix: 256x256 image, 60 degree FOV
size = 256
focal = 256 / 2 / np.tan(60 / 2 * np.pi / 180)
intrinsics = np.array(
    [[focal / size, 0, 0.5], [0, focal / size, 0.5], [0, 0, 1]]
)

sampled_Rts = torch.load(pose_input)['Rts'][:, 0, :, :]  # N x 4 x 4
cam2world_poses = sampled_Rts.inverse()
lhq_files = sorted(os.listdir(os.path.join(output_dir, 'img')))
create_dataset_json(
    output_dir, cam2world_poses[:num_cameras], lhq_files, intrinsics
)
