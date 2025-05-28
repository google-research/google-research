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

"""File containing pose utilities."""
import numpy as np


def recenter_poses(poses, cam_transform=None):
  """Function to recenter poses. Adopted from JaxNerf."""
  poses_ = poses.copy()
  bottom = np.reshape([0, 0, 0, 1.], [1, 4])
  c2w = poses_avg(poses)
  c2w = np.concatenate([c2w[:3, :4], bottom], -2)
  bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
  poses = np.concatenate([poses[:, :3, :4], bottom], -2)
  poses = np.linalg.inv(c2w) @ poses

  if cam_transform is not None:
    poses = poses @ cam_transform

  poses_[:, :3, :4] = poses[:, :3, :4]
  poses = poses_
  return poses


def poses_avg(poses):
  """Function to compute the average poses. Adopted from JaxNerf."""
  hwf = poses[0, :3, -1:]
  center = poses[:, :3, 3].mean(0)
  vec2 = normalize(poses[:, :3, 2].sum(0))
  up = poses[:, :3, 1].sum(0)
  c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
  return c2w


def viewmatrix(z, up, pos):
  """Construct lookat view matrix. Adopted from JaxNerf."""
  vec2 = normalize(z)
  vec1_avg = up
  vec0 = normalize(np.cross(vec1_avg, vec2))
  vec1 = normalize(np.cross(vec2, vec0))
  m = np.stack([vec0, vec1, vec2, pos], 1)
  return m


def viewmatrix_ren(z, up, pos):
  """Construct lookat view matrix. Adopted from JaxNerf."""
  vec2 = normalize(z)
  vec1_avg = up
  vec0 = normalize(np.cross(vec1_avg, vec2))
  vec1 = normalize(np.cross(vec2, vec0))
  m = np.stack([vec0, vec1, vec2, pos], 1)
  return m


def normalize(x):
  return x / np.linalg.norm(x)


def generate_spiral_poses(poses, bds, cam_transform):
  """Generate a spiral path for renderin. Adopted from JaxNerf."""
  c2w = poses_avg(poses)
  # Get average pose.
  up = normalize(poses[:, :3, 1].sum(0))
  # Find a reasonable "focus depth" for this dataset.
  close_depth, inf_depth = bds.min() * .9, bds.max() * 5.
  dt = .75
  mean_dz = 1. / (((1. - dt) / close_depth + dt / inf_depth))
  focal = mean_dz
  # Get radii for spiral path.
  tt = poses[:, :3, 3]
  rads = np.percentile(np.abs(tt), 90, 0)
  c2w_path = c2w
  n_views = 120
  n_rots = 2
  # Generate poses for spiral path.
  render_poses = []
  rads = np.array(list(rads) + [1.])
  hwf = c2w_path[:, 4:5]
  zrate = .5
  for theta in np.linspace(0., 2. * np.pi * n_rots, n_views + 1)[:-1]:
    c = np.dot(
        c2w[:3, :4],
        (np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) *
         rads))
    z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
    render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))

  render_poses = np.array(render_poses).astype(np.float32)[:, :3, :4]
  render_poses = render_poses @ cam_transform
  return render_poses
