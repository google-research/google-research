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

"""Test."""

import argparse
import math
import sys
from my_python_utils.common_utils import imshow
from my_python_utils.common_utils import np
from my_python_utils.common_utils import show_pointcloud
from my_python_utils.common_utils import tonumpy
from my_python_utils.common_utils import totorch
import svox
import svox2
import torch
from util import config_util
from util.dataset import datasets
from util.util import convert_c2w_plenoxel_to_pleonctree

sys.path.append('..')
sys.path.append('../..')


parser = argparse.ArgumentParser()
config_util.define_common_args(parser)

parser.add_argument(
    '--ckpt',
    default='checkpoints/lego_test_low_res_no_viewdir/ckpt.npz',
    type=str,
)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dset_test = datasets[args.dataset_type](
    args.data_dir, split='test', **config_util.build_data_options(args)
)


ALPHA_THRESHOLD = 0.5

grid = svox2.SparseGrid.load(args.ckpt, device=device)


# display debug info:
reso_x, reso_y, reso_z = grid.links.shape
X = torch.arange(0, reso_x, dtype=torch.long, device=device)
Y = torch.arange(0, reso_y, dtype=torch.long, device=device)
Z = torch.arange(0, reso_z, dtype=torch.long, device=device)
X, Y, Z = torch.meshgrid(X, Y, Z)
points = torch.stack([X, Y, Z], dim=-1)

valid_links = grid.links[grid.links >= 0]

valid_colors = grid.sh_data[valid_links.long()]
valid_densities = grid.sh_data[valid_links.long()]
valid_non_zero_densities = valid_densities[:, 0] >= ALPHA_THRESHOLD

grid_valid_points = points[grid.links >= 0].float()

points_with_density = grid_valid_points[valid_non_zero_densities]

c = valid_colors[valid_non_zero_densities]
c = c.clip(-5.0, 5.0)
colors_with_density_normalized = np.array(
    255.0 * tonumpy((c - c.min()) / (c.max() - c.min())), dtype='uint8'
)

show_pointcloud(
    points_with_density, colors_with_density_normalized, title='colors_fitted'
)


img_id = 0
c2w = dset_test.c2w[img_id].to(device=device)
cam = svox2.Camera(
    c2w,
    dset_test.intrins.get('fx', img_id),
    dset_test.intrins.get('fy', img_id),
    dset_test.intrins.get('cx', img_id),
    dset_test.intrins.get('cy', img_id),
    width=dset_test.get_image_size(img_id)[1],
    height=dset_test.get_image_size(img_id)[0],
    ndc_coeffs=dset_test.ndc_coeffs,
)

rgb_pred_test = grid.volume_render_image(cam)
rgb_gt_test = dset_test.gt[img_id]
imshow(rgb_pred_test, title='rgb_pred_from_grid')
imshow(rgb_gt_test, title='rgb_gt')

grid_mse = (rgb_gt_test.cpu() - rgb_pred_test.cpu()) ** 2
mse_num: float = grid_mse.mean().item()
grid_psnr = -10.0 * math.log10(mse_num)


octree_converted = grid.to_svox1()

index_by_corners = False
if index_by_corners:
  curr_reso = grid.links.shape
  X = (
      torch.arange(
          curr_reso[0], dtype=grid.sh_data.dtype, device=grid.links.device
      )
      + 0.5
  ) / curr_reso[0]
  Y = (
      torch.arange(
          curr_reso[1], dtype=grid.sh_data.dtype, device=grid.links.device
      )
      + 0.5
  ) / curr_reso[0]
  Z = (
      torch.arange(
          curr_reso[2], dtype=grid.sh_data.dtype, device=grid.links.device
      )
      + 0.5
  ) / curr_reso[0]
  X, Y, Z = torch.meshgrid(X, Y, Z)
  points_octree = torch.stack((X, Y, Z), dim=-1).view(-1, 3)
  mask = grid.links.view(-1) >= 0
  points_octree = points_octree[mask.to(device=device)]
else:
  points_octree = octree_converted[:].corners
index = svox.LocalIndex(points_octree)
octree_colors = octree_converted[index, :-1].values
octree_alphas = octree_converted[index, -1].values

show_pointcloud(
    points_octree[octree_alphas >= ALPHA_THRESHOLD], title='corners_thresholded'
)

colors = octree_colors.clip(-5, 5)
# normalize by min and max so that it is scaled [0,255] as uint8
colors = (colors - colors.min()) / (colors.max() - colors.min()) * 255
colors = np.array(tonumpy(colors), dtype=np.uint8)
show_pointcloud(
    points_octree[tonumpy(octree_alphas) >= ALPHA_THRESHOLD],
    colors[tonumpy(octree_alphas) >= ALPHA_THRESHOLD],
    title='pcl_octree',
)

octree_loaded = svox.N3Tree.load('tree.npz', map_location=device)

points_octree = octree_loaded[:].corners
index = svox.LocalIndex(points_octree)
octree_colors = octree_loaded[index, :-1].values
octree_alphas = octree_loaded[index, -1].values

show_pointcloud(points_octree, title='corners_thresholded')

focal = dset_test.intrins.get('fx', img_id)
c2w = dset_test.c2w[img_id].to(device=device)
octree_renderer = svox.VolumeRenderer(
    octree_converted, step_size=0.0001, ndc=None
)

# for the converted octree, it uses OpenGL convention on camera but
# the scale is the one used for the plenoxel dataset
c2w_octree = totorch(convert_c2w_plenoxel_to_pleonctree(c2w, scene_scale=1))
rgb_pred_reconstructed_octree = octree_renderer.render_persp(
    c2w_octree,
    fx=focal,
    # cx=dset_test.intrins.get('cx', img_id),
    # cy=dset_test.intrins.get('cy', img_id),
    width=dset_test.get_image_size(img_id)[1],
    height=dset_test.get_image_size(img_id)[0],
)

imshow(rgb_pred_reconstructed_octree, title='rgb_pred_reconstructed_octree')


c2w_octree = totorch(
    convert_c2w_plenoxel_to_pleonctree(c2w, dset_test.scene_scale)
)
octree_renderer = svox.VolumeRenderer(octree_loaded, step_size=0.0001, ndc=None)
rgb_pred_loaded_octree = octree_renderer.render_persp(
    c2w_octree,
    fx=focal,
    # cx=dset_test.intrins.get('cx', img_id),
    # cy=dset_test.intrins.get('cy', img_id),
    width=dset_test.get_image_size(img_id)[1],
    height=dset_test.get_image_size(img_id)[0],
)

imshow(rgb_pred_loaded_octree, title='rgb_pred_loaded_octree')

grid_valid_coordinates = (
    grid_valid_points.cpu() / grid.links.shape[0] * grid.radius * 2
    - grid.center
    - grid.radius / 2
)
colors_a = np.zeros_like(tonumpy(grid_valid_coordinates), dtype='uint8')
colors_b = np.zeros_like(tonumpy(octree_loaded.corners), dtype='uint8')
colors_c = np.zeros_like(tonumpy(octree_converted.corners), dtype='uint8')

colors_a[:, 0] = 255
colors_b[:, 1] = 255
colors_c[:, 2] = 255

# pointclous for converted
show_pointcloud(
    np.concatenate([
        tonumpy(grid_valid_coordinates),
        tonumpy(octree_loaded.corners),
        tonumpy(octree_converted.corners),
    ]),
    np.concatenate([colors_a, colors_b, colors_c]),
)


# DIFFERENCE BETWEEN PLENOXELS AND PLENOCTREES CAMERAS:
# see nerf_dataset.py
# plenoctrees firs c2w # it's the one directly stored:
# tensor([[-1.0000,  0.0000,  0.0000,  0.0000],
#         [ 0.0000, -0.7341,  0.6790,  2.7373],
#         [ 0.0000,  0.6790,  0.7341,  2.9593],
#         [ 0.0000,  0.0000,  0.0000,  1.0000]], device='cuda:0')

# plenoxels transforms with opencv:
# L76
# tensor([[ 1.,  0.,  0.,  0.],
#         [ 0., -1.,  0.,  0.],
#         [ 0.,  0., -1.,  0.],
#         [ 0.,  0.,  0.,  1.]])
#
# and also get's multiplied by scene scale:
# L99 *0.66 for the lego scale
