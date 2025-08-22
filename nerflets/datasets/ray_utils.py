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
from kornia import create_meshgrid


def get_ray_directions(H, W, focal):
  """Get ray directions for all pixels in camera coordinate.

    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W, focal: image height, width and focal length

    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
  """
  grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
  i, j = grid.unbind(-1)
  # the direction here is without +0.5 pixel centering as calibration is not so accurate
  # see https://github.com/bmild/nerf/issues/24
  directions = \
      torch.stack([(i-W/2)/focal, -(j-H/2)/focal, -torch.ones_like(i)], -1) # (H, W, 3)

  return directions


def get_rays(directions, c2w):
  """Get ray origin and normalized directions in world coordinate for all pixels in one image.

    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world
        coordinate

    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world
        coordinate
  """
  # Rotate ray directions from camera coordinate to the world coordinate
  rays_d = directions @ c2w[:, :3].T  # (H, W, 3)
  rays_d /= torch.norm(rays_d, dim=-1, keepdim=True)
  # The origin of all rays is the camera origin in world coordinate
  rays_o = c2w[:, 3].expand(rays_d.shape)  # (H, W, 3)

  rays_d = rays_d.view(-1, 3)
  rays_o = rays_o.view(-1, 3)

  return rays_o, rays_d


def get_ndc_rays(H, W, focal, near, rays_o, rays_d):
  """Transform rays from world coordinate to NDC.

    NDC: Space such that the canvas is a cube with sides [-1, 1] in each axis.
    For detailed derivation, please see:
    http://www.songho.ca/opengl/gl_projectionmatrix.html
    https://github.com/bmild/nerf/files/4451808/ndc_derivation.pdf

    In practice, use NDC "if and only if" the scene is unbounded (has a large
    depth).
    See https://github.com/bmild/nerf/issues/18

    Inputs:
        H, W, focal: image height, width and focal length
        near: (N_rays) or float, the depths of the near plane
        rays_o: (N_rays, 3), the origin of the rays in world coordinate
        rays_d: (N_rays, 3), the direction of the rays in world coordinate

    Outputs:
        rays_o: (N_rays, 3), the origin of the rays in NDC
        rays_d: (N_rays, 3), the direction of the rays in NDC
  """
  # Shift ray origins to near plane
  t = -(near + rays_o[Ellipsis, 2]) / rays_d[Ellipsis, 2]
  rays_o = rays_o + t[Ellipsis, None] * rays_d

  # Store some intermediate homogeneous results
  ox_oz = rays_o[Ellipsis, 0] / rays_o[Ellipsis, 2]
  oy_oz = rays_o[Ellipsis, 1] / rays_o[Ellipsis, 2]

  # Projection
  o0 = -1. / (W / (2. * focal)) * ox_oz
  o1 = -1. / (H / (2. * focal)) * oy_oz
  o2 = 1. + 2. * near / rays_o[Ellipsis, 2]

  d0 = -1. / (W / (2. * focal)) * (rays_d[Ellipsis, 0] / rays_d[Ellipsis, 2] - ox_oz)
  d1 = -1. / (H / (2. * focal)) * (rays_d[Ellipsis, 1] / rays_d[Ellipsis, 2] - oy_oz)
  d2 = 1 - o2

  rays_o = torch.stack([o0, o1, o2], -1)  # (B, 3)
  rays_d = torch.stack([d0, d1, d2], -1)  # (B, 3)

  return rays_o, rays_d
