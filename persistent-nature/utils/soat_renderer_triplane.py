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

# pylint: disable=g-multiple-import,invalid-name
"""render from triplane model.

sample the appropriate features from SOAT stitched xz plane, and sequence of
stiched xy and yz planes
"""
from external.eg3d.training.volumetric_rendering.ray_marcher import MipRayMarcher2
from external.eg3d.training.volumetric_rendering.renderer import ImportanceRenderer, generate_planes, project_onto_planes
import torch


def sample_from_planes_soat(
    plane_axes,
    plane_features,
    coordinates,
    base_resolution,
    mode='bilinear',
    padding_mode='reflection',
    box_warp=None,
):
  """sample feature inputs from all planes given xyz coordinates."""
  # plane features is [xy_list, xz, yz_list]
  assert padding_mode == 'reflection'
  xy_planes = plane_features[0]
  xz_plane = plane_features[1]
  yz_planes = plane_features[2]

  output_features_xy = []
  for plane_feature in xy_planes:
    _, _, H, W = plane_feature.shape  # N = batch size
    # _, M, _ = coordinates.shape  # M = render_h * render_w * num_samples
    normalized_coordinates = (2 / box_warp) * coordinates  # normalize [-1, 1]
    normalized_coordinates[:, :, 0] *= base_resolution / W  # X
    normalized_coordinates[:, :, 1] *= base_resolution / H  # Y
    # first plane: N x 1 x M x 2
    projected_coordinates = project_onto_planes(
        plane_axes[[0]], normalized_coordinates
    ).unsqueeze(1)
    output_features_xy.append(
        torch.nn.functional.grid_sample(
            plane_feature,
            projected_coordinates.float(),
            mode=mode,
            padding_mode=padding_mode,
            align_corners=False,
        ).permute(0, 3, 2, 1)[:, :, 0, :]
        # output of grid sample: N x C x 1 x M
        # output of permute: N x M x 1 x C
        # final output = N x M x C
    )

  # output_feature_xz
  plane_feature = xz_plane
  _, _, H, W = plane_feature.shape  # N = batch size
  # _, M, _ = coordinates.shape  # M = render_h * render_w * num_samples
  normalized_coordinates = (2 / box_warp) * coordinates  # normalize [-1, 1]
  normalized_coordinates[:, :, 0] *= base_resolution / W  # X
  normalized_coordinates[:, :, 2] *= base_resolution / H  # Z
  # second plane: N x 1 x M x 2
  projected_coordinates = project_onto_planes(
      plane_axes[[1]], normalized_coordinates
  ).unsqueeze(1)
  output_features_xz = torch.nn.functional.grid_sample(
      plane_feature,
      projected_coordinates.float(),
      mode=mode,
      padding_mode=padding_mode,
      align_corners=False,
  ).permute(0, 3, 2, 1)[:, :, 0, :]

  output_features_yz = []
  for plane_feature in yz_planes:
    _, _, H, W = plane_feature.shape  # N = batch size
    # _, M, _ = coordinates.shape  # M = render_h * render_w * num_samples
    normalized_coordinates = (2 / box_warp) * coordinates  # normalize [-1, 1]
    normalized_coordinates[:, :, 1] *= base_resolution / W  # Y
    normalized_coordinates[:, :, 2] *= base_resolution / H  # Z
    # first plane: N x 1 x M x 2
    projected_coordinates = project_onto_planes(
        plane_axes[[2]], normalized_coordinates
    ).unsqueeze(1)
    output_features_yz.append(
        torch.nn.functional.grid_sample(
            plane_feature,
            projected_coordinates.float(),
            mode=mode,
            padding_mode=padding_mode,
            align_corners=False,
        ).permute(0, 3, 2, 1)[:, :, 0, :]
    )
  output_features = [output_features_xy, output_features_xz, output_features_yz]
  return output_features


class ImportanceRendererSOAT(ImportanceRenderer):
  """modified renderer that interpolates a set of vertical features."""

  def __init__(self, base_resolution):
    super().__init__()
    self.ray_marcher = MipRayMarcher2()
    self.plane_axes = generate_planes()
    self.base_resolution = base_resolution

  def run_model(
      self, planes, decoder, sample_coordinates, sample_directions, options
  ):
    sampled_features = sample_from_planes_soat(
        self.plane_axes,
        planes,
        sample_coordinates,
        base_resolution=self.base_resolution,
        padding_mode='reflection',
        box_warp=options['box_warp'],
    )

    xy_list = sampled_features[0]
    xz_feature = sampled_features[1]  # N x M x C
    yz_list = sampled_features[2]
    # interpolate xy feature from the planes
    xy_centers = (
        torch.arange(len(xy_list)) - len(xy_list) / 2 + 0.5
    ) * options['box_warp']
    z_coord = sample_coordinates[:, :, [2]]
    weights = [
        (1 - torch.abs(z_coord - center) / options['box_warp']).clamp(0, 1)
        for center in xy_centers
    ]
    sum_features = torch.stack(
        [weight * feature for weight, feature in zip(weights, xy_list)], dim=3
    ).sum(dim=3)
    sum_weights = torch.stack(weights, dim=3).sum(dim=3)
    xy_feature = sum_features / sum_weights

    # interpolate yz feature from the planes
    yz_centers = (
        torch.arange(len(yz_list)) - len(yz_list) / 2 + 0.5
    ) * options['box_warp']
    x_coord = sample_coordinates[:, :, [0]]
    weights = [
        (1 - torch.abs(x_coord - center) / options['box_warp']).clamp(0, 1)
        for center in yz_centers
    ]
    sum_features = torch.stack(
        [weight * feature for weight, feature in zip(weights, yz_list)], dim=3
    ).sum(dim=3)
    sum_weights = torch.stack(weights, dim=3).sum(dim=3)
    yz_feature = sum_features / sum_weights

    # decoder input shape = N x n_planes x M x C
    decoder_in = torch.stack([xy_feature, xz_feature, yz_feature], dim=1)
    out = decoder(decoder_in, sample_directions)
    if options.get('density_noise', 0) > 0:
      out['sigma'] += torch.randn_like(out['sigma']) * options['density_noise']
    return out
