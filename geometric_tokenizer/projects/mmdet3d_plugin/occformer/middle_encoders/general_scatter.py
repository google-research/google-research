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

# pylint: skip-file
import torch
from torch import nn


class GeneralScatter(nn.Module):

  def __init__(self, in_channels, output_shape, only_label_scatter=False):
    super().__init__()
    self.output_shape = output_shape
    self.ny = int(output_shape[0])  # = x value
    self.nx = int(output_shape[1])  # = y value
    self.nz = int(output_shape[2])
    self.in_channels = in_channels
    self.fp16_enabled = False
    self.only_label_scatter = only_label_scatter

  def forward(self, voxel_features, coors, batch_size=None, channel_revise=-1):
    """Foraward function to scatter features."""
    # TODO: rewrite the function in a batch manner
    # no need to deal with different batch cases
    if batch_size is not None:
      return self.forward_batch(
          voxel_features, coors, batch_size, channel_revise
      )
    else:
      return self.forward_single(voxel_features, coors)

  def forward_single(self, voxel_features, coors):
    """Scatter features of single sample.

    Args:
        voxel_features (torch.Tensor): Voxel features in shape (N, M, C).
        coors (torch.Tensor): Coordinates of each voxel. The first column
          indicates the sample ID.
    """
    # Create the canvas for this sample
    canvas = torch.zeros(
        self.in_channels,
        self.nx * self.ny * self.nz,
        dtype=voxel_features.dtype,
        device=voxel_features.device,
    )

    indices = (
        coors[:, 1] * self.nx + coors[:, 2] + coors[:, 3] * (self.nx * self.ny)
    )
    indices = indices.long()
    voxels = voxel_features.t()
    # Now scatter the blob back to the canvas.
    canvas[:, indices] = voxels
    # Undo the column stacking to final 4-dim tensor
    canvas = canvas.view(1, self.in_channels, self.ny, self.nx, self.nz)
    return [canvas]

  def forward_batch(self, voxel_features, coors, batch_size, channel_revise=-1):
    """Scatter features of single sample.

    Args:
        voxel_features (torch.Tensor): Voxel features in shape (N, M, C).
        coors (torch.Tensor): Coordinates of each voxel in shape (N, 4). The
          first column indicates the sample ID.
        batch_size (int): Number of samples in the current batch.
    """
    # batch_canvas will be the final output.
    batch_canvas = []
    in_channels = channel_revise if channel_revise > 0 else self.in_channels

    for batch_itt in range(batch_size):
      # Create the canvas for this sample
      canvas = torch.zeros(
          in_channels,
          self.nx * self.ny * self.nz,
          dtype=voxel_features.dtype,
          device=voxel_features.device,
      )

      # Only include non-empty pillars
      batch_mask = coors[:, 0] == batch_itt
      this_coors = coors[batch_mask, :]
      # indices = this_coors[:, 2] * self.nx + this_coors[:, 3] + this_coors[:, 1] * (self.nx * self.ny)
      # indices = this_coors[:, 2] + this_coors[:, 3] * self.ny + this_coors[:, 1] * (self.nx * self.ny)
      indices = (
          this_coors[:, 2] * (self.nx * self.nz)
          + this_coors[:, 3] * self.nz
          + this_coors[:, 1]
      )
      indices = indices.type(torch.long)
      voxels = voxel_features[batch_mask, :]
      voxels = voxels.t()

      # Now scatter the blob back to the canvas.
      canvas[:, indices] = voxels

      # Append to a list for later stacking.
      batch_canvas.append(canvas)

    # Stack to 3-dim tensor (batch-size, in_channels, nrows*ncols)
    batch_canvas = torch.stack(batch_canvas, 0)

    # Undo the column stacking to final 4-dim tensor
    batch_canvas = batch_canvas.view(
        batch_size, in_channels, self.ny, self.nx, self.nz
    )

    return batch_canvas
