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

"""Multiscale tools."""

import torch


def get_links_to_current_grid(grid, previous_grid):
  """Get links."""
  c_res = grid.shape
  p_res = previous_grid.shape
  # resize into blocks
  assert (
      c_res[0] % p_res[0] == 0
      and c_res[1] % p_res[1] == 0
      and c_res[2] % p_res[2] == 0
  ), "Should be divisible, so that the links from previous to current is exact!"
  links_to_current = grid.links.view(
      p_res[0],
      c_res[0] // p_res[0],
      p_res[1],
      c_res[1] // p_res[1],
      p_res[2],
      c_res[2] // p_res[2],
  ).permute(0, 2, 4, 1, 3, 5)

  previous_valid_links = previous_grid.links > 0
  previous_links = previous_grid.links[previous_valid_links]
  current_links = links_to_current[previous_valid_links].reshape(
      (len(previous_links), -1)
  )

  valid_c_nodes = current_links > 0
  valid_current_nodes = valid_c_nodes.sum(-1) > 0

  previous_links = previous_links[valid_current_nodes]
  current_links = current_links[valid_current_nodes]

  return previous_links.long(), current_links.long()


# removes the averages of coarsest grids from top/bottom
# when finished, the sum of previous grids + grid == the grid when called
def propagate_multiscale_to_parents(
    finest_grid, previous_grids, args, sh_dim_per_scale, grad=False
):
  """Propagate."""
  with torch.no_grad():
    if not previous_grids:
      return
    else:
      assert not grad or finest_grid.sh_data.grad is not None, (
          "Was called with grads not populated for finest grid, using"
          " grad=True, which shouldn't be the case"
      )

      assert (
          finest_grid.multiscale
      ), "Multiscale should be aggregated when removing it!"

      son_g = finest_grid

      for parent_g, (p_id, s_id) in previous_grids:
        # compute current averages
        assert son_g.shape[0] == parent_g.shape[0] * 2

        valid_c_nodes = s_id > 0
        n_valid_c_nodes = valid_c_nodes.sum(-1)
        assert n_valid_c_nodes.min() > 0, (
            "Should have removed nodes from parent that don't have valid"
            " children, as they are not used"
        )

        if args.multiscale:
          if not grad:
            sh_sums = son_g.sh_data[s_id].sum(1)
            sh_averages = sh_sums / n_valid_c_nodes[:, None]
            son_g.sh_data[s_id] -= sh_averages[:, None, :]
            # only replace when at it has at least one valid_c_node
            son_g.sh_data[s_id][valid_c_nodes[:, :]] = 0
            parent_g.sh_data[p_id] = sh_averages
          else:
            sh_sums = son_g.sh_data.grad[s_id].sum(1)
            sh_averages = sh_sums / n_valid_c_nodes[:, None]
            son_g.sh_data.grad[s_id] -= sh_averages[
                :, None, :
            ]  # * valid_c_nodes[:,:,None]
            son_g.sh_data.grad[s_id][valid_c_nodes[:, :]] = 0

            # only replace when at it has at least one valid_c_node
            parent_g.sh_data.grad = torch.zeros_like(parent_g.sh_data).to(
                sh_averages.device
            )
            parent_g.sh_data.grad[p_id] = sh_averages

        if args.multiscale_sigma:
          if not grad:
            # Do the same as above also for sigma
            density_sums = son_g.density_data[s_id].sum(1)
            sigma_averages = density_sums / n_valid_c_nodes[:, None]
            son_g.density_data[s_id] -= sigma_averages[
                :, None, :
            ]  # * valid_c_nodes[:,:,None]
            son_g.density_data[s_id][valid_c_nodes[:, :]] = 0

            # only replace when at it has at least one valid_c_node
            parent_g.density_data[p_id] = sigma_averages
          else:
            density_sums = son_g.density_data.grad[s_id].sum(1)
            sigma_averages = density_sums / n_valid_c_nodes[:, None]
            son_g.density_data.grad[s_id] -= sigma_averages[
                :, None, :
            ]  # * valid_c_nodes[:,:,None]
            son_g.density_data.grad[s_id][valid_c_nodes[:, :, None]] = 0

            # only replace when at it has at least one valid_c_node

            parent_g.density_data.grad = torch.zeros_like(
                parent_g.density_data
            ).to(sigma_averages.device)
            parent_g.density_data.grad[p_id] = sigma_averages

        son_g = parent_g

      finest_grid.multiscale = False

    if sh_dim_per_scale is not None:
      all_scales_grids = [grid for grid, _ in previous_grids[::-1]] + [
          finest_grid
      ]
      for sh_dim, grid in zip(
          sh_dim_per_scale[: len(all_scales_grids)], all_scales_grids
      ):
        if grid.basis_dim != sh_dim:
          # we need to mask the coefficents that correspond to the
          # higher order sh_dim
          to_mask = torch.zeros(3, grid.basis_dim).cuda()
          to_mask[:, sh_dim:] = 1
          grid.sh_data[:, to_mask.bool().flatten()] = 0


# adds the values of previous_grids to grid
# when finishted, the sum of grid == previous grids + the grid when called
def propagate_multiscale_to_leaves(
    finest_grid, previous_grids, args, sh_dim_per_scale, grad=False
):
  """Propagate."""
  del sh_dim_per_scale
  with torch.no_grad():
    if not previous_grids:
      return
    else:
      assert (
          not finest_grid.multiscale
      ), "Multiscale should not be aggregated when adding it!"

      i = 0

      all_son_grids = [p for p, _ in previous_grids[::-1][1:]] + [finest_grid]
      for parent_g, (p_id, s_id) in previous_grids[::-1]:
        son_g = all_son_grids[i]
        assert son_g.shape[0] == parent_g.shape[0] * 2

        # compute current averages
        valid_s_nodes = s_id > 0
        if args.multiscale:
          if not grad:
            son_g.sh_data[s_id] += (
                parent_g.sh_data[p_id][:, None, :] * valid_s_nodes[:, :, None]
            )
            parent_g.sh_data[p_id] -= parent_g.sh_data[p_id]
          else:
            son_g.sh_data.grad[s_id] += (
                parent_g.sh_data.grad[p_id][:, None, :]
                * valid_s_nodes[:, :, None]
            )
            # The parent gradients should all be zero after this, we
            # turn them to None to save memory
            # parent_g.sh_data.grad[p_id] -=
            # parent_g.sh_data.grad[p_id]
            parent_g.sh_data.grad = None
        if args.multiscale_sigma:
          if not grad:
            son_g.density_data[s_id] += (
                parent_g.density_data[p_id][:, None, :]
                * valid_s_nodes[:, :, None]
            )
            parent_g.density_data[p_id] -= parent_g.density_data[p_id]
          else:
            son_g.density_data.grad[s_id] += (
                parent_g.density_data.grad[p_id][:, None, :]
                * valid_s_nodes[:, :, None]
            )
            # parent_g.density_data.grad[p_id] -=
            # parent_g.density_data.grad[p_id]
            parent_g.density_data.grad = None
        i += 1

      finest_grid.multiscale = True
