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

"""ray transparency regularizers."""
import torch


def ray_entropy(weights, clip=1e-8):
  entropy = -torch.sum(weights * torch.log(weights.clip(clip)), dim=-1)
  return entropy.mean()


def ray_distortion(weights):  # , mode='linear'):
  # pylint: disable=g-import-not-at-top,g-multiple-import,invalid-name
  from torch_efficient_distloss import eff_distloss  # , eff_distloss_native

  B, num_rays, N = weights.shape
  # note: this operation assumed that weights were linearly spaced!!
  s = torch.linspace(0, 1, N + 1)
  m = (s[1:] + s[:-1]) * 0.5
  m = m[None][None].repeat(B, num_rays, 1).to(weights.device)
  interval = 1 / N
  loss = eff_distloss(weights, m, interval)
  return loss


def ray_finite_difference(ray_info):
  """penalize visible decreases in alpha."""
  grad_occ = torch.sum(
      ray_info['weights'][:, :, 1:]
      * (-torch.diff(ray_info['alpha'], n=1, dim=-1)).clip(0)
      / ray_info['dists'][:, :, :-1],
      dim=-1,
  )
  # added: remove invalid values (nan can happen when
  # dists is zero, due to perturb=True in xyz
  grad_occ = torch.nan_to_num(grad_occ, nan=0.0, posinf=0.0, neginf=0.0)
  output = grad_occ.mean()
  return output
