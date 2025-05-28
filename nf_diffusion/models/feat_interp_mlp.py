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

"""Implementation of auto-decoder of feature modulated MLP.
"""

import functools
from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from nf_diffusion.models.utils import resample


def batched_resample_fn(feat, locations, res, edge_behavior, coordinate_order):
  """Resampling feature [feat] with coordinates at location [location]."""
  if len(res) == 2:
    feat = resample.resample_2d(
        feat, locations,
        edge_behavior=edge_behavior,
        coordinate_order=coordinate_order)
  elif len(res) == 3:
    feat = resample.resample_3d(
        feat, locations,
        edge_behavior=edge_behavior,
        coordinate_order=coordinate_order)
  else:
    raise ValueError("Invalid dimension: {}".format(len(res)))
  return feat


class FeatIterpNFMLP(nn.Module):
  """Neural fields MLP with grid features."""
  inp_dim: int
  hid_dim: Sequence[int]
  out_dim: int

  # Auto decoder configuration types
  num_emb: int
  feat_dim: int
  res: Sequence[int]
  edge_behavior: str = "CONSTANT_OUTSIDE"
  coordinate_order: str = "xy"

  @nn.compact
  def __call__(self, idx, x):
    assert self.inp_dim == len(self.res), "Wrong cfg: {} {}".format(
        self.inp_dim, self.res)
    assert x.shape[-1] == len(self.res), "x wrong dim: {}".format(x.shape)
    assert x.shape[:len(idx.shape)] == idx.shape, (
        "x and idx shape doesn't matching: x {} idx {}".format(
            x.shape, idx.shape))

    x_shape = x.shape
    idx_shape = idx.shape
    pnt_shape = x.shape[len(idx.shape):-1]
    nexmp = int(np.prod(idx_shape))
    npnts = int(np.prod(pnt_shape))
    idx = jnp.reshape(idx, nexmp)

    # Get the feature
    ttl_feat_dim = self.feat_dim * np.prod(np.array(self.res))
    feat = nn.Embed(self.num_emb, ttl_feat_dim, name="emb")(idx)
    feat = jnp.reshape(feat, [nexmp, 1] + list(self.res) + [self.feat_dim])

    # Interpolate the feature to get the feature for position [x]
    # Since the resampler functions assumed locations are in integers.
    x = jnp.reshape(x, (nexmp, npnts, self.inp_dim))
    resolution = jnp.broadcast_to(jnp.array(self.res), x.shape)
    locations = (x + 0.5) * (resolution - 1)

    batched_resample = functools.partial(
        batched_resample_fn,
        res=self.res, edge_behavior=self.edge_behavior,
        coordinate_order=self.coordinate_order)
    batched_resample = jax.vmap(batched_resample, in_axes=(0, 0), out_axes=0)
    feat = batched_resample(feat, locations)

    for nlayer, hdim in enumerate(self.hid_dim):
      feat = nn.Dense(hdim, name="dense{}".format(nlayer))(feat)
      # TODO(guandao) allow choice of nonlinearity
      feat = jax.nn.leaky_relu(feat, negative_slope=0.01)
    feat = nn.Dense(self.out_dim)(feat)
    feat = jnp.reshape(feat, list(x_shape[:-1]) + [self.out_dim])
    return feat


class FeatNFMLP(nn.Module):
  """Neural fields MLP with different types of features."""
  inp_dim: int
  hid_dim: Sequence[int]
  out_dim: int

  @nn.compact
  def __call__(self, feat):
    for nlayer, hdim in enumerate(self.hid_dim):
      feat = nn.Dense(hdim, name="dense{}".format(nlayer))(feat)
      # TODO(guandao) allow choice of nonlinearity
      feat = jax.nn.leaky_relu(feat, negative_slope=0.01)
    feat = nn.Dense(self.out_dim)(feat)
    return feat
