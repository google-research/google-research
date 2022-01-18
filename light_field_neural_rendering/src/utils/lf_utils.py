# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""File containing light field utils."""
import abc

import jax.numpy as jnp

from light_field_neural_rendering.src.utils import model_utils


class LightField(abc.ABC):
  """A class encapsulating all the utilities for a lightfield.

  Light field parametrizations should use this class as a base class.
  """

  def __init__(self, config):
    """Init Method."""
    self.config = config

  def get_lf_encoding(self, rays):
    """Return the light field and its encoding"""
    lf_samples, non_intersect_mask = self.ray2lightfield(rays)
    lf_samples_enc = self.encode(lf_samples)
    return lf_samples, lf_samples_enc, non_intersect_mask

  def ray2lightfield(self, rays):
    """Convert the rays to light field representation.

    Args:
      rays: data_types.Rays

    Returns:
      lf_samples: Light field representation of rays
      non_intersect_mask: [Optional] To indcate rays that dont intersect the
        light field manifold.
    """
    raise NotImplementedError

  def encode(self, lf_samples):
    """Feature encoding for the light field samples.

    Args:
      lf_samples: Light field input.

    Returns:
      lf_samples_enc : Encoded light field representation.
    """
    if self.config.encoding_name == "positional_encoding":
      lf_samples_enc = model_utils.posenc(
          lf_samples,
          self.config.min_deg_point,
          self.config.max_deg_point,
      )
    elif self.config.encoding_name == "identity":
      lf_samples_enc = lf_samples
    else:
      raise ValueError("Mapping type {} not implemented".format(
          self.config.encoding_name))

    return lf_samples_enc


class LightSlab(LightField):
  """A class encapsulation the LightSlab utilities."""

  def ray_plane_intersection(self, zconst, rays):
    """Compute intersection of the ray with a plane of the form z=const.

    Args:
      zconst: Fixed z-value for the plane.
      rays: data_type.Rays.

    Returns:
      xy: The free-coordinates of intersection.
    """
    t1 = (zconst - rays.origins[Ellipsis, -1]) / rays.directions[Ellipsis, -1]
    xy = rays.origins[Ellipsis, :2] + (t1[Ellipsis, None] * rays.directions)[Ellipsis, :2]

    return xy

  def ray2lightfield(self, rays):
    """Compute the lightslab representation."""
    st = self.ray_plane_intersection(self.config.st_plane, rays)
    uv = self.ray_plane_intersection(self.config.uv_plane, rays)

    lf_samples = jnp.concatenate([st, uv], -1)
    # Assuming there are no non-intersecting rays.
    non_intersect_mask = jnp.array([False] * lf_samples.shape[0])[:, None]

    return lf_samples, non_intersect_mask


def get_lightfield_obj(lf_config):
  """Return the lightfield object"""
  if lf_config.name == "lightslab":
    lightfield_obj = LightSlab(lf_config)
  else:
    raise ValueError("Parametrization:{} not supported for light field".format(
        lf_config.name))
  return lightfield_obj
