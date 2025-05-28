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

"""A 3D U-Net Architecture from Denoising Diffusion Probabilistic Models."""

from typing import Any, Dict, Optional
from flax import linen as nn
import jax
import jax.numpy as jnp
import ml_collections
from nf_diffusion.models.score_model import ddpm_univdiff_3d


class UNet3D(nn.Module):
  """A UNet architecture.

  Attributes:
    1. density UNet configuration: input and output 1 dim.
    2. color UNet confiugration: input 5d and output 5d.
  """

  density_config: ml_collections.ConfigDict
  color_config: ml_collections.ConfigDict
  stop_den_grad: bool = False
  concat_input_den: bool = True
  use_denoise_density_cond: bool = False

  @nn.compact
  def __call__(self,
               x=None,
               t=None,
               cond = None,
               train = True,
               **kwargs):
    """UNet forward pass with input x, conditioning input c, and time t.

    Args:
      x: noisy image for diffusion models otherwise None
      cond: conditioning dict
      t: time-step for diffusion models otherwise None
      train: whether this is during training (for dropout, etc.)
      **kwargs: anything else.

    Returns:
      network output.
    """
    assert cond is None or isinstance(cond, dict), "cond isn't None or dict."

    assert x is not None  # diffusion model
    assert t is not None
    B, _, _, _, _ = x.shape  # pylint: disable=invalid-name
    if isinstance(t, tuple):
      assert t[0].shape == (B,)
      assert t[1].shape == (B,)
      t_density = t[0]
      t_color = t[1]
    else:
      assert t.shape == (B,)
      t_color = t
      t_density = t
    assert x.dtype in (jnp.float32, jnp.float64)

    density = x[Ellipsis, :1]
    colors = x[Ellipsis, 1:]
    density_out = ddpm_univdiff_3d.UNet3D(self.density_config)(density,
                                                               t_density,
                                                               cond,
                                                               train,
                                                               **kwargs)
    if self.use_denoise_density_cond:
      raise NotImplementedError
    else:
      density_out_cond = density_out
    if self.stop_den_grad:
      density_out_cond = jax.lax.stop_gradient(density_out)
    colors_input = jnp.concatenate([density_out_cond, colors], axis=-1)
    if self.concat_input_den:
      colors_input = jnp.concatenate([density, colors_input], axis=-1)
    colors_output = ddpm_univdiff_3d.UNet3D(self.color_config)(colors_input,
                                                               t_color, cond,
                                                               train, **kwargs)
    if self.concat_input_den:
      colors_output = colors_output[Ellipsis, 2:]
    else:
      colors_output = colors_output[Ellipsis, 1:]

    out = jnp.concatenate([density_out, colors_output], axis=-1)
    return out
