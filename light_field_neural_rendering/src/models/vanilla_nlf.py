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

"""Vanilla Light Field Neural Rendering model"""

from flax import linen as nn
from jax import random
import jax.numpy as jnp

from light_field_neural_rendering.src.models import mlp
from light_field_neural_rendering.src.utils import config_utils
from light_field_neural_rendering.src.utils import lf_utils


class VanillaNLF(nn.Module):
  """Vanilla NLF model. With epipolar geometric bias."""
  mlp_config: config_utils.MLPParams
  render_config: config_utils.RenderParams
  encoding_config: config_utils.EncodingParams
  lf_config: config_utils.LightFieldParams

  def setup(self):
    """Setup Model and LightField object"""
    self.rgb_mlp = mlp.SimpleMLP(self.mlp_config)
    self.lightfield = lf_utils.get_lightfield_obj(self.lf_config)

    # Set fill value for background
    self.fill_value = 1. if self.render_config.white_bkgd else 0.

  def _predict_color(self, lf_samples_enc):
    """Predict color for encoded lightfield."""
    raw_rgb = self.rgb_mlp(lf_samples_enc)
    rgb = self.render_config.rgb_activation(raw_rgb)
    return rgb

  def _fill_background(self, rgb, bkgd_mask):
    """Fill color for background rays and set them to background color (black or white)

    Args:
      rgb: rgb predictions
      bkgd_mask: If true then ray is background

    Returns:
      rgb: rgb with background filled in with white or black.
    """
    return rgb * (1. - bkgd_mask) + self.fill_value * bkgd_mask

  def __call__(self, rng_0, rng_1, batch, randomized):
    """Vanilla NLF Model.

    Args:
      rng_0: jnp.ndarray, random number generator for coarse model sampling.
      rng_1: jnp.ndarray, random number generator for fine model sampling.
      batch: data batch data_types.Batch.
      randomized: bool, use randomized stratified sampling.

    Returns:
      ret: list, [(rgb, None, foreground_mask)]
    """

    del rng_0, rng_1
    # Sample the light field representation and it encoding.
    _, lf_samples_enc, bkgd_mask = self.lightfield.get_lf_encoding(
        batch.target_view.rays)

    # Get the color prediction.
    rgb = self._predict_color(lf_samples_enc)

    # Fill the background region with white or black.
    rgb = self._fill_background(rgb, bkgd_mask=bkgd_mask)

    # Construct the return tuples keeping the signature close to NeRF as
    # possible
    ret = []
    ret.append((rgb, None, 1. * ~bkgd_mask))

    return ret


def construct_model(key, example_batch, args):
  """Construct a Vanilla Light Field Neural Renderer.

  Args:
    key: jnp.ndarray. Random number generator.
    example_batch: dict, an example of a batch of data.
    args: FLAGS class. Hyperparameters of nerf.

  Returns:
    model: nn.Model. Nerf model with parameters.
    state: flax.Module.state. Nerf model state for stateful parameters.
  """
  net_activation = getattr(nn, str(args.model.net_activation))
  rgb_activation = getattr(nn, str(args.model.rgb_activation))
  sigma_activation = getattr(nn, str(args.model.sigma_activation))

  # Assert that rgb_activation always produces outputs in [0, 1], and
  # sigma_activation always produce non-negative outputs.
  x = jnp.exp(jnp.linspace(-90, 90, 1024))
  x = jnp.concatenate([-x[::-1], x], 0)

  rgb = rgb_activation(x)
  if jnp.any(rgb < 0) or jnp.any(rgb > 1):
    raise NotImplementedError(
        "Choice of rgb_activation `{}` produces colors outside of [0, 1]"
        .format(args.rgb_activation))

  sigma = sigma_activation(x)
  if jnp.any(sigma < 0):
    raise NotImplementedError(
        "Choice of sigma_activation `{}` produces negative densities".format(
            args.sigma_activation))

  # We have defined some wrapper functions to extract the relevant cofiguration
  # so are to allow for efficient reuse
  mlp_config = config_utils.get_mlp_config(args, net_activation)
  render_config = config_utils.get_render_params(args, rgb_activation,
                                                 sigma_activation)
  encoding_config = config_utils.get_encoding_params(args)
  lf_config = config_utils.get_lightfield_params(args)

  model = VanillaNLF(
      mlp_config=mlp_config,
      render_config=render_config,
      encoding_config=encoding_config,
      lf_config=lf_config,
  )

  key1, key2, key3 = random.split(key, num=3)

  init_variables = model.init(
      key1,
      rng_0=key2,
      rng_1=key3,
      batch=example_batch,
      randomized=args.model.randomized)

  return model, init_variables
