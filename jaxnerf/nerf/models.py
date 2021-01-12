# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

# Lint as: python3
"""Different model implementation plus a general port for all the models."""
from flax import nn
from jax import random
import jax.numpy as jnp

from jaxnerf.nerf import model_utils


def get_model(key, example_batch, args):
  return model_dict[args.model](key, example_batch, args)


class NerfModel(nn.Module):
  """Nerf NN Model with both coarse and fine MLPs."""

  def apply(self, rng_0, rng_1, origins, directions, viewdirs,
            num_coarse_samples, num_fine_samples, use_viewdirs, near, far,
            noise_std, net_depth, net_width, net_depth_condition,
            net_width_condition, net_activation, skip_layer, num_rgb_channels,
            num_sigma_channels, randomized, white_bkgd, deg_point, deg_view,
            lindisp, rgb_activation, sigma_activation, tf_posenc_order):
    """Nerf Model.

    Args:
      rng_0: jnp.ndarray, random number generator for coarse model sampling.
      rng_1: jnp.ndarray, random number generator for fine model sampling.
      origins: jnp.ndarray(float32), [batch_size, 3], each ray origin.
      directions: jnp.ndarray(float32), [batch_size, 3], each ray direction.
      viewdirs: jnp.ndarray(float32), [batch_size, 3], the viewing direction for
        each ray. This is only used if NDC rays are used, as otherwise
        `directions` is equal to viewdirs.
      num_coarse_samples: int, the number of samples for coarse nerf.
      num_fine_samples: int, the number of samples for fine nerf.
      use_viewdirs: bool, use viewdirs as a condition.
      near: float, near clip.
      far: float, far clip.
      noise_std: float, std dev of noise added to regularize sigma output.
      net_depth: int, the depth of the first part of MLP.
      net_width: int, the width of the first part of MLP.
      net_depth_condition: int, the depth of the second part of MLP.
      net_width_condition: int, the width of the second part of MLP.
      net_activation: function, the activation function used within the MLP.
      skip_layer: int, add a skip connection to the output vector of every
        skip_layer layers.
      num_rgb_channels: int, the number of RGB channels.
      num_sigma_channels: int, the number of density channels.
      randomized: bool, use randomized stratified sampling.
      white_bkgd: bool, use white background.
      deg_point: degree of positional encoding for positions.
      deg_view: degree of positional encoding for viewdirs.
      lindisp: bool, sampling linearly in disparity rather than depth if true.
      rgb_activation: function, the activation used to generate RGB.
      sigma_activation: function, the activation used to generate density.
      tf_posenc_order: bool, keep the same ordering as the original tf code.

    Returns:
      ret: list, [(rgb_coarse, disp_coarse, acc_coarse), (rgb, disp, acc)]
    """
    # Stratified sampling along rays
    key, rng_0 = random.split(rng_0)
    z_vals, samples = model_utils.sample_along_rays(key, origins, directions,
                                                    num_coarse_samples, near,
                                                    far, randomized, lindisp)
    samples_enc = model_utils.posenc(samples, deg_point, tf_posenc_order)
    # Point attribute predictions
    if use_viewdirs:
      viewdirs_enc = model_utils.posenc(
          viewdirs / jnp.linalg.norm(viewdirs, axis=-1, keepdims=True),
          deg_view, tf_posenc_order)
      raw_rgb, raw_sigma = model_utils.MLP(
          samples_enc,
          viewdirs_enc,
          net_depth=net_depth,
          net_width=net_width,
          net_depth_condition=net_depth_condition,
          net_width_condition=net_width_condition,
          net_activation=net_activation,
          skip_layer=skip_layer,
          num_rgb_channels=num_rgb_channels,
          num_sigma_channels=num_sigma_channels,
      )
    else:
      raw_rgb, raw_sigma = model_utils.MLP(
          samples_enc,
          net_depth=net_depth,
          net_width=net_width,
          net_depth_condition=net_depth_condition,
          net_width_condition=net_width_condition,
          net_activation=net_activation,
          skip_layer=skip_layer,
          num_rgb_channels=num_rgb_channels,
          num_sigma_channels=num_sigma_channels,
      )
    # Add noises to regularize the density predictions if needed
    key, rng_0 = random.split(rng_0)
    raw_sigma = model_utils.add_gaussian_noise(key, raw_sigma, noise_std,
                                               randomized)
    rgb = rgb_activation(raw_rgb)
    sigma = sigma_activation(raw_sigma)
    # Volumetric rendering.
    comp_rgb, disp, acc, weights = model_utils.volumetric_rendering(
        rgb,
        sigma,
        z_vals,
        directions,
        white_bkgd=white_bkgd,
    )
    ret = [
        (comp_rgb, disp, acc),
    ]
    # Hierarchical sampling based on coarse predictions
    if num_fine_samples > 0:
      z_vals_mid = .5 * (z_vals[Ellipsis, 1:] + z_vals[Ellipsis, :-1])
      key, rng_1 = random.split(rng_1)
      z_vals, samples = model_utils.sample_pdf(
          key,
          z_vals_mid,
          weights[Ellipsis, 1:-1],
          origins,
          directions,
          z_vals,
          num_fine_samples,
          randomized,
      )
      samples_enc = model_utils.posenc(samples, deg_point, tf_posenc_order)
      if use_viewdirs:
        raw_rgb, raw_sigma = model_utils.MLP(samples_enc, viewdirs_enc)
      else:
        raw_rgb, raw_sigma = model_utils.MLP(samples_enc)
      key, rng_1 = random.split(rng_1)
      raw_sigma = model_utils.add_gaussian_noise(key, raw_sigma, noise_std,
                                                 randomized)
      rgb = rgb_activation(raw_rgb)
      sigma = sigma_activation(raw_sigma)
      comp_rgb, disp, acc, unused_weights = model_utils.volumetric_rendering(
          rgb,
          sigma,
          z_vals,
          directions,
          white_bkgd=white_bkgd,
      )
      ret.append((comp_rgb, disp, acc))
    return ret


def nerf(key, example_batch, args):
  """Neural Randiance Field.

  Args:
    key: jnp.ndarray. Random number generator.
    example_batch: dict, an example of a batch of data.
    args: FLAGS class. Hyperparameters of nerf.

  Returns:
    model: nn.Model. Nerf model with parameters.
    state: flax.Module.state. Nerf model state for stateful parameters.
  """
  net_activation = getattr(nn, str(args.net_activation))
  rgb_activation = getattr(nn, str(args.rgb_activation))
  sigma_activation = getattr(nn, str(args.sigma_activation))

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

  model_fn = NerfModel.partial(
      deg_point=args.deg_point,
      deg_view=args.deg_view,
      num_coarse_samples=args.num_coarse_samples,
      num_fine_samples=args.num_fine_samples,
      use_viewdirs=args.use_viewdirs,
      near=args.near,
      far=args.far,
      noise_std=args.noise_std,
      randomized=args.randomized,
      white_bkgd=args.white_bkgd,
      net_depth=args.net_depth,
      net_width=args.net_width,
      net_depth_condition=args.net_depth_condition,
      net_width_condition=args.net_width_condition,
      skip_layer=args.skip_layer,
      num_rgb_channels=args.num_rgb_channels,
      num_sigma_channels=args.num_sigma_channels,
      lindisp=args.lindisp,
      net_activation=net_activation,
      rgb_activation=rgb_activation,
      sigma_activation=sigma_activation,
      tf_posenc_order=args.tf_posenc_order)
  with nn.stateful() as init_state:
    rays = example_batch["rays"]
    key1, key2, key3 = random.split(key, num=3)
    # TODO(barron): Determine why the rays have an unused first dimension.
    _, init_params = model_fn.init(key1, key2, key3, rays.origins[0],
                                   rays.directions[0], rays.viewdirs[0])

    model = nn.Model(model_fn, init_params)
  return model, init_state


model_dict = {
    "nerf": nerf,
}
