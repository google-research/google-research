# coding=utf-8
# Copyright 2020 The Google Research Authors.
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
import jax.numpy as jnp

from jaxnerf.nerf import model_utils


def get_model(key, args):
  return model_dict[args.model](key, args)


class NerfModel(nn.Module):
  """Nerf NN Model with both coarse and fine MLPs."""

  def apply(self, key_0, key_1, rays, n_samples, n_fine_samples, use_viewdirs,
            near, far, noise_std, net_depth, net_width, net_depth_condition,
            net_width_condition, activation, skip_layer, alpha_channel,
            rgb_channel, randomized, white_bkgd, deg_point, deg_view, lindisp):
    """Nerf Model.

    Args:
      key_0: jnp.ndarray, random number generator for coarse model sampling.
      key_1: jnp.ndarray, random number generator for fine model sampling.
      rays: jnp.ndarray(float32), [batch_size, 6/9], each ray is a 6-d vector
        where the first 3 dimensions represent the ray origin and the last 3
        dimensions represent the unormalized ray direction. Note that if ndc
        rays are used, rays are 9-d where the extra 3-dimensional vector is the
        view direction before transformed to ndc rays.
      n_samples: int, the number of samples for coarse nerf.
      n_fine_samples: int, the number of samples for fine nerf.
      use_viewdirs: bool, use viewdirs as a condition.
      near: float, near clip.
      far: float, far clip.
      noise_std: float, std dev of noise added to regularize sigma output.
      net_depth: int, the depth of the first part of MLP.
      net_width: int, the width of the first part of MLP.
      net_depth_condition: int, the depth of the second part of MLP.
      net_width_condition: int, the width of the second part of MLP.
      activation: function, the activation function used in the MLP.
      skip_layer: int, add a skip connection to the output vector of every
        skip_layer layers.
      alpha_channel: int, the number of alpha_channels.
      rgb_channel: int, the number of rgb_channels.
      randomized: bool, use randomized stratified sampling.
      white_bkgd: bool, use white background.
      deg_point: degree of positional encoding for positions.
      deg_view: degree of positional encoding for viewdirs.
      lindisp: bool, sampling linearly in disparity rather than depth if true.

    Returns:
      ret: list, [(rgb, disp, acc), (rgb_coarse, disp_coarse, acc_coarse)]
    """
    # Extract viewdirs from the ray array
    if rays.shape[-1] > 6:  # viewdirs different from rays_d
      viewdirs = rays[Ellipsis, -3:]
      rays = rays[Ellipsis, :-3]
    else:  # viewdirs are normalized rays_d
      viewdirs = rays[Ellipsis, 3:6]
    # Stratified sampling along rays
    z_vals, samples = model_utils.sample_along_rays(key_0, rays, n_samples,
                                                    near, far, randomized,
                                                    lindisp)
    samples = model_utils.posenc(samples, deg_point)
    # Point attribute predictions
    if use_viewdirs:
      norms = jnp.linalg.norm(viewdirs, axis=-1, keepdims=True)
      viewdirs = viewdirs / norms
      viewdirs = model_utils.posenc(viewdirs, deg_view)
      raw = model_utils.MLP(
          samples, viewdirs, net_depth=net_depth, net_width=net_width,
          net_depth_condition=net_depth_condition,
          net_width_condition=net_width_condition,
          activation=activation, skip_layer=skip_layer,
          alpha_channel=alpha_channel, rgb_channel=rgb_channel,
      )
    else:
      raw = model_utils.MLP(
          samples, net_depth=net_depth, net_width=net_width,
          net_depth_condition=net_depth_condition,
          net_width_condition=net_width_condition,
          activation=activation, skip_layer=skip_layer,
          alpha_channel=alpha_channel, rgb_channel=rgb_channel,
      )
    # Add noises to regularize the density predictions if needed
    raw = model_utils.noise_regularize(key_0, raw, noise_std, randomized)
    # Volumetric rendering.
    rgb, disp, acc, weights = model_utils.volumetric_rendering(
        raw,
        z_vals,
        rays[Ellipsis, 3:6],
        white_bkgd=white_bkgd,
    )
    ret = [
        (rgb, disp, acc),
    ]
    # Hierarchical sampling based on coarse predictions
    if n_fine_samples > 0:
      z_vals_mid = .5 * (z_vals[Ellipsis, 1:] + z_vals[Ellipsis, :-1])
      z_vals, samples = model_utils.sample_pdf(
          key_1,
          z_vals_mid,
          weights[Ellipsis, 1:-1],
          rays,
          z_vals,
          n_fine_samples,
          randomized,
      )
      samples = model_utils.posenc(samples, deg_point)
      if use_viewdirs:
        raw = model_utils.MLP(samples, viewdirs)
      else:
        raw = model_utils.MLP(samples)
      raw = model_utils.noise_regularize(key_1, raw, noise_std, randomized)
      rgb, disp, acc, unused_weights = model_utils.volumetric_rendering(
          raw,
          z_vals,
          rays[Ellipsis, 3:6],
          white_bkgd=white_bkgd,
      )
      ret.append((rgb, disp, acc))
    return ret


def nerf(key, args):
  """Neural Randiance Field.

  Args:
    key: jnp.ndarray. Random number generator.
    args: FLAGS class. Hyperparameters of nerf.

  Returns:
    model: nn.Model. Nerf model with parameters.
    state: flax.Module.state. Nerf model state for stateful parameters.
  """
  deg_point = args.deg_point
  deg_view = args.deg_view
  n_samples = args.n_samples
  n_fine_samples = args.n_fine_samples
  use_viewdirs = args.use_viewdirs
  near = args.near
  far = args.far
  noise_std = args.noise_std
  randomized = args.randomized
  white_bkgd = args.white_bkgd
  net_depth = args.net_depth
  net_width = args.net_width
  net_depth_condition = args.net_depth_condition
  net_width_condition = args.net_width_condition
  if args.activation == "relu":
    activation = nn.relu
  else:
    raise NotImplementedError("Invalid choice of activation {}".format(
        args.activation))
  skip_layer = args.skip_layer
  alpha_channel = args.alpha_channel
  rgb_channel = args.rgb_channel
  lindisp = args.lindisp

  ray_shape = (args.batch_size, 6 if args.dataset != "llff" else 9)
  model_fn = NerfModel.partial(
      n_samples=n_samples,
      n_fine_samples=n_fine_samples,
      use_viewdirs=use_viewdirs,
      near=near,
      far=far,
      noise_std=noise_std,
      net_depth=net_depth,
      net_width=net_width,
      net_depth_condition=net_depth_condition,
      net_width_condition=net_width_condition,
      activation=activation,
      skip_layer=skip_layer,
      alpha_channel=alpha_channel,
      rgb_channel=rgb_channel,
      randomized=randomized,
      white_bkgd=white_bkgd,
      deg_point=deg_point,
      deg_view=deg_view,
      lindisp=lindisp)
  with nn.stateful() as init_state:
    unused_outspec, init_params = model_fn.init_by_shape(
        key,
        [
            (key.shape, key.dtype),
            (key.shape, key.dtype),
            (ray_shape, jnp.float32),
        ],
    )
    model = nn.Model(model_fn, init_params)
  return model, init_state


model_dict = {
    "nerf": nerf,
}
