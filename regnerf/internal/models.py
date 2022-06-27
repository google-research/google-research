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

"""Different model implementation plus a general port for all the models."""

import functools
from typing import Any, Callable

from flax import linen as nn
import gin
from internal import mip, utils  # pylint: disable=g-multiple-import
import jax
from jax import random
import jax.numpy as jnp


@gin.configurable
class MipNerfModel(nn.Module):
  """Nerf NN Model with both coarse and fine MLPs."""
  config: Any = None  # A Config class, must be set upon construction.
  num_samples: int = 128  # The number of samples per level.
  num_levels: int = 2  # The number of sampling levels.
  stop_level_grad: bool = True  # If True, don't backprop across levels.
  use_viewdirs: bool = True  # If True, use view directions as input.
  genspace_fn: Callable[Ellipsis, Any] = None  # The genspace() curve function.
  ray_shape: str = 'cone'  # The shape of cast rays ('cone' or 'cylinder').
  disable_integration: bool = False  # If True, use PE instead of IPE.
  single_jitter: bool = False  # If True, jitter whole rays instead of samples.

  @nn.compact
  def __call__(
      self,
      rng,
      rays,
      resample_padding,
      compute_extras,
  ):
    """The mip-NeRF Model.

    Args:
      rng: random number generator (or None for deterministic output).
      rays: util.Rays, a pytree of ray origins, directions, and viewdirs.
      resample_padding: float, the histogram padding to use when resampling.
      compute_extras: bool, if True, compute extra quantities besides color.

    Returns:
      ret: list, [*(rgb, distance, acc)]
    """
    # Construct the MLP.
    mlp = MLP()

    renderings = []
    for i_level in range(self.num_levels):
      if rng is None:
        key = None
      else:
        key, rng = random.split(rng)

      if i_level == 0:
        # Stratified sampling along rays
        t_vals, samples = mip.sample_along_rays(
            key,
            rays.origins,
            rays.directions,
            rays.radii,
            self.num_samples,
            rays.near,
            rays.far,
            self.genspace_fn,
            self.ray_shape,
            self.single_jitter,
        )
      else:
        t_vals, samples = mip.resample_along_rays(
            key,
            rays.origins,
            rays.directions,
            rays.radii,
            t_vals,
            weights,
            self.ray_shape,
            self.stop_level_grad,
            resample_padding,
            self.single_jitter,
        )

      if self.disable_integration:
        samples = (samples[0], jnp.zeros_like(samples[1]))

      # Point attribute predictions
      if self.use_viewdirs:
        (rgb, density, normals) = mlp(rng, samples, rays.viewdirs)
      else:
        (rgb, density, normals) = mlp(rng, samples, None)

      # Volumetric rendering.
      weights, _, _, delta = mip.compute_alpha_weights(
          density, t_vals, rays.directions)
      rendering = mip.volumetric_rendering(
          rgb,
          weights,
          normals,
          t_vals,
          self.config.white_background,
          self.config.vis_num_rays,
          compute_extras,
          delta,
      )
      renderings.append(rendering)
    return renderings


def construct_mipnerf(rng, rays, config):
  """Construct a Neural Radiance Field.

  Args:
    rng: jnp.ndarray. Random number generator.
    rays: an example of input Rays.
    config: A Config class.

  Returns:
    model: nn.Model. Nerf model with parameters.
    state: flax.Module.state. Nerf model state for stateful parameters.
  """
  # Grab just 10 rays, to minimize memory overhead during construction.
  ray = jax.tree_map(lambda x: jnp.reshape(x, [-1, x.shape[-1]])[:10], rays)
  model = MipNerfModel(config=config)
  init_variables = model.init(
      rng, rng=None, rays=ray, resample_padding=0., compute_extras=False)
  return model, init_variables


def cosine_easing_window(alpha, min_freq_log2=0, max_freq_log2=16):
  """Eases in each frequency one by one with a cosine.

  This is equivalent to taking a Tukey window and sliding it to the right
  along the frequency spectrum.

  Args:
    alpha: will ease in each frequency as alpha goes from 0.0 to num_freqs.
    min_freq_log2: the lower frequency band.
    max_freq_log2: the upper frequency band.

  Returns:
    A 1-d numpy array with num_sample elements containing the window.
  """
  num_bands = max_freq_log2 - min_freq_log2
  bands = jnp.linspace(min_freq_log2, max_freq_log2, num_bands)
  x = jnp.clip(alpha - bands, 0.0, 1.0)
  values = 0.5 * (1 + jnp.cos(jnp.pi * x + jnp.pi))

  # always set first 4 freqs to 1
  values = values.reshape(-1)
  values = jnp.concatenate([jnp.ones_like(values[:4]), values[4:]])

  values = jnp.repeat(values.reshape(-1, 1), 3, axis=1).reshape(-1)
  return jnp.stack([values, values])


@gin.configurable
class MLP(nn.Module):
  """A simple MLP."""
  net_depth: int = 8  # The depth of the first part of MLP.
  net_width: int = 256  # The width of the first part of MLP.
  net_depth_viewdirs: int = 1  # The depth of the second part of MLP.
  net_width_viewdirs: int = 128  # The width of the second part of MLP.
  net_activation: Callable[Ellipsis, Any] = nn.relu  # The activation function.
  # Initializer for the weights of the MLP.
  weight_init: Callable[Ellipsis, Any] = jax.nn.initializers.glorot_uniform()
  skip_layer: int = 4  # Add a skip connection to the output of every N layers.
  num_rgb_channels: int = 3  # The number of RGB channels.
  min_deg_point: int = 0  # Min degree of positional encoding for 3D points.
  max_deg_point: int = 16  # Max degree of positional encoding for 3D points.
  deg_view: int = 4  # Degree of positional encoding for viewdirs.
  density_activation: Callable[Ellipsis, Any] = nn.softplus  # Density activation.
  density_noise: float = 0.  # Standard deviation of noise added to raw density.
  density_bias: float = -1.  # The shift added to raw densities pre-activation.
  rgb_activation: Callable[Ellipsis, Any] = nn.sigmoid  # The RGB activation.
  rgb_padding: float = 0.001  # Padding added to the RGB outputs.
  disable_normals: bool = False  # If True, don't bother computing normals.

  @nn.compact
  def __call__(self, rng, samples, viewdirs=None):
    """Evaluate the MLP.

    Args:
      rng: random number generator (or None for deterministic output).
      samples: a tuple containing:
        - mean: [..., num_samples, 3], coordinate means, and
        - cov: [..., num_samples, 3{, 3}], coordinate covariance matrices.
      viewdirs: jnp.ndarray(float32), [batch, 3], if not None, this variable
        will be part of the input to the second part of the MLP concatenated
        with the output vector of the first part of the MLP. If None, only the
        first part of the MLP will be used with input x. In the original paper,
        this variable is the view direction.

    Returns:
      rgb: jnp.ndarray(float32), with a shape of [..., num_rgb_channels].
      density: jnp.ndarray(float32), with a shape of [...].
      normals: jnp.ndarray(float32), with a shape of [..., 3].
    """

    dense_layer = functools.partial(nn.Dense, kernel_init=self.weight_init)

    def predict_density(rng, means, covs):
      """Helper function to output density."""
      # Encode input positions
      inputs = mip.integrated_pos_enc(
          (means, covs), self.min_deg_point, self.max_deg_point)
      # Evaluate network to output density
      x = inputs
      for i in range(self.net_depth):
        x = dense_layer(self.net_width)(x)
        x = self.net_activation(x)
        if i % self.skip_layer == 0 and i > 0:
          x = jnp.concatenate([x, inputs], axis=-1)
      raw_density = dense_layer(1)(x)[Ellipsis, 0]  # Hardcoded to a single channel.
      # Add noise to regularize the density predictions if needed.
      if (rng is not None) and (self.density_noise > 0):
        key, rng = random.split(rng)
        raw_density += self.density_noise * random.normal(
            key, raw_density.shape, dtype=raw_density.dtype)
      # Apply bias and activation to raw density
      density = self.density_activation(raw_density + self.density_bias)
      return density, x

    means, covs = samples
    if self.disable_normals:
      density, x = predict_density(rng, means, covs)
      normals = jnp.full_like(means, fill_value=jnp.nan)
    else:
      # Flatten the input so value_and_grad can be vmap'ed.
      means_flat = means.reshape([-1, means.shape[-1]])
      covs_flat = covs.reshape([-1] + list(covs.shape[len(means.shape) - 1:]))
      # Evaluate the network and its gradient on the flattened input.
      predict_density_and_grad_fn = jax.vmap(
          jax.value_and_grad(predict_density, argnums=1, has_aux=True),
          in_axes=(None, 0, 0))
      (density_flat, x_flat), density_grad_flat = (
          predict_density_and_grad_fn(rng, means_flat, covs_flat))

      # Unflatten the output.
      density = density_flat.reshape(means.shape[:-1])
      x = x_flat.reshape(list(means.shape[:-1]) + [x_flat.shape[-1]])
      density_grad = density_grad_flat.reshape(means.shape)

      # Compute surface normals as negative normalized density gradient
      eps = jnp.finfo(jnp.float32).eps
      normals = -density_grad / jnp.sqrt(
          jnp.maximum(jnp.sum(density_grad**2, axis=-1, keepdims=True), eps))

    if viewdirs is not None:
      viewdirs_enc = mip.pos_enc(
          viewdirs, min_deg=0, max_deg=self.deg_view, append_identity=True)
      # Output of the first part of MLP.
      bottleneck = dense_layer(self.net_width)(x)
      viewdirs_enc = jnp.broadcast_to(
          viewdirs_enc[Ellipsis, None, :],
          list(bottleneck.shape[:-1]) + [viewdirs_enc.shape[-1]])
      x = jnp.concatenate([bottleneck, viewdirs_enc], axis=-1)
      # Here use 1 extra layer to align with the original nerf model.
      for _ in range(self.net_depth_viewdirs):
        x = dense_layer(self.net_width_viewdirs)(x)
        x = self.net_activation(x)
    rgb = self.rgb_activation(dense_layer(self.num_rgb_channels)(x))
    rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding

    return (rgb, density, normals)


def render_image(render_fn, rays, rng, config):
  """Render all the pixels of an image (in test mode).

  Args:
    render_fn: function, jit-ed render function.
    rays: a `Rays` pytree, the rays to be rendered.
    rng: jnp.ndarray, random number generator (used in training mode only).
    config: A Config class.

  Returns:
    rgb: jnp.ndarray, rendered color image.
    disp: jnp.ndarray, rendered disparity image.
    acc: jnp.ndarray, rendered accumulated weights per pixel.
  """
  height, width = rays.origins.shape[:2]
  num_rays = height * width
  rays = jax.tree_map(lambda r: r.reshape((num_rays, -1)), rays)

  host_id = jax.host_id()
  chunks = []
  idx0s = range(0, num_rays, config.render_chunk_size)
  for i_chunk, idx0 in enumerate(idx0s):
    # pylint: disable=cell-var-from-loop
    if i_chunk % max(1, len(idx0s) // 10) == 0:
      print(f'Rendering chunk {i_chunk}/{len(idx0s)-1}')
    chunk_rays = (
        jax.tree_map(lambda r: r[idx0:idx0 + config.render_chunk_size], rays))
    actual_chunk_size = chunk_rays.origins.shape[0]
    rays_remaining = actual_chunk_size % jax.device_count()
    if rays_remaining != 0:
      padding = jax.device_count() - rays_remaining
      chunk_rays = jax.tree_map(
          lambda r: jnp.pad(r, ((0, padding), (0, 0)), mode='edge'), chunk_rays)
    else:
      padding = 0
    # After padding the number of chunk_rays is always divisible by host_count.
    rays_per_host = chunk_rays.origins.shape[0] // jax.host_count()
    start, stop = host_id * rays_per_host, (host_id + 1) * rays_per_host
    chunk_rays = jax.tree_map(lambda r: utils.shard(r[start:stop]), chunk_rays)
    chunk_renderings = render_fn(rng, chunk_rays)

    # Unshard the renderings
    chunk_renderings = [{k: utils.unshard(v[0], padding)
                         for k, v in r.items()}
                        for r in chunk_renderings]

    chunk_rendering = chunk_renderings[-1]
    keys = [k for k in chunk_renderings[0] if k.find('ray_') == 0]
    for k in keys:
      chunk_rendering[k] = [r[k] for r in chunk_renderings]

    chunks.append(chunk_rendering)

  rendering = {}
  for k in chunks[0]:
    if isinstance(chunks[0][k], list):
      rendering[k] = [r[k] for r in chunks]
      ds = range(len(rendering[k][0]))
      rendering[k] = [jnp.concatenate([r[d] for r in rendering[k]]) for d in ds]
    else:
      rendering[k] = jnp.concatenate([r[k] for r in chunks])
      rendering[k] = (
          rendering[k].reshape((height, width) + chunks[0][k].shape[1:]))

  # After all of the ray bundles have been concatenated together, extract a
  # new random bundle (deterministically) from the concatenation that is the
  # same size as one of the individual bundles.
  keys = [k for k in rendering if k.find('ray_') == 0]
  if keys:
    ray_idx = random.permutation(
        random.PRNGKey(0), rendering[keys[0]][0].shape[0])[:config.vis_num_rays]
  for k in keys:
    rendering[k] = [r[ray_idx] for r in rendering[k]]

  return rendering
