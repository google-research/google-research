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

"""General math functions for NeRF."""
from absl import logging
import flax.jax_utils as flax_utils
import jax
import jax.numpy as jnp
from nf_diffusion.models.utils import resample


def matmul(a, b):
  """jnp.matmul defaults to bfloat16, but this helper function doesn't."""
  return jnp.matmul(a, b, precision=jax.lax.Precision.HIGHEST)


# Pose/ray math.
def generate_rays(pixel_coords, pix2cam, cam2world):
  """Generate camera rays from pixel coordinates and poses."""
  homog = jnp.ones_like(pixel_coords[Ellipsis, :1])
  pixel_dirs = jnp.concatenate([pixel_coords + 0.5, homog], axis=-1)[Ellipsis, None]
  cam_dirs = matmul(pix2cam, pixel_dirs)
  ray_dirs = matmul(cam2world[Ellipsis, :3, :3], cam_dirs)[Ellipsis, 0]
  ray_origins = jnp.broadcast_to(cam2world[Ellipsis, :3, 3], ray_dirs.shape)
  return ray_origins, ray_dirs


def pix2cam_matrix(height, width, focal, scene_inverse_y=False):
  """Inverse intrinsic matrix for a pinhole camera."""
  if scene_inverse_y:
    return jnp.array([
        [1.0 / focal, 0, -0.5 * width / focal],
        [0, -1.0 / focal, 0.5 * height / focal],
        [0, 0, -1.0],
    ])
  else:
    return jnp.array([
        [1.0 / focal, 0, -0.5 * width / focal],
        [0, 1.0 / focal, -0.5 * height / focal],
        [0, 0, 1.0],
    ])


def camera_ray_batch(cam2world, hwf):
  """Generate rays for a pinhole camera with given extrinsic and intrinsic."""
  height, width = int(hwf[0]), int(hwf[1])
  pix2cam = pix2cam_matrix(*hwf)
  pixel_coords = jnp.stack(
      jnp.meshgrid(jnp.arange(width), jnp.arange(height)), axis=-1
  )
  return generate_rays(pixel_coords, pix2cam, cam2world)


def _random_ray_batch_(rng, batch_size, data):
  """Generate a random batch of ray data."""
  keys = jax.random.split(rng, 3)
  cam_ind = jax.random.randint(keys[0], batch_size, 0, data['c2w'].shape[0])
  y_ind = jax.random.randint(keys[1], batch_size, 0, data['images'].shape[1])
  x_ind = jax.random.randint(keys[2], batch_size, 0, data['images'].shape[2])
  pixel_coords = jnp.stack([x_ind, y_ind], axis=-1)
  pix2cam = pix2cam_matrix(*data['hwf'])
  cam2world = data['c2w'][cam_ind, :3, :4]
  rays = generate_rays(pixel_coords, pix2cam, cam2world)
  pixels = data['images'][cam_ind, y_ind, x_ind]
  return rays, pixels


random_ray_batch = jax.jit(_random_ray_batch_, static_argnums=(1,))


def safe_exp(x):
  """jnp.exp() but with finite output and gradients for large inputs."""
  return jnp.exp(jnp.minimum(x, 88.0))  # jnp.exp(89) is infinity.


# Weighted resampling.


def interp_nd(*args):
  sh = args[0].shape
  args = [x.reshape([-1, x.shape[-1]]) for x in args]
  ret = jax.vmap(jnp.interp)(*args)
  ret = ret.reshape(sh)
  return ret


def resample_along_rays(t, weights, n_samples, rng):
  """resample_along_rays."""
  pdf = jnp.concatenate([jnp.zeros_like(weights[Ellipsis, :1]), weights + 1e-8], -1)
  pdf = pdf / pdf.sum(axis=-1, keepdims=True)
  cdf = jnp.cumsum(pdf, axis=-1)
  u = jnp.linspace(0.0, 1.0, n_samples)
  u = jnp.broadcast_to(u, cdf.shape[:-1] + (n_samples,))
  t = jnp.broadcast_to(t, cdf.shape)
  if rng is not None:
    delta = 1.0 / (n_samples - 1)
    u += (
        jax.random.uniform(
            rng, pdf.shape[:-1] + (n_samples,), minval=-0.5, maxval=0.5
        )
        * delta
    )

  # Inverse transform sampling
  t_new = interp_nd(u, cdf, t)
  return t_new


# Grid/hash gathers with trilinear interpolation.
def grid_trilerp(grid_values, coordinates, config):
  """Grid trilinear interpolation."""
  # Note: unlike hash_resample_3d, resample_3d expects integer coordinate voxel
  # centers, so we need to offset the coordinates by 0.5 here.
  coordinates_flat = coordinates.reshape((-1, 3)) - 0.5
  coordinates_flat = jnp.flip(coordinates_flat, axis=1)
  coordinates_3d = coordinates_flat.reshape([1, 1, -1, 3])
  result = resample.resample_3d(grid_values, coordinates_3d)
  num_channels = result.shape[-1]
  result = result.reshape(list(coordinates.shape[0:-1]) + [num_channels])
  result = result * config.model.preconditioner
  return result


# Rendering utilities
def pts_from_rays(rays, t, grid_min, grid_max, grid_size):
  t_mids = 0.5 * (t[Ellipsis, 1:] + t[Ellipsis, :-1])
  end_pts = rays[0][Ellipsis, None, :] + rays[1][Ellipsis, None, :] * t[Ellipsis, None]
  pts = rays[0][Ellipsis, None, :] + rays[1][Ellipsis, None, :] * t_mids[Ellipsis, None]
  pts_grid = (pts - grid_min) / (grid_max - grid_min)
  pts_grid *= grid_size
  return pts_grid, pts, end_pts, t_mids


def compute_volumetric_rendering_weights(density, end_pts):
  delta = jnp.linalg.norm(end_pts[Ellipsis, 1:, :] - end_pts[Ellipsis, :-1, :], axis=-1)
  density_delta = density * delta
  density_delta_shifted = jnp.concatenate(
      [jnp.zeros_like(density_delta[Ellipsis, :1]), density_delta[Ellipsis, :-1]], axis=-1
  )
  alpha = 1.0 - jnp.exp(-density_delta)
  trans = jnp.exp(-jnp.cumsum(density_delta_shifted, axis=-1))
  weights = alpha * trans
  return weights


def density_activation(feature, config):
  activation_type = config.model.get('density_activation_type', 'exp')
  if activation_type == 'exp':
    density = safe_exp(feature + config.model.density_offset)
  elif activation_type == 'softplus':
    density = jax.nn.softplus(feature + config.model.density_offset)
  else:
    raise ValueError('Invalid argument activation_type: %s' % activation_type)
  return density


def render_rays(rays, vox, rng, config, jitter=True):
  """Given rays, and voxel [reluf], and config, output colors."""
  grid_min = jnp.array([-1, -1, -1]) * config.trainer.scene_grid_scale
  grid_max = jnp.array([1, 1, 1]) * config.trainer.scene_grid_scale
  grid_size = vox.shape[-2]

  # First sample the acceleration grid at uniformly spaced sample points.
  sh = list(rays[0].shape[:-1])
  t = jnp.linspace(
      config.trainer.near, config.trainer.far, config.trainer.num_samples + 1
  )
  if jitter:
    delta = (
        config.trainer.far - config.trainer.near
    ) / config.trainer.num_samples
    key, rng = jax.random.split(rng)
    t += (
        jax.random.uniform(rng, sh + [t.shape[0]], minval=-0.5, maxval=0.5)
        * delta
    )

  pts_grid, _, end_pts, _ = pts_from_rays(
      rays, t, grid_min, grid_max, grid_size
  )
  mlp_density_features = grid_trilerp(vox, pts_grid, config)
  coarse_density = density_activation(mlp_density_features[Ellipsis, 0], config)

  weights = compute_volumetric_rendering_weights(coarse_density, end_pts)

  key = None
  if jitter:
    key, rng = jax.random.split(rng)

  # Resample fewer points according to the alpha compositing weights computed
  # from the acceleration grid.
  t = resample_along_rays(t, weights, config.trainer.num_resamples + 1, key)
  t = jax.lax.stop_gradient(t)
  pts_grid, _, end_pts, t_mids = pts_from_rays(
      rays, t, grid_min, grid_max, grid_size
  )

  # Now use the MLP to compute density...
  mlp_density_features = grid_trilerp(vox, pts_grid, config)
  mlp_density = density_activation(mlp_density_features[Ellipsis, 0], config)
  if config.trainer.get('color_act', True):
    colors = jax.nn.sigmoid(mlp_density_features[Ellipsis, 1:4])
  # If the first stage if trained with sigmoid(interp(x, feature)),
  # then we need to inverse this so that we don't run into artifacts.
  elif config.trainer.get('inverse_rgb', True):
    # Inverse the whole color grid, then do interpolation
    # TODO(guandao): this is waste of compute, but for debugging purpose
    # def inverse_sigmoid(y, eps=1e-6):
    # y = jnp.clip(y, eps, 1 - eps)
    # x = jnp.log(y / (1  - y))
    # return x
    def inverse_sigmoid(y):
      def safe_log(x, eps=1e-8):
        return jnp.log(jnp.clip(x, min=eps))

      return safe_log(y) - safe_log(1 - y)

    # NOTE: this is tricky since my data pipeline as divided by
    # config.model.preconditioner, so we need to recover that before inversing
    color_feature = (
        inverse_sigmoid(vox[Ellipsis, 1:4] * config.model.preconditioner)
        / config.model.preconditioner
    )
    colors_feature = grid_trilerp(color_feature, pts_grid, config)
    colors = jax.nn.sigmoid(colors_feature)
  else:
    colors = mlp_density_features[Ellipsis, 1:4]
  weights = compute_volumetric_rendering_weights(mlp_density, end_pts)
  depth = jnp.sum(weights * t_mids, axis=-1)
  acc = jnp.sum(weights, axis=-1)
  rgb = jnp.sum(weights[Ellipsis, None] * colors, axis=-2)

  # Composite onto the background color.
  if config.data.white_bkgd:
    rgb = rgb + (1.0 - acc[Ellipsis, None])

  return rgb, depth, acc, coarse_density, mlp_density, weights, t


def make_render_loop(
    vox, render_config, multi=False, send_to_cpu=False, verbose=False
):
  """Create rendering loop for a particular voxel."""

  def render_rays_test(rays, rng):
    rng = jax.random.fold_in(rng, jax.lax.axis_index('batch'))
    return render_rays(rays, vox, rng, render_config, jitter=False)

  if multi:
    render_rays_p = jax.pmap(render_rays_test, axis_name='batch')
  else:
    render_rays_p = jax.vmap(render_rays_test, axis_name='batch')

  def render_test(rays, rng):
    sh = rays[0].shape
    rays = [x.reshape((jax.local_device_count(), -1) + sh[1:]) for x in rays]
    out = render_rays_p(rays, flax_utils.replicate(rng))
    out = [x.reshape(sh[:-1] + (-1,)) for x in out]
    return out

  def render_loop(rays, rng):
    sh = list(rays[0].shape[:-1])
    rays = [x.reshape([-1, 3]) for x in rays]
    l = rays[0].shape[0]
    n = jax.local_device_count()
    p = ((l - 1) // n + 1) * n - l
    rays = [jnp.pad(x, ((0, p), (0, 0))) for x in rays]

    outs = []
    cnt = 0
    for i in range(0, rays[0].shape[0], render_config.trainer.chunk):
      if verbose and cnt % 100 == 0:
        logging.info('Render ray batch [%04d/%04d]', i, rays[0].shape[0])
      cnt += 1
      outi = render_test(
          [x[i : i + render_config.trainer.chunk] for x in rays],
          jax.random.fold_in(rng, i),
      )
      if send_to_cpu:
        outi = [jnp.array(x) for x in outi]
      outs.append(outi)
    outs = [
        jnp.reshape(jnp.concatenate([z[i] for z in outs])[:l], sh + [-1])
        for i in range(3)
    ]
    return outs

  return render_loop
