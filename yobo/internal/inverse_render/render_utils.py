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

# pylint: skip-file

import dataclasses
import inspect
import jax.numpy as jnp
import jax
from typing import Any
from types import FunctionType
import flax
from functools import partial

from google_research.yobo.internal import ref_utils
from google_research.yobo.internal import math as math_utils
from google_research.yobo.internal.inverse_render import math

# Make pytype skip this file. EnvmapSampler has Sampler1D as one of its fields,
# which breaks pytype for some reason...
# pytype: skip-file


def get_directions(envmap_H, envmap_W):
  omega_phi, omega_theta = jnp.meshgrid(
      jnp.linspace(-jnp.pi, jnp.pi, envmap_W + 1)[:-1]
      + 2.0 * jnp.pi / (2.0 * envmap_W),
      jnp.linspace(0.0, jnp.pi, envmap_H + 1)[:-1] + jnp.pi / (2.0 * envmap_H),
  )

  dtheta_dphi = (omega_theta[1, 1] - omega_theta[0, 0]) * (
      omega_phi[1, 1] - omega_phi[0, 0]
  )

  omega_theta = omega_theta.flatten()
  omega_phi = omega_phi.flatten()

  omega_x = jnp.sin(omega_theta) * jnp.cos(omega_phi)
  omega_y = jnp.sin(omega_theta) * jnp.sin(omega_phi)
  omega_z = jnp.cos(omega_theta)
  omega_xyz = jnp.stack([omega_x, omega_y, omega_z], axis=-1)

  return omega_theta, omega_phi, omega_xyz, dtheta_dphi


def get_rays(H, W, focal, c2w, rand_ort=False, key=None):
  """c2w: 4x4 matrix

  output: two arrays of shape [H, W, 3]
  """
  j, i = jnp.meshgrid(
      jnp.arange(W, dtype=jnp.float32) + 0.5,
      jnp.arange(H, dtype=jnp.float32) + 0.5,
  )

  if rand_ort:
    k1, k2 = jax.random.split(key)

    i += jax.random.uniform(k1, shape=(H, W)) - 0.5
    j += jax.random.uniform(k2, shape=(H, W)) - 0.5

  dirs = jnp.stack(
      [
          (j.flatten() - 0.5 * W) / focal,
          -(i.flatten() - 0.5 * H) / focal,
          -jnp.ones((H * W,), dtype=jnp.float32),
      ],
      -1,
  )  # shape [HW, 3]

  rays_d = math.matmul(dirs, c2w[:3, :3].T)  # shape [HW, 3]
  rays_o = c2w[:3, -1:].T.repeat(H * W, 0)
  return rays_o.reshape(H, W, 3), rays_d.reshape(H, W, 3)


def get_rays_at_pixel_coords(pixel_coords, H, W, focal, c2w, rand_ort=False, key=None):
  """c2w: 4x4 matrix

  output: two arrays of shape [pixel_coords.shape[0], 3]
  """
  sh = pixel_coords.shape[0]
  i, j = jnp.unravel_index(pixel_coords, (H, W))
  i += 0.5
  j += 0.5

  if rand_ort:
    k1, k2 = jax.random.split(key)

    i += jax.random.uniform(k1, shape=i.shape) - 0.5
    j += jax.random.uniform(k2, shape=j.shape) - 0.5

  dirs = jnp.stack(
      [
          (j.flatten() - 0.5 * W) / focal,
          -(i.flatten() - 0.5 * H) / focal,
          -jnp.ones_like(i.flatten(), dtype=jnp.float32),
      ],
      -1,
  )  # shape [sh, 3]

  rays_d = math.matmul(dirs, c2w[:3, :3].T)  # shape [sh, 3]
  rays_o = c2w[:3, -1:].T.repeat(sh, 0)
  return rays_o.reshape(sh, 3), rays_d.reshape(sh, 3)


def get_random_ray_offsets(N, focal, c2w, key, randomness='uniform'):
  """c2w: 4x4 matrix

  output: N random vector shifts for d
  """

  if randomness == 'uniform':
    di, dj = jax.random.uniform(key, shape=(2, N)) - 0.5
  elif randomness == 'gaussian':
    di, dj = jax.random.normal(key, shape=(2, N)) * 0.5
  else:
    raise ValueError('Only uniform or gaussian')

  delta_dirs = jnp.stack(
      [dj / focal, -di / focal, jnp.zeros((N,), dtype=jnp.float32)], -1
  )  # shape [N, 3]

  return math.matmul(delta_dirs, c2w[:3, :3].T)  # shape [N, 3]


def get_rotation_matrix(normal):
  # Get rotation matrix mapping [0, 0, 1] to normal.
  # new_z = normal
  # old_x = jnp.array([1.0, 0.0, 0.0])
  # dp = normal[0] #normal.dot(old_x) # if this is 1 we're not going to be happy...
  # new_x = (old_x - dp * normal) / jnp.sqrt(1.0 - dp ** 2)
  # new_y = jnp.cross(new_z, new_x)

  old_z = jnp.array([0.0, 0.0, 1.0])[None]
  old_y = jnp.array([0.0, 1.0, 0.0])[None]
  up = jnp.where(jnp.abs(normal[Ellipsis, 2:3]) < 0.9, old_z, old_y)
  new_x = jnp.cross(up, normal)
  new_x = new_x / (jnp.linalg.norm(new_x, axis=-1, keepdims=True) + 1e-10)
  new_z = normal
  new_y = jnp.cross(new_z, new_x)
  new_y = new_y / (jnp.linalg.norm(new_y, axis=-1, keepdims=True) + 1e-10)

  R = jnp.stack([new_x, new_y, new_z], axis=-1)
  return R


def test_get_rotation_matrix():
  normals = get_directions(50, 100)[2]
  diff = (
      jnp.abs(
          jax.vmap(lambda M: math.matmul(M.T, M))(
              jax.vmap(get_rotation_matrix)(normals)
          )
          - jnp.eye(3)[None, :, :]
      )
      .sum(-1)
      .sum(-1)
  )
  assert jnp.all(diff < 1e-5), "Rotation matrix isn't unitary"


# test_get_rotation_matrix()


def get_all_camera_rays(N_cameras, camera_dist, H, W, focal, hemisphere=True):
  theta, phi = math.nice_uniform_spherical(N_cameras, hemisphere)

  camera_x_vec = jnp.sin(theta) * jnp.cos(phi)
  camera_y_vec = jnp.sin(theta) * jnp.sin(phi)
  camera_z_vec = jnp.cos(theta)

  rays_o_vec = []
  rays_d_vec = []
  cameras = []
  for i in range(N_cameras):
    # camera = jnp.eye(4)
    # camera[0, 3] = camera_x_vec[i] * camera_dist
    # camera[1, 3] = camera_y_vec[i] * camera_dist
    # camera[2, 3] = camera_z_vec[i] * camera_dist

    camera_position = (
        jnp.array([camera_x_vec[i], camera_y_vec[i], camera_z_vec[i]])
        * camera_dist
    )
    zdir = camera_position / jnp.linalg.norm(camera_position)

    ydir = jnp.array([0.0, 0.0, 1.0])
    ydir -= zdir * zdir.dot(ydir)
    ydir += 1e-10 * jnp.array(
        [1.0, 0.0, 0.0]
    )  # make sure that cameras pointing straight down/up have a defined ydir
    ydir /= jnp.linalg.norm(ydir)

    xdir = jnp.cross(ydir, zdir)

    # camera[:3, 0] = xdir
    # camera[:3, 1] = ydir
    # camera[:3, 2] = zdir

    # camera = jnp.array([[]])
    camera_top_3x4 = jnp.stack([xdir, ydir, zdir, camera_position], axis=1)
    camera = jnp.concatenate(
        [camera_top_3x4, jnp.array([[0.0, 0.0, 0.0, 1.0]])], axis=0
    )

    cameras.append(camera)

    rays_o, rays_d = get_rays(H, W, focal, camera)

    rays_o_vec.append(rays_o)
    rays_d_vec.append(rays_d)

  rays_o_vec = jnp.stack(rays_o_vec, 0)
  rays_d_vec = jnp.stack(rays_d_vec, 0)
  cameras_vec = jnp.stack(cameras, 0)

  return rays_o_vec, rays_d_vec, cameras_vec


def get_all_cameras(N_cameras, camera_dist, hemisphere=True):
  theta, phi = math.nice_uniform_spherical(N_cameras, hemisphere)

  camera_x_vec = jnp.sin(theta) * jnp.cos(phi)
  camera_y_vec = jnp.sin(theta) * jnp.sin(phi)
  camera_z_vec = jnp.cos(theta)

  cameras = []
  for i in range(N_cameras):
    # camera = jnp.eye(4)
    # camera[0, 3] = camera_x_vec[i] * camera_dist
    # camera[1, 3] = camera_y_vec[i] * camera_dist
    # camera[2, 3] = camera_z_vec[i] * camera_dist

    camera_position = (
        jnp.array([camera_x_vec[i], camera_y_vec[i], camera_z_vec[i]])
        * camera_dist
    )
    zdir = camera_position / jnp.linalg.norm(camera_position)

    ydir = jnp.array([0.0, 0.0, 1.0])
    ydir -= zdir * zdir.dot(ydir)
    ydir += 1e-10 * jnp.array(
        [1.0, 0.0, 0.0]
    )  # make sure that cameras pointing straight down/up have a defined ydir
    ydir /= jnp.linalg.norm(ydir)

    xdir = jnp.cross(ydir, zdir)

    # camera[:3, 0] = xdir
    # camera[:3, 1] = ydir
    # camera[:3, 2] = zdir

    # camera = jnp.array([[]])
    camera_top_3x4 = jnp.stack([xdir, ydir, zdir, camera_position], axis=1)
    camera = jnp.concatenate(
        [camera_top_3x4, jnp.array([[0.0, 0.0, 0.0, 1.0]])], axis=0
    )

    cameras.append(camera)


  cameras_vec = jnp.stack(cameras, 0)

  return cameras_vec


@flax.struct.dataclass
class Sampler1D:
  x: Any
  n: Any
  cdf: Any
  integral: Any
  global_dirs: bool = False

  @classmethod
  @partial(jax.jit, static_argnums=(0,))
  def create(cls, x):
    # assert jnp.all(x >= 0.0)
    n = x.shape[0]
    cdf = jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(x) / n])
    integral = cdf[-1]
    cdf /= integral
    return cls(x, n, cdf, integral)

  # @partial(jax.jit, static_argnums=(2,))
  @jax.jit
  def sample(self, u):
    bin_ = jnp.searchsorted(self.cdf, u, side='right') - 1
    du = u - self.cdf[bin_]
    du /= self.cdf[bin_ + 1] - self.cdf[bin_]
    pdf = self.x[bin_] / self.integral

    sample = (bin_ + du) / self.n
    return sample, jax.lax.stop_gradient(pdf), bin_


# @flax.struct.dataclass
# class Sampler2D:
#   x: Any  # type: ignore
#   h: Any  # type: ignore
#   w: Any  # type: ignore
#   integral: Any  # type: ignore
#   v_sampler: Any  # type: ignore
#   u_given_v_cdfs: Any  # type: ignore

#   @classmethod
#   @partial(jax.jit, static_argnums=(0,))
#   def create(cls, x):
#     # assert jnp.all(x >= 0.0)
#     h, w = x.shape
#     integral = x.mean()  # This is assuming area = 1.
#     v_dist = x.mean(axis=1) / integral

#     v_sampler = Sampler1D.create(v_dist)

#     u_given_v_cdfs = jnp.concatenate(
#         [
#             jnp.zeros((h, 1)),
#             jnp.cumsum(x / integral / v_dist[:, None], axis=1) / w,
#         ],
#         axis=1,
#     )

#     conditional_integrals = u_given_v_cdfs[:, -1]
#     u_given_v_cdfs /= conditional_integrals[:, None]

#     return cls(x, h, w, integral, v_sampler, u_given_v_cdfs)

#   # @partial(jax.jit, static_argnums=(0, 3))
#   @jax.jit
#   def sample(self, u1, u2):
#     v_samples, marginal_pdf, v_indices = self.v_sampler.sample(u1)

#     distributions = self.u_given_v_cdfs[v_indices]

#     print(
#         'Can we avoid instantiating all of these distributions? Yes! Count'
#         ' u_indices and generate points according to that. Is that slow though?'
#     )
#     u_indices = (
#         jax.vmap(jnp.searchsorted, in_axes=(0, 0, None))(
#             distributions, u2, 'right'
#         )
#         - 1
#     )

#     du = u2 - self.u_given_v_cdfs[v_indices, u_indices]
#     du /= (
#         self.u_given_v_cdfs[v_indices, u_indices + 1]
#         - self.u_given_v_cdfs[v_indices, u_indices]
#     )

#     # print(u_indices.shape, v_indices.shape, self.conditional_integrals.shape)
#     # print(x[v_indices[:, None].repeat(self.w, 1),
#     #        u_indices[None, :].repeat(self.h, 0)].shape)
#     pdf = (
#         self.x[v_indices, u_indices] / self.integral
#     )  # / self.conditional_integrals[u_indices]

#     u_samples = (u_indices + du) / self.w
#     # print(distributions)
#     # u_sample = jax.vmap(sample_1d, in_axes=(0, 0, 0))(distributions, u2, v_indices)
#     return (
#         u_samples,
#         v_samples,
#         jax.lax.stop_gradient(pdf),
#         u_indices,
#         v_indices,
#     )



@jax.jit
def apply_3x3_binomial(envmap):
  padded = jnp.concatenate([envmap[:1, Ellipsis], envmap, envmap[-1:, Ellipsis]], axis=0)
  horizontal_blurred = (
      2.0 * padded + jnp.roll(padded, 1, axis=1) + jnp.roll(padded, -1, axis=1)
  )
  blurred = (
      2.0 * horizontal_blurred[1:-1, Ellipsis]
      + horizontal_blurred[:-2, Ellipsis]
      + horizontal_blurred[2:, Ellipsis]
  )

  return blurred / 16.0



@flax.struct.dataclass
class EnvmapSampler:
  x: Any
  h: Any
  w: Any
  integral: Any
  v_sampler: Any
  u_given_v_cdfs: Any
  sintheta: Any
  global_dirs: bool = False

  @classmethod
  @partial(jax.jit, static_argnums=(0,))
  def create(cls, envmap):
    # assert jnp.all(x >= 0.0)
    h, w = envmap.shape[:2]
    omega_theta = get_directions(h, w)[0]

    sintheta = jnp.sin(omega_theta).reshape(h, w)
    x = envmap.sum(-1) * sintheta
    #x = apply_3x3_binomial(x)

    integral = x.mean()  # This is assuming area = 1.
    v_dist = x.mean(axis=1) / integral

    v_sampler = Sampler1D.create(v_dist)

    u_given_v_cdfs = jnp.concatenate(
        [
            jnp.zeros((h, 1)),
            jnp.cumsum(x / integral / v_dist[:, None], axis=1) / w,
        ],
        axis=1,
    )

    conditional_integrals = u_given_v_cdfs[:, -1]
    u_given_v_cdfs /= conditional_integrals[:, None]

    return cls(x, h, w, integral, v_sampler, u_given_v_cdfs, sintheta)

  @jax.jit
  def sample(self, u1, u2):
    v_samples, marginal_pdf, v_indices = self.v_sampler.sample(u1)

    distributions = self.u_given_v_cdfs[v_indices]

    print(
        'Can we avoid instantiating all of these distributions? Yes! Count'
        ' u_indices and generate points according to that. Is that slow though?'
    )
    u_indices = (
        jax.vmap(jnp.searchsorted, in_axes=(0, 0, None))(
            distributions, u2, 'right'
        )
        - 1
    )

    du = u2 - self.u_given_v_cdfs[v_indices, u_indices]
    du /= (
        self.u_given_v_cdfs[v_indices, u_indices + 1]
        - self.u_given_v_cdfs[v_indices, u_indices]
    )

    # print(u_indices.shape, v_indices.shape, self.conditional_integrals.shape)
    # print(x[v_indices[:, None].repeat(self.w, 1),
    #        u_indices[None, :].repeat(self.h, 0)].shape)
    pdf = (
        self.x[v_indices, u_indices] / self.integral
    )  # / self.conditional_integrals[u_indices]

    u_samples = (u_indices + du) / self.w
    # print(distributions)
    # u_sample = jax.vmap(sample_1d, in_axes=(0, 0, 0))(distributions, u2, v_indices)
    return (
        u_samples,
        v_samples,
        jax.lax.stop_gradient(pdf),
        u_indices,
        v_indices,
    )

  @jax.jit
  def sample_directions(self, rng, u1, u2):
    envmap_u, envmap_v, pdf, ubins, vbins = self.sample(u1, u2)

    phi = envmap_u * 2.0 * jnp.pi - jnp.pi
    theta = envmap_v * jnp.pi
    costheta = jnp.cos(theta)
    sintheta = jnp.sin(theta)
    wi_x = sintheta * jnp.cos(phi)
    wi_y = sintheta * jnp.sin(phi)
    wi_z = costheta
    pdf /= 2.0 * jnp.pi**2 * self.sintheta[vbins, ubins]

    return jnp.stack([wi_x, wi_y, wi_z], axis=-1), jax.lax.stop_gradient(pdf)

  @jax.jit
  def pdf(self, directions):
    theta, phi = math.cart2sph(directions)
    envmap_u = (phi + jnp.pi) / 2.0 / jnp.pi
    envmap_v = theta / jnp.pi

    ubins = jnp.int32(jnp.floor(envmap_u * self.w))
    vbins = jnp.int32(jnp.floor(envmap_v * self.h))

    pdf = self.x[vbins, ubins] / self.integral

    # Apply spherical Jacobian correction factor
    pdf /= 2.0 * jnp.pi**2 * self.sintheta[vbins, ubins]

    return jax.lax.stop_gradient(pdf)



class MirrorSampler:
  global_dirs: bool = False

  def __init__(self):
    pass

  def sample_directions(self, rng, u1, u2, wo, _, kwargs):
    wi = math.reflect(wo)[None, :]
    pdf = jnp.ones_like(wi[Ellipsis, 0])  # jnp.ones_like(u1)
    return wi, pdf

  def pdf(self, wo, wi, _):
    return jnp.zeros_like(wi[Ellipsis, 2])


@flax.struct.dataclass
class QuadratureEnvmapSampler:
  sintheta: Any
  omega_xyz: Any
  global_dirs: bool = False

  @classmethod
  @partial(jax.jit, static_argnums=(0, 1, 2))
  def create(cls, envmap_H, envmap_W):
    omega_theta, _, omega_xyz, _ = get_directions(envmap_H, envmap_W)
    sintheta = jnp.sin(omega_theta)
    return cls(sintheta, omega_xyz)

  @jax.jit
  def sample_directions(self, rng, _, __):
    pdf = 1.0 / (2.0 * jnp.pi**2 * self.sintheta)
    return self.omega_xyz, pdf

  @jax.jit
  def pdf(self, directions):
    """This shouldn't really be called by multiple importance sampling, but I think it should still work?"""
    curr_sintheta = jnp.sqrt(1.0 - directions[Ellipsis, 2] ** 2)
    pdf = 1.0 / (2.0 * jnp.pi**2 * curr_sintheta)


@flax.struct.dataclass
class RandomGenerator2D:
  h_blocks: Any
  w_blocks: Any
  stratified: Any

  @classmethod
  def create(cls, n, stratified):
    h_blocks = int(2 ** jnp.int32(jnp.floor((jnp.log2(n) - 1) / 2.0)))
    w_blocks = h_blocks * 2
    h_shifts = (
        jnp.linspace(0.0, 1.0, w_blocks + 1)[:-1][None, :]
        .repeat(n // w_blocks, 0)
        .flatten()
    )
    w_shifts = (
        jnp.linspace(0.0, 1.0, h_blocks + 1)[:-1][:, None]
        .repeat(n // h_blocks, 1)
        .flatten()
    )
    return cls(h_blocks, w_blocks, h_shifts, w_shifts, stratified)

  # @functools.partial(jax.jit, static_argnames=['n', 'stratified'])
  def sample(self, rng, n, _):
    # Generate uniform samples on the top hemisphere
    key, rng = random_split(rng)
    u = jax.random.uniform(key, shape=(n, 2))
    uh = u[Ellipsis, 0]
    uw = u[Ellipsis, 1]

    if self.stratified:
      h_shifts = (
          jnp.linspace(0.0, 1.0, self.w_blocks + 1)[:-1][None, :]
          .repeat(n // self.w_blocks, 0)
          .flatten()
      )
      w_shifts = (
          jnp.linspace(0.0, 1.0, self.h_blocks + 1)[:-1][:, None]
          .repeat(n // self.h_blocks, 1)
          .flatten()
      )

      uh = jnp.clip(
          h_shifts + uh / self.w_blocks,
          0.0,
          1.0 - jnp.finfo(jnp.float32).eps,
      )
      uw = jnp.clip(
          w_shifts + uw / self.h_blocks,
          0.0,
          1.0 - jnp.finfo(jnp.float32).eps,
      )

    return uh, uw


@flax.struct.dataclass
class DummySampler2D:
  global_dirs: bool = False

  def sample(self, _, __, ___):
    return None, None


@flax.struct.dataclass
class UniformSphereSampler:
  global_dirs: bool = False

  @jax.jit
  def sample_directions(self, rng, u1, u2, wo, _, kwargs):
    costheta = 1.0 - 2.0 * u1
    sintheta = jnp.sqrt((1.0 - u1) * 4.0 * u1)
    phi = u2 * 2.0 * jnp.pi - jnp.pi
    wi_x = sintheta * jnp.cos(phi)
    wi_y = sintheta * jnp.sin(phi)
    wi_z = costheta
    pdf = 1 / 4.0 / jnp.pi * jnp.ones_like(phi)
    return jnp.stack([wi_x, wi_y, wi_z], axis=-1), pdf

  @jax.jit
  def pdf(self, wo, wi, _, kwargs):
    return 1 / 4.0 / jnp.pi * jnp.ones_like(wi[Ellipsis, 2])


# alpha = 0.5


# @flax.struct.dataclass
# class HalfEnvmapSampler(Sampler2D):

#   @classmethod
#   def create(self, envmap):
#     x = envmap.sum(-1) * jnp.sin(omega_theta).reshape(envmap_H, envmap_W)
#     return super().create(x)

#   @jax.jit
#   def sample_directions(self, u1, u2):
#     selector = u1 < alpha
#     u1 = jnp.mod(2.0 * u1, 1.0)
#     envmap_u, envmap_v, pdf, ubins, vbins = self.sample(u1, u2)

#     phi = envmap_u * 2.0 * jnp.pi - jnp.pi
#     theta = envmap_v * jnp.pi
#     costheta = jnp.cos(theta)
#     sintheta = jnp.sin(theta)
#     wi_x = sintheta * jnp.cos(phi)
#     wi_y = sintheta * jnp.sin(phi)
#     wi_z = costheta
#     pdf /= (
#         2.0
#         * jnp.pi**2
#         * jnp.sin(omega_theta.reshape(envmap_H, envmap_W)[vbins, ubins])
#     )

#     envmap_directions = jnp.stack([wi_x, wi_y, wi_z], axis=-1)

#     uniform_directions, _ = UniformSphereSampler().sample_directions(
#         u1, u2, None
#     )

#     directions = jnp.where(
#         selector[..., None], envmap_directions, uniform_directions
#     )
#     pdf = jax.lax.stop_gradient(pdf) * alpha + (1.0 - alpha) / jnp.pi / 4.0

#     return directions, pdf

#   @jax.jit
#   def pdf(self, directions):
#     theta, phi = math.cart2sph(directions)
#     envmap_u = (phi + jnp.pi) / 2.0 / jnp.pi
#     envmap_v = theta / jnp.pi

#     ubins = jnp.int32(jnp.floor(envmap_u * envmap_W))
#     vbins = jnp.int32(jnp.floor(envmap_v * envmap_H))

#     pdf = self.x[vbins, ubins] / self.integral

#     # Apply spherical Jacobian correction factor
#     pdf /= (
#         2.0
#         * jnp.pi**2
#         * jnp.sin(omega_theta.reshape(envmap_H, envmap_W)[vbins, ubins])
#     )

#     return jax.lax.stop_gradient(pdf) * alpha + (1.0 - alpha) / jnp.pi / 4.0


class UniformHemisphereSampler:
  global_dirs: bool = False

  def __init__(self):
    pass

  def sample_directions(self, rng, u1, u2, wo, _, kwargs):
    costheta = 1.0 - u1
    sintheta = jnp.sqrt((2.0 - u1) * u1)
    phi = u2 * 2.0 * jnp.pi - jnp.pi
    wi_x = sintheta * jnp.cos(phi)
    wi_y = sintheta * jnp.sin(phi)
    wi_z = costheta
    pdf = 1 / 2.0 / jnp.pi * jnp.ones_like(phi)
    return jnp.stack([wi_x, wi_y, wi_z], axis=-1), pdf

  def pdf(self, wo, wi, _, kwargs):
    pdf = 1 / 2.0 / jnp.pi * jnp.ones_like(wi[Ellipsis, 2])

    pdf = jnp.where(
        wi[Ellipsis, 2] < 0,
        jnp.zeros_like(pdf),
        pdf,
    )

    return pdf


class CosineSampler:
  global_dirs: bool = False

  def __init__(self):
    pass

  def sample_directions(self, rng, u1, u2, wo, _, kwargs):
    r = jnp.sqrt(u1)
    phi = u2 * 2.0 * jnp.pi - jnp.pi
    wi_x = r * jnp.cos(phi)
    wi_y = r * jnp.sin(phi)
    eps = jnp.finfo(jnp.float32).eps
    wi_z = jnp.sqrt(jnp.maximum(eps, 1.0 - wi_x**2 - wi_y**2))
    pdf = wi_z / jnp.pi
    return jnp.stack([wi_x, wi_y, wi_z], axis=-1), pdf

  def pdf(self, wo, wi, _, kwargs):
    pdf = wi[Ellipsis, 2] / jnp.pi

    pdf = jnp.where(
        wi[Ellipsis, 2] < 0,
        jnp.zeros_like(pdf),
        pdf,
    )

    return pdf


def GGX_D(costheta, a):
  eps = jnp.finfo(jnp.float32).eps
  return a**2 / jnp.maximum(eps, jnp.pi * ((costheta ** 2 * (a ** 2 - 1.) + 1.)) ** 2)


@flax.struct.dataclass
class MicrofacetSampler:
  sample_visible: bool = False
  global_dirs: bool = False

  def trowbridge_reitz_sample_11(self, u1, u2, costheta):
    pass

  def trowbridge_reitz_sample(self, u1, u2, alpha, wi):
    """
    https://github.com/mmp/pbrt-v3/blob/aaa552a4b9cbf9dccb71450f47b268e0ed6370e2/src/core/microfacet.cp284
    """
    pass

  def sample_normals(self, u1, u2, alpha):
    if self.sample_visible:
      raise NotImplementedError('')
    else:
      eps = jnp.finfo(jnp.float32).eps
      tantheta2 = alpha ** 2 * u1 / jnp.maximum(1.0 - u1, eps)
      costheta = 1.0 / jnp.sqrt(jnp.maximum(1.0 + tantheta2, eps))
      sintheta = jnp.sqrt(jnp.maximum(eps, 1.0 - costheta ** 2))
      phi = u2 * 2.0 * jnp.pi - jnp.pi
      nx = sintheta * jnp.cos(phi)
      ny = sintheta * jnp.sin(phi)
      nz = costheta

      pdf = GGX_D(costheta, alpha) * jnp.abs(costheta)

      return jnp.stack([nx, ny, nz], axis=-1), pdf

  def sample_directions(self, rng, u1, u2, wo, alpha, kwargs):
    normals, normal_pdf = self.sample_normals(u1, u2, alpha[Ellipsis, 0])
    directions = jax.vmap(math.reflect, in_axes=(0, 0))(wo, normals)
    eps = jnp.finfo(jnp.float32).eps
    jac = 1.0 / jnp.maximum(4.0 * jnp.sum(wo * normals, axis=-1), eps)
    pdf = normal_pdf * jac
    return directions, pdf


  def pdf(self, wo, wi, alpha, kwargs):
    wh = math.normalize(wo + wi)
    eps = jnp.finfo(jnp.float32).eps
    jac = 1.0 / jnp.maximum(4.0 * jnp.sum(wo * wh, axis=-1), eps)
    pdf = GGX_D(wh[Ellipsis, 2], alpha[Ellipsis, 0]) * jnp.abs(wh[Ellipsis, 2]) * jac

    pdf = jnp.where(
        wh[Ellipsis, 2] < 0,
        jnp.zeros_like(pdf),
        pdf,
    )

    return pdf


class DefensiveMicrofacetSampler:
  microfacet_sampler: MicrofacetSampler
  cosine_sampler: CosineSampler
  global_dirs: bool = False


  def sample_directions(self, rng, u1, u2, wo, alpha):
    pass


  def pdf(self, wo, wi, alpha):
    pass


# @functools.partial(jax.jit, static_argnums=(3,))
def get_lobe(wi, wo, materials, brdf_correction, config):
  """Compute BRDF in local coordinates.

  wi: incoming light directions, shape [N, 3]
  wo: outgoing light direction, shape [3]
  materials: dictionary with elements of shapes [3]

  return values of BRDF evaluated at wi, wo. Shape is [N, 3]
  """
  # print(wi.shape, wo.shape, materials['specular_albedo'].shape)
  # assert opts.shading in ['lambertian', 'phong', 'blinnphong', 'mirror']

  if config.shading in ['mirror']:
    return 1.0  # / wo[..., 2][..., None]

  lobe = 0.0
  if config.shading in ['lambertian', 'phong', 'blinnphong', 'microfacet']:
    lobe = (
        jnp.maximum(0.0, wi[Ellipsis, 2:])
        * materials['albedo'][Ellipsis, None, :]
        / jnp.pi
    )

  if config.shading == 'microfacet':
    assert 'roughness' in materials.keys() and 'F_0' in materials.keys()
    eps = jnp.finfo(jnp.float32).eps
    roughness = materials['roughness'][Ellipsis, None, :]
    F_0 = materials['F_0'][Ellipsis, None, :]

    albedo = materials['albedo'][Ellipsis, None, :]
    specular_albedo = materials['specular_albedo'][Ellipsis, None, :]
    metalness = materials['metalness'][Ellipsis, None, :]
    diffuseness = materials['diffuseness'][Ellipsis, None, :]
    F_0 = specular_albedo * metalness + F_0 * (1.0 - metalness)

    halfdirs = math.normalize(wi + wo)
    n_dot_v = jnp.maximum(0., wo[Ellipsis, 2:])
    n_dot_l = jnp.maximum(0., wi[Ellipsis, 2:])
    n_dot_h = jnp.maximum(0., halfdirs[Ellipsis, 2:])
    l_dot_h = jnp.maximum(0., jnp.sum(wi * halfdirs, axis=-1, keepdims=True))
    a = roughness ** 2

    def fresnel(cos_theta):
      return F_0 + (1. - F_0) * jnp.power(jnp.maximum(eps, 1. - cos_theta), 5)

    D = GGX_D(n_dot_h, a)
    F = fresnel(l_dot_h)
    k = (roughness + 1.) ** 2 / 8.
    G = (n_dot_v / jnp.maximum(eps, n_dot_v * (1. - k) + k)) * (n_dot_l / jnp.maximum(eps, n_dot_l * (1. - k) + k))
    ggx_lobe = D * F * G / jnp.maximum(eps, 4. * n_dot_v * n_dot_l)
    lambertian_lobe = n_dot_l * albedo / jnp.pi

    if config.use_brdf_correction:
      lobe = (
          (
              ggx_lobe
              * brdf_correction[Ellipsis, 0:1]
          ) * (1.0 - diffuseness)
          + (
              lambertian_lobe
              * brdf_correction[Ellipsis, 1:2]
          ) * diffuseness
      )
    else:
      diffuse_weight = (1. - fresnel(n_dot_l)) * (1. - fresnel(l_dot_h))
      lobe = (
          (
              ggx_lobe + diffuse_weight * lambertian_lobe
          ) * (1.0 - diffuseness)
          + (
              lambertian_lobe
          ) * diffuseness
      )

  if config.shading == 'phong':
    assert 'specular_albedo' in materials.keys()
    specular_albedo = materials['specular_albedo'][Ellipsis, None, :]
    exponent = materials['specular_exponent'][Ellipsis, None, :]
    refdir = math.reflect(wo)

    # No need to normalize because ||n|| = 1 and ||d|| = 1, so ||2(n.d)n - d|| = 1.
    print('Not normalizing here (because unnecessary, at least theoretically).')
    # refdirs /= (jnp.linalg.norm(refdirs, axis=-1, keepdims=True) + 1e-10)  # [N, HW, envmap_H, envmap_W, 3]

    lobe += (
        specular_albedo
        * jnp.maximum(0.0, (refdir * wi).sum(-1, keepdims=True)) ** exponent
    )

    # lobe += jnp.maximum(0.0, wi[..., 2:]) * materials['albedo'][..., None, :] / jnp.pi

  """
    if shading == 'blinnphong':
        assert 'specular_albedo' in materials.keys()
        specular_albedo = materials['specular_albedo'][:, None, None, :]
        exponent = materials['specular_exponent'][:, None, None, :]

        d_norm_sq = (rays_d ** 2).sum(-1, keepdims=True)
        rays_d_norm = -rays_d / jnp.sqrt(d_norm_sq + 1e-10)

        halfvectors = omega_xyz.reshape(1, envmap_H, envmap_W, 3) + rays_d_norm[:, None, None, :]
        halfvectors /= (jnp.linalg.norm(halfvectors, axis=-1, keepdims=True) + 1e-10)  # [N, envmap_H, envmap_W, 3]

        lobes += jnp.maximum(0.0, (halfvectors * normals[:, None, None, :]).sum(-1, keepdims=True)) ** exponent * specular_albedo

    """
  return lobe


def global_to_local(directions, R):
  return (
      directions[Ellipsis, 0:1] * R[Ellipsis, 0, :]
      + directions[Ellipsis, 1:2] * R[Ellipsis, 1, :]
      + directions[Ellipsis, 2:3] * R[Ellipsis, 2, :]
  )

def local_to_global(directions, R):
  return (
      directions[Ellipsis, 0:1] * R[Ellipsis, 0]
      + directions[Ellipsis, 1:2] * R[Ellipsis, 1]
      + directions[Ellipsis, 2:3] * R[Ellipsis, 2]
  )


def query_spherical_img(envmap, theta, phi, method='bilinear'):
  envmap_H, envmap_W = envmap.shape[:2]
  if method == 'nearest':
    r = envmap_H * theta / jnp.pi  # map from [0, pi) to [0, envmap_H)
    c = (
        envmap_W * (phi + jnp.pi) / jnp.pi / 2.0
    )  # map from [-pi, pi) to [0, envmap_W)

    r = jnp.mod(r, envmap_H).astype(
        jnp.int32
    )  # jnp.clip(jnp.int32(jnp.floor(r)), 0, envmap_H-1)
    c = jnp.mod(c, envmap_W).astype(
        jnp.int32
    )  # jnp.clip(jnp.int32(jnp.floor(c)), 0, envmap_W-1)

    # masked_envmap = envmap[r, c, :] * mask[r, c, None]
    return envmap[r, c, :]
  elif method == 'bilinear':
    y = (
        (envmap_H - 1) * theta / jnp.pi
    )  # - 0.5  # map from [0, pi) to [0, envmap_H)
    print(
        'I think this is wrong (similarly to mitsuba). It needs to be envmap_H'
        ' * theta / np.pi'
    )
    x = (
        envmap_W * (phi + jnp.pi) / jnp.pi / 2.0 - 0.5
    )  # map from [-pi, pi) to [0, envmap_W)

    x1_float = jnp.floor(x)
    x1 = jnp.mod(
        x1_float.astype(jnp.int32), envmap_W
    )  # This might be unnecessary?
    x2 = jnp.mod(x1 + 1, envmap_W)  # Wrap phi around
    y1_float = jnp.floor(y)
    y1 = jnp.clip(
        y1_float.astype(jnp.int32), 0, envmap_H - 1
    )  # This might be unnecessary?
    y2 = jnp.clip(
        y1 + 1, 0, envmap_H - 1
    )  # Clamp theta to {0, ..., envmap_H-1}

    envmap_11 = envmap[y1, x1, :]
    envmap_12 = envmap[y2, x1, :]
    envmap_21 = envmap[y1, x2, :]
    envmap_22 = envmap[y2, x2, :]

    # First interpolate in the x direction
    wx2 = x - x1_float
    wx1 = 1.0 - wx2

    interp_1 = wx1[Ellipsis, None] * envmap_11 + wx2[Ellipsis, None] * envmap_21
    interp_2 = wx1[Ellipsis, None] * envmap_12 + wx2[Ellipsis, None] * envmap_22

    # Now interpolate along y
    wy2 = y - y1_float
    wy1 = 1.0 - wy2

    return wy1[Ellipsis, None] * interp_1 + wy2[Ellipsis, None] * interp_2

  else:
    raise RuntimeError('Not implemented')


def render_pixel_mc(
    rng,
    normal,
    global_viewdirs,
    pts,
    material,
    envmap,
    mask,
    inv_rad,
    envmap_sampler,
    material_sampler,
    random_generator_2d,
    get_visibility,
    config,
):
  # Get random samples in [0, 1)^2
  key, rng = random_split(rng)
  uh, uw = random_generator_2d.sample(
      key, config.num_shadow_rays, config.stratified
  )
  assert (material_sampler is None) ^ (envmap_sampler is None)
  R = get_rotation_matrix(normal)

  local_viewdirs = global_to_local(global_viewdirs, R)
  # Get material samples
  if material_sampler is not None:
    local_lightdirs, pdf = jax.lax.stop_gradient(material_sampler.sample_directions(
        uh, uw, local_viewdirs, material.get('roughness', None)
    ))

    global_lightdirs = local_to_global(local_lightdirs, R)

  # Get light samples
  else:
    global_lightdirs, pdf = jax.lax.stop_gradient(envmap_sampler.sample_directions(uh, uw))
    local_lightdirs = global_to_local(global_lightdirs, R)

  theta_global, phi_global = math.cart2sph(global_lightdirs)

  dp = (pts * global_lightdirs).sum(axis=-1, keepdims=True) * inv_rad
  norm_o_sq = (pts**2).sum(axis=-1, keepdims=True)

  eps = jnp.finfo(jnp.float32).eps
  occlusion_directions = (
      pts * inv_rad
      + (jnp.sqrt(jnp.maximum(eps, dp**2 + 1.0 - norm_o_sq * inv_rad**2)) - dp)
      * global_lightdirs
  )
  theta_occ, phi_occ = math.cart2sph(occlusion_directions)

  masked_envmap = query_spherical_img(
      envmap, theta_global, phi_global, 'nearest'
  ) * query_spherical_img(
      mask[Ellipsis, None], theta_occ, phi_occ, 'bilinear'
  )  # This is different! interpolate(x)*interpolate(y) != interpolate(x*y)
  # masked_envmap = query_spherical_img(envmap * mask[..., None], theta_global, phi_global)

  if config.self_occlusions:
    visibility = get_visibility(pts+1e-6*normal, global_lightdirs)
    masked_envmap = visibility * masked_envmap

  # local_viewdirs = global_to_local(global_viewdirs, R)
  lobe = get_lobe(local_lightdirs, local_viewdirs, material, config)

  denominator = jnp.maximum(pdf[Ellipsis, None], eps)
  res = (masked_envmap * lobe / denominator).mean(0)
  #return jnp.where(jnp.sum(normal ** 2) > eps, res, jnp.zeros_like(res))
  return jnp.where(jnp.any(jnp.isinf(pts)), jnp.zeros_like(res), res)



def render_pixel_mc_mis(
    rng,
    normal,
    global_viewdirs,
    pts,
    material,
    envmap,
    mask,
    inv_rad,
    envmap_sampler,
    material_sampler,
    random_generator_2d,
    get_visibility,
    config,
):
  """Multiple importance sampling version of `render_pixel_mc`."""
  # Get random samples in [0, 1)^2

  n1 = config.num_shadow_rays // 2
  n2 = config.num_shadow_rays - n1
  key, rng = random_split(rng)
  key1, key2 = jax.random.split(key)
  uh1, uw1 = random_generator_2d.sample(
      key1, n1, config.stratified
  )
  uh2, uw2 = random_generator_2d.sample(
      key2, n2, config.stratified
  )

  R = get_rotation_matrix(normal)
  local_viewdirs = global_to_local(global_viewdirs, R)


  # Get material samples
  local_viewdirs = global_to_local(global_viewdirs, R)
  local_lightdirs1, pdf1 = jax.lax.stop_gradient(material_sampler.sample_directions(
      uh1, uw1, local_viewdirs, material.get('roughness', None)
  ))
  global_lightdirs1 = local_to_global(local_lightdirs1, R)

  # Get light samples
  global_lightdirs2, pdf2 = jax.lax.stop_gradient(envmap_sampler.sample_directions(uh2, uw2))
  local_lightdirs2 = global_to_local(global_lightdirs2, R)

  theta_global1, phi_global1 = math.cart2sph(global_lightdirs1)
  theta_global2, phi_global2 = math.cart2sph(global_lightdirs2)

  # (dot(x, wi) * kappa)^2 + 1 - ||x||^2 * kappa ^2 > (dot(x, wi) * kappa)^2 - 8 >

  eps = jnp.finfo(jnp.float32).eps

  dp1 = (pts * global_lightdirs1).sum(axis=-1, keepdims=True) * inv_rad
  norm_o_sq = (pts**2).sum(axis=-1, keepdims=True)

  occlusion_directions1 = (
      pts * inv_rad
      + (jnp.sqrt(jnp.maximum(dp1**2 + 1.0 - norm_o_sq * inv_rad**2, eps)) - dp1)
      * global_lightdirs1
  )
  theta_occ1, phi_occ1 = math.cart2sph(occlusion_directions1)

  dp2 = (pts * global_lightdirs2).sum(axis=-1, keepdims=True) * inv_rad
  norm_o_sq = (pts**2).sum(axis=-1, keepdims=True)
  occlusion_directions2 = (
      pts * inv_rad
      + (jnp.sqrt(jnp.maximum(dp2**2 + 1.0 - norm_o_sq * inv_rad**2, eps)) - dp2)
      * global_lightdirs2
  )
  theta_occ2, phi_occ2 = math.cart2sph(occlusion_directions2)


  masked_envmap1 = query_spherical_img(
      envmap, theta_global1, phi_global1, 'nearest'
  ) * query_spherical_img(
      mask[Ellipsis, None], theta_occ1, phi_occ1, 'bilinear'
  )
  masked_envmap2 = query_spherical_img(
      envmap, theta_global2, phi_global2, 'nearest'
  ) * query_spherical_img(
      mask[Ellipsis, None], theta_occ2, phi_occ2, 'bilinear'
  )

  if config.self_occlusions:
    pts2 = (pts+1e-6*normal)[None, :].repeat(global_lightdirs2.shape[0], 0)
    visibility1 = get_visibility(pts2, global_lightdirs1)
    visibility2 = get_visibility(pts2, global_lightdirs2)
    print(visibility1.shape, visibility2.shape)
    masked_envmap1 = visibility1 * masked_envmap1
    masked_envmap2 = visibility2 * masked_envmap2


  # masked_envmap1 = envmap[r1, c1, :] * mask[r1, c1, None]
  # masked_envmap2 = envmap[r2, c2, :] * mask[r2, c2, None]
  # print(directions_global)
  lobe1 = get_lobe(local_lightdirs1, local_viewdirs, material, config)
  lobe2 = get_lobe(local_lightdirs2, local_viewdirs, material, config)

  beta = 2
  denom1 = (
      (n1 * pdf1) ** beta
      + (n2 * envmap_sampler.pdf(global_lightdirs1)) ** beta
      + 1e-10
  )
  weights1 = (n1 * pdf1) ** beta / denom1
  denom2 = (
      (n1 * material_sampler.pdf(local_viewdirs, local_lightdirs2, material.get('roughness', None))) ** beta
      + (n2 * pdf2) ** beta
      + 1e-10
  )
  weights2 = (n2 * pdf2) ** beta / denom2

  denominator1 = jnp.maximum(pdf1[Ellipsis, None], eps)
  denominator2 = jnp.maximum(pdf2[Ellipsis, None], eps)

  res = (
      masked_envmap1 * lobe1 * weights1[Ellipsis, None] / denominator1
  ).mean(0)
  res += (
      masked_envmap2 * lobe2 * weights2[Ellipsis, None] / denominator2
  ).mean(0)
  return jnp.where(jnp.any(jnp.isinf(pts)), jnp.zeros_like(res), res)


def render(
    rng,
    envmap,
    mask,
    inv_radii,
    pixel_inds,
    hwf,
    camera,
    material_params,
    normal_params=None,
    envmap_sampler=None,
    material_sampler=None,
    random_generator_2d=None,
    pts_to_materials=lambda: None,
    get_intersection_and_normals=lambda: None,
    get_refined_normals=None,
    get_visibility=lambda: None,
    config=None,
):
  """envmap:     shape [h, w, 3]

  mask:       shape [h, w]
  materials:  dictionary with entries of shape [N, 3]
  normals:    shape [N, 3]
  pixel_inds:     shape [N]
  alpha:      shape [N, 1]

  output: rendered colors, shape [N, 3]

  ***

  If envmap_sampler is None, this function creates its own internal sampler.
  """
  rng, key = jax.random.split(rng)

  rays_o, rays_d = get_rays_at_pixel_coords(pixel_inds, hwf[0], hwf[1], hwf[2],
                                            camera, rand_ort=config.jitter,
                                            key=key)

  pts, alpha, normals = get_intersection_and_normals(
      rays_o, rays_d
  )


  if get_refined_normals is not None:
    normals = get_refined_normals(normal_params, pts)
    normals = math.normalize(normals)

  materials = pts_to_materials(material_params, pts, rays_o, rays_d)

  keys = jax.random.split(rng, num=rays_d.shape[0])

  # if envmap_sampler is None and opts.shading != 'mirror':
  #  envmap_sampler = EnvmapSampler.create(envmap * mask[:, :, None])

  if material_sampler is None or envmap_sampler is None:
    if material_sampler is None and envmap_sampler is None:
      material_sampler = UniformSphereSampler()
    single_pixel_render_fn = render_pixel_mc
  else:
    single_pixel_render_fn = render_pixel_mc_mis

  wo = -math.normalize(rays_d)  # Flip from view direction to outgoing direction, and normalize
  colors = jax.vmap(
      single_pixel_render_fn,
      in_axes=(0, 0, 0, 0, 0, None, None, None, None, None, None, None, None),
  )(
      keys,
      normals,
      wo,
      pts,
      materials,
      envmap,
      mask,
      inv_radii,
      envmap_sampler,
      material_sampler,
      random_generator_2d,
      get_visibility,
      config,
  )

  return colors, alpha


# def get_gt_normals(xyz, shape='sphere'):
#   assert shape in ['sphere']
#   if shape == 'sphere':
#     normals = xyz / jnp.linalg.norm(xyz, axis=-1, keepdims=True)
#   return normals


@partial(jax.jit, static_argnums=(2, 3, 4))
def get_materials_gt(
    xyz, rad, texture='white', shading='lambertian'
):
  """xyz:  shape [N, 3]

  rad:  float

  returns: materials: dictionary with entries of shape [N, 3]
         normals: shape [N, 3]
  """
  x = xyz[:, 0:1]
  y = xyz[:, 1:2]
  z = xyz[:, 2:3]

  if texture == 'white':
    albedo = jnp.ones((x.shape[0], 3), dtype=jnp.float32)

  elif texture == 'checkerboard':
    theta = jnp.arctan2(jnp.sqrt(x**2 + y**2), z)
    phi = jnp.arctan2(y, x)

    N_phi_stripes = 10
    N_theta_stripes = 10
    phi_mult = jnp.where(
        jnp.remainder(N_phi_stripes * (phi + jnp.pi) / (2 * jnp.pi), 2) < 1,
        1.0,
        -1.0,
    )
    theta_mult = jnp.where(
        jnp.remainder(N_theta_stripes * theta / jnp.pi, 2) < 1, 1.0, -1.0
    )

    clr1 = jnp.array([1.0, 1.0, 1.0])[None, :]
    # clr2 = jnp.array([1.0, 0.3, 0.3])[None, :]
    clr2 = jnp.array([0.3, 1.0, 0.3])[None, :]
    albedo = jnp.where(phi_mult * theta_mult > 0.0, clr1, clr2)

  elif texture == 'gradient_checkerboard':
    theta = jnp.arctan2(jnp.sqrt(x**2 + y**2), z)
    phi = jnp.arctan2(y, x)

    N_phi_stripes = 8
    N_theta_stripes = 10
    # triangle wave
    triangle = (
        lambda x, num: 2.0
        * num
        * jnp.abs(jnp.remainder(x - num / 8.0, 2.0 / num) - 1.0 / num)
        - 1.0
    )

    # Triangle wave in [-1, 1] with period 2/number_of_stripes
    phi_mult = triangle((phi + jnp.pi) / (2.0 * jnp.pi), N_phi_stripes)
    theta_mult = triangle(theta / jnp.pi, N_theta_stripes)

    mix = phi_mult * theta_mult * 0.5 + 0.5  # map to [0, 1]

    clr1 = jnp.array([1.0, 1.0, 1.0])[None, :]
    clr2 = jnp.array([1.0, 0.3, 0.3])[None, :]
    albedo = mix * clr1 + (1.0 - mix) * clr2
  elif texture == 'noisy_checkerboard':
    theta = jnp.arctan2(jnp.sqrt(x**2 + y**2), z)
    phi = jnp.arctan2(y, x)

    N_phi_stripes = 30
    N_theta_stripes = 25
    phi_ind = jnp.int32(
        jnp.floor(N_phi_stripes * (phi + jnp.pi) / (2 * jnp.pi))
    )  # [N, 1]
    theta_ind = jnp.int32(jnp.floor(N_theta_stripes * theta / jnp.pi))  # [N, 1]

    ind = phi_ind * N_theta_stripes + theta_ind

    rng = jax.random.PRNGKey(0)
    coeffs = jax.random.uniform(
        rng, (N_theta_stripes * (N_phi_stripes + 1) + 1,), dtype=jnp.float32
    )

    mix = coeffs[ind]

    clr1 = jnp.array([1.0, 1.0, 1.0])[None, :]
    # clr2 = jnp.array([1.0, 0.3, 0.3])[None, :]
    clr2 = jnp.array([0.3, 1.0, 0.3])[None, :]

    albedo = mix * clr1 + (1.0 - mix) * clr2

  else:
    raise ValueError(f'Texture {texture} not implemented.')

  materials = {}
  materials['albedo'] = albedo
  if shading == 'phong':
    materials['specular_albedo'] = jnp.ones_like(albedo) * 3.0  # 10.0
    # materials['specular_exponent'] = jnp.ones_like(albedo) * 1000.0
    materials['specular_exponent'] = jnp.where(
        (jnp.abs(x) < 0.15 * rad) | (jnp.abs(y) < 0.15 * rad), 20.0, 100.0
    )
  elif shading == 'blinnphong':
    materials['specular_albedo'] = jnp.ones_like(albedo) * 5.0  # 10.0
    # materials['specular_exponent'] = jnp.ones_like(albedo) * 300.0
    materials['specular_exponent'] = jnp.where(
        (jnp.abs(x) < 0.15 * rad) | (jnp.abs(y) < 0.15 * rad), 50.0, 300.0
    )
  elif shading == 'microfacet':
    materials['F_0'] = jnp.ones_like(albedo[Ellipsis, :1]) * 0.04
    materials['roughness'] = jnp.ones_like(albedo[Ellipsis, :1]) * 0.4

  return materials

def cylinders_into_primitives(params):
  if params is None:
    return None
  new_params = []
  for prms in params:
    if prms['shape'] == 'cylinder':
      bottom_cap = {'shape': 'disk',
                    'radius': prms['radius'],
                    'origin': jnp.array(prms['origin']) - jnp.array([0.0, 0.0, 1.0]) * prms['height']/2.0,
                    'normal_flip': True}
      top_cap = {'shape': 'disk',
                    'radius': prms['radius'],
                    'origin': jnp.array(prms['origin']) + jnp.array([0.0, 0.0, 1.0]) * prms['height']/2.0 ,
                    'normal_flip': False}
      hollow_cylinder = {'shape': 'hollow_cylinder',
                         'radius': prms['radius'],
                         'origin': prms['origin'],
                         'height': prms['height']}
      new_params.append(bottom_cap)
      new_params.append(top_cap)
      new_params.append(hollow_cylinder)
    else:
      new_params.append(prms)
  return new_params

def get_intersection_and_normals_gt_(rays_o, rays_d, params):
  if type(params) == dict:
    # Find point on surface of a sphere centered at c:
    # ||o + td - c||^2 = ||o-c||^2 + 2tdot(o-c, d) + t^2||d||^2 = R^2
    # So ||d||^2 * t^2 + 2dot(o-c, d) * t + ||o-c||^2 - R^2 = 0.
    # A = ||d||^2, B = 2*dot(o-c, d), C = ||o-c||^2 - R^2.
    # t = (-B - sqrt(disc)) / 2A, where disc = 4 * dot(o-c, d)^2 - 4||d||^2 * (||o-c||^2 - R^2)
    if params['shape'] == 'sphere':
      oxyz = jnp.float32(params['origin'])
      rad = params['radius']
      d_norm_sq = (rays_d**2).sum(-1, keepdims=True)
      o_norm_sq = ((rays_o-oxyz)**2).sum(-1, keepdims=True)
      d_dot_o = ((rays_o-oxyz) * rays_d).sum(-1, keepdims=True)
      disc = d_norm_sq * (rad**2 - o_norm_sq) + d_dot_o**2
      t_ = -jnp.sqrt(jnp.abs(disc)) - d_dot_o
      alpha = jnp.bitwise_and(disc > 0, t_ > 0.0)  # A point is visible if t > 0 and disc > 0
      # t_surface = jnp.where(disc > 0, -jnp.sqrt(disc) - d_dot_o, jnp.inf) / d_norm_sq  # [H, W, 1]
      t_surface = (
          jnp.where(alpha, t_ / d_norm_sq, 1000.0)
      )  # [H, W, 1]

      pts = rays_o + rays_d * t_surface
      normals = (pts - oxyz) / jnp.linalg.norm(pts - oxyz, axis=-1, keepdims=True)

    # elif params['shape'] == 'disk':  # z-aligned disk
    #   t_surface = (params['origin'][..., 2:3] - rays_d[..., 2:3]) / rays_o[..., 2:3]
    #   pts = rays_o + t_surface * rays_d
    #   print(pts)
    #   alpha = jnp.float32(pts[..., 0:1] ** 2 + pts[..., 1:2] ** 2 < params['radius'] ** 2)
    #   normals = jnp.array([0.0, 0.0, 1.0]) * jnp.ones_like(pts)


    elif params['shape'] == 'disk':
      t_surface = (params['origin'][2] - rays_o[Ellipsis, 2:3]) / rays_d[Ellipsis, 2:3]
      pts = rays_o + t_surface * rays_d
      alpha = jnp.float32(jnp.bitwise_and(t_surface > 0.0, (pts[Ellipsis, 0:1] - params['origin'][0]) ** 2 + (pts[Ellipsis, 1:2] - params['origin'][1]) ** 2 < params['radius'] ** 2))
      normals = jnp.array([0.0, 0.0, -1.0 if params['normal_flip'] else 1.0]) * jnp.ones_like(pts)

    elif params['shape'] == 'hollow_cylinder':
      oxyz = jnp.float32(params['origin'])
      rad = params['radius']
      d_norm_sq = (rays_d**2)[Ellipsis, :2].sum(-1, keepdims=True)
      o_norm_sq = ((rays_o-oxyz)**2)[Ellipsis, :2].sum(-1, keepdims=True)
      d_dot_o = ((rays_o-oxyz) * rays_d)[Ellipsis, :2].sum(-1, keepdims=True)
      disc = d_norm_sq * (rad**2 - o_norm_sq) + d_dot_o**2
      t_ = -jnp.sqrt(jnp.abs(disc)) - d_dot_o

      alpha = jnp.bitwise_and(jnp.abs((rays_o + rays_d * t_ / d_norm_sq)[Ellipsis, 2:] - params['origin'][2]) < params['height'] / 2.0, jnp.bitwise_and(disc > 0, t_ > 0.0))  # A point is visible if t > 0 and disc > 0
      # t_surface = jnp.where(disc > 0, -jnp.sqrt(disc) - d_dot_o, jnp.inf) / d_norm_sq  # [H, W, 1]
      t_surface = (
          jnp.where(alpha, t_ / d_norm_sq, 1000.0)
      )  # [H, W, 1]

      pts = rays_o + rays_d * t_surface
      normals = (pts - oxyz) * jnp.array([1.0, 1.0, 0.0])
      normals /= jnp.linalg.norm(normals, axis=-1, keepdims=True)

    elif params['shape'] == 'cube':
      origin = jnp.array(params['origin'])
      t1 = (origin - params['side'] / 2.0 - rays_o) / rays_d
      t2 = (origin + params['side'] / 2.0 - rays_o) / rays_d

      #t_min = jnp.maximum(t1.min(axis=-1, keepdims=True), t2.min(axis=-1, keepdims=True))
      #t_max = jnp.minimum(t1.max(axis=-1, keepdims=True), t2.max(axis=-1, keepdims=True))

      t_min = jnp.max(jnp.minimum(t1, t2), axis=-1, keepdims=True)
      t_max = jnp.min(jnp.maximum(t1, t2), axis=-1, keepdims=True)

      t_surface = jnp.minimum(t_min, t_max)
      alpha = jnp.bitwise_and(t_min > 0.0, t_max > t_min)
      pts = rays_o + rays_d * t_surface

      active = jnp.argmax(jnp.abs(pts - origin), axis=-1, keepdims=True)
      sgn = 2.0 * jnp.float32(pts - origin > 0.0) - 1.0


      normals = jnp.eye(3)[active][Ellipsis, 0, :] * sgn


  elif type(params) == list or type(params) == tuple:
    ps = []
    as_ = []
    ns = []
    ts = []

    for prm in params:
      p, a, n, t = get_intersection_and_normals_gt_(rays_o, rays_d, prm)
      ps.append(p)
      as_.append(a)
      ns.append(n)
      ts.append(t)

    pts_ = jnp.stack(ps, axis=-1)
    alphas_ = jnp.stack(as_, axis=-1)
    normals_ = jnp.stack(ns, axis=-1)
    t_surface_ = jnp.stack(ts, axis=-1)

    inds = jnp.argmin(jnp.where(alphas_ > 0.5, t_surface_, 10000.0), axis=-1)[Ellipsis, 0]

    extract = jax.vmap(lambda arr, ind: arr[Ellipsis, ind], in_axes=0)
    pts = extract(pts_, inds)
    alpha = extract(alphas_, inds)
    normals = extract(normals_, inds)
    t_surface = extract(t_surface_, inds)

  else:
    raise RuntimeError(f'params must be tuple, list, or dict, but got {type(params)}')

  return pts, jnp.float32(alpha), normals, t_surface



def get_intersection_and_normals_gt(rays_o, rays_d, params):
  # Same function but only returns pts, alphas, normals
  return get_intersection_and_normals_gt_(rays_o, rays_d, params)[:3]


@flax.struct.dataclass
class PrecomputedIntersection:
  H: Any
  W: Any
  focal: Any
  cameras_vec: Any
  points_gt: Any
  alpha_gt: Any
  normals_gt: Any

  @jax.jit
  def get_intersection_and_normals_precomputed_hacky(self, ray_o, ray_d):
    # Get index of camera
    cameras_o = self.cameras_vec[:, :3, 3]

    dists = jnp.linalg.norm(ray_o[None, :] - cameras_o, axis=-1)
    camera_ind = jnp.argmin(dists)

    camera = self.cameras_vec[camera_ind]

    # Now get index of ray
    res = ray_d @ camera[:3, :3]
    j = jnp.int32(res[0] * self.focal + 0.5 * self.W - 0.5)
    i = jnp.int32(-res[1] * self.focal + 0.5 * self.H - 0.5)



    # Grab pants
    #p = self.points_gt[camera_ind, i, j]
    #a = self.alpha_gt[camera_ind, i, j]
    #n = self.normals_gt[camera_ind, i, j]
    p = self.points_gt[camera_ind, i * self.W + j]
    a = self.alpha_gt[camera_ind, i * self.W + j]
    n = self.normals_gt[camera_ind, i * self.W + j]

    return p, a, n

@flax.struct.dataclass
class PrecomputedAlbedo:
  H: Any
  W: Any
  focal: Any
  cameras_vec: Any
  albedo_gt: Any

  @jax.jit
  def get_albedo_precomputed_hacky(self, ray_o, ray_d):
    # Get index of camera
    cameras_o = self.cameras_vec[:, :3, 3]

    dists = jnp.linalg.norm(ray_o[None, :] - cameras_o, axis=-1)
    camera_ind = jnp.argmin(dists)

    camera = self.cameras_vec[camera_ind]

    # Now get index of ray
    res = ray_d @ camera[:3, :3]
    j = jnp.int32(res[0] * self.focal + 0.5 * self.W - 0.5)
    i = jnp.int32(-res[1] * self.focal + 0.5 * self.H - 0.5)

    return self.albedo_gt[camera_ind, i * self.W + j]

def random_split(rng):
  if rng is None:
    key = None
  else:
    key, rng = jax.random.split(rng)

  return key, rng

def importance_sample_rays(
    rng,
    global_viewdirs,
    normal,
    material,
    random_generator_2d=None,
    stratified_sampling=False,
    use_mis=True,
    samplers=None,
    num_secondary_samples=None,
    light_sampler_results=None,
):
  rotation_mat = get_rotation_matrix(normal)
  local_viewdirs = global_to_local(global_viewdirs, rotation_mat)
  roughness = material.get('roughness', jnp.ones_like(local_viewdirs))

  # Resample
  num_real_samples = sum(sample_count for _, sample_count in samplers)
  resample = num_real_samples > num_secondary_samples

  # Calculate MIS samples (directions)
  local_lightdirs = []
  pdf = []
  weight = []

  for sampler, sample_count in samplers:
    if resample:
      real_sample_count = sample_count
    else:
      real_sample_count = int(
          round(
              (float(sample_count) / num_real_samples) * num_secondary_samples
          )
      )

    # Get random samples in [0, 1)^2
    key, rng = random_split(rng)
    uh, uw = random_generator_2d.sample(
        key, local_viewdirs.shape[0] * real_sample_count, stratified_sampling
    )
    uh = uh.reshape(local_viewdirs.shape[0], real_sample_count)
    uw = uw.reshape(local_viewdirs.shape[0], real_sample_count)

    # Current inputs
    cur_local_viewdirs = jnp.repeat(
        local_viewdirs[Ellipsis, None, :], real_sample_count, axis=-2
    )
    cur_roughness = jnp.repeat(
        roughness[Ellipsis, None, :], real_sample_count, axis=-2
    )

    # Set sample rng
    key, rng = random_split(rng)

    # Importance sample
    cur_local_lightdirs, cur_pdf = sampler.sample_directions(
        key,
        uh,
        uw,
        cur_local_viewdirs,
        cur_roughness,
        light_sampler_results
    )

    if sampler.global_dirs:
      cur_local_lightdirs = global_to_local(
          cur_local_lightdirs, rotation_mat[Ellipsis, None, :, :]
      )

    # Calculate MIS weights
    if (
        use_mis
        and len(samplers) > 1
    ):
      denominator = jnp.finfo(jnp.float32).eps

      for sampler_p, sample_count_p in samplers:
        if sampler_p.global_dirs:
          temp_viewdirs = local_to_global(
              cur_local_viewdirs, rotation_mat[Ellipsis, None, :, :]
          )
          temp_lightdirs = local_to_global(
              cur_local_lightdirs, rotation_mat[Ellipsis, None, :, :]
          )
        else:
          temp_viewdirs = cur_local_viewdirs
          temp_lightdirs = cur_local_lightdirs

        # Heuristic weight denominator
        denominator += jnp.square(
            sampler_p.pdf(
                temp_viewdirs,
                temp_lightdirs,
                cur_roughness,
                light_sampler_results
            ) * sample_count_p
        )

      # Heuristic weight
      cur_weight = (
          jnp.square(sample_count * cur_pdf) / denominator
      )

      # Correct total energy
      cur_weight = cur_weight * (
          float(num_real_samples) / float(sample_count)
      )
      cur_weight = jnp.where(
          cur_pdf > 0.0,
          cur_weight,
          jnp.zeros_like(cur_weight)
      )
    else:
      cur_weight = jnp.ones_like(cur_pdf)

    # Append
    local_lightdirs.append(cur_local_lightdirs)
    pdf.append(cur_pdf)
    weight.append(cur_weight)

  # Concatenate
  local_lightdirs = jnp.concatenate(local_lightdirs, axis=-2)
  local_viewdirs = jnp.repeat(
      local_viewdirs[Ellipsis, None, :],
      num_secondary_samples,
      axis=-2
  )
  global_viewdirs = jnp.repeat(
      global_viewdirs[Ellipsis, None, :],
      num_secondary_samples,
      axis=-2
  )
  pdf = jnp.concatenate(pdf, axis=-1)[Ellipsis, None]
  weight = jnp.concatenate(weight, axis=-1)[Ellipsis, None]

  # Global bounce directions
  global_lightdirs = local_to_global(
      local_lightdirs,
      rotation_mat[Ellipsis, None, :, :]
  )

  # Samples
  samples = {
      'local_lightdirs': local_lightdirs,
      'local_viewdirs': local_viewdirs,
      'global_lightdirs': global_lightdirs,
      'global_viewdirs': global_viewdirs,
      'pdf': jax.lax.stop_gradient(pdf),
      'weight': jax.lax.stop_gradient(weight),
  }

  # Select one secondary sample
  if resample:
    probs = jnp.ones_like(pdf)

    key, rng = random_split(rng)
    inds = jax.random.categorical(
        key,
        math_utils.safe_log(probs),
        axis=-2,
        shape=(
            pdf.shape[:-2] + (num_secondary_samples,)
        )
    )[Ellipsis, None]

    samples = jax.tree_util.tree_map(
        lambda x: jnp.take_along_axis(x, inds, axis=-2),
        samples
    )

  return samples


def get_secondary_rays(
    rng,
    rays,
    means,
    viewdirs,
    normals,
    material,
    normal_eps=1e-4,
    refdir_eps=1e-2,
    random_generator_2d=None,
    stratified_sampling=False,
    use_mis=True,
    samplers=None,
    num_secondary_samples=None,
    light_sampler_results=None,
    light_rotation=None,
    offset_origins=False,
):
  if rng is None:
    rng = jax.random.PRNGKey(0)

  # Reflected ray origins
  ref_origins = means + normals * normal_eps
  ref_origins = jnp.repeat(
      ref_origins[Ellipsis, None, :], num_secondary_samples, axis=-2
  )

  # Reflected ray directions
  global_viewdirs = -viewdirs[Ellipsis, None, :] * jnp.ones_like(means)
  material = jax.tree_util.tree_map(
      lambda x: x.reshape(-1, x.shape[-1]),
      material
  )

  if light_sampler_results is not None:
    light_sampler_results = jax.tree_util.tree_map(
        lambda x: x.reshape((-1,) + x.shape[-2:]),
        light_sampler_results
    )

  key, rng = random_split(rng)
  ref_samples = importance_sample_rays(
      key,
      global_viewdirs.reshape(-1, 3),
      normals.reshape(-1, 3),
      material,
      random_generator_2d=random_generator_2d,
      stratified_sampling=stratified_sampling,
      use_mis=use_mis,
      samplers=samplers,
      num_secondary_samples=num_secondary_samples,
      light_sampler_results=light_sampler_results,
  )

  # Create reflect rays
  new_sh = (-1, num_secondary_samples, 3,)

  ref_rays = rays.replace(
      near=(
          refdir_eps * jnp.ones_like(ref_origins[Ellipsis, :1])
      ).reshape(new_sh[:-1] + (1,)),
      far=(
          rays.far[Ellipsis, None, None] * jnp.ones_like(ref_origins[Ellipsis, :1])
      ).reshape(new_sh[:-1] + (1,)),
      origins=ref_origins.reshape(new_sh),
      directions=ref_samples['global_lightdirs'].reshape(new_sh),
      viewdirs=ref_samples['global_lightdirs'].reshape(new_sh),
  )

  ref_rays = ref_rays.replace(
      radii=jnp.ones_like(ref_rays.directions[Ellipsis, :1]),
      lossmult=jnp.ones_like(ref_rays.directions),
  )

  if offset_origins:
    ref_rays = ref_rays.replace(
        origins=ref_rays.origins + ref_rays.directions * ref_rays.near,
        near=jnp.zeros_like(ref_rays.near),
    )

  if light_rotation is not None:
    ref_rays = ref_rays.replace(
        directions=local_to_global(
            ref_rays.directions, light_rotation.reshape(-1, 1, 3, 3)
        ),
        viewdirs=local_to_global(
            ref_rays.viewdirs, light_rotation.reshape(-1, 1, 3, 3)
        ),
    )

  # Reshape sample outputs
  ref_samples = jax.tree_util.tree_map(
      lambda x: x.reshape(new_sh[:-1] + (x.shape[-1],)),
      ref_samples
  )

  return ref_rays, ref_samples


def get_outgoing_rays(
    rng,
    rays,
    viewdirs,
    normals,
    material,
    random_generator_2d=None,
    stratified_sampling=False,
    use_mis=True,
    samplers=None,
    num_secondary_samples=None,
):
  if rng is None:
    rng = jax.random.PRNGKey(0)

  # Reflected ray directions
  global_viewdirs = -viewdirs[Ellipsis, None, :] * jnp.ones_like(normals)
  material = jax.tree_util.tree_map(
      lambda x: x.reshape(-1, x.shape[-1]),
      material
  )

  key, rng = random_split(rng)
  ref_samples = importance_sample_rays(
      key,
      global_viewdirs.reshape(-1, 3),
      normals.reshape(-1, 3),
      material,
      random_generator_2d=random_generator_2d,
      stratified_sampling=stratified_sampling,
      use_mis=use_mis,
      samplers=samplers,
      num_secondary_samples=num_secondary_samples,
  )

  # Create reflect rays
  ref_rays = rays.replace(
      viewdirs=-ref_samples['global_lightdirs'].reshape(
          rays.viewdirs.shape
      )
  )

  return ref_rays


def eval_vmf(x, means, kappa):
  # Evaluate vmf at directions x
  eps = jnp.finfo(jnp.float32).eps
  vmf_vals = kappa * math.safe_exp(
      kappa * (jnp.sum(x * means, axis=-1))
  ) / (4 * jnp.pi * jnp.sinh(kappa))
  out = jnp.where(
      jnp.less_equal(kappa, eps),
      jnp.ones_like(means[Ellipsis, 0]) / (4. * jnp.pi),
      vmf_vals
  )
  return out

def integrate_reflect_rays(
    material_type,
    use_brdf_correction,
    material,
    samples,
):
  eps = jnp.finfo(jnp.float32).eps
  config = type('', (), {})()
  config.shading = material_type
  config.use_brdf_correction = use_brdf_correction

  material = jax.tree_util.tree_map(
      lambda x: x.reshape(-1, x.shape[-1]),
      material
  )
  material_lobe = get_lobe(
      samples['local_lightdirs'],
      samples['local_viewdirs'],
      material,
      samples['brdf_correction'],
      config
  )

  denominator = jnp.maximum(
      samples['pdf'],
      eps
  )
  weight = jnp.maximum(
      samples['weight'],
      0.0
  )
  weight = jnp.where(
      samples['local_lightdirs'][Ellipsis, 2:] > 0.0, weight, jnp.zeros_like(weight)
  )

  # Outgoing radiance
  radiance_out = (
      samples['radiance_in']
      * material_lobe
      * weight / denominator
  ).mean(1)

  # Incoming irradiance
  diffuse_lobe = (
      jnp.maximum(0., samples['local_lightdirs'][Ellipsis, 2:]) / jnp.pi
  )
  irradiance = (
      samples['radiance_in']
      * diffuse_lobe
      * weight / denominator
  ).mean(1)

  # Multipliers
  if use_brdf_correction:
    integrated_multiplier = (
        samples['brdf_correction']
        * weight / denominator
    ).mean(1) / (2 * jnp.pi)

    integrated_multiplier_irradiance = (
        samples['brdf_correction'][Ellipsis, 1:2]
        * samples['radiance_in']
        * diffuse_lobe
        * weight / denominator
    ).mean(1)
  else:
    integrated_multiplier = samples['brdf_correction'][:, 0]
    integrated_multiplier_irradiance = samples['brdf_correction'][:, 0, :1]

  return dict(
      radiance_out=radiance_out,
      irradiance=irradiance,
      integrated_multiplier=integrated_multiplier,
      integrated_multiplier_irradiance=integrated_multiplier_irradiance,
  )


def expand_vmf_vars(vars, x):
  means, kappas, weights = vars

  means = jnp.repeat(means[None], x.shape[0], axis=0)
  kappas = jnp.repeat(kappas[None], x.shape[0], axis=0)

  return means, kappas, weights


def sample_vmf_vars(rng, vars, x):
  key, rng = jax.random.split(rng)
  latents = jax.random.categorical(
      key, logits=math_utils.safe_log(vars[2]), axis=-1, shape=(x.shape[0],)
  )
  means = jnp.take_along_axis(
      vars[0], latents[Ellipsis, None, None], axis=-2
  )[Ellipsis, 0, :]
  kappas = jnp.take_along_axis(
      vars[1], latents[Ellipsis, None], axis=-1
  )[Ellipsis, 0]

  return means, kappas, vars[2]


def filter_vmf_vars(vars, sample_normals, t1=0.1, t2=0.09):
  means, kappas, weights = vars

  # Mask
  dotprod = (
      ref_utils.l2_normalize(means) * sample_normals[Ellipsis, None, :]
  ).sum(axis=-1)

  logits = math_utils.safe_log(weights)
  new_logits = logits + jax.lax.stop_gradient(dotprod - t2) / (t1 - t2)

  logits = jnp.where(
      dotprod > t1, logits, new_logits
  )
  weights = math_utils.safe_exp(logits)

  return means, kappas, weights


def vmf_loss_fn(
    vars, sample_normals, sample_dirs, function_vals, samples
):
  # vars = filter_vmf_vars(
  #     vars,
  #     sample_normals,
  # )
  means = ref_utils.l2_normalize(vars[0])
  kappas = vars[1][Ellipsis, 0]
  weights = vars[2][Ellipsis, 0]

  # KL Divergence Loss
  likelihood = jnp.sum(
      weights[Ellipsis, None, :] * eval_vmf(
          sample_dirs[Ellipsis, None, :],
          means[Ellipsis, None, :, :],
          kappas[Ellipsis, None, :]
      ),
      axis=-1
  )

  # Denominator
  denominator = jnp.maximum(
      samples['pdf'][Ellipsis, 0],
      1e-1
  )
  # denominator = jnp.ones_like(denominator)

  # Weight (MIS)
  dotprod = (
      sample_dirs * sample_normals[Ellipsis, None, :]
  ).sum(axis=-1)
  weight = jnp.maximum(
      samples['weight'][Ellipsis, 0],
      0.0
  )
  weight = jnp.ones_like(weight)
  weight = jnp.where(
      dotprod > 0.0,
      weight,
      jnp.zeros_like(weight),
  )

  return jnp.mean(
      jnp.square(
          jnp.power(function_vals, 1.0)
          - jnp.power(likelihood, 1.0)
      )
      * weight
      / denominator
  )


def sample_vmf(rng, vars, x, n_dirs):
  """Sample random directions from vmf distribution.

  Args:
    rng: jnp.ndarray, random generator key.
    mean: jnp.ndarray(float32), [..., 3].
    kappa: jnp.ndarray(float32), [...]., vmf kappa parameter.
    n_dirs: int.

  Returns:
    rand_dirs: jnp.ndarray(float32), [..., n_dirs, 3]
  """
  key, rng = jax.random.split(rng)
  mean, kappa, weights = sample_vmf_vars(key, vars, x)
  mean = ref_utils.l2_normalize(mean)

  t_vec = jnp.stack([-mean[Ellipsis, 1], mean[Ellipsis, 0],
                     jnp.zeros_like(mean[Ellipsis, 0])], axis=-1)
  t_vec = ref_utils.l2_normalize(t_vec)
  b_vec = jnp.cross(mean, t_vec)
  b_vec = ref_utils.l2_normalize(b_vec)
  rotmat = jnp.stack([t_vec, b_vec, mean], axis=-1)

  key, rng = jax.random.split(rng)
  # vmf sampling (https://www.mitsuba-renderer.org/~wenzel/files/vmf.pdf)
  v = jax.random.normal(key, shape=mean.shape[:-1] + (n_dirs, 2))
  v = ref_utils.l2_normalize(v)

  key, rng = jax.random.split(rng)
  tmp = jax.random.uniform(key, shape=mean.shape[:-1] + (n_dirs,))
  w = 1. + (1. / kappa[Ellipsis, None]) * math_utils.safe_log(
      tmp + (1. - tmp) * jnp.exp(-2. * kappa[Ellipsis, None])
  )
  rand_dirs = jnp.stack([math_utils.safe_sqrt(1. - w**2) * v[Ellipsis, 0],
                         math_utils.safe_sqrt(1. - w**2) * v[Ellipsis, 1],
                         w], axis=-1)
  rand_dirs = jnp.matmul(rotmat[Ellipsis, None, :, :], rand_dirs[Ellipsis, None])[Ellipsis, 0]

  return rand_dirs


class LightSampler:
  global_dirs: bool = True

  def __init__(self):
    pass

  def sample_directions(self, rng, u1, u2, wo, _, kwargs):
    vars = (
        kwargs['vmf_means'],
        kwargs['vmf_kappas'][Ellipsis, 0],
        kwargs['vmf_weights'][Ellipsis, 0]
    )
    vars = filter_vmf_vars(
        vars,
        kwargs['vmf_normals'][Ellipsis, 0, :],
    )
    means = ref_utils.l2_normalize(vars[0])
    kappas = vars[1]
    weights = jax.nn.softmax(math_utils.safe_log(vars[2]))

    key, rng = jax.random.split(rng)
    sample_dirs = sample_vmf(
        key,
        (means, kappas, weights),
        wo,
        n_dirs=u1.shape[-1],
    )

    # Get pdf
    pdf = jnp.sum(
        weights[Ellipsis, None, :] * eval_vmf(
            sample_dirs[Ellipsis, None, :],
            means[Ellipsis, None, :, :],
            kappas[Ellipsis, None, :],
        ),
        axis=-1
    )

    return sample_dirs, pdf

  def pdf(self, wo, wi, _, kwargs):
    vars = (
        kwargs['vmf_means'],
        kwargs['vmf_kappas'][Ellipsis, 0],
        kwargs['vmf_weights'][Ellipsis, 0]
    )
    vars = filter_vmf_vars(
        vars,
        kwargs['vmf_normals'][Ellipsis, 0, :],
    )
    means = ref_utils.l2_normalize(vars[0])
    kappas = vars[1]
    weights = jax.nn.softmax(math_utils.safe_log(vars[2]))

    pdf = jnp.sum(
        weights[Ellipsis, None, :] * eval_vmf(
            wi[Ellipsis, None, :],
            means[Ellipsis, None, :, :],
            kappas[Ellipsis, None, :],
        ),
        axis=-1
    )

    return pdf


IMPORTANCE_SAMPLER_BY_NAME = {
    'light': LightSampler,
    'microfacet': MicrofacetSampler,
    'cosine': CosineSampler,
    'uniform': UniformHemisphereSampler,
    'uniform_sphere': UniformSphereSampler,
}
