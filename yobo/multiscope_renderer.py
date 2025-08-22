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
"""Multiscope renderer for MipNerf360."""

import dataclasses
import functools
import time

import flax
from google_research.yobo.internal import camera_utils
from google_research.yobo.internal import datasets
from google_research.yobo.internal import image
from google_research.yobo.internal import math
from google_research.yobo.internal import models
from google_research.yobo.internal import ref_utils
from google_research.yobo.internal import train_utils
from google_research.yobo.internal import utils
from google_research.yobo.internal import vis
from google_research.yobo.internal.inverse_render import render_utils
import jax
from jax import random
import jax.numpy as jnp
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
import scipy.interpolate

WIDTH_INIT = 400  # Width (pixels) of render window.
HEIGHT_INIT = WIDTH_INIT  # Height (pixels) of render window.
FOCAL_INIT = 1111 * (WIDTH_INIT / 800.0)  # Pinhole camera focal length.

HIGH_RES_MULTIPLIER = 4

BASE_STEP = 0.01

TRAIN_ITERS_PER_STEP = 10  # Number of training iterations per viewer step.

matmul = np.matmul


def get_directions(height, width):
  # Returns coordinates (spherical and cartesian) for equirect points on
  # sphere, with half-pixel offsets (centers of equirect-spaced pixels).
  phi, theta = jnp.meshgrid(
      jnp.linspace(-jnp.pi, jnp.pi, width, endpoint=False)
      + 2.0 * jnp.pi / (2.0 * width),
      jnp.linspace(0.0, jnp.pi, height, endpoint=False)
      + jnp.pi / (2.0 * height),
  )

  # \delta theta * \delta phi (area of pixels in spherical coordinates),
  # for use in quadrature integration.
  dtheta_dphi = (2.0 * jnp.pi / width) * (jnp.pi / height)

  theta = theta.flatten()
  phi = phi.flatten()

  x = jnp.sin(theta) * jnp.cos(phi)
  y = jnp.sin(theta) * jnp.sin(phi)
  z = jnp.cos(theta)
  xyz = jnp.stack([x, y, z], axis=-1)

  return theta, phi, xyz, dtheta_dphi


def cart2sph(xyz):
  x, y, z = jnp.moveaxis(xyz, -1, 0)
  r = math.safe_sqrt(x**2 + y**2)
  phi = math.safe_arctan2(y, x)
  theta = math.safe_arctan2(r, z)
  return jnp.stack([phi, theta], axis=-1)


def plot_poses(poses, eps=0.05):
  """Plot 2D projection of camera poses onto each cartesian axis."""
  t = poses[:, :3, 3]
  axis_pts = t.reshape([-1, 3, 1]) + poses[:, :3, :3] * eps
  plt.figure(figsize=(15, 4))
  for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.scatter(t[:, i], t[:, (i + 1) % 3], s=5)
    for j in range(3):
      z = np.stack([t, axis_pts[:, :, j]], 0)
      plt.plot(z[Ellipsis, i], z[Ellipsis, (i + 1) % 3], c='rgb'[j])
    plt.axis('equal')
  plt.show()


def trans_xyz(x, y, z, xnp=np):
  return xnp.array(
      [
          [1, 0, 0, x],
          [0, 1, 0, y],
          [0, 0, 1, z],
          [0, 0, 0, 1],
      ],
      dtype=xnp.float32,
  )


def rot_phi(phi, xnp=np):
  return xnp.array(
      [
          [1, 0, 0, 0],
          [0, xnp.cos(phi), -xnp.sin(phi), 0],
          [0, xnp.sin(phi), xnp.cos(phi), 0],
          [0, 0, 0, 1],
      ],
      dtype=xnp.float32,
  )


def rot_theta(th, xnp=np):
  return xnp.array(
      [
          [xnp.cos(th), 0, -xnp.sin(th), 0],
          [0, 1, 0, 0],
          [xnp.sin(th), 0, xnp.cos(th), 0],
          [0, 0, 0, 1],
      ],
      dtype=xnp.float32,
  )


def rot_psi(th, xnp=np):
  return xnp.array(
      [
          [xnp.cos(th), -xnp.sin(th), 0, 0],
          [xnp.sin(th), xnp.cos(th), 0, 0],
          [0, 0, 1, 0],
          [0, 0, 0, 1],
      ],
      dtype=xnp.float32,
  )


def pose_spherical(theta, phi, psi, xnp=np):
  pca_flip = xnp.array([
      1,
      0,
      0,
      0,
      0,
      0,
      -1,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      1,
  ]).reshape(4, 4)
  c2w = xnp.eye(4)
  matmul = math.matmul if xnp == jnp else np.matmul
  c2w = matmul(rot_psi(psi / 180.0 * xnp.pi, xnp=xnp), c2w)
  c2w = matmul(rot_phi(phi / 180.0 * xnp.pi, xnp=xnp), c2w)
  c2w = matmul(rot_theta(theta / 180.0 * xnp.pi, xnp=xnp), c2w)
  c2w = matmul(pca_flip, c2w)
  return c2w


def invert_pose(pose):
  position = pose[:3, -1]
  rot = pose[:3, :3]
  pca_flip = np.array([
      1,
      0,
      0,
      0,
      0,
      0,
      -1,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      1,
  ]).reshape(4, 4)
  pca_flip_inv = pca_flip.T
  rot = pca_flip_inv[:3, :3] @ rot
  phi = np.arcsin(-rot[1, 2])
  theta = np.arctan2(-rot[0, 2], rot[2, 2])
  theta = np.remainder(theta, 2 * np.pi)
  psi = np.arctan2(rot[1, 0], rot[1, 1])
  return position, np.array((theta, phi, psi)) * 180 / np.pi


def add_reticle(z, c, r=5, t=0):
  z = np.copy(z)
  y = z.shape[0] // 2
  x = z.shape[1] // 2
  z[y - r : y + r + 1, x - t : x + t + 1] = c
  z[y - t : y + t + 1, x - r : x + r + 1] = c
  return z


def add_reticle_xy(z, c, r=5, t=0, x=0, y=0):
  z = np.copy(z)
  z[y - r : y + r + 1, x - t : x + t + 1] = c
  z[y - t : y + t + 1, x - r : x + r + 1] = c
  return z


def add_border(z, c, r=5):
  z = np.copy(z)
  zc = np.copy(z[r:-r, r:-r])
  z[:, :] = c
  z[r:-r, r:-r] = zc
  return z


def get_spline(values):
  n = len(values)
  k = min(n - 1, 3)
  # s = n - np.sqrt(2*n)
  tck = scipy.interpolate.splrep(np.arange(n), values, k=k, s=0.01)
  spline = scipy.interpolate.BSpline(*tck)
  return spline


class Spliner:

  def __init__(self, fps=20):
    self.clear()
    self.dt = 1.0 / fps

  def clear(self):
    self.keyframes = []
    self.t_int = 0
    self.t = 0.0

  def make_splines(self):
    if len(self.keyframes) >= 3:
      kf = np.array(self.keyframes).T
      self.splines = [get_spline(v) for v in kf]
    else:
      self.splines = None

  def push(self, kf):
    self.keyframes.append(kf)
    if len(self.keyframes) >= 3:
      self.make_splines()

  def insert(self, kf):
    kfs = self.keyframes
    self.keyframes = kfs[: self.t_int] + [kf] + kfs[self.t_int :]
    if len(self.keyframes) >= 3:
      self.make_splines()

  def edit(self, kf):
    # self.keyframes[int(self.t)] = kf
    self.keyframes[self.t_int] = kf
    self.make_splines()

  def pop(self):
    if self.keyframes:
      self.keyframes = self.keyframes[:-1]
    self.make_splines()

  def eval(self, t):
    kf = np.array([spline(t) for spline in self.splines])
    return kf

  def incr(self, up):
    self.t_int += 1 if up else -1
    self.t = float(self.t_int)
    self.clamp_t()
    return self.keyframes[self.t_int]

  def clamp_t(self):
    t = self.t
    self.t_int = self.t_int % len(self.keyframes)
    if t > len(self.keyframes) - 1 + 1e-3:
      t = 0
    if t < 0:
      t = len(self.keyframes) - 1
    self.t = t

  def play(self):
    kf = self.eval(self.t)
    self.t += self.dt
    self.t_int = int(self.t)
    self.clamp_t()
    return kf

  def get_pose_path(self, frames_per_kf):
    n_kf = len(self.keyframes)
    tt = np.linspace(0, n_kf - 1, frames_per_kf * (n_kf - 1), endpoint=False)
    poses = np.array([keyframe_to_pose(self.eval(t)) for t in tt])
    return poses


def keyframe_to_pose(kf):
  return matmul(trans_xyz(*kf[-3:]), pose_spherical(*kf[:3]))


class Controller:

  def __init__(self, theta=0, phi=0, psi=0, t=np.zeros((3,)), spl=Spliner()):
    self.phi = phi
    self.theta = theta
    self.psi = psi

    self.cache_phi = 0
    self.cache_theta = 0
    self.cache_psi = 0

    self.position = t
    self._mouse_speed = 0.4
    self._phi_ratio = -0.5
    self._keyboard_speed = 0.02
    self.keyframes = []
    self.keypos = []
    self.i_frame = 0

    self.spl = spl

    self.focus = False

  def rotation(self):
    return pose_spherical(self.theta, self.phi, self.psi)

  def reset_light_rotation(self):
    self.cache_theta = 0
    self.cache_phi = 0
    self.cache_psi = 0

  def light_rotation(self):
    light_rotation = pose_spherical(
        self.cache_theta, self.cache_phi, self.cache_psi
    )[:3, :3]

    return light_rotation

  def pose(self):
    p = matmul(trans_xyz(*self.position), self.rotation())
    if self.focus:
      p = matmul(p, trans_xyz(0, 0, self.z))
    return p

  def invpose(self, pose):
    if self.focus:
      pose = matmul(pose, trans_xyz(0, 0, -self.z))
    position, angles = invert_pose(pose)
    self.position = position
    self.theta, self.phi, self.psi = angles
    self.aligned = True

  def mouse(self, mx, my):
    self.aligned = False
    self.theta += mx * self._mouse_speed
    self.phi += my * self._mouse_speed * self._phi_ratio

  def cache_mouse(self, mx, my):
    self.cache_theta += mx * self._mouse_speed
    self.cache_phi += my * self._mouse_speed * self._phi_ratio

  def keyboard(
      self,
      key_up=False,
      key_left=False,
      key_down=False,
      key_right=False,
      key_zup=False,
      key_zdown=False,
      key_barrel_left=False,
      key_barrel_right=False,
      key_level=False,
  ):
    if (
        key_up
        or key_left
        or key_down
        or key_right
        or key_zup
        or key_zdown
        or key_barrel_left
        or key_barrel_right
        or key_level
    ):
      self.aligned = False
    rot = self.rotation()
    # Left/right movement with A/D keys and pose x-axis.
    right_dir = np.array(rot[:3, 0])
    # Forward/backward movement with W/S keys and negated pose z-axis.
    fwd_dir = -np.array(rot[:3, 2])
    # Down/up movement with Q/E keys and world-space z-axis.
    z_dir = np.array([0, 0, 1])
    if self.focus:
      if key_up:
        self.z -= self._keyboard_speed
      if key_down:
        self.z += self._keyboard_speed
    else:
      if key_up:
        self.position += self._keyboard_speed * fwd_dir
      if key_down:
        self.position -= self._keyboard_speed * fwd_dir
    if key_right:
      self.position += self._keyboard_speed * right_dir
    if key_left:
      self.position -= self._keyboard_speed * right_dir
    if key_zup:
      self.position += self._keyboard_speed * z_dir
    if key_zdown:
      self.position -= self._keyboard_speed * z_dir
    if key_barrel_left:
      self.psi -= 2.5
    if key_barrel_right:
      self.psi += 2.5
    if key_level:
      self.psi = 0
      self.phi = 0

  def get_kf(self):
    return (self.theta, self.phi, self.psi, *self.pose()[:3, -1])

  def set_kf(self, kf):
    self.theta, self.phi, self.psi = kf[:3]
    self.position = np.array(kf[-3:])

  def push(self):
    self.spl.push(self.get_kf())

  def insert(self):
    self.spl.insert(self.get_kf())

  def edit(self):
    self.spl.edit(self.get_kf())

  def play(self):
    kf = self.spl.play()
    self.set_kf(kf)

  def incr(self, up):
    kf = self.spl.incr(up)
    self.set_kf(kf)

  def switch_mode(self):
    if self.focus:
      self.position = self.pose()[:3, -1]
      self.focus = False
    else:
      self.focus = True
      rot = self.rotation()
      focus_pos = self.position - self.z * rot[:3, 2]
      self.position = focus_pos


class RenderTracker:

  def __init__(self, levels, shape):
    self.shape = shape
    self.reset_params(levels)

  def reset_params(self, levels):
    self.levels = levels
    shape = self.shape

    py, px = jnp.meshgrid(
        jnp.arange(shape[0]), jnp.arange(shape[1]), indexing='ij'
    )

    if levels == 0:
      self.xyr = [px, py, (0, 0)]
      return

    def i2b(i):
      return jnp.mod(i // 2 ** jnp.arange(levels), 2)

    def b2i(b):
      return (b * 2 ** jnp.arange(levels * 2)).sum(-1)

    def i2b_alt(i):
      return jnp.mod(i // 2 ** np.arange(levels * 2), 2)

    def rfn(i):
      r = 2 ** (jnp.argmax(i2b_alt(i)[::-1]) // 2)
      if i == 0:
        r = 2**levels
      rx, ry = jnp.mgrid[:r, :r]
      rx = rx.reshape(-1, 1)
      ry = ry.reshape(-1, 1)
      return rx, ry

    by = i2b(py[Ellipsis, None] % 2**levels)[Ellipsis, ::-1]
    bx = i2b(px[Ellipsis, None] % 2**levels)[Ellipsis, ::-1]
    b = jnp.stack([bx, by], -1).reshape(by.shape[:-1] + (by.shape[-1] * 2,))
    ivals = b2i(b)

    self.xyr = [
        (px[i == ivals], py[i == ivals], rfn(i)) for i in range(4**levels)
    ]
    self.i = 0

  def get_idx(self, loop):
    if self.i == len(self.xyr):
      if not loop:
        return None, None
      else:
        self.i = 0
    x, y, (rx, ry) = self.xyr[self.i]
    self.i += 1
    return (y, x), (y + ry, x + rx)

  def reset(self):
    self.i = 0


class MultiscopeRenderer:
  """Interactive NeRF renderer using Multiscope."""

  def __init__(
      self,
      dataset,
      config,
      model,
      state,
      train_pstep,
      spl=Spliner(),
      highres_window=False,
      hwf_init=(HEIGHT_INIT, WIDTH_INIT, FOCAL_INIT),  # Height, width, focal
  ):
    self.image_writer = multiscope.ImageWriter('Rendering')
    self.light_image_writer = multiscope.ImageWriter('Incoming Light')
    self.text_writer = multiscope.TextWriter('Render log')
    self.cursor_info_writer = multiscope.TextWriter('Cursor info')
    self.input_writer = multiscope.TextWriter('Input window')
    self.fps_writer = multiscope.ScalarWriter('Speed')
    self.psnr_writer = multiscope.ScalarWriter('Train PSNR')
    if config.optimize_cameras:
      self.camera_rotation_writer = multiscope.ScalarWriter('Rotation')
      self.camera_translation_writer = multiscope.ScalarWriter('Translation')
      self.camera_focal_length_writer = multiscope.ScalarWriter('Focal Length')
      self.camera_principal_point_writer = multiscope.ScalarWriter(
          'Principal Point'
      )
      self.camera_radial_distortion1_writer = multiscope.ScalarWriter(
          'Radial Distortion 1'
      )
      self.camera_radial_distortion2_writer = multiscope.ScalarWriter(
          'Radial Distortion 2'
      )
      self.camera_radial_distortion3_writer = multiscope.ScalarWriter(
          'Radial Distortion 3'
      )
      self.camera_radial_distortion4_writer = multiscope.ScalarWriter(
          'Radial Distortion 4'
      )
      self.precondition_matrix_writer = multiscope.TensorWriter('Precondition')
    self.camera_writer = multiscope.TextWriter('Camera Transform')
    self.highres_window = highres_window
    if self.highres_window:
      self.highres_image_writer = multiscope.ImageWriter('High Res Rendering')
    multiscope.events.register_keyboard_callback(self._keyboard_callback)
    multiscope.events.register_mouse_callback(self._mouse_callback)

    self.camera_writer.set_display_size(50, 400)

    self.h, self.w, self.focal = hwf_init
    self.h = (self.h // jax.local_device_count()) * jax.local_device_count()
    self.w = self.w

    self._mouse_speed = 1.0
    self._geom_ratio = 2 ** (1.0 / 8)
    self._res_ratio = 2 ** (1.0 / 2)
    self.reset_default_values()
    self.near = config.near
    self.far = config.far
    self.exposure = None if dataset.exposures is None else dataset.exposures[0]

    rng = jax.random.PRNGKey(1)
    rng, key = jax.random.split(rng)

    self.light_H, self.light_W = 128, 256
    theta, phi, xyz, dtheta_dphi = get_directions(self.light_H, self.light_W)
    self.light_xyz = xyz

    self.rng = random.PRNGKey(0)
    self.render_rng = random.PRNGKey(0)
    self.must_update = True
    self.used_mouse = False
    self.deltas = [0.0, 0.0]
    self.select_x, self.select_y = self.w // 2, self.h // 2
    self.out_show = np.zeros((self.h, self.w, 3))
    self.show_difference = False
    self.rotate_cache = False
    self.avg_step = 0
    self.averaging = False
    self.is_target_averageable = True
    self.avg_key = 'rgb'
    self.avg_is_linear = True
    self.avg_multiplier = 0.5
    self.convert_to_srgb = False
    self.out_show_override = None

    self.controller = Controller(spl=spl)
    self.play_spline = False
    self.edit_spline = False
    self.i_pose = 0
    self.controller.invpose(dataset.camtoworlds[0])

    self.last_key = None
    self.modes = ['c', 'z', 'n', 'm', 'i']
    self.mode = self.modes[0]
    self.sub_mode = 0
    self.exposure_level = 0
    self.shift_pause = True
    self.select_mode = False
    self.keydict = {}

    # Ben changes
    self.grad_accum_steps = config.grad_accum_steps
    self.train_batch = None
    # Ben changes

    self.dataset = dataset
    self.p_raybatcher = flax.jax_utils.prefetch_to_device(
        datasets.RayBatcher(dataset), 3
    )
    np_to_jax = lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x
    self.cameras = tuple(
        np_to_jax(x) for x in dataset.get_train_cameras(config)
    )
    image_sizes = jnp.array([(x.shape[1], x.shape[0]) for x in dataset.images])
    self.jax_cameras = dataset.get_train_cameras(
        config, return_jax_cameras=True
    )
    self.jax_cameras_gt = dataset.jax_cameras
    self.config = config

    def cast_fn(pose, h, w, focal, near, rng):
      key, rng = jax.random.split(rng)
      return camera_utils.cast_general_rays(
          pose,
          camera_utils.get_pixtocam(focal, w, h, xnp=jnp),
          h,
          w,
          near,
          config.far,
          camtype=self.dataset.camtype,
          rng=key,
          jitter=self.config.jitter_rays,
          jitter_scale=float(dataset.width) / float(w),
          xnp=jnp,
      )

    def cast_fn_no_jitter(pose, h, w, focal, near, rng):
      return camera_utils.cast_general_rays(
          pose,
          camera_utils.get_pixtocam(focal, w, h, xnp=jnp),
          h,
          w,
          near,
          config.far,
          camtype=self.dataset.camtype,
          rng=rng,
          jitter=0,
          xnp=jnp,
      )

    self.get_rays = cast_fn
    self.get_rays = jax.jit(
        self.get_rays,
        static_argnums=(
            1,
            2,
        ),
    )

    self.get_rays_no_jitter = cast_fn_no_jitter
    self.get_rays_no_jitter = jax.jit(
        self.get_rays_no_jitter,
        static_argnums=(
            1,
            2,
        ),
    )

    def render_rays(
        rays,
        variables,
        rng,
        train_frac=1.0,
        zero_backfacing=None,
        light_rotation=None,
    ):
      key, rng = jax.random.split(rng)
      r = model.apply(
          variables,
          key,
          rays,
          train_frac=train_frac,
          compute_extras=True,
          train=False,
          mesh=dataset.mesh,
          zero_backfacing=zero_backfacing,
          light_rotation=light_rotation,
          cameras=self.cameras,
          camtype=self.dataset.camtype,
      )
      return r['render'], rng

    render_rays_p = jax.pmap(render_rays, in_axes=(0, 0, 0, None, None, None))

    def render_rays_batched(
        rays,
        variables,
        train_frac=1.0,
        zero_backfacing=None,
        light_rotation=None,
    ):
      out_sharded, self.render_rngs = render_rays_p(
          utils.shard(rays),
          variables,
          self.render_rngs,
          train_frac,
          zero_backfacing,
          light_rotation,
      )
      return jax.tree.map(utils.unshard, out_sharded)

    self.render_rays = render_rays_batched
    self.render_pfn = train_utils.create_render_fn(model)
    self.state = flax.jax_utils.replicate(state)
    self.init_state = self.state
    self.train_pstep = train_pstep

    self.training = False

    self.rngs = random.split(random.PRNGKey(0), jax.local_device_count())
    self.render_rngs = random.split(random.PRNGKey(0), jax.local_device_count())

    # self.rngs = random.PRNGKey(0)
    # self.render_rngs = random.PRNGKey(0)

    self.reset_displays()
    self.chunk = 2**15
    self.fullframe = True

    self.reticle = True

    self.render_tracker = RenderTracker(levels=0, shape=(self.h, self.w))

  def set_target(self, key, multiplier, out, bias=0.0):
    self.text_writer.write(f'Target = {key}')

    if key not in out:
      return out['rgb']

    self.is_target_averageable = False
    return jnp.nan_to_num(out[key]) * multiplier + bias

  def set_average_target(
      self,
      key,
      is_linear,
      multiplier,
      out,
      convert_to_srgb=False,
      use_max=False,
  ):
    self.text_writer.write(f'Target = {key}')

    if key not in out:
      return out['rgb']

    multiplier = (2.0**self.exposure_level) * multiplier

    self.avg_key = key
    self.avg_is_linear = is_linear
    self.avg_multiplier = multiplier
    self.convert_to_srgb = convert_to_srgb
    self.is_target_averageable = True
    self.use_max = use_max
    out = out[self.avg_key]

    if not self.avg_is_linear:
      out = image.srgb_to_linear(out) * self.avg_multiplier
    else:
      out = out * self.avg_multiplier

    if convert_to_srgb:
      out = image.linear_to_srgb(out)

    return out

  def get_average(self, out):
    if self.avg_step == 0:
      self.out_accum = out[self.avg_key]

      if not self.avg_is_linear:
        self.out_accum = (
            image.srgb_to_linear(self.out_accum) * self.avg_multiplier
        )
      else:
        self.out_accum = self.out_accum * self.avg_multiplier

      if self.convert_to_srgb:
        out = image.linear_to_srgb(self.out_accum)
      else:
        out = self.out_accum
    else:
      # Convert to linear
      new_out = out[self.avg_key]

      if not self.avg_is_linear:
        new_out = image.srgb_to_linear(new_out) * self.avg_multiplier
      else:
        new_out = new_out * self.avg_multiplier

      # Average
      a = float(self.avg_step) / (self.avg_step + 1)

      if self.use_max:
        self.out_accum = jnp.maximum(self.out_accum, new_out)
      else:
        self.out_accum = self.out_accum * a + new_out * (1 - a)

      # Convert to sRGB
      if self.convert_to_srgb:
        out = image.linear_to_srgb(self.out_accum)
      else:
        out = self.out_accum

    # Increment
    self.avg_step += 1

    return out

  def render_rays_chunked(self, rays, train_frac=1.0):
    state = flax.jax_utils.unreplicate(self.state)
    render_fn = functools.partial(self.render_pfn, state.params, train_frac)

    return models.render_image(
        render_fn, rays, self.render_rngs, self.config, verbose=False
    )

  def _keyboard_callback(self, _, event):
    self.input_writer.write(str(event))
    self.used_mouse = False

    if event.event_type == event.DOWN:
      self.last_key = event.key
      self.must_update = True

    if event.key == 'Shift' and event.event_type == event.UP:
      self.shift_pause = True

    if event.key == 'l' and event.event_type == event.UP:
      self.select_mode = not self.select_mode
      self.text_writer.write(f'Setting select mode to {self.select_mode}')

    self.keydict[event.key.lower()] = event.event_type == event.DOWN

  def check_key(self, key):
    f = lambda key: (key in self.keydict and self.keydict[key])
    return f(key) or f(key.upper())

  def _mouse_callback(self, path, event):
    self.input_writer.write(str(event))
    self.mouse = event
    self.used_mouse = True

    if event.event_type in [1, 4]:
      # "up" and "leave" events
      self.shift_pause = True

    elif event.event_type == 2:
      # "down" event
      self.shift_pause = False

    if path[0] == 'Rendering' and not self.shift_pause:
      if not self.rotate_cache:
        # When mouse is hovering over rendered output and not in pause mode,
        # add translation into the tracked deltas.
        self.deltas[0] += event.translation_x
        self.deltas[1] += event.translation_y

        if not self.select_mode:
          self.controller.mouse(event.translation_x, event.translation_y)

        self.must_update = True
      else:
        self.deltas[0] += event.translation_x
        self.deltas[1] += event.translation_y

        if not self.select_mode:
          self.controller.cache_mouse(event.translation_x, event.translation_y)

        self.must_update = True

  def reset_default_values(self):
    self.x = 0
    self.y = 0.5

  def reset_displays(self):
    h = int(self.h)
    w = int(self.w)
    highres_mult = int(HIGH_RES_MULTIPLIER)
    self.image_writer.set_display_size(h, w)
    self.light_image_writer.set_display_size(self.light_H, self.light_W)
    if self.highres_window:
      self.highres_image_writer.set_display_size(
          h * highres_mult, w * highres_mult
      )

  def train(self):
    step = self.state.step[0]

    if self.train_batch is None or step % self.grad_accum_steps == 0:
      self.train_batch = next(self.p_raybatcher)

    train_frac = jnp.clip((step - 1) / (self.config.max_steps - 1), 0, 1)

    self.state, stats, self.rngs = self.train_pstep(
        self.rngs, self.state, self.train_batch, self.cameras, train_frac
    )

    return stats

  def render_high_res(self):
    h = int(self.h * HIGH_RES_MULTIPLIER)
    w = int(self.w * HIGH_RES_MULTIPLIER)
    focal = int(self.focal * HIGH_RES_MULTIPLIER)
    pose = self.controller.pose()

    self.text_writer.write('Rendering high res...\nThis may take a minute')
    self.highres_image_writer.set_display_size(h, w)

    key, self.rng = jax.random.split(self.rng)
    rays = self.get_rays(pose, h, w, focal, self.near, key)
    step = self.state.step[0]
    train_frac = jnp.clip((step - 1) / (self.config.max_steps - 1), 0, 1)
    out = self.render_rays_chunked(rays, train_frac)['rgb']
    out_8b = np.array((out * 255).astype(np.uint8))

    self.highres_image_writer.write(out_8b)

    self.text_writer.write('Completed rendering high res')

  def _plot_camera_optim(self, params):
    camera_delta = self.config.camera_delta_cls()
    new_cameras = camera_delta.apply(params, self.jax_cameras)
    rotation_diffs = jax.vmap(camera_utils.rotation_distance)(
        self.jax_cameras_gt.orientation, new_cameras.orientation
    )
    pos_diffs = jnp.linalg.norm(
        self.jax_cameras_gt.position - new_cameras.position, axis=-1
    )
    self.camera_rotation_writer.write({'train': rotation_diffs.mean()})
    self.camera_translation_writer.write({'train': pos_diffs.mean()})

    focal_diffs = abs(
        self.jax_cameras_gt.focal_length - new_cameras.focal_length
    )
    self.camera_focal_length_writer.write({'train': focal_diffs.mean()})

    principal_point_diffs = jnp.linalg.norm(
        self.jax_cameras_gt.principal_point - new_cameras.principal_point,
        axis=-1,
    )
    self.camera_principal_point_writer.write(
        {'train': principal_point_diffs.mean()}
    )
    if new_cameras.has_radial_distortion:
      radial_distortion_gt = jnp.zeros(4)
      if self.jax_cameras_gt.has_radial_distortion:
        radial_distortion_gt = self.jax_cameras_gt.radial_distortion
      radial_distortion_diffs = abs(
          radial_distortion_gt - new_cameras.radial_distortion
      )
      self.camera_radial_distortion1_writer.write(
          {'train': radial_distortion_diffs[Ellipsis, 0].mean()}
      )
      self.camera_radial_distortion2_writer.write(
          {'train': radial_distortion_diffs[Ellipsis, 1].mean()}
      )
      self.camera_radial_distortion3_writer.write(
          {'train': radial_distortion_diffs[Ellipsis, 2].mean()}
      )
      self.camera_radial_distortion4_writer.write(
          {'train': radial_distortion_diffs[Ellipsis, 3].mean()}
      )

    if 'precondition' in params and 'jtj' in params['precondition']:
      jtj = params['precondition']['jtj']
      matrix = jax.vmap(camera_delta.precondition_matrix_from_jtj)(jtj)
      matrix = jnp.log(abs(matrix + 1e-4))
      matrix = np_utils.assemble_arrays(matrix, shape=(-1, 8), spacing=1)
      self.precondition_matrix_writer.write(matrix)

  def set_gt_view(self):
    gt_show = self.dataset.images[self.i_pose]
    gt_show = jax.image.resize(gt_show, (self.h, self.w, 3), 'bilinear')

    if not self.config.linear_to_srgb:
      gt_show = image.linear_to_srgb(gt_show)

    gt_show = np.array((jnp.clip(gt_show, 0.0, 1.0) * 255).astype(np.uint8))
    gt_show = add_border(gt_show, c=[128, 128, 128])
    self.out_show_override = gt_show

  def set_difference_view(self, out, multiplier=2.0):
    gt_show = self.dataset.images[self.i_pose]
    gt_show = jax.image.resize(gt_show, (self.h, self.w, 3), 'bilinear')

    if not self.config.linear_to_srgb:
      gt_show = image.linear_to_srgb(gt_show)

    return jnp.abs(gt_show - out) * multiplier

  def step(self):
    """Runs the UI forward by one step."""

    step_time = time.time()

    if self.training:
      for _ in range(TRAIN_ITERS_PER_STEP):
        stats = self.train()
      x = float(stats['psnr'][0])
      d = {'train': x}
      self.psnr_writer.write(d)
      step = self.state.step[0]
      self.text_writer.write(f'psnr = {x:0.2f}, step = {step}')
      if self.config.optimize_cameras and 'camera_params' in self.state.params:
        camera_params = flax.jax_utils.unreplicate(
            self.state.params['camera_params']
        )
        self._plot_camera_optim(camera_params)

    if self.must_update or self.play_spline:
      dx, dy = self.deltas

      if self.select_mode:
        self.select_x += dx
        self.select_y += dy
      else:
        self.x += dx / self.w * self._mouse_speed
        self.y += dy / self.w * self._mouse_speed

      for i in range(2):
        self.deltas[i] = 0

      self.must_update = False

      self.round = 0
      self.i = 0

      arrows = ['w', 'a', 's', 'd', 'e', 'q', '1', '2']
      self.controller.keyboard(*[self.check_key(k) for k in arrows])

      if (
          dx != 0
          or dy != 0
          or (not self.used_mouse and self.last_key not in ['g', 't', 'p'])
      ):
        self.avg_step = 0

      if self.last_key in self.modes:
        self.mode = self.last_key
        self.text_writer.write(f'mode = {self.mode}')
      elif self.last_key == 'v':
        self.averaging = not self.averaging
        self.text_writer.write(f'Toggle averaging: {self.averaging}')
      elif self.last_key == 'p' and self.mode == 'c':
        self.show_difference = not self.show_difference
        self.text_writer.write(
            f'Toggle show difference: {self.show_difference}'
        )
      elif self.last_key == 'b':
        pass

        # self.rotate_cache = (not self.rotate_cache)
        # self.text_writer.write(
        #     f'Toggle rotate cache: {self.rotate_cache}'
        # )

        # if not self.rotate_cache:
        #   self.controller.reset_light_rotation()

      elif self.last_key == '3':
        self.focal /= self._geom_ratio
        self.text_writer.write(f'focal = {self.focal:0.2f}')
      elif self.last_key == '4':
        self.focal *= self._geom_ratio
        self.text_writer.write(f'focal = {self.focal:0.2f}')
      elif self.last_key == '5':
        self.near /= 2.0
        self.text_writer.write(f'near = {self.near:0.2f}')
      elif self.last_key == '6':
        self.near *= 2.0
        self.text_writer.write(f'near = {self.near:0.2f}')
      elif self.last_key == '-' and self.exposure is not None:
        self.exposure /= 2.0
        self.text_writer.write(f'exposure = {self.exposure:0.2f}')
      elif self.last_key == '=' and self.exposure is not None:
        self.exposure *= 2.0
        self.text_writer.write(f'exposure = {self.exposure:0.2f}')
      elif self.last_key == 'm':
        pass
      elif self.last_key == 'k':
        self.edit_spline = not self.edit_spline
      elif self.last_key == 'f':
        self.controller.switch_mode()
        self.text_writer.write(f'controller.focus = {self.controller.focus}')
      elif self.last_key == 'r':
        self.reticle = not self.reticle
      elif self.last_key == 't':
        self.training = not self.training
        self.text_writer.write(f'training = {self.training}')
      elif self.last_key == 'o':
        self.controller.position = np.zeros((3,))
        self.text_writer.write('Reset position')
      elif self.last_key == '[':
        self.exposure_level -= 1

        # if self.render_tracker.levels > 0:
        #   self.render_tracker.reset_params(self.render_tracker.levels - 1)
        # self.text_writer.write(f'Progressive levels {self.render_tracker.levels}')
      elif self.last_key == ']':
        self.exposure_level += 1

        # self.render_tracker.reset_params(self.render_tracker.levels + 1)
        # self.text_writer.write(f'Progressive levels {self.render_tracker.levels}')
      elif self.last_key == ' ':
        pass
      elif self.last_key == 'i':
        pass
      elif self.last_key == 'ArrowLeft':
        self.sub_mode -= 1
      elif self.last_key == 'ArrowRight':
        self.sub_mode += 1
      elif self.last_key == 'ArrowDown':
        self.i_pose = (self.i_pose - 1) % len(self.dataset.camtoworlds)
        self.controller.invpose(self.dataset.camtoworlds[self.i_pose])
        if self.out_show_override is not None:
          self.set_gt_view()
        self.text_writer.write(f'Pose index {self.i_pose}')
      elif self.last_key == 'ArrowUp':
        self.i_pose = (self.i_pose + 1) % len(self.dataset.camtoworlds)
        self.controller.invpose(self.dataset.camtoworlds[self.i_pose])
        if self.out_show_override is not None:
          self.set_gt_view()
        self.text_writer.write(f'Pose index {self.i_pose}')
      elif self.last_key == 'Backspace':
        self.controller.spl.pop()
      elif self.last_key == 'h' and self.highres_window:
        self.render_high_res()
      elif self.last_key == 'g':
        self.controller.invpose(self.dataset.camtoworlds[self.i_pose])
        if self.out_show_override is None:
          self.set_gt_view()
        else:
          self.out_show_override = None
      self.last_key = None

      if not self.controller.aligned:
        self.out_show_override = None

      if self.play_spline:
        self.controller.play()
        self.text_writer.write('Frame', self.controller.spl.t)
      elif self.edit_spline:
        self.controller.edit()

      self.render_tracker.reset()

    # Update rays
    pose = self.controller.pose()

    key, self.rng = jax.random.split(self.rng)
    self.all_rays_jitter = self.get_rays(
        pose,
        int(self.h),
        int(self.w),
        self.focal,
        self.near,
        key,
    )
    self.all_rays_no_jitter = self.get_rays_no_jitter(
        pose,
        int(self.h),
        int(self.w),
        self.focal,
        self.near,
        self.rng,
    )

    if self.is_target_averageable and not self.select_mode:
      self.all_rays = self.all_rays_jitter
    else:
      self.all_rays = self.all_rays_no_jitter

    if self.exposure is not None:
      self.all_rays = self.all_rays.replace(
          exposure_values=self.all_rays.near * 0.0 + self.exposure
      )

    if self.render_tracker.levels > 0:
      ray_idx, img_idx = self.render_tracker.get_idx(self.training)
      if ray_idx is not None:

        def select(z):
          return z[ray_idx[0], ray_idx[1]] if z is not None else None

        ray_subset = jax.tree.map(select, self.all_rays)
        step = self.state.step[0]
        train_frac = jnp.clip((step - 1) / (self.config.max_steps - 1), 0, 1)
        out = self.render_rays(
            ray_subset,
            self.state.params,
            train_frac,
            None,
            self.controller.light_rotation() if self.rotate_cache else None,
        )
        self.out_show[img_idx[0], img_idx[1]] = out['rgb']
    else:
      step = self.state.step[0]
      train_frac = jnp.clip((step - 1) / (self.config.max_steps - 1), 0, 1)
      out = self.render_rays(
          self.all_rays,
          self.state.params,
          train_frac,
          None,
          self.controller.light_rotation() if self.rotate_cache else None,
      )
      depth = out['distance_median']

      if not self.controller.focus:
        x = self.w // 2
        y = self.h // 2
        origin = self.all_rays.origins[y, x]
        direction = self.all_rays.directions[y, x]
        xyz = origin + direction * depth[y, x]
        # self.controller.z = depth[y, x]
        self.controller.z = np.linalg.norm(self.controller.pose()[:3, -1])
        self.cursor_info_writer.write(
            f'depth = {depth[y, x]:.04f}\n'
            f'x = {xyz[0]:.04f}\n'
            f'y = {xyz[1]:.04f}\n'
            f'z = {xyz[2]:.04f}\n'
            f'r = {jnp.linalg.norm(xyz):.04f}'
        )

      # Save dict
      out_dict = out

      # Render estimate of lighting
      if self.select_mode:
        select_x = int(self.select_x)
        select_y = int(self.select_y)

        positions = (
            self.all_rays.origins[select_y, select_x]
            + self.all_rays.directions[select_y, select_x]
            * depth[select_y, select_x, Ellipsis, None]
        )
        # normals = out['normals_pred'][select_y, select_x]
        normals = out['normals'][select_y, select_x]
        positions = positions + 1e-5 * normals
        rot_mat = render_utils.get_rotation_matrix(-normals[None])[0]
        cam_to_world = np.eye(4)
        cam_to_world[:3, :3] = rot_mat
        cam_to_world[Ellipsis, :3, -1] = positions

        sphere_rays = camera_utils.cast_spherical_rays(
            cam_to_world,
            self.light_H,
            self.light_W,
            0.0,
            # 2e-1,
            # 0.5,
            self.far,
            xnp=jnp,
        )

        new_rays = dataclasses.replace(sphere_rays)
        new_rays = dataclasses.replace(
            new_rays,
            directions=self.light_xyz.reshape(sphere_rays.directions.shape),
        )
        new_rays = dataclasses.replace(
            new_rays,
            viewdirs=self.light_xyz.reshape(sphere_rays.viewdirs.shape),
        )

        light_mask = (normals[None, None] * new_rays.directions).sum(
            axis=-1, keepdims=True
        ) > 0.0

        out_dict_light = self.render_rays(
            new_rays,
            self.state.params,
            train_frac,
            True,
            self.controller.light_rotation() if self.rotate_cache else None,
        )

        out_light = (
            out_dict_light['cache_rgb']
            if 'cache_rgb' in out_dict_light
            else out_dict_light['rgb']
        )
        out_light_max = out_light.max()

        # Display
        num_sub_modes = 2 if 'vmf_means' in out_dict else 1

        if (self.sub_mode % num_sub_modes) == 0:
          out_light = image.linear_to_srgb(out_light)
          out_show_8b = np.array(
              (out_light * light_mask * 255).astype(np.uint8)
          )
          self.light_image_writer.write(out_show_8b)

        elif (self.sub_mode % num_sub_modes) == 1:
          true_means = ref_utils.l2_normalize(
              out_dict['vmf_means'][select_y, select_x]
          )
          true_kappas = out_dict['vmf_kappas'][select_y, select_x, Ellipsis, 0]
          true_weights = out_dict['vmf_weights'][select_y, select_x, Ellipsis, 0]

          # true_means, true_kappas, true_weights = render_utils.filter_vmf_vars(
          #     (true_means, true_kappas, true_weights),
          #     normals,
          # )

          true_weights = jax.nn.softmax(math.safe_log(true_weights))
          out_light_vmf = jnp.sum(
              true_weights
              * render_utils.eval_vmf(
                  self.light_xyz[Ellipsis, None, :], true_means, true_kappas
              ),
              axis=-1,
          ).reshape(self.light_H, self.light_W, 1)

          out_light_vmf = jnp.repeat(out_light_vmf, 3, axis=-1)
          out_light_vmf = out_light_vmf / out_light_vmf.max()

          out_show_8b = np.array((out_light_vmf * 255).astype(np.uint8))
          self.light_image_writer.write(out_show_8b)

      # Averaging
      self.is_target_averageable = False

      # Set output array
      if self.mode == 'z':
        depth_curve_fn = lambda x: -jnp.log(x + jnp.finfo(jnp.float32).eps)
        out = vis.visualize_cmap(
            depth,
            out_dict['acc'],
            cm.get_cmap('turbo'),
            curve_fn=depth_curve_fn,
        )
      elif self.mode == 'n':
        num_sub_modes = 3

        if (self.sub_mode % num_sub_modes) == 0:
          out = self.set_target('normals', 0.5, out_dict, 0.5)
        elif (self.sub_mode % num_sub_modes) == 1:
          out = self.set_target('normals_pred', 0.5, out_dict, 0.5)
        elif (self.sub_mode % num_sub_modes) == 2:
          out = self.set_target('point_offset_contract', 2.0, out_dict, 0.5)
        else:
          out = self.set_target('normals_pred', 0.5, out_dict, 0.5)
      elif self.mode == 'i':
        num_sub_modes = 5

        if (self.sub_mode % num_sub_modes) == 0:
          out = self.set_average_target(
              'cache_rgb', True, 1.0, out_dict, convert_to_srgb=True
          )
        elif (self.sub_mode % num_sub_modes) == 1:
          out = self.set_average_target(
              'lighting_irradiance',
              True,
              0.5,
              out_dict,
              convert_to_srgb=True,
              use_max=False,
          )
        elif (self.sub_mode % num_sub_modes) == 2:
          out = self.set_average_target(
              'integrated_multiplier_specular', True, 0.5, out_dict
          )
        elif (self.sub_mode % num_sub_modes) == 3:
          out = self.set_average_target(
              'integrated_multiplier_diffuse', True, 0.5, out_dict
          )
        elif (self.sub_mode % num_sub_modes) == 4:
          out = self.set_average_target(
              'lighting_emission', True, 1.0, out_dict, convert_to_srgb=True
          )
        else:
          out = self.set_average_target(
              'cache_rgb', True, 1.0, out_dict, convert_to_srgb=True
          )

      elif self.mode == 'm':
        num_sub_modes = 7

        if (self.sub_mode % num_sub_modes) == 0:
          out = self.set_average_target(
              'material_total_albedo', True, 1.0, out_dict
          )
        elif (self.sub_mode % num_sub_modes) == 1:
          out = self.set_target('material_residual_albedo', 1.0, out_dict)
        elif (self.sub_mode % num_sub_modes) == 2:
          out = self.set_target('material_albedo', 1.0, out_dict)
        elif (self.sub_mode % num_sub_modes) == 3:
          out = self.set_target('material_roughness', 1.0, out_dict)
        elif (self.sub_mode % num_sub_modes) == 4:
          out = self.set_target('material_F_0', 1.0, out_dict)
        elif (self.sub_mode % num_sub_modes) == 5:
          out = self.set_target('material_metalness', 1.0, out_dict)
        elif (self.sub_mode % num_sub_modes) == 6:
          out = self.set_target('material_diffuseness', 1.0, out_dict)
        else:
          out = self.set_average_target(
              'material_total_albedo', True, 1.0, out_dict
          )
      else:
        num_sub_modes = 2

        if (self.sub_mode % num_sub_modes) == 0:
          out = self.set_average_target(
              'rgb',
              not self.config.linear_to_srgb,
              1.0,
              out_dict,
              convert_to_srgb=True,
          )
        elif (
            self.sub_mode % num_sub_modes
        ) == 1 and 'material_rgb' in out_dict:
          out = self.set_average_target(
              'material_rgb',
              not self.config.linear_to_srgb,
              1.0,
              out_dict,
              convert_to_srgb=True,
          )
        else:
          out = self.set_average_target(
              'rgb',
              not self.config.linear_to_srgb,
              1.0,
              out_dict,
              convert_to_srgb=True,
          )

      # Average
      if not self.is_target_averageable:
        self.avg_step = 0
      elif self.averaging:
        out = self.get_average(out_dict)

      # Show difference
      if self.mode == 'c' and self.show_difference:
        out = self.set_difference_view(out, 5.0)

      self.out_show = np.array(out)

    out_show_8b = np.array(
        (jnp.clip(self.out_show, 0.0, 1.0) * 255).astype(np.uint8)
    )

    if self.reticle:
      out_show_8b = add_reticle(out_show_8b, c=[255, 0, 0])
    if self.play_spline:
      out_show_8b = add_border(out_show_8b, c=[0, 255, 0])
    elif self.edit_spline:
      out_show_8b = add_border(out_show_8b, c=[255, 0, 0])
    if self.out_show_override is not None:
      out_show_8b = self.out_show_override

    if self.select_mode:
      out_show_8b = add_reticle_xy(
          out_show_8b,
          c=[255, 0, 0],
          x=select_x,
          y=select_y,
      )

    self.image_writer.write(out_show_8b)
    self.camera_writer.write(str(self.controller.pose()))

    t = time.time() - step_time
    fps = 1.0 / t
    self.fps_writer.write({'updates per sec': fps})
