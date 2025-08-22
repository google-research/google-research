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

import jax
import jax.numpy as jnp


def matmul(a, b):
  """jnp.matmul defaults to bfloat16, but this helper function doesn't."""
  return jnp.matmul(a, b, precision=jax.lax.Precision.HIGHEST)


def parse_bin(s):
  return int(s[1:], 2) / 2.0 ** (len(s) - 1)


def phi2(i):
  return parse_bin('.' + f'{i:b}'[::-1])


def nice_uniform(N):
  u = []
  v = []
  for i in range(N):
    u.append(i / float(N))
    v.append(phi2(i))

  return u, v


def safe_arctan2(x1, x2):
  eps = jnp.finfo(jnp.float32).eps
  safe_x1 = jnp.where(x1 < 0., -1., +1.) * jnp.maximum(jnp.abs(x1), eps)
  safe_x2 = jnp.where(x2 < 0., -1., +1.) * jnp.maximum(jnp.abs(x2), eps)
  return jnp.arctan2(safe_x1, safe_x2)


def cart2sph(xyz):
  eps = jnp.finfo(jnp.float32).eps
  x, y, z = xyz[Ellipsis, 0], xyz[Ellipsis, 1], xyz[Ellipsis, 2]
  # phi = jnp.where(jnp.bitwise_or(jnp.abs(x) > eps, jnp.abs(y) > eps), jnp.arctan2(y, x), 0.0)
  # theta = jnp.arctan2(jnp.sqrt(x**2 + y**2), z)
  r = jnp.sqrt(jnp.maximum(eps, x**2 + y**2))
  phi = safe_arctan2(y, x)
  theta = safe_arctan2(r, z)
  return theta, phi


def reflect(w, v=None):
  """Reflect w about v."""
  if v is None:
    v = jnp.array([0.0, 0.0, 1.0])
  #return 2.0 * v.dot(w) * v - w
  return 2.0 * (v * w).sum(axis=-1, keepdims=True) * v - w


def nice_uniform_spherical(N, hemisphere=True):
  """implementation of http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html"""
  u, v = nice_uniform(N)

  theta = jnp.arccos(1.0 - jnp.array(u)) * (2.0 - int(hemisphere))
  phi = 2.0 * jnp.pi * jnp.array(v)

  return theta, phi


def normalize(v):
    return v / jnp.sqrt(1e-10 + jnp.sum(v ** 2, axis=-1, keepdims=True))


def mse_to_psnr(mse, max_val=1.0):
  """Compute PSNR given an MSE (we assume the maximum pixel value is 1)."""
  return (20.0 * jnp.log(max_val) - 10.0 * jnp.log(mse)) / jnp.log(10.0)


def spherical_mse(img1, img2, jacobian):
  dtheta_dphi = 2 * jnp.pi ** 2 / img1.shape[0] / img1.shape[1]
  mse = (jacobian[:, None] * (img1 - img2).reshape(-1, 3) ** 2).sum() * dtheta_dphi / 4.0 / jnp.pi / 3.0
  return mse

def smape(img1, img2):
  return jnp.mean(jnp.abs(img1 - img2) * 2.0 / (jnp.abs(img1) + jnp.abs(img2)))

def spherical_smape(img1, img2, jacobian):
  dtheta_dphi = 2 * jnp.pi ** 2 / img1.shape[0] / img1.shape[1]
  return (jacobian[:, None] * (jnp.abs(img1 - img2) * 2.0 / (jnp.abs(img1) + jnp.abs(img2))).reshape(-1, 3)).sum() * dtheta_dphi / 4.0 / jnp.pi / 3.0



def per_channel_multiplier_invariant_spherical_mse(img1, img2, jacobian):
  w = jacobian[:, None]
  best_mult = (w * (img1 * img2).reshape(-1, 3)).sum(0) / (w * (img1 ** 2).reshape(-1, 3)).sum(0)

  return spherical_mse(best_mult * img1, img2, jacobian), best_mult


@jax.jit
def safe_exp(x):
  return jnp.exp(jnp.minimum(x, 80.0))

