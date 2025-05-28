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


def _repeated_dot_product(x, y):
  """npgc,mc->npgm"""
  N, P, G, C = x.shape
  z = jnp.einsum('nc,mc->nm', x[:, 0, 0, :], y)
  return jnp.tile(jnp.reshape(z, [N, 1, 1, -1]), [1, P, G, 1])


@jax.custom_vjp
def repeated_dot_product_custom_vjp(x, y):
  return _repeated_dot_product(x, y)


def repeated_dot_product_fwd_(x, y):
  return _repeated_dot_product(x, y), (x, y)


def repeated_dot_product_bwd_(res, g):
  # Warning dy is always zero.
  x, y = res
  N, P, G, C = x.shape
  g_ = g[:, 0, 0, :]  # [n,m]
  dx = jnp.reshape(jnp.einsum('mc,nm->nc', y, g_), [N, 1, 1, -1])
  dx = jnp.tile(dx, [1, P, G, 1])
  return dx, jnp.zeros_like(y)


repeated_dot_product_custom_vjp.defvjp(repeated_dot_product_fwd_,
                                       repeated_dot_product_bwd_)


@jax.custom_jvp
def repeated_dot_product_custom_jvp(x, y):
  return _repeated_dot_product(x, y)


def repeated_dot_product_jvp_(primals, tangents):
  x, y = primals
  N, P, G, C = x.shape
  dx, dy = tangents
  dz = jnp.einsum('npgc,mc->npgm', dx, y)
  return _repeated_dot_product(x, y), dz


repeated_dot_product_custom_jvp.defjvp(repeated_dot_product_jvp_)


def repeated_dot_product_v2(x, y):
  return jnp.einsum('npgc,mc->npgm', x, y)


def _spatial_avg_group_linear(x, w, b):
  # N,P,G,C -> N,G,C
  N, P, G, C = x.shape
  x_avg = jnp.mean(x, axis=1)
  x_grp = jnp.reshape(x_avg, [x_avg.shape[0], -1])
  print(x_grp.shape, w.shape, b.shape)
  y = jnp.einsum('nc,cd->nd', x_grp, w) + b
  return jnp.tile(jnp.reshape(y, [N, 1, 1, -1]), [1, P, G, 1])


@jax.custom_vjp
def spatial_avg_group_linear_custom_vjp(x, w, b):
  return _spatial_avg_group_linear(x, w, b)


@jax.custom_jvp
def spatial_avg_group_linear_custom_jvp(x, w, b):
  return _spatial_avg_group_linear(x, w, b)


def spatial_avg_group_linear_jvp_(primals, tangents):
  x, w, b = primals
  dx, dw, db = tangents
  N, P, G, C = x.shape
  dx_avg = dx / float(P)
  w_ = jnp.reshape(w, [G, C, -1])
  b = jnp.reshape(b, [-1])
  x_avg = jnp.mean(x, axis=1)
  x_grp = jnp.reshape(x_avg, [x_avg.shape[0], -1])
  dy = jnp.einsum('npgc,gcd->npgd', dx_avg, w_) + jnp.einsum(
      'nc,cd->nd', x_grp, dw)[:, None, None, :] + db
  y = jnp.einsum('nc,cd->nd', x_grp, w)[:, None, None, :] + b
  y = jnp.tile(y, [1, P, G, 1])
  return y, dy


def spatial_avg_group_linear_fwd_(x, w, b):
  return _spatial_avg_group_linear(x, w, b), (x, w)


def spatial_avg_group_linear_bwd_(res, g):
  x, w = res
  N, P, G, C = x.shape
  x_avg = jnp.mean(x, axis=1)
  x_grp = jnp.reshape(x_avg, [x_avg.shape[0], -1])
  g_ = g[:, 0, 0, :]
  db = jnp.reshape(jnp.sum(g_, axis=[0]), [-1]) * float(P * G)
  dw = jnp.reshape(jnp.einsum('nc,nd->cd', x_grp, g_), [G * C, -1]) * float(
      P * G)
  dx = jnp.einsum('ngd,gcd->ngc', g[:, 0, :, :], jnp.reshape(
      w, [G, C, -1])) / float(P)
  dx = jnp.tile(dx[:, None, :, :], [1, P, 1, 1])
  return dx, dw, db


spatial_avg_group_linear_custom_jvp.defjvp(spatial_avg_group_linear_jvp_)
spatial_avg_group_linear_custom_vjp.defvjp(spatial_avg_group_linear_fwd_,
                                           spatial_avg_group_linear_bwd_)
