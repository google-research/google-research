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
from jax.scipy.special import logsumexp


def _spatial_avg_group_linear_cross_entropy(x, w, b, labels):
  # N,P,G,C -> N,G,C
  N, P, G, C = x.shape
  x_avg = jnp.mean(x, axis=1)
  x_grp = jnp.reshape(x_avg, [x_avg.shape[0], -1])
  logits = jnp.einsum('nc,cd->nd', x_grp, w) + b
  logits = logits - logsumexp(logits, axis=-1, keepdims=True)
  loss = -jnp.sum(logits * labels, axis=-1)
  return jnp.tile(jnp.reshape(loss, [N, 1, 1]), [1, P, G])


@jax.custom_vjp
def spatial_avg_group_linear_cross_entropy_custom_vjp(x, w, b, labels):
  return _spatial_avg_group_linear_cross_entropy(x, w, b, labels)


@jax.custom_jvp
def spatial_avg_group_linear_cross_entropy_custom_jvp(x, w, b, labels):
  return _spatial_avg_group_linear_cross_entropy(x, w, b, labels)


def spatial_avg_group_linear_cross_entropy_jvp_(primals, tangents):
  x, w, b, labels = primals
  dx, dw, db, dlabels = tangents
  N, P, G, C = x.shape
  dx_avg = dx / float(P)
  w_ = jnp.reshape(w, [G, C, -1])
  b = jnp.reshape(b, [-1])
  x_avg = jnp.mean(x, axis=1)
  x_grp = jnp.reshape(x_avg, [x_avg.shape[0], -1])
  logits = jnp.einsum('nc,cd->nd', x_grp, w) + b
  logits = logits - logsumexp(logits, axis=-1, keepdims=True)
  loss = -jnp.sum(logits * labels, axis=-1)
  dlogits_bwd = jax.nn.softmax(logits, axis=-1) - labels  # [N, D]
  dloss = jnp.einsum('npgc,gcd,nd->npg', dx_avg, w_, dlogits_bwd) + jnp.einsum(
      'nd,nd->n',
      (jnp.einsum('nc,cd->nd', x_grp, dw) + db), dlogits_bwd)[:, None, None]
  return jnp.tile(jnp.reshape(loss, [N, 1, 1]), [1, P, G]), dloss


def spatial_avg_group_linear_cross_entropy_fwd_(x, w, b, labels):
  N, P, G, C = x.shape
  x_avg = jnp.mean(x, axis=1)
  x_grp = jnp.reshape(x_avg, [x_avg.shape[0], -1])
  logits = jnp.einsum('nc,cd->nd', x_grp, w) + b
  logits = logits - logsumexp(logits, axis=-1, keepdims=True)
  loss = -jnp.sum(logits * labels, axis=-1)
  return jnp.tile(jnp.reshape(loss, [N, 1, 1]),
                  [1, P, G]), (x, w, logits, labels)


def spatial_avg_group_linear_cross_entropy_bwd_(res, g):
  x, w, logits, labels = res
  N, P, G, C = x.shape
  x_avg = jnp.mean(x, axis=1)
  x_grp = jnp.reshape(x_avg, [x_avg.shape[0], -1])
  g_ = g[:, 0:1, 0]
  dlogits = g_ * (jax.nn.softmax(logits, axis=-1) - labels)  # [N, D]

  db = jnp.reshape(jnp.sum(dlogits, axis=[0]), [-1]) * float(P * G)
  dw = jnp.reshape(jnp.einsum('nc,nd->cd', x_grp, dlogits),
                   [G * C, -1]) * float(P * G)
  dx = jnp.einsum('nd,gcd->ngc', dlogits, jnp.reshape(w,
                                                      [G, C, -1])) / float(P)
  dx = jnp.tile(dx[:, None, :, :], [1, P, 1, 1])
  return dx, dw, db, None


spatial_avg_group_linear_cross_entropy_custom_jvp.defjvp(
    spatial_avg_group_linear_cross_entropy_jvp_)
spatial_avg_group_linear_cross_entropy_custom_vjp.defvjp(
    spatial_avg_group_linear_cross_entropy_fwd_,
    spatial_avg_group_linear_cross_entropy_bwd_)


def spatial_avg_group_linear_cross_entropy_v2(x, w, b, labels):
  # Concate everything.
  N, P, G, C = x.shape
  avg_pool_p = jnp.mean(x, axis=1, keepdims=True)
  x_div_p = x / float(P)
  x = x_div_p + jax.lax.stop_gradient(avg_pool_p - x_div_p)

  x = jnp.tile(jnp.reshape(x, [N, P, 1, G, -1]), [1, 1, G, 1, 1])
  mask = jnp.eye(G)[None, None, :, :, None]
  x = mask * x + jax.lax.stop_gradient((1.0 - mask) * x)
  x = jnp.reshape(x, [N, P, G, -1])
  logits = jnp.einsum('npgc,cd->npgd', x, w) + b
  logits = logits - logsumexp(logits, axis=-1, keepdims=True)
  loss = -jnp.sum(logits * labels[:, None, None, :], axis=-1)
  return loss
