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

"""
SASS utilities used for experimenting in MSG.

This stuff is experimental and has not made it into the paper.
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp as jax_logsumexp

from acme.jax import networks as networks_lib


SMALL_LOGIT = -1e10
# SMALL_LOGIT = 0.


def fill_diagonal(a, val):
  "From https://github.com/google/jax/issues/2680#issuecomment-804269672"
  assert a.ndim >= 2
  i, j = jnp.diag_indices(min(a.shape[-2:]))
  return a.at[Ellipsis, i, j].set(val)


def build_simclr_loss(encoder_network):
  def simclr_loss_fn(
      encoder_params,
      h1,
      h2,
      temp = 1.,):

    z1 = encoder_network(encoder_params, h1)
    # z1 = z1 / jnp.linalg.norm(z1, keepdims=True)
    z2 = encoder_network(encoder_params, h2)
    # z2 = z2 / jnp.linalg.norm(z2, keepdims=True)

    z = jnp.concatenate([z1, z2], axis=0)
    s = (z @ z.T) / temp
    s = fill_diagonal(s, SMALL_LOGIT) # 2N x 2N

    ######### implementation v1
    # n1 = jnp.diagonal(s, offset=z1.shape[0])
    # n2 = jnp.diagonal(s, offset=-z1.shape[0])
    # n = jnp.concatenate([n1, n2])
    # logsumexp = jax_logsumexp(s, axis=-1)
    # log_probs = n - logsumexp
    # loss = -1. * jnp.mean(log_probs)
    #########

    labels1 = jnp.arange(z1.shape[0], 2 * z1.shape[0])
    labels2 = jnp.arange(z1.shape[0])
    labels = jnp.concatenate([labels1, labels2], axis=0)

    # # for debugging see if you can predict yourself reliably
    # labels = jnp.arange(z.shape[0])

    one_hot_labels = jax.nn.one_hot(labels, num_classes=z.shape[0], dtype=z.dtype)
    log_softmax = jax.nn.log_softmax(s, axis=-1)
    log_prob = jnp.sum(log_softmax * one_hot_labels, axis=-1)
    loss = -1. * jnp.mean(log_prob)
    pred_inds = jnp.argmax(log_softmax, axis=-1)
    acc = jnp.mean(pred_inds == labels)

    return loss, acc

  return simclr_loss_fn
