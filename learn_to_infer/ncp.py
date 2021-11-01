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

"""Implementation of Neural Clustering Processes.
"""
from functools import partial

from . import jax_nn

import jax
from jax import grad
from jax import jit
from jax import vmap
import jax.numpy as np
import jax.random
import jax.scipy as scipy


class NCP(object):

  def __init__(self, h_dim, u_dim, g_dim, x_dim,
               hidden_layer_dim, num_hidden_layers, key):
    keys = jax.random.split(key, num=5)
    hidden_layers = [hidden_layer_dim] * num_hidden_layers
    self.h_params = jax_nn.init_network_params(
        [x_dim] + hidden_layers + [h_dim], keys[0])
    self.u_params = jax_nn.init_network_params(
        [x_dim] + hidden_layers + [u_dim], keys[1])
    self.g_params = jax_nn.init_network_params(
        [h_dim] + hidden_layers + [g_dim], keys[2])
    self.f_params = jax_nn.init_network_params(
        [g_dim + u_dim] + hidden_layers + [1], keys[3])
    self.f_fn = lambda g, u, p: jax_nn.nn_fwd(
        np.concatenate([g, u], axis=0), p).squeeze()
    self.params = (self.h_params, self.u_params, self.g_params, self.f_params)
    self.key = keys[4]

  @partial(jit, static_argnums=0)
  def ll(self, xs, cs):
    return self._ll(xs, cs, self.params)

  def _ll(self, xs, cs, params):
    """Computes log likelihoods under the neural clustering process.
    """
    h_params, u_params, g_params, f_params = params
    num_data_points, unused_data_dim = xs.shape
    hs = vmap(jax_nn.nn_fwd, in_axes=(0, None))(xs, h_params)
    h_dim = hs.shape[1]
    us = vmap(jax_nn.nn_fwd, in_axes=(0, None))(xs, u_params)
    U = np.sum(us[1:], axis=0)
    Hs = np.zeros([num_data_points, h_dim])
    Hs = Hs.at[0].set(hs[0])
    G = jax_nn.nn_fwd(Hs[0], g_params)
    K = 1

    def scan_body(state, i):
      U, K, G, Hs, ll = state
      U = U - us[i]

      def inner_map_body(k):
        H_k = Hs[k] + hs[i]
        G_k = G - jax_nn.nn_fwd(Hs[k], g_params) + jax_nn.nn_fwd(H_k, g_params)
        return self.f_fn(G_k, U, f_params)

      def map_body(k):
        return jax.lax.cond(np.less(k, K+1),
                            inner_map_body,
                            lambda x: -np.inf,
                            k)

      log_potentials = jax.lax.map(map_body, np.arange(num_data_points))
      log_Z_hat = scipy.special.logsumexp(log_potentials, keepdims=True)
      log_q = log_potentials - log_Z_hat
      ll += log_q[cs[i]]
      K = jax.lax.cond(np.equal(cs[i], K), lambda x: x + 1, lambda x: x, K)
      G = G - jax_nn.nn_fwd(Hs[cs[i]], g_params) + jax_nn.nn_fwd(
          Hs[cs[i]] + hs[i], g_params)
      Hs = Hs.at[cs[i]].set(Hs[cs[i]] + hs[i])
      return (U, K, G, Hs, ll), None

    out = jax.lax.scan(
        scan_body,
        (U, K, G, Hs, 0.),
        np.arange(1, num_data_points))
    return out[0][4]

  @partial(jit, static_argnums=0)
  def grad_ll(self, xs, cs):
    return grad(
        lambda x, c, p: self._ll(x, c, p), argnums=2)(xs, cs, self.params)

  @partial(jit, static_argnums=0)
  def sample(self, xs):
    self.key, key = jax.random.split(self.key)
    return self._sample(xs, self.params, key)

  def _sample(self, xs, params, key):
    """Runs the neural clustering process.
    """
    h_params, u_params, g_params, f_params = params
    num_data_points, unused_data_dim = xs.shape
    hs = vmap(jax_nn.nn_fwd, in_axes=(0, None))(xs, h_params)
    h_dim = hs.shape[1]
    us = vmap(jax_nn.nn_fwd, in_axes=(0, None))(xs, u_params)
    U = np.sum(us[1:], axis=0)
    Hs = np.zeros([num_data_points, h_dim])
    Hs = Hs.at[0].set(hs[0])
    G = jax_nn.nn_fwd(Hs[0], g_params)
    cs = np.zeros([num_data_points], dtype=np.int32)
    K = 1

    def for_body(i, state):
      U, K, G, Hs, cs, key = state
      U = U - us[i]

      def inner_map_body(k):
        H_k = Hs[k] + hs[i]
        G_k = G - jax_nn.nn_fwd(Hs[k], g_params) + jax_nn.nn_fwd(H_k, g_params)
        return self.f_fn(G_k, U, f_params)

      def map_body(k):
        return jax.lax.cond(np.less(k, K+1),
                            inner_map_body,
                            lambda x: -np.inf,
                            k)

      log_potentials = vmap(map_body)(np.arange(num_data_points))
      log_Z_hat = scipy.special.logsumexp(log_potentials, keepdims=True)
      log_q = log_potentials - log_Z_hat
      key, subkey = jax.random.split(key)
      c = jax.random.categorical(subkey, log_q)
      K = jax.lax.cond(np.equal(c, K), lambda x: x + 1, lambda x: x, K)
      G = G - jax_nn.nn_fwd(Hs[c], g_params) + jax_nn.nn_fwd(
          Hs[c] + hs[i], g_params)
      Hs = Hs.at[c].set(Hs[c] + hs[i])
      cs = cs.at[i].set(c)
      return (U, K, G, Hs, cs, key)

    out = jax.lax.fori_loop(1, num_data_points,
                            for_body,
                            (U, K, G, Hs, cs, jax.random.PRNGKey(0)))
    return out[4]
