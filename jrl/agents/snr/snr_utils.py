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

"""Implementation of various SNR components"""

from collections import OrderedDict
import time
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Tuple
from acme.jax import networks as networks_lib
import jax
from jax.experimental import host_callback
import jax.numpy as jnp
from jax.scipy.special import logsumexp as jax_logsumexp
import numpy as np
import tensorflow_probability
from jrl.agents.snr import kmeans
from jrl.agents.snr.config import SNRKwargs
tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions
distributional = networks_lib.distributional


class SNRState(NamedTuple):
  """Contains training state for the SNR."""
  snr_matrix: Optional[jnp.ndarray] = None
  u: Optional[jnp.ndarray] = None
  v: Optional[jnp.ndarray] = None
  phiT_phi_prime: Optional[jnp.ndarray] = None
  phiT_phi: Optional[jnp.ndarray] = None
  centroids: Optional[jnp.ndarray] = None
  counts: Optional[jnp.ndarray] = None


def host_eig(A):
  return np.linalg.eig(A)


def jax_eig(A):
  complex_dtype = jax.lax.complex(A, A).dtype
  result_shape = (
      jax.ShapeDtypeStruct((A.shape[0],), complex_dtype),
      jax.ShapeDtypeStruct(A.shape, complex_dtype)
  )
  w, v = host_callback.call(
      host_eig,
      jax.lax.stop_gradient(A),
      result_shape=result_shape
  )
  return w, v


def compute_spec_norm_vectors(
    A,
    key):
  def spec_norm_step(i, state):
    u, v = state
    v = A.T @ u
    v = v / jnp.linalg.norm(v)
    u = A @ v
    u = u / jnp.linalg.norm(u)
    return (u, v)

  key, sub_key = jax.random.split(key)
  u = jax.random.normal(sub_key, shape=[A.shape[0], 1])
  u = u / jnp.linalg.norm(u)
  key, sub_key = jax.random.split(key)
  v = jax.random.normal(sub_key, shape=[A.shape[1], 1])
  v = v / jnp.linalg.norm(v)

  u, v = jax.lax.fori_loop(
      0,
      20,
      spec_norm_step,
      (u, v))
  return u, v


def compute_gelfand_approx(
    A,
    num_squarings = 8):
  """Approximates spectral radius of A using Gelfand's theorem.

  Approximates ||A^k||^{1/k}, where k = 2 ** num_squarings and ||.|| is the
  Frobenius norm.
  """
  initial_coeff = (1e-6 + jnp.sum(A ** 2)) ** 0.5
  coeffs = []
  A /= initial_coeff
  for _ in range(num_squarings):
    next_A = A @ A
    next_coeff = (1e-6 + jnp.sum(next_A ** 2)) ** 0.5
    A = next_A / next_coeff
    coeffs.append(next_coeff)

  gelfand_coeff = initial_coeff * jnp.exp(
      sum([jnp.log(c) * 2 ** (-i - 1)
           for (i, c) in enumerate(coeffs)]))
  gelfand_approx = (gelfand_coeff *
                    (1e-6 + jnp.sum(A ** 2)) ** (0.5 / 2.0 ** num_squarings))
  return gelfand_approx


def snr_state_init(
    c_dim,
    key,
    snr_kwargs,):
  if snr_kwargs.use_log_space_matrix:
    snr_state = SNRState(snr_matrix=jnp.ones([c_dim, c_dim]))
  else:
    if snr_kwargs.snr_loss_type == 'svd_kamyar_v3':
      snr_state = SNRState(
          phiT_phi_prime=jnp.zeros([256, 256]),
          phiT_phi=jnp.zeros([256, 256]),)
    elif snr_kwargs.snr_loss_type == 'svd_kamyar_v1_with_kmeans':
      key, sub_key = jax.random.split(key)
      snr_state = SNRState(
          centroids=jax.random.normal(sub_key, [snr_kwargs.snr_num_centroids, c_dim]),
          counts=jnp.ones([snr_kwargs.snr_num_centroids, 1]),)
    elif snr_kwargs.snr_loss_type in ['svd_ofir_v2', 'gelfand_ofir_v1']:
      snr_state = SNRState(
          phiT_phi_prime=jnp.zeros([c_dim, c_dim]),
          phiT_phi=jnp.zeros([c_dim, c_dim]),)
    # elif snr_kwargs.snr_loss_type == 'spec_norm_v0':
    #   key, sub_key = jax.random.split(key)
    #   u = jax.random.normal(sub_key, shape=[c_dim, 1])
    #   u = u / jnp.linalg.norm(u)
    #   key, sub_key = jax.random.split(key)
    #   v = jax.random.normal(sub_key, shape=[c_dim, 1])
    #   v = v / jnp.linalg.norm(v)
    #   snr_state = SNRState(
    #       snr_matrix=None,
    #       u=u,
    #       v=v,)
    else:
      snr_state = SNRState()

  return snr_state


def build_snr_loss_fn(
    snr_kwargs,
    discount,
    networks,
    compute_kernel_features, # for when snr model is based on params
    ):
  def snr_loss_fn(
      next_dist_params, # a Tanh action distribution
      obs,
      acts,
      next_obs,
      step_discount,
      key,
      snr_state,
      q_params,
      target_q_params,):
    distribution = tfd.Normal(
        loc=next_dist_params[0], scale=next_dist_params[1])
    next_dist_params = tfd.Independent(
        distributional.TanhTransformedDistribution(distribution),
        reinterpreted_batch_ndims=1)
    policy_next_acts = networks.sample(next_dist_params, key)

    X = jnp.concatenate([obs, acts], axis=-1)
    X_prime = jnp.concatenate([next_obs, policy_next_acts], axis=-1)

    if snr_kwargs.snr_mode == 'ntk':
      K_X_X = networks.q_kernel_fn(X, X)
      K_X_X = K_X_X.ntk
    elif snr_kwargs.snr_mode == 'nngp':
      K_X_X = networks.q_kernel_fn(X, X)
      K_X_X = K_X_X.nngp
    elif snr_kwargs.snr_mode == 'params_kernel':
      # dQ_dX0 = compute_kernel_features(q_params[0], X)
      dQ_dX0 = compute_kernel_features(q_params, X)
      K_X_X = dQ_dX0 @ dQ_dX0.T
    else:
      raise NotImplementedError()
    K_X_X = K_X_X + (1e-4) * jnp.eye(K_X_X.shape[0])
    K_X_X_inv = jnp.linalg.inv(K_X_X)

    if snr_kwargs.snr_mode == 'ntk':
      K_Xprime_X = networks.q_kernel_fn(X_prime, X)
      K_Xprime_X = K_Xprime_X.ntk
    elif snr_kwargs.snr_mode == 'nngp':
      K_Xprime_X = networks.q_kernel_fn(X_prime, X)
      K_Xprime_X = K_Xprime_X.nngp
    elif snr_kwargs.snr_mode == 'params_kernel':
      if snr_kwargs.use_target_for_phi_prime:
        # dQ_dX1 = compute_kernel_features(target_q_params[0], X_prime)
        # dQ_dX1 = compute_kernel_features(target_q_params, X_prime)
        dQ_dX1 = compute_kernel_features(q_params, X_prime)
      else:
        # dQ_dX1 = compute_kernel_features(q_params[0], X_prime)
        dQ_dX1 = compute_kernel_features(q_params, X_prime)
      K_Xprime_X = dQ_dX1 @ dQ_dX0.T
    else:
      raise NotImplementedError()

    if snr_kwargs.use_log_space_matrix:
      assert snr_kwargs.snr_mode == 'params_kernel', snr_kwargs.snr_mode
      C = dQ_dX0.T @ (step_discount * discount * dQ_dX1 - dQ_dX0)
      snr_matrix = snr_state.snr_matrix
      snr_matrix = (1. - snr_kwargs.snr_matrix_tau) * snr_matrix + snr_kwargs.snr_matrix_tau * C
      max_eig = 0.
    else:
      C = K_Xprime_X @ K_X_X_inv
      snr_matrix = C
      max_eig = 1.

    # s, v = jnp.linalg.eigh(jax.lax.stop_gradient(
    #     0.5 * (avg_matrix + avg_matrix.T)))

    # mask = (s > max_eig)
    # masked_s = s * mask

    # sn = jnp.transpose(v) @ (0.5 * (C + C.T)) @ v
    # sn = jnp.sum(jnp.diag(sn) * mask)

    if snr_kwargs.snr_loss_type == 'full':
      w, v = jax_eig(snr_matrix)
      sn = jnp.sum(
          jax.nn.relu(
              jnp.diag(jnp.linalg.inv(v) @ snr_matrix @ v).real - max_eig
          ) + max_eig
      )
      s = w.real
      mask = (s > max_eig)
      masked_s = s * mask
    elif snr_kwargs.snr_loss_type == 'trace_relu':
      sn = jax.nn.relu(jnp.trace(snr_matrix))
      masked_s = jnp.zeros((snr_matrix.shape[0],), dtype=snr_matrix.dtype)
    elif snr_kwargs.snr_loss_type == 'trace_abs':
      sn = jnp.sum(jnp.abs(jnp.diag(snr_matrix)))
      masked_s = jnp.zeros((snr_matrix.shape[0],), dtype=snr_matrix.dtype)
    elif snr_kwargs.snr_loss_type == 'svd_kamyar':
      # doing a full svd
      u, s, vh = jnp.linalg.svd(jax.lax.stop_gradient(snr_matrix))
      sn = jnp.diag(u.T @ snr_matrix @ vh.T)
      mask = (sn > max_eig)
      masked_s = sn * mask
      sn = jnp.sum(masked_s)
    elif snr_kwargs.snr_loss_type == 'svd_kamyar_v1':
      # doing 20 iters of spec norm in each step
      key, sub_key = jax.random.split(key)
      u, v = compute_spec_norm_vectors(
          jax.lax.stop_gradient(snr_matrix),
          sub_key)
      sn = jnp.diag(u.T @ snr_matrix @ v)
      mask = (sn > max_eig)
      masked_s = sn * mask
      sn = jnp.sum(masked_s)
    elif snr_kwargs.snr_loss_type == 'svd_kamyar_v1_pinv':
      # doing 20 iters of spec norm in each step
      snr_matrix = dQ_dX1 @ jnp.linalg.pinv(dQ_dX0.T).T
      key, sub_key = jax.random.split(key)
      u, v = compute_spec_norm_vectors(
          jax.lax.stop_gradient(snr_matrix),
          sub_key)
      sn = jnp.diag(u.T @ snr_matrix @ v)
      mask = (sn > max_eig)
      masked_s = sn * mask
      sn = jnp.sum(masked_s)
    elif snr_kwargs.snr_loss_type == 'svd_kamyar_v1_with_kmeans':
      key, sub_key = jax.random.split(key)

      centroids = snr_state.centroids
      phi = jnp.concatenate([centroids, dQ_dX0], axis=0)
      phi_phiT = phi @ phi.T
      phi_prime_phiT = dQ_dX1 @ phi.T
      snr_matrix = phi_prime_phiT @ jnp.linalg.inv(
          phi_phiT + 1e-4 * jnp.eye(phi_phiT.shape[0]))

      # doing 20 iters of spec norm in each step
      key, sub_key = jax.random.split(key)
      u, v = compute_spec_norm_vectors(
          jax.lax.stop_gradient(snr_matrix),
          sub_key)
      sn = jnp.diag(u.T @ snr_matrix @ v)
      mask = (sn > max_eig)
      masked_s = sn * mask
      sn = jnp.sum(masked_s)
    elif snr_kwargs.snr_loss_type == 'svd_kamyar_v2':
      # same as v1 but with inner product matrices
      key, sub_key = jax.random.split(key)

      phiT_phi_prime = dQ_dX0.T @ dQ_dX1
      phiT_phi = dQ_dX0.T @ dQ_dX0
      snr_matrix = phiT_phi_prime @ jnp.linalg.inv(
          phiT_phi + 1e-4 * jnp.eye(phiT_phi.shape[0]))

      u, v = compute_spec_norm_vectors(
          jax.lax.stop_gradient(snr_matrix),
          sub_key)
      sn = jnp.diag(u.T @ snr_matrix @ v)
      mask = (sn > max_eig)
      masked_s = sn * mask
      sn = jnp.sum(masked_s)
    elif snr_kwargs.snr_loss_type == 'svd_kamyar_v3':
      # same as v2 but with EMA matrices
      key, sub_key = jax.random.split(key)

      phiT_phi_prime = dQ_dX0.T @ dQ_dX1
      phiT_phi_prime = (1. - snr_kwargs.snr_matrix_tau) * snr_state.phiT_phi_prime + \
          snr_kwargs.snr_matrix_tau * phiT_phi_prime
      phiT_phi = dQ_dX0.T @ dQ_dX0
      phiT_phi = (1. - snr_kwargs.snr_matrix_tau) * snr_state.phiT_phi + \
          snr_kwargs.snr_matrix_tau * phiT_phi
      snr_matrix = phiT_phi_prime @ jnp.linalg.inv(
          phiT_phi + 1e-4 * jnp.eye(phiT_phi.shape[0]))

      u, v = compute_spec_norm_vectors(
          jax.lax.stop_gradient(snr_matrix),
          sub_key)
      sn = jnp.diag(u.T @ snr_matrix @ v)
      mask = (sn > max_eig)
      masked_s = sn * mask
      sn = jnp.sum(masked_s)
      sn = sn / snr_kwargs.snr_matrix_tau
    elif snr_kwargs.snr_loss_type == 'svd_ofir_v1':
      phiT_phi_prime = dQ_dX0.T @ (discount * step_discount[:, None] * dQ_dX1)
      phiT_phi = dQ_dX0.T @ dQ_dX0

      snr_matrix = phiT_phi_prime @ jnp.linalg.inv(
          phiT_phi + 1e-4 * jnp.eye(phiT_phi.shape[0]))

      u, s, vh = jnp.linalg.svd(jax.lax.stop_gradient(snr_matrix))
      sn = jnp.diag(u.T @ snr_matrix @ vh.T)
      mask = (sn > max_eig)
      masked_s = sn * mask
      sn = jnp.sum(masked_s)
    elif snr_kwargs.snr_loss_type == 'svd_ofir_v2':
      phiT_phi_prime = snr_state.phiT_phi_prime
      phiT_phi = snr_state.phiT_phi

      phiT_phi_prime = (1. - snr_kwargs.snr_matrix_tau) * phiT_phi_prime + \
          snr_kwargs.snr_matrix_tau * (dQ_dX0.T @ (discount * step_discount[:, None] * dQ_dX1))
      phiT_phi = (1. - snr_kwargs.snr_matrix_tau) * phiT_phi + \
          snr_kwargs.snr_matrix_tau * (dQ_dX0.T @ dQ_dX0)

      snr_matrix = phiT_phi_prime @ jnp.linalg.inv(
          phiT_phi + 1e-4 * jnp.eye(phiT_phi.shape[0]))

      u, s, vh = jnp.linalg.svd(jax.lax.stop_gradient(snr_matrix))
      sn = jnp.diag(u.T @ snr_matrix @ vh.T)
      mask = (sn > max_eig)
      masked_s = sn * mask
      sn = jnp.sum(masked_s)
    elif snr_kwargs.snr_loss_type == 'gelfand_ofir_v1':
      phiT_phi_prime = snr_state.phiT_phi_prime
      phiT_phi = snr_state.phiT_phi

      phiT_phi_prime = (1. - snr_kwargs.snr_matrix_tau) * phiT_phi_prime + \
          snr_kwargs.snr_matrix_tau * (dQ_dX0.T @ (discount * step_discount[:, None] * dQ_dX1))
      phiT_phi = (1. - snr_kwargs.snr_matrix_tau) * phiT_phi + \
          snr_kwargs.snr_matrix_tau * (dQ_dX0.T @ dQ_dX0)

      snr_matrix = phiT_phi_prime @ jnp.linalg.inv(
          phiT_phi + 1e-4 * jnp.eye(phiT_phi.shape[0]))

      s = compute_gelfand_approx(snr_matrix)
      masked_s = s * (s > max_eig)
      sn = masked_s
      masked_s = jnp.expand_dims(masked_s, 0)

    # elif snr_kwargs.snr_loss_type == 'spec_norm_v0':
    #   u = snr_state.u
    #   v = snr_state.v
    #   v = snr_matrix @ u
    #   v = v / jnp.linalg.norm(v)
    #   u = snr_matrix @ v
    #   u = u / jnp.linalg.norm(u)
    #   sn = (u.T @ snr_matrix @ v)[0, 0]
    #   sn = jax.nn.relu(sn - 1.) + 1.
    #   masked_s = 0.

    # elif snr_kwargs.snr_loss_type == 'spectral_norm':
    #   u = snr_state.u
    #   v = snr_state.v
    #   stop_grad_snr_matrix = jax.lax.stop_gradient(snr_matrix)
    #   v = stop_grad_snr_matrix.T @ u
    #   v = v / jnp.linalg.norm(v)
    #   u = stop_grad_snr_matrix @ v
    #   u = u / jnp.linalg.norm(u)
    else:
      raise NotImplementedError()

    # Update the SNRState
    if snr_kwargs.use_log_space_matrix:
      new_snr_state = SNRState(snr_matrix=snr_matrix)
    else:
      if snr_kwargs.snr_loss_type == 'svd_kamyar_v3':
        new_snr_state = SNRState(
            phiT_phi_prime=phiT_phi_prime,
            phiT_phi=phiT_phi,)
      elif snr_kwargs.snr_loss_type == 'svd_kamyar_v1_with_kmeans':
        centroids, counts = kmeans._kmeans_update_step(
          data=jax.lax.stop_gradient(dQ_dX0),
          prev_centroids=snr_state.centroids,
          prev_counts=snr_state.counts,
          key=sub_key,
          iters=snr_kwargs.snr_kmeans_iters,
          decay=0.9,
          counts_decay=0.9,
          dead_tolerance=0.01,)
        new_snr_state = SNRState(
            centroids=centroids,
            counts=counts,)
      elif snr_kwargs.snr_loss_type in ['svd_ofir_v2', 'gelfand_ofir_v1']:
        new_snr_state = SNRState(
            phiT_phi_prime=phiT_phi_prime,
            phiT_phi=phiT_phi,)
      # elif snr_kwargs.snr_loss_type == 'spec_norm_v0':
      #   new_snr_state = SNRState(
      #       snr_matrix=None,
      #       u=u,
      #       v=v,)
      else:
        new_snr_state = SNRState()

    return sn, (masked_s, C, new_snr_state)

  return snr_loss_fn
