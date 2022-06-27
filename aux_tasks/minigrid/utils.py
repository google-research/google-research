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

"""Several objective functions."""
import jax.numpy as jnp


def grassman_distance(Y1, Y2):  # pylint: disable=invalid-name
  """Grassman distance between subspaces spanned by Y1 and Y2."""
  Q1, _ = jnp.linalg.qr(Y1)  # pylint: disable=invalid-name
  Q2, _ = jnp.linalg.qr(Y2)  # pylint: disable=invalid-name

  _, sigma, _ = jnp.linalg.svd(Q1.T @ Q2)
  # sigma = jnp.clip(sigma, -1., 1.)
  sigma = jnp.round(sigma, decimals=6)
  return jnp.linalg.norm(jnp.arccos(sigma))


def inner_objective_mc(Phi, Psi, W):  # pylint: disable=invalid-name
  # Inner Objective function: $\|\Phi W - \Psi \|^2_F$
  return jnp.linalg.norm(Phi @ W - Psi, ord='fro')**2


def outer_objective_mc(Phi, Psi):  # pylint: disable=invalid-name
  # Outer objective function: $J(\Phi) =\min_W \|\Phi W - \Psi \|^2_F$
  W_star, _, _, _ = jnp.linalg.lstsq(Phi, Psi, rcond=1e-5)  # pylint: disable=invalid-name
  return inner_objective_mc(Phi, Psi, W_star)


def transition_matrix(env, policy):
  return jnp.einsum('ijk,ij->ik', env.transition_probs, policy)


def approx_error(F, k, v):  # pylint: disable=invalid-name
  S = F.shape[0]  # pylint: disable=invalid-name
  F_k = F[:, :k]  # pylint: disable=invalid-name
  P_perp_term = (jnp.eye(S) - F_k@F_k.T) @ v  # pylint: disable=invalid-name
  approx_err = jnp.linalg.norm(P_perp_term, ord='fro')**2
  return approx_err
