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

"""Computes Generalization error.
"""

import os

from absl import flags
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS


def orthogonal_projection(k, F, v, norm='2'):  # pylint: disable=invalid-name
  r"""Computes the P^\perp term.

  Args:
    k: int, dimension
    F: n times k array
    v: S \times 1 matrix, targets
    norm: str, '2' or 'infinity'

  Returns:
    P^\perp: float
  """
  assert np.allclose(np.transpose(F) @ F, np.eye(F.shape[1]))
  F_k = F[:, :k]  # pylint: disable=invalid-name
  n, k = F_k.shape
  P_perp = (np.eye(n) - F_k @ F_k.T) @ v  # pylint: disable=invalid-name
  if norm == '2':
    return (P_perp**2).sum()
  elif norm == 'infinity':
    return np.max(P_perp**2)


orth_proj_vec = np.vectorize(orthogonal_projection, excluded=[1, 2, 3])


def stylized_theoretical_bound(F, k, v, n):  # pylint: disable=invalid-name
  r"""Computes the sloppy version theoretical estimation error bound.

  Args:
    F: n times k array
    k: int, dimension
    v: S \times 1 matrix, targets
    n: int, daatset size

  Returns:
    theoretical bound: float
  """
  # 1/S*|P^perp_{F_k} v|^2 + k mu(F) / n + k mu(F)/n * |P^perp_{F_k} v|^2_inf/n
  S = F.shape[0]  # pylint: disable=invalid-name
  F_k = F[:, :k]  # pylint: disable=invalid-name
  P_perp_term = (np.eye(S) - F_k@F_k.T) @ v  # pylint: disable=invalid-name
  approx_error = 1/S * np.sum(np.square(P_perp_term))
  est_error = k * coherence(k, F) / n
  lower_order_error = k * coherence(k, F) / n * np.linalg.norm(
      P_perp_term, ord=np.inf)**2 / n
  return approx_error + est_error + lower_order_error


def coherence(k, F):  # pylint: disable=invalid-name
  """Computes the coherence of k-truncated matrix F.

  Args:
    k: int, dimension
    F: n times k array

  Returns:
    coherence: float
  """
  assert np.allclose(np.transpose(F) @ F, np.eye(F.shape[0]))
  F_k = F[:, :k]  # pylint: disable=invalid-name
  n, k = F_k.shape
  return n / k * np.max(np.sum(np.square(F_k), axis=1))

coherence_vec = np.vectorize(coherence, excluded=[1])


def w_hat(d, F, V, indices, noise):  # pylint: disable=invalid-name
  r"""Empirical risk minimizers.

  Args:
    d: int, top d projection of the left singular vectors
    F: S \times S matrix, left singular vectors
    V: S \times 1 matrix, targets
    indices: array of ints, indices of the training dataset
    noise: float: noise added to true values

  Returns:
    Numpy array with indices
  """
  w, _, _, _ = np.linalg.lstsq(F[indices, :d], V[indices] + noise)
  return w


def estimation_error(d, F, V, indices, noise):  # pylint: disable=invalid-name
  r"""Computes the estimation error for a d-dimensional linear model from F.

  Args:
    d: int, top d projection of the left singular vectors
    F: S \times S matrix, left singular vectors
    V: S \times 1 matrix, targets
    indices: array of ints, indices of the training dataset
    noise: float: noise added to true values
  Returns:
    estimation error: Float.
  """
  assert len(indices.shape) == 1
  S = F.shape[0]  # pylint: disable=invalid-name
  return 1/S * np.sum(np.square(F[:, :d] @ w_hat(d, F, V, indices, noise) - V))


def generate_and_estime(n, d, F, V, trials, seed=None):  # pylint: disable=invalid-name
  r"""Computes the estimation error for a d-dimensional linear model.

  The model is linear wrt some features F and the estimator is computed from
  data by ERM.

  Args:
    n: int, experience dataset size
    d: int, top d projection of the left singular vectors
    F: S \times S matrix, left singular vectors
    V: S \times 1 matrix, targets
    trials: int, number of repetitions
    seed: seed

  Returns:
    estimation error: Float.
  """
  if seed is None:
    rng = np.random.default_rng()
  else:
    rng = np.random.default_rng(seed)
  return np.array([
      estimation_error(d, F, V, rng.integers(0, F.shape[0], size=n),
                       0.1 * rng.normal(size=n)) for _ in range(trials)
  ])


def plot_excess_error_rewards(base_dir, k_range, data, rewards, num_samples,
                              num_states, gamma, env_name):
  """Plot estimation error vs dimnension."""
  plt.subplots(1, 1, figsize=(7, 5))
  colors = sns.color_palette('colorblind')
  idx = -1
  for reward_type in rewards:
    idx += 1
    plt.semilogy(
        k_range,
        np.median(data[reward_type], axis=1),
        'o',
        color=colors[idx],
        linewidth=3,
        markevery=12,
        ls='-',
        ms=8)
    plt.fill_between(
        k_range,
        np.percentile(data[reward_type], 10, axis=1),
        np.percentile(data[reward_type], 90, axis=1),
        alpha=0.3)
  plt.xlabel('Number of features', fontsize='xx-large')
  plt.ylabel('Empirical excess risk', fontsize='xx-large')
  plt.title(
      'n={}, S={}, MDP={}, gamma={}'.format(num_samples, num_states, env_name,
                                            gamma),
      fontsize='xx-large')
  xticks = np.arange(0, len(k_range), 20)
  xticks[0] = 1
  plt.xticks(ticks=xticks)
  plt.tick_params(length=0.1, width=0.1, labelsize='x-large')
  plt.gca().spines['right'].set_visible(False)
  plt.gca().spines['top'].set_visible(False)
  pdf_file = os.path.join(base_dir,
                          f'{env_name}_excess_error_compare_rewards.pdf')
  with tf.io.gfile.GFile(pdf_file, 'w') as f:
    plt.savefig(f, format='pdf', dpi=300, bbox_inches='tight')
  plt.clf()
  plt.close('all')


def plot_coherence(base_dir, data):
  """Plot the coherence and effective dimension."""
  num_policies, num_states = data.shape
  dimension = np.arange(1, num_states+1)
  plt.subplots(1, 1, figsize=(8, 6))
  for i in range(num_policies):
    plt.plot(
        dimension, data[i], lw=3)
  plt.xlabel('Feature Dimension', fontsize=24)
  plt.ylabel('Coherence', fontsize=24)
  plt.legend(fontsize=18, loc=7)
  pdf_file = os.path.join(base_dir, 'coherence.pdf')
  with tf.io.gfile.GFile(pdf_file, 'w') as f:
    plt.savefig(f, format='pdf', dpi=300, bbox_inches='tight')
  plt.clf()
  plt.close('all')
  plt.subplots(1, 1, figsize=(8, 6))
  for i in range(num_policies):
    plt.plot(
        dimension,
        data[i] * dimension,
        lw=3)
  plt.xlabel('Feature Dimension', fontsize=24)
  plt.ylabel('Effective Dimension', fontsize=24)
  plt.legend(fontsize=18, loc=7)
  pdf_file = os.path.join(base_dir, 'effective_dimension.pdf')
  with tf.io.gfile.GFile(pdf_file, 'w') as f:
    plt.savefig(f, format='pdf', dpi=300, bbox_inches='tight')
  plt.clf()
  plt.close('all')


def plot_sty_bound_rewards(base_dir, k_range, data, rewards, num_samples,
                           num_states, gamma, env_name):
  """Plot th estimation error vs dimnension."""
  plt.subplots(1, 1, figsize=(7, 5))
  colors = sns.color_palette('colorblind')
  idx = -1
  for reward_type in rewards:
    idx += 1
    plt.plot(
        k_range,
        data[reward_type],
        'o',
        color=colors[idx],
        linewidth=3,
        markevery=12,
        ls='-',
        ms=8)
  plt.xlabel('Number of features', fontsize='xx-large')
  plt.ylabel('Stylized excess risk error', fontsize='xx-large')
  plt.legend(rewards, fontsize=18)
  plt.title(
      'n={}, S={}, MDP={}, gamma={}'.format(num_samples, num_states, env_name,
                                            gamma),
      fontsize='xx-large')
  xticks = np.arange(0, len(k_range), 20)
  xticks[0] = 1
  plt.xticks(ticks=xticks)
  plt.tick_params(length=0.1, width=0.1, labelsize='x-large')
  plt.gca().spines['right'].set_visible(False)
  plt.gca().spines['top'].set_visible(False)
  pdf_file = os.path.join(base_dir,
                          f'{env_name}_stylized_error_compare_rewards.pdf')
  with tf.io.gfile.GFile(pdf_file, 'w') as f:
    plt.savefig(f, format='pdf', dpi=300, bbox_inches='tight')
  plt.clf()
  plt.close('all')
