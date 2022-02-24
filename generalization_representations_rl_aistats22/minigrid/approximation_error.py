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

"""Computes Approximation error.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow.compat.v1 as tf


def w_star(d, F, V):  # pylint: disable=invalid-name
  r"""Best d-dimensional linear model.

  Args:
    d: int, top d projection of the left singular vectors.
    F: S \times S matrix, left singular vectors.
    V: S \times 1 matrix, values of the states.

  Returns:
    weight vector: array of size d
  """
  return np.linalg.pinv(F[:, :d]) @ V


def approx_error(d, F, V):  # pylint: disable=invalid-name
  r"""Computes the approximation error for a d-dimensional linear model.

  Features from the SR in a noiseless setting.

  If the parameter reward_function is None, we use the environment's reward. If
  the parameter reward_function is a non-None vector, we use those rewards
  instead.

  Args:
    d: int, top d projection of the left singular vectors
    F: S \times S matrix, left singular vectors
    V: S \times 1 matrix, targets
  Returns:
    approximation error: Float.
  """
  return np.mean((F[:, :d] @ w_star(d, F, V) - V)**2)

approx_error_vec = np.vectorize(approx_error, excluded=[1, 2])


def plot_approx_error_rewards(base_dir, k_range, data, rewards, num_states,
                              gamma, env_name):
  """Plot estimation error vs dimnension."""
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
        markevery=34,
        ls='-',
        ms=8)
  plt.xlabel('Number of features', fontsize='xx-large')
  plt.ylabel('Approximation error', fontsize='xx-large')
  # plt.legend(rewards, fontsize='xx-large')
  plt.title(
      'S={}, MDP={}, gamma={}'.format(num_states, env_name, gamma),
      fontsize='xx-large')
  xticks = np.arange(0, len(k_range), 50)
  xticks[0] = 1
  plt.xticks(ticks=xticks)
  plt.tick_params(length=0.1, width=0.1, labelsize='x-large')
  plt.gca().spines['right'].set_visible(False)
  plt.gca().spines['top'].set_visible(False)
  pdf_file = os.path.join(base_dir,
                          f'{env_name}_approx_error_compare_rewards.pdf')
  with tf.io.gfile.GFile(pdf_file, 'w') as f:
    plt.savefig(f, format='pdf', dpi=300, bbox_inches='tight')
  plt.clf()
  plt.close('all')


def plot_singular_values(base_dir, data):
  """Plot singular values for all policies."""
  plt.subplots(1, 1, figsize=(8, 6))
  num_policies, num_states = data.shape
  dimension = np.arange(1, num_states+1)
  epsilons = np.linspace(0, 1, num_policies)
  for i in range(num_policies):
    plt.plot(
        dimension, data[i], lw=3, label=f'epsilon {epsilons[i]:.1f}')
  plt.xlabel('Index', fontsize=24)
  plt.ylabel('Singular Value', fontsize=24)
  plt.yscale('log')
  plt.legend(fontsize=18, loc=7)
  pdf_file = os.path.join(base_dir, 'singular_value.pdf')
  with tf.io.gfile.GFile(pdf_file, 'w') as f:
    plt.savefig(f, format='pdf', dpi=300, bbox_inches='tight')
  plt.clf()
  plt.close('all')
