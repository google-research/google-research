# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Ripple and K-norm mechanism library for the experiments and plots.

The script run_experiments.py calls the functions in this file to run the
experiments and generate the plots.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from dp_ripple import k_norm
from dp_ripple import ripple_count
from dp_ripple import ripple_sum
from dp_ripple import ripple_vote

# Import mechanism functions from the k_norm module
compute_eulerian_numbers = k_norm.compute_eulerian_numbers
count_mechanism = k_norm.count_mechanism
sum_mechanism = k_norm.sum_mechanism
vote_mechanism = k_norm.vote_mechanism

# pylint: disable=g-docstring-has-escape
# pylint: disable=anomalous-backslash-in-string
# pylint: disable=invalid-name

###############################################################################
# Sum experiments
###############################################################################


def sum_norm(v, k):
  """Returns the sum induced K-norm of the vector v.

  Args:
    v: Numpy array.
    k: Integer l_0 bound.
  """
  l_1_norm_over_k = np.linalg.norm(v, ord=1) / k
  l_inf_norm = np.linalg.norm(v, ord=np.inf)
  return max(l_1_norm_over_k, l_inf_norm)


def sum_norm_mean_norm(d, k, eps, num_samples):
  """Computes mean ball norms for K-norm sum mechanism.

  Args:
    d: Integer dimension.
    k: Integer l_0 bound.
    eps: Float privacy parameter.
    num_samples: Integer number of samples.

  Returns:
    Float mean norm.
  """
  eulerian_numbers = compute_eulerian_numbers(d)
  sum_mechanism_samples = sum_mechanism(
      np.zeros(d), eulerian_numbers, k, eps, num_samples
  )
  mean_norm = np.mean(
      np.array([sum_norm(sample, k) for sample in sum_mechanism_samples])
  )
  return mean_norm


def sum_ripple_mean_norm(d, k, eps, num_samples):
  """Computes mean sum norm for Ripple sum mechanism.

  Args:
    d: Integer dimension.
    k: Integer l_0 bound.
    eps: Float privacy parameter.
    num_samples: Integer number of samples.

  Returns:
    Float mean sum norm.
  """
  ripple_sum_samples = ripple_sum.sample_ripple_sum_point(
      d, k, eps, num_samples
  )
  mean_norm = np.mean(
      np.array([sum_norm(sample, k) for sample in ripple_sum_samples])
  )
  return mean_norm


def sum_plot_norm_vs_eps_for_multiple_k(
    d, num_samples, k_values, eps_values, output_dir="."
):
  """Plots Ripple mechanism error reduction against K-norm mechanism for sum.

  Creates a plot with a line plotting reduction vs epsilon for each k in
  k_values.

  Args:
    d: Integer dimension.
    num_samples: Integer number of samples.
    k_values: Sequence of integer k bounds to check.
    eps_values: Sequence of float epsilons to check.
    output_dir: String output directory path.
  """
  # Set a base font size variable for easy adjustments
  base_fs = 20

  plt.figure(figsize=(12, 8))  # Increased figure size for better spacing

  for k in k_values:
    ripple_norms = [
        sum_ripple_mean_norm(d, k, eps, num_samples) for eps in eps_values
    ]
    k_norms = [
        sum_norm_mean_norm(d, k, eps, num_samples) for eps in eps_values
    ]
    error_reduction_percent = [
        (k_norms[i] - ripple_norms[i]) / k_norms[i]
        for i in range(len(ripple_norms))
    ]

    plt.plot(
        eps_values,
        error_reduction_percent,
        marker="o",
        linewidth=2.5,
        markersize=8,
        label=f"k = {k}",
    )

  # Formatting the plot with larger fonts
  plt.xlabel(r"$\epsilon$", fontsize=base_fs + 4)

  plt.ylabel(r"Average Sum Norm Error Reduction", fontsize=base_fs + 2)
  plt.title(f"Sum ($d={d}$)", fontsize=base_fs + 6, pad=20)

  # Tick labels (the numbers on the axes)
  plt.xticks(eps_values, fontsize=base_fs)
  plt.yticks(fontsize=base_fs)

  # Legend and Grid
  plt.legend(
      title="k values", title_fontsize=base_fs, fontsize=base_fs - 2, loc="best"
  )
  plt.grid(True, linestyle="--", alpha=0.7)

  plt.tight_layout()  # Ensures labels don't get cut off
  plt.savefig(os.path.join(output_dir, "sum_plot.png"))
  plt.close()


################################################################################
# Count experiments
################################################################################


def count_norm(v, k):
  """Returns the count induced K-norm of the vector v.

  Args:
    v: Numpy array.
    k: Integer l_0 bound.
  """
  p = np.where(v > 0, v, 0)
  n = np.where(v < 0, v, 0)
  return sum_norm(p, k) + sum_norm(n, k)


def count_norm_mean_norms(d, k, eps, num_samples):
  """Computes mean count norm for the K-norm count mechanism.

  Args:
    d: Integer dimension.
    k: Integer l_0 bound.
    eps: Float privacy parameter.
    num_samples: Integer number of samples.

  Returns:
    Float mean count norm.
  """
  eulerian_numbers = compute_eulerian_numbers(d)
  count_mechanism_samples = count_mechanism(
      np.zeros(d), eulerian_numbers, k, eps, num_samples
  )
  mean_norm = np.mean(
      np.array([count_norm(v, k) for v in count_mechanism_samples])
  )
  return mean_norm


def count_ripple_mean_norms(d, k, eps, num_samples):
  """Computes mean count norm for the Ripple count mechanism.

  Args:
    d: Integer dimension.
    k: Integer l_0 bound.
    eps: Float privacy parameter.
    num_samples: Integer number of samples.

  Returns:
    Float mean count norm.
  """
  ripple_count_samples = ripple_count.sample_ripple_count_point(
      d, k, eps, num_samples
  )
  mean_norm = np.mean(
      np.array([count_norm(v, k) for v in ripple_count_samples])
  )
  return mean_norm


def count_plot_norm_vs_eps_for_multiple_k(
    d, num_samples, eps_values, k_values, output_dir="."
):
  """Plots Ripple mechanism error reduction against K-norm mechanism for count.

  Returns a plot with a line plotting reduction vs epsilon for each k in
  k_values.

  Args:
    d: Integer dimension.
    num_samples: Integer number of samples.
    eps_values: Sequence of float epsilons to check.
    k_values: Sequence of integer k bounds to check.
    output_dir: String output directory path.
  """
  # Set a base font size variable for easy adjustments
  base_fs = 20

  plt.figure(figsize=(12, 8))  # Increased figure size for better spacing

  for k in k_values:
    # Calculate the ripple sum norm for each epsilon at this specific k
    ripple_norms = [
        count_ripple_mean_norms(d, k, eps, num_samples) for eps in eps_values
    ]
    k_norms = [
        count_norm_mean_norms(d, k, eps, num_samples) for eps in eps_values
    ]
    error_reduction_percent = [
        (k_norms[i] - ripple_norms[i]) / k_norms[i]
        for i in range(len(ripple_norms))
    ]

    # Plot this k-value as a separate line
    plt.plot(
        eps_values,
        error_reduction_percent,
        marker="o",
        linewidth=2.5,
        markersize=8,
        label=f"k = {k}",
    )

  # Formatting the plot
  plt.xlabel(r"$\epsilon$", fontsize=base_fs + 4)
  plt.ylabel(r"Average Count Norm Error Reduction", fontsize=base_fs + 2)
  plt.title(f"Count ($d={d}$)", fontsize=base_fs + 6, pad=20)

  # Tick labels (the numbers on the axes)
  plt.xticks(eps_values, fontsize=base_fs)
  plt.yticks(fontsize=base_fs)

  # Legend and Grid
  plt.legend(
      title="k values", title_fontsize=base_fs, fontsize=base_fs - 2, loc="best"
  )
  plt.grid(True, linestyle="--", alpha=0.7)

  plt.tight_layout()  # Ensures labels don't get cut off
  plt.savefig(os.path.join(output_dir, "count_plot.png"))
  plt.close()


################################################################################
# Vote experiments
################################################################################
def vote_norm(d, v):
  """Returns the vote induced K-norm of the vector v.

  Args:
    d: Integer dimension.
    v: Numpy array.
  """
  # Sort and reverse
  v_desc = np.sort(v)[::-1]
  all_constraint_As = np.zeros((d + 1, d))
  all_constraint_Bs = np.zeros(d + 1)
  # handle cap constraints
  all_constraint_As[0] = -np.ones(d)
  all_constraint_As[1] = np.ones(d)
  all_constraint_Bs[0] = d * (d - 1) / 2
  all_constraint_Bs[1] = d * (d - 1) / 2
  for s in range(1, d):
    all_constraint_As[s + 1] = np.concatenate(
        [np.ones(s), np.zeros(d - s)]
    ) - s / d * np.ones(d)
    all_constraint_Bs[s + 1] = s * (d - s) / 2
  all_ax_over_b = np.array([
      np.dot(all_constraint_As[s], v_desc) / all_constraint_Bs[s]
      for s in range(d + 1)
  ])
  return np.max(all_ax_over_b)


def vote_norm_mean_norms(d, eps, num_samples):
  """Computes mean vote norm for the K-norm vote mechanism.

  Args:
    d: Integer dimension.
    eps: Float privacy parameter.
    num_samples: Integer number of samples.

  Returns:
    Float mean vote norm.
  """
  vote_mechanism_samples = []
  for _ in range(num_samples):
    vote_mechanism_samples.append(vote_mechanism(np.zeros(d), eps))
  mean_norm = np.mean(
      np.array([vote_norm(d, v) for v in vote_mechanism_samples])
  )
  return mean_norm


def vote_ripple_mean_norms(d, eps, num_samples):
  """Computes mean vote norm for the Ripple vote mechanism.

  Args:
    d: Integer dimension.
    eps: Float privacy parameter.
    num_samples: Integer number of samples.

  Returns:
    Float mean vote norm.
  """
  ripple_vote_samples = ripple_vote.sample_ripple_vote_point(
      d, eps, num_samples
  )
  mean_norm = np.mean(
      np.array([vote_norm(d, v) for v in ripple_vote_samples])
  )
  return mean_norm


def vote_plot_norm_vs_eps_multiple_d(
    d_values, num_samples, eps_values, output_dir="."
):
  """Plots Ripple mechanism error reduction against K-norm mechanism for vote.

  Returns a plot with a line plotting reduction vs epsilon for each k in
  k_values.

  Args:
    d_values: Sequence of integer dimensions to check.
    num_samples: Integer number of samples.
    eps_values: Sequence of float epsilons to check.
    output_dir: String output directory path.
  """
  # Set a base font size variable for easy adjustments
  base_fs = 20

  plt.figure(figsize=(12, 8))  # Increased figure size for better spacing

  # Iterate through each d in the list
  for d in d_values:
    ripple_norms = [
        vote_ripple_mean_norms(d, eps, num_samples) for eps in eps_values
    ]

    k_norms = [
        vote_norm_mean_norms(d, eps, num_samples) for eps in eps_values
    ]

    error_reduction_percent = [
        (c - d_norm) / c for c, d_norm in zip(k_norms, ripple_norms)
    ]

    # Plot a separate line for each d with a label for the legend
    plt.plot(
        eps_values,
        error_reduction_percent,
        marker="o",
        linewidth=2.5,
        markersize=8,
        label=f"d = {d}",
    )

  # Formatting the plot
  plt.xlabel(r"$\epsilon$", fontsize=base_fs + 4)
  plt.ylabel(r"Average Vote Norm Error Reduction", fontsize=base_fs + 2)
  plt.title("Vote", fontsize=base_fs + 6, pad=20)

  # Tick labels (the numbers on the axes)
  plt.xticks(eps_values, fontsize=base_fs)
  plt.yticks(fontsize=base_fs)

  # Add a legend to distinguish between the d values
  plt.legend(
      title="Dimension (d)",
      title_fontsize=base_fs,
      fontsize=base_fs - 2,
      loc="best",
  )
  plt.grid(True, linestyle="--", alpha=0.7)
  plt.tight_layout()  # Ensures labels don't get cut off
  plt.savefig(os.path.join(output_dir, "vote_plot.png"))
  plt.close()
