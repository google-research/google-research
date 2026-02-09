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

"""Fuse method for two-sample hypothesis tests.

Implements the 'fuse' method (arXiv:2306.08777) for two-sample hypothesis
testing, enabling efficient hyperparameter exploration without Bonferroni-like
power loss. Extends the original MMD-based method to support diverse
divergences.
"""

import functools
from typing import Any, Union
from absl import logging
import jax
import jax.numpy as jnp
from f_divergence_tests.divergence import kernel as kernel_module
from f_divergence_tests.hypothesis_test import hypothesis_test_utils


def _make_shuffled_estimator(
    divergence_estimator,
    n_samples,
):
  """Makes a shuffled estimator function."""

  def shuffled_estimator(idx, kernel_matrix):
    idx_x, idx_y = idx[:n_samples], idx[n_samples:]
    return divergence_estimator(
        kernel_matrix[idx_x[:, None], idx_x],
        kernel_matrix[idx_y[:, None], idx_y],
        kernel_matrix[idx_x[:, None], idx_y],
    )

  return shuffled_estimator


class PermutationTest:
  """Exact hypothesis test that uses a permutations.

  Attributes:
    num_permutations: Number of permutations to use.
    divergence_class: Class of the divergence to use for the two-sample test.
    significance: Significance level of the test.
    divergence_params: Parameters of the divergence to use.
    min_memory_kernel: Whether to use min_memory for the kernel computation.
    min_memory_permutations: Whether to use min_memory for the permutations.
    batch_size: Batch size for the permutations in case of
      min_memory_permutations.
    kernel: Kernel to use.
    bandwidth: Bandwidth to use.
  """

  def __init__(
      self,
      divergence_class,
      kernel,
      significance,
      bandwidth,
      num_permutations,
      min_memory_kernel,
      min_memory_permutations,
      batch_size_permutations = 500,
      divergence_params = None,
  ):
    divergence_params = divergence_params or {}
    self.num_permutations = num_permutations
    self.divergence_class = divergence_class
    self.significance = significance
    self.divergence_params = divergence_params or {}
    self.divergence_params = divergence_params
    self.min_memory_kernel = min_memory_kernel
    self.min_memory_permutations = min_memory_permutations
    self.batch_size = batch_size_permutations
    self.kernel = kernel
    self.bandwidth = bandwidth

  def get_permutations(self, key, n_samples):
    return hypothesis_test_utils.get_permutations(
        key, num_permutations=self.num_permutations, m=n_samples, n=n_samples
    )


class FuseTest(PermutationTest):
  """Fuses hypothesis tests over multiple kernels and hyperparameter settings.

  This class implements the 'fuse' method for two-sample hypothesis testing,
  enabling efficient hyperparameter exploration without Bonferroni-like power
  loss. It extends the original MMD-based method to support diverse
  distribution metrics from divergences.
  """

  def __init__(
      self,
      divergence_class,
      kernels,
      significance,
      number_bandwidths,
      num_permutations,
      min_memory_kernel,
      min_memory_permutations,
      batch_size_permutations = 500,
      divergence_params = None,
      divergence_sweep_hyperparams = None,
      bandwidths = None,
      lambda_fuse_multiplier = 1.0,
  ):
    """Initializes the FuseTest class.

    Args:
      divergence_class: Class of the divergence to use for the two-sample test.
      kernels: Kernels to use for the two-sample test.
      significance: Significance level of the test.
      number_bandwidths: Number of bandwidths to fuse for the two-sample test.
      num_permutations: Number of permutations to use.
      min_memory_kernel: Whether to use min_memory for the kernel computation.
      min_memory_permutations: Whether to use min_memory for the permutations.
      batch_size_permutations: Batch size for the permutations in case of
        min_memory_permutations.
      divergence_params: Parameters of the divergence to use.
      divergence_sweep_hyperparams: Hyperparameters to sweep over for the
        divergence. Only one hyperparameter is supported per divergence.
      bandwidths: Bandwidths to use for the two-sample test instead of selecting
        them.
      lambda_fuse_multiplier: Multiplier for the lambda fuse parameter. See
        https://arxiv.org/abs/2306.08777 for more details.
    """
    super().__init__(
        divergence_class=divergence_class,
        kernel=kernel_module.Kernel.EMPTY,  # unused for fuse.
        significance=significance,
        bandwidth=jnp.inf,  # unused for fuse.
        num_permutations=num_permutations,
        min_memory_kernel=min_memory_kernel,
        min_memory_permutations=min_memory_permutations,
        batch_size_permutations=batch_size_permutations,
        divergence_params=divergence_params,
    )
    if divergence_sweep_hyperparams:
      self.hyperparam_name = list(divergence_sweep_hyperparams.keys())[0]
      num_hyperparams = len(divergence_sweep_hyperparams[self.hyperparam_name])
    else:
      self.hyperparam_name = "unused"
      num_hyperparams = 1
      divergence_sweep_hyperparams = {self.hyperparam_name: [jnp.inf]}

    self.divergence_sweep_hyperparams = divergence_sweep_hyperparams

    if len(divergence_sweep_hyperparams) > 1:
      raise ValueError(
          "Only sweeping over one divergence-specific hyperparameter is"
          " supported."
      )
    if not isinstance(kernels, list):
      kernels = [kernels]

    if bandwidths and number_bandwidths:
      raise ValueError("Only specify number_bandwiths or bandwidths.")
    elif bandwidths:
      self.bandwidths = bandwidths
      self.select_bandwidths = False
      self.number_bandwidths = len(bandwidths)
    else:
      logging.info("Selecting bandwidths")
      self.select_bandwidths = True
      self.number_bandwidths = number_bandwidths
    self.kernels = {
        kernel_module.Norm.L1: [
            k for k in kernels if k.norm == kernel_module.Norm.L1
        ],
        kernel_module.Norm.L2: [
            k for k in kernels if k.norm == kernel_module.Norm.L2
        ],
    }
    self.number_kernels = sum([len(ks) for ks in self.kernels.values()])

    self.number_statistics = (
        self.number_bandwidths * self.number_kernels * num_hyperparams
    )
    self.lambda_fuse_multiplier = lambda_fuse_multiplier

  @functools.partial(jax.jit, static_argnums=0)
  def run(
      self,
      samples_x,
      samples_y,
      key_seed,
  ):
    """Run the test.

    Args:
      samples_x: Array of shape (n_samples, d).
      samples_y: Array of shape (n_samples, d).
      key_seed: Seed for JAX random number generator.

    Returns:
      A TestResult object containing the results of the test.
    """
    if samples_x.shape[0] != samples_y.shape[0]:
      raise ValueError(
          "samples_x and samples_y must have the same number of samples."
      )

    n_samples = samples_x.shape[0]
    all_samples = jnp.concatenate((samples_x, samples_y)).astype(jnp.float32)

    key = jax.random.PRNGKey(key_seed)  # You can use any seed value here
    key, subkey = jax.random.split(key)

    _, all_idx = self.get_permutations(subkey, n_samples)

    all_statistics = jnp.zeros(
        (self.number_statistics, self.num_permutations + 1)
    ).astype(jnp.float32)

    current_hyperparams_idx = 0

    std_devs = jnp.zeros(self.number_statistics).astype(jnp.float32)
    lambda_reg = self.lambda_fuse_multiplier * jnp.sqrt(
        float(n_samples * (n_samples - 1.0))
    )
    for norm_l, kernels_l in self.kernels.items():
      pairwise_matrix = kernel_module.get_distances(
          all_samples,
          all_samples,
          norm=norm_l.value,
          min_memory=self.min_memory_kernel,
      ).astype(jnp.float32)
      if self.select_bandwidths:
        distances = pairwise_matrix[jnp.triu_indices(pairwise_matrix.shape[0])]
        self.bandwidths = kernel_module.compute_bandwidths(
            distances, self.number_bandwidths
        ).astype(jnp.float32)

      for kernel in kernels_l:
        for param in self.divergence_sweep_hyperparams[self.hyperparam_name]:
          if self.hyperparam_name != "unused":
            self.divergence_params[self.hyperparam_name] = param

          divergence_estimator = self.divergence_class(**self.divergence_params)

          for bw in self.bandwidths:
            kernel_matrix = kernel_module.get_kernel_matrix(
                pairwise_matrix,
                kernel=kernel,
                bandwidth=bw,
            ).astype(jnp.float32)

            scaled_std = jnp.sqrt(
                jnp.sum(kernel_matrix**2) / (n_samples * (n_samples - 1))
            )
            std_devs = std_devs.at[current_hyperparams_idx].set(scaled_std)
            if divergence_estimator.has_vectorized_kernel:
              kernel_matrix = kernel_matrix.at[
                  jnp.diag_indices(kernel_matrix.shape[0])
              ].set(0)
              key, subkey = jax.random.split(key)
              permuted_statistics = divergence_estimator(
                  kernel_matrix,
                  self.num_permutations,
                  subkey,
              )
            else:
              shuffled_estimator_fn = _make_shuffled_estimator(
                  divergence_estimator, n_samples
              )

              if self.min_memory_permutations:

                def mapped_fn(idx):
                  return shuffled_estimator_fn(idx, kernel_matrix)  # pylint: disable=cell-var-from-loop

                permuted_statistics = jax.lax.map(
                    mapped_fn, all_idx, batch_size=self.batch_size
                )
              else:
                vestimate = jax.vmap(shuffled_estimator_fn, in_axes=0)
                permuted_statistics = vestimate(all_idx, kernel_matrix)

            all_statistics = all_statistics.at[current_hyperparams_idx].set(
                permuted_statistics
            )
            current_hyperparams_idx += 1

    all_statistics_norm = (
        jax.scipy.special.logsumexp(
            lambda_reg * all_statistics / std_devs[:, None],
            axis=0,
            b=1 / self.number_statistics,
        )
        / lambda_reg
    )

    all_statistics_norm = all_statistics_norm.flatten()

    original_statistic_norm = all_statistics_norm[-1]
    p_val_norm = jnp.mean(all_statistics_norm >= original_statistic_norm)
    output_norm = p_val_norm <= self.significance

    return hypothesis_test_utils.TestResult(
        all_statistics=all_statistics_norm,
        p_val=p_val_norm,
        result_test=output_norm,
        additional_info={
            "statistics_matrix": all_statistics,
            "bw": self.bandwidths,
        },
    )
