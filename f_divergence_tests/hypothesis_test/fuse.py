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

Implements the 'fuse' method (Schrab et al., 2023; arXiv:2306.08777) for
two-sample hypothesis testing, enabling efficient hyperparameter exploration
without Bonferroni-like power loss, extended to regularized f-divergence kernel
tests (https://arxiv.org/abs/2409.14980):

  lambda_reg = multiplier * sqrt(n * (n - 1))
  T_fused = (1 / lambda_reg) * log(
      (1 / S_valid) * sum_{valid s} exp(lambda_reg * T_s / sigma_s)
  )
  p = mean_b 1[T_fused^(b) >= T_fused^(orig)]

Non-finite hyperparameter rows (where any column of `lambda_reg * T_s / sigma_s`
is non-finite, e.g. from an ill-conditioned Cholesky factorization or zero
kernel normalizer `sigma_s`) are masked symmetrically across all permutation
columns including the original column, preserving permutation exchangeability
under H0.

Also provides fused witness density-ratio scoring
(`FuseTest.compute_fused_witness_scores`) to estimate
`dP_numerator / dP_denominator` at query points using softmax weights
`w = softmax(lambda_reg * T_s^dir / sigma_s)` over valid hyperparameter rows.

Note:
  All computations run in `float32` (`jax_enable_x64` is assumed off). Do not
  mutate a `FuseTest` instance after construction, as `jax.jit` with
  `static_argnums=0` hashes `self` by object identity.
"""

from collections.abc import Iterator
import functools
from typing import Any, Union
from absl import logging
import jax
import jax.numpy as jnp
from f_divergence_tests.divergence import kernel as kernel_module
from f_divergence_tests.divergence import likelihood_ratio
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


def _validate_sample_pair(
    samples_a,
    samples_b,
    name_a,
    name_b,
):
  """Validates a pair of equal-sized 2D sample matrices with at least 2 rows."""
  if samples_a.ndim != 2 or samples_b.ndim != 2:
    raise ValueError(f"{name_a} and {name_b} must be 2D arrays.")
  if samples_a.shape[0] != samples_b.shape[0]:
    raise ValueError(
        f"{name_a} and {name_b} must have the same number of samples."
    )
  if samples_a.shape[0] < 2:
    raise ValueError(f"{name_a} and {name_b} must have at least 2 samples.")
  if samples_a.shape[1] != samples_b.shape[1]:
    raise ValueError(
        f"{name_a} and {name_b} must have the same feature dimension."
    )


def _kernel_normalizer(kernel_matrix, n_samples):
  """MMD-FUSE kernel scale `||K||_F / sqrt(n * (n - 1))` (arXiv:2306.08777)."""
  return jnp.sqrt(jnp.sum(kernel_matrix**2) / (n_samples * (n_samples - 1)))


def _ratio_on_queries(
    estimator,
    numerator_samples,
    denominator_samples,
    query_samples,
    kernel_matrix,
    kernel,
    bandwidth,
    min_memory_kernel,
    query_batch_size,
):
  """Computes directional divergence and query likelihood ratios for one row.

  This helper is the single place that maps the caller's `numerator` /
  `denominator` roles onto `LikelihoodRatioEstimator`'s `y` (numerator) and `x`
  (denominator) roles:
    - `x` role: `denominator_samples` (`k_xx = k_dd`)
    - `y` role: `numerator_samples` (`k_yy = k_nn`, `k_xy = k_dn`)

  Args:
    estimator: A `LikelihoodRatioEstimator` (e.g. `KernelFDivergence` or
      `DrMMDSymmetric`) providing `.lmbda`, `.directional(...)`, and
      `.likelihood_ratio_yx_on_z(...)`.
    numerator_samples: Samples from the numerator distribution, shape `[n, d]`.
    denominator_samples: Samples from the denominator distribution, shape `[n,
      d]`.
    query_samples: Query samples at which to evaluate `dP_numerator /
      dP_denominator`, shape `[q, d]`.
    kernel_matrix: Kernel matrix over `jnp.concatenate((numerator_samples,
      denominator_samples))`, shape `[2 * n, 2 * n]`.
    kernel: `Kernel` enum specifying the kernel function and norm.
    bandwidth: Scalar kernel bandwidth (`float32` array).
    min_memory_kernel: Whether to compute pairwise distances sequentially.
    query_batch_size: Positive batch size for evaluating queries via
      `jax.lax.map`.

  Returns:
    A `(stat, query_ratios)` tuple where `stat` is the scalar directional
    divergence `D(P_numerator || P_denominator)` and `query_ratios` has shape
    `[q]`.
  """
  n_samples = numerator_samples.shape[0]
  n_queries = query_samples.shape[0]

  k_nn = kernel_matrix[:n_samples, :n_samples]
  k_dd = kernel_matrix[n_samples:, n_samples:]
  k_dn = kernel_matrix[n_samples:, :n_samples]

  cho = likelihood_ratio.regularized_cho_factor(k_dd, estimator.lmbda)
  stat = estimator.directional(k_xx=k_dd, k_yy=k_nn, k_xy=k_dn)

  effective_batch_size = min(query_batch_size, n_queries)
  num_batches = (n_queries + effective_batch_size - 1) // effective_batch_size
  pad_len = num_batches * effective_batch_size - n_queries
  if pad_len > 0:
    padded_queries = jnp.pad(query_samples, ((0, pad_len), (0, 0)), mode="edge")
  else:
    padded_queries = query_samples
  batched_queries = padded_queries.reshape(
      num_batches, effective_batch_size, query_samples.shape[1]
  )

  def _score_query_batch(q_batch):
    batch_len = q_batch.shape[0]
    dist_qd = (
        kernel_module.get_distances(
            q_batch,
            denominator_samples,
            norm=kernel.norm.value,
            min_memory=min_memory_kernel,
        )
        .reshape(batch_len, n_samples)
        .astype(jnp.float32)
    )
    dist_qn = (
        kernel_module.get_distances(
            q_batch,
            numerator_samples,
            norm=kernel.norm.value,
            min_memory=min_memory_kernel,
        )
        .reshape(batch_len, n_samples)
        .astype(jnp.float32)
    )
    k_qd = kernel_module.get_kernel_matrix(
        dist_qd, kernel=kernel, bandwidth=bandwidth
    ).astype(jnp.float32)
    k_qn = kernel_module.get_kernel_matrix(
        dist_qn, kernel=kernel, bandwidth=bandwidth
    ).astype(jnp.float32)
    return estimator.likelihood_ratio_yx_on_z(
        k_zx=k_qd,
        k_zy=k_qn,
        k_xx=k_dd,
        cho_factor_k_xx=cho,
        k_xy=k_dn,
    )

  batched_ratios = jax.lax.map(_score_query_batch, batched_queries)
  query_ratios = batched_ratios.reshape(-1)[:n_queries]
  return stat, query_ratios


class PermutationTest:
  """Exact hypothesis test that uses permutations.

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
    """Initializes the PermutationTest instance.

    Args:
      divergence_class: Class of the divergence to use for the two-sample test.
      kernel: Kernel to use.
      significance: Significance level of the test in `(0, 1)`.
      bandwidth: Bandwidth to use for the kernel.
      num_permutations: Number of permutations to use.
      min_memory_kernel: Whether to use min_memory for the kernel computation.
      min_memory_permutations: Whether to use min_memory for the permutations.
      batch_size_permutations: Batch size for the permutations in case of
        min_memory_permutations.
      divergence_params: Optional parameters of the divergence to use.
    """
    self.num_permutations = num_permutations
    self.divergence_class = divergence_class
    self.significance = significance
    self.divergence_params = (
        dict(divergence_params) if divergence_params else {}
    )
    self.min_memory_kernel = min_memory_kernel
    self.min_memory_permutations = min_memory_permutations
    self.batch_size = batch_size_permutations
    self.kernel = kernel
    self.bandwidth = bandwidth

  def get_permutations(
      self, key, n_samples
  ):
    """Generates permutation sign assignments and indices for two equal splits.

    Args:
      key: JAX PRNG key.
      n_samples: Number of samples per distribution (`m = n = n_samples`).

    Returns:
      A `(permuted_assignments, sorted_indices)` tuple of arrays of shape
      `[num_permutations + 1, 2 * n_samples]`.
    """
    return hypothesis_test_utils.get_permutations(
        key, num_permutations=self.num_permutations, m=n_samples, n=n_samples
    )


class FuseTest(PermutationTest):
  """Fuses hypothesis tests over multiple kernels and hyperparameter settings.

  Implements the 'fuse' method (arXiv:2306.08777) for two-sample hypothesis
  testing, enabling efficient hyperparameter exploration without Bonferroni-like
  power loss. Extends the original MMD-based method to support diverse
  distribution metrics from regularized f-divergences (arXiv:2409.14980).

  Do not mutate a `FuseTest` instance after construction: `run` is compiled with
  `jax.jit(..., static_argnums=0)`, which hashes `self` by object identity.

  Attributes:
    hyperparam_name: Name of the swept divergence hyperparameter (or `"unused"`
      when no sweep dictionary is provided).
    divergence_sweep_hyperparams: Mapping from `hyperparam_name` to the list of
      swept values.
    select_bandwidths: Whether bandwidths are computed adaptively from pairwise
      distances per norm.
    number_bandwidths: Number of bandwidths evaluated per `(kernel, param)`.
    kernels: Mapping from `Norm` to the list of `Kernel` enums using that norm.
    number_kernels: Total number of kernels across all norms.
    number_statistics: Total number of hyperparameter rows `S`.
    lambda_fuse_multiplier: Multiplier for the FUSE temperature parameter.
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
      kernels: Kernel or sequence of kernels to fuse over.
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

    Raises:
      ValueError: If more than one hyperparameter is specified in
        `divergence_sweep_hyperparams`, or if both or neither of `bandwidths`
        and `number_bandwidths` are specified.
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
      if len(divergence_sweep_hyperparams) > 1:
        raise ValueError(
            "Only sweeping over one divergence-specific hyperparameter is"
            " supported."
        )
      self.hyperparam_name = list(divergence_sweep_hyperparams.keys())[0]
      num_hyperparams = len(divergence_sweep_hyperparams[self.hyperparam_name])
      self.divergence_sweep_hyperparams = {
          self.hyperparam_name: list(
              divergence_sweep_hyperparams[self.hyperparam_name]
          )
      }
    else:
      self.hyperparam_name = "unused"
      num_hyperparams = 1
      self.divergence_sweep_hyperparams = {self.hyperparam_name: [jnp.inf]}

    if isinstance(kernels, kernel_module.Kernel):
      kernels = [kernels]
    else:
      kernels = list(kernels)

    if bool(bandwidths) == bool(number_bandwidths):
      raise ValueError("Only specify number_bandwiths or bandwidths.")
    elif bandwidths:
      self._fixed_bandwidths = tuple(float(b) for b in bandwidths)
      self.select_bandwidths = False
      self.number_bandwidths = len(self._fixed_bandwidths)
    else:
      logging.info("Selecting bandwidths")
      self._fixed_bandwidths = None
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
        self.number_bandwidths * self.number_kernels * num_hyperparams  # pyrefly: ignore[unsupported-operation]
    )
    self.lambda_fuse_multiplier = lambda_fuse_multiplier

  def _iterate_hyperparams(self, pooled_samples):
    """Yields `(norm, kernel, param, bandwidth, kernel_matrix, estimator)` in a fixed order.

    Bandwidths are computed into a local variable per norm without mutating
    `self`, avoiding JAX tracer leaks under `jax.jit`.

    Args:
      pooled_samples: Pooled samples of shape `[2 * n_samples, d]`, dtype
        `float32`.

    Yields:
      Tuples `(norm, kernel, param, bandwidth, kernel_matrix, estimator)` for
      each hyperparameter setting in a deterministic order.
    """
    estimators = []
    for param in self.divergence_sweep_hyperparams[self.hyperparam_name]:
      params = dict(self.divergence_params)
      if self.hyperparam_name != "unused":
        params[self.hyperparam_name] = param
      estimators.append((param, self.divergence_class(**params)))

    for norm_l, kernels_l in self.kernels.items():
      if not kernels_l:
        continue
      pairwise_matrix = kernel_module.get_distances(
          pooled_samples,
          pooled_samples,
          norm=norm_l.value,
          min_memory=self.min_memory_kernel,
      ).astype(jnp.float32)
      if self.select_bandwidths:
        distances = pairwise_matrix[jnp.triu_indices(pairwise_matrix.shape[0])]
        bandwidths = kernel_module.compute_bandwidths(
            distances, self.number_bandwidths
        ).astype(jnp.float32)
      else:
        bandwidths = jnp.asarray(self._fixed_bandwidths, dtype=jnp.float32)

      for kernel in kernels_l:
        for bw in bandwidths:
          kernel_matrix = kernel_module.get_kernel_matrix(
              pairwise_matrix,
              kernel=kernel,
              bandwidth=bw,
          ).astype(jnp.float32)
          for param, divergence_estimator in estimators:
            yield (
                norm_l,
                kernel,
                param,
                bw,
                kernel_matrix,
                divergence_estimator,
            )

  @functools.partial(jax.jit, static_argnums=0)
  def run(
      self,
      samples_x,
      samples_y,
      key_seed,
  ):
    """Runs the fused two-sample permutation test.

    Invalid hyperparameter rows (where any column of the standardized statistic
    `lambda_reg * stat / kernel_norm` is non-finite) are masked out prior to the
    `logsumexp` aggregation. Because the validity mask is a symmetric function
    of all permutation columns (including the original unpermuted column), the
    masked fused statistic applies the same fixed function to every column and
    preserves permutation exchangeability under the null hypothesis H0. If no
    hyperparameter rows are valid, returns `p_val = 1.0` and
    `result_test = False`.

    Args:
      samples_x: Array of shape `(n_samples, d)` from the first distribution,
        with `n_samples >= 2`.
      samples_y: Array of shape `(n_samples, d)` from the second distribution,
        with `n_samples >= 2`.
      key_seed: Integer seed for the JAX random number generator.

    Returns:
      A `TestResult` containing:
        - `all_statistics`: Fused statistics across permutations, shape
          `(num_permutations + 1,)`, with the original sample order at `[-1]`.
        - `p_val`: Permutation p-value in `[1 / (num_permutations + 1), 1.0]`.
        - `result_test`: Boolean array indicating rejection at
          `self.significance`.
        - `additional_info`: Dict with `"statistics_matrix"` (`[S, B + 1]`),
          `"bw"` (dict mapping norm name to its local bandwidth array),
          `"num_valid_hyperparams"`, and `"num_hyperparams"`.

    Raises:
      ValueError: If `samples_x` and `samples_y` are not 2D arrays with the same
        shape and at least 2 samples.

    Example:
      ```python
      test = FuseTest(
          divergence_class=likelihood_ratio.SymmetricHockeyStick,
          kernels=[kernel_module.Kernel.GAUSSIAN],
          significance=0.05,
          number_bandwidths=3,
          num_permutations=50,
          min_memory_kernel=False,
          min_memory_permutations=True,
          batch_size_permutations=25,
          divergence_sweep_hyperparams={"lmbda": [0.1, 1.0]},
      )
      result = test.run(samples_x, samples_y, key_seed=0)
      ```
    """
    _validate_sample_pair(samples_x, samples_y, "samples_x", "samples_y")

    n_samples = samples_x.shape[0]
    all_samples = jnp.concatenate((samples_x, samples_y)).astype(jnp.float32)

    key = jax.random.PRNGKey(key_seed)
    key, subkey = jax.random.split(key)

    _, all_idx = self.get_permutations(subkey, n_samples)

    all_statistics = jnp.zeros(
        (self.number_statistics, self.num_permutations + 1),
        dtype=jnp.float32,
    )
    kernel_norms = jnp.zeros(self.number_statistics, dtype=jnp.float32)
    lambda_reg = self.lambda_fuse_multiplier * jnp.sqrt(
        float(n_samples * (n_samples - 1.0))
    )

    num_hyperparams = len(
        self.divergence_sweep_hyperparams[self.hyperparam_name]
    )
    bandwidths_by_norm: dict[str, list[jax.Array]] = {}
    for current_hyperparams_idx, (
        norm_l,
        _,
        _,
        bw,
        kernel_matrix,
        divergence_estimator,
    ) in enumerate(self._iterate_hyperparams(all_samples)):
      norm_key = norm_l.value
      if norm_key not in bandwidths_by_norm:
        bandwidths_by_norm[norm_key] = []
      if (
          current_hyperparams_idx % num_hyperparams == 0
          and len(bandwidths_by_norm[norm_key]) < self.number_bandwidths
      ):
        bandwidths_by_norm[norm_key].append(bw)

      kernel_norm = _kernel_normalizer(kernel_matrix, n_samples)
      kernel_norms = kernel_norms.at[current_hyperparams_idx].set(kernel_norm)
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
        shuffled_estimator_fn = functools.partial(
            _make_shuffled_estimator(divergence_estimator, n_samples),
            kernel_matrix=kernel_matrix,
        )
        if self.min_memory_permutations:
          permuted_statistics = jax.lax.map(
              shuffled_estimator_fn, all_idx, batch_size=self.batch_size
          )
        else:
          permuted_statistics = jax.vmap(shuffled_estimator_fn)(all_idx)

      all_statistics = all_statistics.at[current_hyperparams_idx].set(
          permuted_statistics
      )

    scaled = lambda_reg * all_statistics / kernel_norms[:, None]
    valid = jnp.all(jnp.isfinite(scaled), axis=1, keepdims=True)
    num_valid = jnp.sum(valid)
    safe = jnp.where(valid, scaled, -jnp.inf)
    fused = (
        jax.scipy.special.logsumexp(
            safe,
            axis=0,
            b=1.0 / jnp.maximum(num_valid, 1),
        )
        / lambda_reg
    ).flatten()

    p_val = jnp.where(num_valid > 0, jnp.mean(fused >= fused[-1]), 1.0)
    output_norm = (num_valid > 0) & (p_val <= self.significance)
    bw_info = {
        norm_key: jnp.stack(bws) for norm_key, bws in bandwidths_by_norm.items()
    }

    return hypothesis_test_utils.TestResult(
        all_statistics=fused,
        p_val=p_val,
        result_test=output_norm,
        additional_info={
            "statistics_matrix": all_statistics,
            "bw": bw_info,
            "num_valid_hyperparams": num_valid,
            "num_hyperparams": jnp.asarray(
                self.number_statistics, dtype=num_valid.dtype
            ),
        },
    )

  @functools.partial(jax.jit, static_argnums=(0, 4))
  def compute_fused_witness_scores(
      self,
      numerator_samples,
      denominator_samples,
      query_samples,
      query_batch_size = 4096,
  ):
    """Returns the fused estimate of dP_numerator/dP_denominator at query_samples.

    Weights are `softmax(lambda_reg * stat / kernel_norm)` over valid
    hyperparameter rows, where `stat` is the directional divergence
    `D(P_numerator || P_denominator)` computed on the original (unpermuted)
    samples and `kernel_norm` is the MMD-FUSE kernel normalizer
    `||K||_F / sqrt(n * (n - 1))`, matching `run`. No permutations are needed,
    so this method takes no PRNG key.

    A hyperparameter row is masked (its logit set to `-jnp.inf`, receiving zero
    weight) if its scaled statistic `lambda_reg * stat / kernel_norm` is
    non-finite or any of its `[q]` query likelihood ratios is non-finite. When
    no rows are valid (`num_valid == 0`), returns all-`jnp.nan` scores of shape
    `[q]` and zero weights of shape `[S]`.

    Args:
      numerator_samples: Fit samples from the numerator distribution (e.g.
        `train_fit`), shape `[n, d]` with `n >= 2`.
      denominator_samples: Fit samples from the denominator distribution (e.g.
        `holdout_fit`), shape `[n, d]` with `n >= 2`.
      query_samples: Query samples at which to evaluate the fused witness ratio
        `dP_numerator / dP_denominator` (higher means more numerator-like),
        shape `[q, d]` with `q >= 1`.
      query_batch_size: Positive batch size for evaluating `query_samples` via
        `jax.lax.map`.

    Returns:
      A `(scores, info)` tuple where:
        - `scores`: Fused likelihood-ratio scores at `query_samples`, shape
          `[q]`, dtype `float32` (all `jnp.nan` if `num_valid == 0`).
        - `info`: Dict with `"weights"` of shape `[S]` (`float32` softmax
          weights summing to 1 when `num_valid > 0`, or zeros when
          `num_valid == 0`) and scalar `"num_valid"` (count of valid
          hyperparameter rows).

    Raises:
      ValueError: If inputs are not 2D arrays with matching feature dimension,
        `numerator_samples` and `denominator_samples` have different sample
        counts or fewer than 2 samples, `query_samples` is empty, or
        `query_batch_size <= 0`.

    Example:
      ```python
      test = FuseTest(
          divergence_class=likelihood_ratio.SymmetricHockeyStick,
          kernels=[kernel_module.Kernel.GAUSSIAN],
          significance=0.05,
          number_bandwidths=3,
          num_permutations=50,
          min_memory_kernel=False,
          min_memory_permutations=True,
          divergence_sweep_hyperparams={"lmbda": [0.1, 1.0]},
      )
      scores, info = test.compute_fused_witness_scores(
          numerator_samples=train_fit,
          denominator_samples=holdout_fit,
          query_samples=queries,
      )
      ```
    """
    _validate_sample_pair(
        numerator_samples,
        denominator_samples,
        "numerator_samples",
        "denominator_samples",
    )
    if query_samples.ndim != 2:
      raise ValueError("query_samples must be a 2D array.")
    if numerator_samples.shape[1] != query_samples.shape[1]:
      raise ValueError(
          "numerator_samples, denominator_samples, and query_samples must have"
          " the same feature dimension."
      )
    if query_samples.shape[0] == 0:
      raise ValueError("query_samples must have at least one sample.")
    if query_batch_size <= 0:
      raise ValueError("query_batch_size must be positive.")

    n_samples = numerator_samples.shape[0]
    n_queries = query_samples.shape[0]
    numerator_f32 = numerator_samples.astype(jnp.float32)
    denominator_f32 = denominator_samples.astype(jnp.float32)
    query_f32 = query_samples.astype(jnp.float32)
    pooled_samples = jnp.concatenate((numerator_f32, denominator_f32))

    lambda_reg = self.lambda_fuse_multiplier * jnp.sqrt(
        float(n_samples * (n_samples - 1.0))
    )

    logits = jnp.zeros(self.number_statistics, dtype=jnp.float32)
    all_ratios = jnp.zeros(
        (self.number_statistics, n_queries), dtype=jnp.float32
    )

    for current_hyperparams_idx, (
        _,
        kernel,
        _,
        bw,
        kernel_matrix,
        divergence_estimator,
    ) in enumerate(self._iterate_hyperparams(pooled_samples)):
      kernel_norm = _kernel_normalizer(kernel_matrix, n_samples)
      stat, query_ratios = _ratio_on_queries(
          estimator=divergence_estimator,
          numerator_samples=numerator_f32,
          denominator_samples=denominator_f32,
          query_samples=query_f32,
          kernel_matrix=kernel_matrix,
          kernel=kernel,
          bandwidth=bw,
          min_memory_kernel=self.min_memory_kernel,
          query_batch_size=query_batch_size,
      )
      logits = logits.at[current_hyperparams_idx].set(
          lambda_reg * stat / kernel_norm
      )
      all_ratios = all_ratios.at[current_hyperparams_idx].set(query_ratios)

    valid = jnp.isfinite(logits) & jnp.all(jnp.isfinite(all_ratios), axis=1)
    num_valid = jnp.sum(valid)
    safe_logits = jnp.where(valid, logits, -jnp.inf)
    softmax_weights = jax.nn.softmax(safe_logits) * valid.astype(jnp.float32)
    weights = jnp.where(
        num_valid > 0,
        softmax_weights,
        jnp.zeros_like(logits),
    )
    safe_ratios = jnp.where(valid[:, None], all_ratios, 0.0)
    scores = jnp.where(
        num_valid > 0,
        jnp.matmul(weights, safe_ratios),
        jnp.full((n_queries,), jnp.nan, dtype=jnp.float32),
    )
    return scores, {"weights": weights, "num_valid": num_valid}
