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

"""Differentially Private Alternating Minimization."""

import abc
import collections
import copy
import dataclasses
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import dp_accounting
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from dp_alternating_minimization import dataset as dataset_lib

_DEFAULT_RDP_ORDERS = list(np.linspace(1.1, 10.9, 99)) + list(range(11, 64))
# Type aliases
_Bucket = np.ndarray  # Represents a bucket of ids, used in sliced metrics.
InputMatrix = dataset_lib.InputMatrix
InputBatch = Dict[str, tf.Tensor]


class Sanitizer:
  """Provides a set of functions for pre-processing the data.

  This class provides utilities for pre-processing of the data, including:
    1) centers ratings, 2) estimate counts, 3) restricts to
    head items and reindexes items by frequency, 4) adaptively samples items,
    5) re-estimates counts on sampled data.
  """

  def __init__(
      self,
      budget: Optional[float] = None,
      method: str = "adaptive_weights",
      item_frac: float = 1,
      center: bool = False,
      center_users: bool = False,
      count_stddev: float = 0,
      count_sample: Optional[int] = None,
      weight_exponent: float = 0,
      exact_split: bool = False,
      random_seed: Optional[int] = None):
    """Initializes a Sanitizer.

    Args:
      budget: the per-user budget.
      method: one of
        - "uniform": k items are sampled uniformly per user (k=budget).
        - "tail": for each user, sample the k least frequent items (k=budget).
        - "adaptive_weights": each user contributes to all items (no sampling).
          For a given item i, the user contributes with a weight cᵢ, such that
          Σᵢcᵢ² = budget. cᵢ is chosen proportional to countᵢ^weight_exponent,
          where countᵢ is the item count.
      item_frac: only train on top items, determined by noisy counts.
      center: whether to center the data.
      center_users: whether to center the data of each user.
      count_stddev: the standard deviation of the noise used in
        `self._noisy_count`.
      count_sample: the sample size (per user) used in `self._noisy_count`.
        If None, defaults to `budget`. If set to 0, this disables noisy counts.
      weight_exponent: the exponent used in `adaptive_weights`.
      exact_split: the partitioning into frequent and rare is done exactly.
      random_seed: the random seed to use.
    """
    if random_seed:
      np.random.seed(random_seed)
    if method == "adaptive_weights" and (
        weight_exponent is None or weight_exponent > 0):
      raise ValueError(
          f"weight_exponent must be <= 0, but got {weight_exponent}")
    self.budget = budget
    self.method = method
    self.item_frac = item_frac
    self.center = center
    self.center_users = center_users
    self.count_stddev = count_stddev
    if count_sample is None:
      count_sample = budget
    self.count_sample = count_sample
    self.weight_exponent = weight_exponent
    self.exact_split = exact_split

  def preprocess_data(self,
                      dataset: "Dataset",
                      value_key: str) -> Tuple[InputMatrix, np.ndarray]:
    """Preprocesses the data.

    All steps below are optional. By default, no filtering and no sampling is
    done, and counts are exact.
    1. Center the data, using a noisy estimate of the global average rating.
    2. Compute approximate counts by subsampling each user (uniformly) and
       adding Gaussian noise.
    3. Split the items into frequent and rare based on approximate counts, and
       restrict to frequent.
    4. Sample the training data, then recompute the noisy counts, based only on
       filtered and sampled data.

    Args:
      dataset: a Dataset object.
      value_key: DataFrame key of the values.
    Returns:
      sp_data: InputMatrix of sampled and transposed training data.
      noisy_counts: private estimate of item counts (useful for frequency-based
        regularization).
    """
    num_users = dataset.num_users
    num_items = dataset.num_items
    train_data = dataset.train.data

    # 1. Center the data
    if self.center:
      if "rating" not in train_data.columns:
        raise ValueError("cannot center when the dataset has no ratings.")
      ratings = train_data["rating"]
      noise_scale = 0
      if self.count_sample:
        ratings = train_data.sample(frac=1).groupby("uid").head(
            n=self.count_sample)["rating"]
        noise_scale = ratings.max()*np.sqrt(self.count_sample)*self.count_stddev
      mean = (
          (ratings.sum() + np.random.normal(scale=noise_scale)) /
          (len(ratings) + np.random.normal(scale=self.count_stddev)))
      dataset.remove_global_bias(mean)

    if self.center_users:
      if "rating" not in train_data.columns:
        raise ValueError("cannot center when the dataset has no ratings.")
      user_means = dataset.train.data.groupby("uid").agg({"rating": "mean"})
      dataset.remove_user_bias(user_means)

    # 2. Compute counts
    noisy_item_counts = self._noisy_count(train_data, num_items)
    dataset.train.data["count"] = noisy_item_counts[dataset.train.data["sid"]]

    # 3. Compute counts and restrict to frequent items.
    if self.item_frac == 1:
      num_frequent_items = num_items
      print("No head/tail splitting.")
    else:
      if not self.exact_split:
        counts_for_split = noisy_item_counts
      else:
        print("Using exact counts for head/tail splitting.")
        counts_for_split = self._noisy_count(train_data, num_items, exact=True)
      num_frequent_items = int(num_items * self.item_frac)
      print(f"Training on head items: {num_frequent_items}/{num_items}")
      sorted_ids = np.argsort(-counts_for_split)
      dataset.reindex_items(num_frequent_items, sorted_ids)

    # 4. Sample, then recompute counts.
    df = dataset.get_frequent(dataset.train.data)
    if not self.budget:
      print("No per-user sampling.")
      noisy_counts = self._noisy_count(df, dataset.num_frequent_items)
      sampled_st = dataset_lib.df_to_input_matrix(
          df, num_frequent_items, num_users, "sid", "uid", value_key)
    elif self.method == "uniform":
      print(f"Uniform sampling: {self.budget} items per user")
      sampled_df = df.sample(frac=1).groupby("uid").head(n=self.budget)  # pytype: disable=wrong-arg-types  # pandas-drop-duplicates-overloads
      noisy_counts = self._noisy_count(sampled_df, dataset.num_frequent_items)
      sampled_st = dataset_lib.df_to_input_matrix(
          sampled_df, num_frequent_items, num_users, "sid", "uid", value_key)
    elif self.method == "tail":
      print(f"Tail sampling: {self.budget} items per user")
      df = copy.deepcopy(df)
      sampled_df = df.sort_values("count").groupby("uid").head(n=self.budget)  # pytype: disable=wrong-arg-types  # pandas-drop-duplicates-overloads
      noisy_counts = self._noisy_count(sampled_df, dataset.num_frequent_items)
      sampled_st = dataset_lib.df_to_input_matrix(
          sampled_df, num_frequent_items, num_users, "sid", "uid", value_key)
    elif self.method == "adaptive_weights":
      print(f"Adaptive weights with a budget of {self.budget} per user")
      # Use all the data, but adapt the weights.
      noisy_counts = self._noisy_count(df, dataset.num_frequent_items)
      sampled_df = df
      sampled_st = dataset_lib.df_to_input_matrix(
          sampled_df, num_frequent_items, num_users, "sid", "uid", value_key)
      sampled_st.weights = _compute_adaptive_weights(
          sampled_st.to_sparse_tensor(), noisy_counts, self.weight_exponent,
          self.budget).numpy()
    else:
      raise ValueError(f"Unknown method {self.method}")
    return sampled_st, noisy_counts

  def _noisy_count(self, df, num_items, exact=False):
    """Computes noisy counts.

    Args:
      df: a DataFrame.
      num_items: the number of items.
      exact: whether to return exact or noisy counts.
    Returns:
      An array of item counts.
    """
    if not exact and self.count_sample:
      print(f"Computing approximate counts with count_stddev="
            f"{self.count_stddev} and count_sample={self.count_sample}")
      df = df.sample(frac=1).groupby("uid").head(n=self.count_sample)
    count_df = df.groupby("sid")["sid"].count()
    counts = np.zeros([num_items], dtype=np.int32)
    counts[count_df.index.values] = count_df.values
    if not exact and self.count_stddev:
      counts = counts + np.random.normal(
          scale=self.count_stddev, size=num_items)
    return np.maximum(1, counts)

  def get_dp_event_noisy_count(self) -> dp_accounting.DpEvent:
    """Gets DpEvent for noisy count."""
    if not (self.budget and self.count_sample):
      return dp_accounting.NonPrivateDpEvent()
    if self.count_stddev == 0:
      return dp_accounting.NoOpDpEvent()
    # Privacy for the noisy count.
    # The accounting is done as follows: whenever we compute a statistic with
    # L2 sensitivity k and add Gaussian noise of scale σ, the procedure is
    # (α, α/2/β²)-RDP with β = σ/k. This is represented by a `GaussianDpEvent`
    # with noise_multiplier β.
    events = []
    if self.center:
      # noisy mean rating:
      # k = √count_sample, σ = count_stddev * √count_sample, twice (numerator
      # and denominator)
      events.append(dp_accounting.SelfComposedDpEvent(
          dp_accounting.GaussianDpEvent(self.count_stddev), count=2
      ))
    # noisy counts (for head/tail partition): β = count_stddev/√count_sample
    # (k = √count_sample, σ = count_stddev)
    if self.item_frac < 1 and not self.exact_split:
      events.append(dp_accounting.GaussianDpEvent(
          self.count_stddev / np.sqrt(self.count_sample)))
    # noisy counts for tail/adaptive sampling, and regularization
    events.append(dp_accounting.GaussianDpEvent(
        self.count_stddev / np.sqrt(self.count_sample)))
    return dp_accounting.ComposedDpEvent(events)


class DPALSOptimizer:
  """Optimizer for DPALS."""

  def __init__(
      self,
      budget: float = 0,
      max_norm: Optional[float] = None,
      s0: float = 0,
      s1: float = 0,
      s2: float = 0):
    """Initializes a DPALSOptimizer.

    Args:
      budget: the budget (total squared L2 sensitivity) per user.
      max_norm: clips the user embeddings to max_norm.
      s0: the noise factor for global gramian.
      s1: the noise factor for local gramian.
      s2: the noise factor for rhs.
    """
    self.budget = budget
    self.max_norm = max_norm
    self.sigmas = [s0, s1, s2]

  def clip(self, embeddings: tf.Tensor) -> tf.Tensor:
    if not self.max_norm:
      return embeddings
    return tf.clip_by_norm(embeddings, self.max_norm, axes=[1])

  def _noisy(self, x, stddev):
    if 0 < stddev < np.inf:
      return x + tf.random.normal(tf.shape(x), stddev=stddev, dtype=tf.float32)
    return x

  def _apply_noise_lhs_rhs(self, lhs, rhs):
    """Applies noise to the sufficient statistics of a column solve."""
    if not self.max_norm:
      return lhs, rhs
    max_norm = self.max_norm
    lhs = self._noisy(lhs, self.sigmas[1] * (max_norm ** 2))
    lhs = _project_psd(lhs, rank=3)
    rhs = self._noisy(rhs, self.sigmas[2] * max_norm)
    return lhs, rhs

  def apply_noise_gramian(self, gram: tf.Tensor) -> tf.Tensor:
    """Applies noise to the global Gramian."""
    if not self.max_norm:
      return gram
    gram = self._noisy(gram, self.sigmas[0] * (self.max_norm ** 2))
    return _project_psd(gram, rank=2)

  @tf.function(reduce_retracing=True)
  def solve(self,
            input_batch: InputBatch,
            variable: tf.Variable,
            embeddings: tf.Tensor,
            gramian: tf.Tensor,
            reg: float,
            unobserved_weight: float,
            private: bool) -> tf.Tensor:
    """Computes the private least squares solution.

    Args:
      input_batch: a batch of rows.
      variable: the variable to update.
      embeddings: the embeddings of the other side.
      gramian: the gramian of the other side.
      reg: the regularization constant (can be a vector).
      unobserved_weight: the unobserved weight.
      private: whether to add noise to the LHS and RHS.

    Returns:
      The solution.
    """
    embedding_dim = embeddings.shape[1]
    # Compute LHS and RHS and add noise if needed.
    lhs, rhs = _get_lhs_rhs(input_batch, embeddings)
    if private:
      lhs, rhs = self._apply_noise_lhs_rhs(lhs, rhs)
    # Add Regularization and Gravity terms then solve.
    lhs = lhs + unobserved_weight*tf.expand_dims(gramian, 0)
    lhs = lhs + reg*_reg_tensor(input_batch, embedding_dim)
    solution = tf.squeeze(tf.linalg.solve(lhs, rhs), [2])
    update_indices = input_batch["update_indices"]
    return variable.scatter_update(tf.IndexedSlices(solution, update_indices))

  def get_dp_event_training(self) -> dp_accounting.DpEvent:
    """Gets DpEvent for DP training."""
    # The accounting is done as follows: whenever we compute a statistic with
    # L2 sensitivity k and add Gaussian noise of scale σ, the procedure is
    # (α, α/2/β²)-RDP with β = σ/k. This is represented by a `GaussianDpEvent`
    # with noise_multiplier β.
    # Privacy for ALS steps, where each step involves:
    # - one global Gramian (k² = 1, σ = s0)
    # - local Gramians (k² = budget, σ = s1)
    # - RHSs (k² = budget, σ = s2)
    if not (self.budget and all(self.sigmas)):
      return dp_accounting.NonPrivateDpEvent()
    s0, s1, s2 = self.sigmas
    return dp_accounting.ComposedDpEvent([
        dp_accounting.GaussianDpEvent(s0),
        dp_accounting.GaussianDpEvent(s1 / np.sqrt(self.budget)),
        dp_accounting.GaussianDpEvent(s2 / np.sqrt(self.budget)),
    ])


class DPALSAccountant:
  """Privacy accounting for DPALS.

  Training:
    clip: clips the norm of each user embedding. Used during row solves.
    apply_noise_gramian and apply_noise_lhs_rhs: adds noise to the input
      sufficient statistics involving the user data. Used during column solves.
  """

  def __init__(
      self,
      sanitizer: Sanitizer,
      optimizer: DPALSOptimizer,
      steps: int,
  ):
    """Initializes the DPALSAccountant.

    Args:
      sanitizer: used for data pre-processing.
      optimizer: used to optimize the item embeddings.
      steps: the number of ALS steps.
    """
    self.sanitizer = sanitizer
    self.optimizer = optimizer
    self.steps = steps

  def compute_epsilon(
      self,
      target_delta: float,
      orders: Optional[Sequence[float]] = None,
  ) -> float:
    """Computes epsilon."""
    orders = orders or _DEFAULT_RDP_ORDERS
    accountant = dp_accounting.rdp.RdpAccountant(orders)
    accountant.compose(self.sanitizer.get_dp_event_noisy_count())
    accountant.compose(dp_accounting.SelfComposedDpEvent(
        self.optimizer.get_dp_event_training(), self.steps))
    # Convert RDP to (ε, δ)-DP.
    eps, best_order = accountant.get_epsilon_and_optimal_order(target_delta)
    if best_order == orders[0]:
      print("Consider decreasing the range of order.")
    elif best_order == orders[-1]:
      print("Consider increasing the range of order.")
    return eps

  def set_sigmas(
      self,
      target_epsilon: float,
      target_delta: float,
      sigma_ratio0: float = 1,
      sigma_ratio1: float = 1):
    """Sets sigmas to get the target (epsilon, delta).

    Args:
      target_epsilon: the desired epsilon.
      target_delta: the desired delta.
      sigma_ratio0: the ratio sigma0/sigma2.
      sigma_ratio1: the ratio sigma1/sigma2.
    """
    s_lower = 1e-4
    s_upper = 1e4
    def get_epsilon(s):
      self.optimizer.sigmas = [sigma_ratio0*s, sigma_ratio1*s, s]
      return self.compute_epsilon(target_delta)
    eps = get_epsilon(s_lower)
    i = 0
    while np.abs(eps/target_epsilon - 1) > 0.00001:
      s = (s_lower+s_upper)/2
      eps = get_epsilon(s)
      if eps > target_epsilon:
        s_lower = s
      else:
        s_upper = s
      i += 1
      if i > 1000:
        raise ValueError(
            f"No value of sigmas found for the desired (epsilon, delta)="
            f"={target_epsilon, target_delta}. Consider increasing the "
            "count_stddev.")
    s0, s1, s2 = self.optimizer.sigmas
    print(f"Setting sigmas to [{s0:.2f}, {s1:.2f}, {s2:.2f}], given target "
          f"(epsilon, delta)={target_epsilon, target_delta}")


class DPAMAccountant:
  """Provides privacy accounting for DPAM."""

  def __init__(self,
               sanitizer: Sanitizer,
               steps: int,
               num_users_per_batch: int,
               num_users: int,
               gradient_accumulation_steps: int,
               noise_multiplier: float = 0.0):
    """Initializes the DPAM sanitizer.

    Args:
      sanitizer: a Sanitizer.
      steps: number of steps in DP training.
      num_users_per_batch: (physical) batch size. It is assumed that each user's
        data is grouped in one microbatch.
      num_users: number of users in the dataset, used for amplification by
        sampling.
      gradient_accumulation_steps: the number of accumulation steps before an
        update.
      noise_multiplier: the noise multiplier of DP training.
    """
    self.sanitizer = sanitizer
    self.noise_multiplier = noise_multiplier
    self.steps = steps
    logical_batch_size = num_users_per_batch*gradient_accumulation_steps
    if logical_batch_size > num_users:
      raise ValueError(
          f"num_users_per_batch*gradient_accumulation_steps "
          f"({logical_batch_size}) should not exceed the total number of users "
          f"({num_users})")
    self.sampling_probability = logical_batch_size / num_users

  def get_dp_event_training(self) -> dp_accounting.DpEvent:
    """Gets DpEvent for DP training."""
    if self.steps == 0:
      return dp_accounting.NoOpDpEvent()
    return dp_accounting.SelfComposedDpEvent(
        dp_accounting.PoissonSampledDpEvent(
            self.sampling_probability,
            dp_accounting.GaussianDpEvent(self.noise_multiplier)
        ),
        self.steps,
    )

  def compute_epsilon(
      self,
      target_delta: float,
      account_for_count: bool = True,
      orders: Optional[Sequence[float]] = None,
  ) -> float:
    """Computes epsilon for the DPAM algorithm.

    Args:
      target_delta: the DP delta.
      account_for_count: whether to account for the privacy of noisy count.
      orders: the Renyi orders to use.

    Returns:
      The DP epsilon.
    """
    orders = orders or _DEFAULT_RDP_ORDERS
    accountant = dp_accounting.rdp.RdpAccountant(orders)
    if account_for_count:
      accountant.compose(self.sanitizer.get_dp_event_noisy_count())
    accountant.compose(self.get_dp_event_training())

    # Convert RDP to (ε, δ)-DP.
    eps, best_order = accountant.get_epsilon_and_optimal_order(target_delta)

    if best_order == orders[0]:
      print("Consider decreasing the range of order.")
    elif best_order == orders[-1]:
      print("Consider increasing the range of order.")
    return eps

  def set_noise_multiplier(self, target_epsilon: float, target_delta: float):
    """Sets the noise multiplier to get the target (epsilon, delta)."""
    s_lower = 1e-4
    s_upper = 1e4
    def get_epsilon(noise_multiplier):
      self.noise_multiplier = noise_multiplier
      return self.compute_epsilon(target_delta)
    eps = get_epsilon(s_lower)
    i = 0
    while np.abs(eps/target_epsilon - 1) > 0.00001:
      s = (s_lower+s_upper)/2
      eps = get_epsilon(s)
      if eps > target_epsilon:
        s_lower = s
      else:
        s_upper = s
      i += 1
      if i > 1000:
        raise ValueError(
            f"No value of noise found for the desired (epsilon, delta)="
            f"={target_epsilon, target_delta}. Consider increasing the "
            "count_stddev.")
    print(f"Setting noise to {self.noise_multiplier:.3f}, given target "
          f"(epsilon, delta)={target_epsilon, target_delta}")


@dataclasses.dataclass
class SparseMatrix:
  """Represents a sparse matrix of user feedback."""
  # ["uid", "sid", (optional) "rating"]
  # user rating information
  data: pd.DataFrame
  # ["uid", "sid"] of the history. The history is always excluded from
  # the set of candidates (in recall computation), and always used to compute
  # the projection.
  history: Optional[pd.DataFrame] = None
  # InputMatrix objects created by Dataset.preprocess()
  sp_history: Optional[InputMatrix] = None
  sp_labels: Any = None
  sp_data: Any = None
  sp_data_sampled_transposed: Any = None

  @property
  def num_users(self) -> int:
    num_users = max(self.data.uid) + 1
    if self.history is not None:
      num_users = max(num_users, max(self.history.uid) + 1)
    return num_users


class Dataset:
  """Dataset and processing utilities."""

  def __init__(
      self,
      num_users: int,
      num_items: int,
      metadata_df: pd.DataFrame,
      train: SparseMatrix,
      validation: SparseMatrix,
      test: SparseMatrix):
    self.num_users = num_users
    self.num_items = num_items
    self.num_frequent_items = num_items
    self.metadata_df = metadata_df
    self.train = train
    self.validation = validation
    self.test = test
    self.user_counts_train = self._count_nnz(train.data, "uid", num_users)
    self.user_counts_valid = self._count_nnz(validation.data, "uid", num_users)
    self.user_counts_test = self._count_nnz(test.data, "uid", num_users)
    # The following are set by self.preprocess()
    self.num_features = 0
    self.noisy_frequent_item_counts = None

  def _count_nnz(self, df, key, num_elements):
    """Counts the number of occurrences of df[key]."""
    count_df = df.groupby(key)[key].count()
    counts = np.zeros([num_elements], dtype=np.int32)
    counts[count_df.index.values] = count_df.values
    return counts

  @property
  def item_counts_train(self) -> np.ndarray:
    # The item counts are recomputed because items may be reindexed.
    return self._count_nnz(self.train.data, "sid", self.num_items)

  @property
  def item_counts_valid(self) -> np.ndarray:
    return self._count_nnz(self.validation.data, "sid", self.num_items)

  @property
  def item_counts_test(self) -> np.ndarray:
    return self._count_nnz(self.test.data, "sid", self.num_items)

  def preprocess(
      self,
      sanitizer: Optional[Sanitizer] = None,
      binarize: bool = False):
    """Pre-processes the data using a Sanitizer.

    Creates the following InputMatrix objects (all contain frequent items only):
    For training:
    - train.sp_data
    - train.sp_data_sampled_transposed
    For evaluation:
    - train.sp_labels
    - test.sp_labels
    - test.sp_history
    - validation.sp_labels
    - validation.sp_history

    Args:
      sanitizer: a Sanitizer object.
      binarize: whether to binarize the data.
    """
    if sanitizer is None:
      sanitizer = Sanitizer()
    value_key = None if binarize else "rating"
    train_st, noisy_frequent_item_counts = sanitizer.preprocess_data(
        self, value_key)
    self.noisy_frequent_item_counts = noisy_frequent_item_counts
    # Training data
    self.train.sp_data = dataset_lib.df_to_input_matrix(
        self.get_frequent(self.train.data), num_rows=self.num_users,
        num_cols=self.num_frequent_items, value_key=value_key)
    self.train.sp_data_sampled_transposed = train_st
    # Eval data
    for matrix in [self.train, self.validation, self.test]:
      matrix.sp_labels = dataset_lib.df_to_input_matrix(
          matrix.data, matrix.num_users, self.num_items, value_key=value_key)
    for matrix in [self.validation, self.test]:
      if matrix.history is not None:
        matrix.sp_history = dataset_lib.df_to_input_matrix(
            self.get_frequent(matrix.history), matrix.num_users, self.num_items,
            value_key=value_key)

  def _update_all(self, fn):
    """Applies a function to all data sets."""
    self.train.data = fn(self.train.data)
    self.validation.data = fn(self.validation.data)
    self.test.data = fn(self.test.data)
    if self.validation.history is not None:
      self.validation.history = fn(self.validation.history)
    if self.test.history is not None:
      self.test.history = fn(self.test.history)

  def remove_global_bias(self, global_bias: float):
    if "rating" not in self.train.data:
      raise ValueError("Cannot center the data if it has no ratings.")
    print(f"Removing the global bias: {global_bias:.2f}", flush=True)
    def remove_bias(df):
      new_df = df.copy()
      new_df.loc[:, "rating"] = df["rating"] - global_bias
      return new_df
    self._update_all(remove_bias)

  def reindex_items(self, num_frequent_items: int, sorted_ids: np.ndarray):
    """Split items into frequent/rare items."""
    self.num_frequent_items = num_frequent_items
    index_map = np.zeros(self.num_items, dtype=np.int32)
    index_map[sorted_ids] = np.arange(self.num_items)
    self._sid_new_from_old = index_map
    self._sid_old_from_new = sorted_ids
    def map_ids(df):
      new_df = df.copy()
      new_df["old_sid"] = new_df["sid"]
      new_df.loc[:, "sid"] = index_map[df["sid"]]
      return new_df
    self._update_all(map_ids)
    if self.metadata_df is not None:
      self.metadata_df = map_ids(self.metadata_df)

  def get_frequent(self, df: pd.DataFrame) -> pd.DataFrame:
    return df[df.sid < self.num_frequent_items]

  def get_rare(self, df: pd.DataFrame) -> pd.DataFrame:
    return df[df.sid >= self.num_frequent_items]

  def set_reg_weights(
      self,
      row_reg_exponent: float,
      col_reg_exponent: float,
      apply_in_eval: bool = True):
    """Sets weights for frequency-based regularization.

    Args:
      row_reg_exponent: The regularization coefficient of a row is scaled by its
        frequency raised to this power (see compute_weights). Defaults to 0,
        i.e. uniform weights.
      col_reg_exponent: see row_reg_exponent.
      apply_in_eval: whether to also apply frequency regularization to the eval
        (when computing the WALS projection).
    """
    if row_reg_exponent:
      row_reg = _compute_weights(self.user_counts_train, row_reg_exponent)
      self.train.sp_data.row_reg = row_reg
      if apply_in_eval:
        for x in [self.validation.sp_history, self.test.sp_history]:
          if x is not None:
            x.row_reg = row_reg
    if col_reg_exponent:
      self.train.sp_data_sampled_transposed.row_reg = _compute_weights(
          self.noisy_frequent_item_counts, col_reg_exponent)


class EvalMetrics:
  """Computes evaluation metrics."""

  def __init__(self,
               name: str,
               model: "_TwoTowerModel",
               matrix: SparseMatrix,
               eval_batch_size: Optional[int],
               recall_positions: Sequence[int]):
    """Computes eval metrics from a dataset.

    Compute user embeddings by projecting, using the history (only uses frequent
    items for projections).
    Note: some users may have 0 frequent items in their history, and will
    have a 0 embedding, i.e. their predictions are all 0.
    Evaluation includes all items, frequent and rare.

    Args:
      name: the name of the evaluation task
      model: a TwoTowerModel
      matrix: a SparseMatrix (train, validation, or test)
      eval_batch_size: used in wals_projection.
      recall_positions: a list of recall positions
    """
    self.name = name
    self.model = model
    self.num_items = model.num_items
    self.num_frequent_items = model.num_frequent_items
    self.matrix = matrix
    self.recall_positions = recall_positions
    # Batch the data for wals_projection.
    self.do_project = False
    if matrix.sp_history is not None:
      self.do_project = True
      self.history_ds = matrix.sp_history.batch_ls(eval_batch_size)
    # Pre-compute user-means for use in the metrics.
    self.user_avg_ratings = None
    if matrix.history is not None and not model.binarize:
      means = matrix.history.groupby("uid").mean()
      user_avg_ratings = np.zeros(matrix.num_users)
      user_avg_ratings[means.index] = means["rating"]
      self.user_avg_ratings = tf.cast(user_avg_ratings, tf.float32)

  def sparse_predictions(self) -> tf.Tensor:
    """Computes predictions at the positions of matrix.data (used for RMSE)."""
    row_embeddings = self.model.row_embeddings
    if self.do_project:
      row_embeddings = self.model.wals_projection(self.history_ds)
    return tf.gather_nd(
        tf.matmul(row_embeddings, self.model.col_embeddings, transpose_b=True),
        self.matrix.sp_labels.indices)

  def predictions(self) -> tf.Tensor:
    """Computes the full prediction matrix (used for Recall)."""
    row_embeddings = self.model.row_embeddings
    if self.do_project:
      row_embeddings = self.model.wals_projection(self.history_ds)
    preds = tf.matmul(
        row_embeddings, self.model.col_embeddings, transpose_b=True)
    if self.do_project:
      # Exclude history labels.
      hist = self.matrix.sp_history.to_sparse_tensor()
      hist_indicator = hist.with_values(
          tf.ones_like(hist.values, dtype=tf.float32))
      preds = preds - 1000000*tf.sparse.to_dense(hist_indicator)
    return preds

  def metrics(self) -> Dict[str, tf.Tensor]:
    """Returns a dict of metrics.

    Note: combining all metrics in one function avoids recomputing the
    predictions matrix.

    Defines the following metrics:
      rmse: RMSE without rare items.
      rmse_useravg: RMSE, where we predict the user's average on rare items.
      recall: dict of Recall@k, without rare items.
    """
    metrics = {}
    name = self.name
    labels = self.matrix.sp_labels.to_sparse_tensor()
    # RMSE
    rmse_preds = self.sparse_predictions()
    rmse = _rmse(labels, rmse_preds)
    metrics[f"{name}_RMSE"] = rmse
    # RMSE with user average:
    # Uses the user's mean rating (computed on the full history) for rare items.
    if self.user_avg_ratings is not None:
      user_indices = labels.indices[:, 0]
      item_indices = labels.indices[:, 1]
      preds_useravg = tf.where(
          item_indices < self.num_frequent_items,
          rmse_preds,
          tf.gather(self.user_avg_ratings, user_indices))
      metrics[f"{name}_RMSE_user"] = _rmse(labels, preds_useravg)
    # Recall
    if self.recall_positions:
      recall_labels = labels.with_values(labels.indices[:, 1])
      recall_preds = self.predictions()
      for k in self.recall_positions:
        metrics[f"{name}_R@{k}"] = _recall(recall_labels, recall_preds, k)
    return metrics

  def rmse_per_item(self, user_avg: bool = True) -> tf.Tensor:
    preds = self.sparse_predictions()
    labels = self.matrix.sp_labels.to_sparse_tensor()
    if user_avg and self.user_avg_ratings is not None:
      user_indices = labels.indices[:, 0]
      item_indices = labels.indices[:, 1]
      preds = tf.where(
          item_indices < self.num_frequent_items,
          preds,
          tf.gather(self.user_avg_ratings, user_indices))
    return _rmse(labels, preds, axis=0)

  def _item_count(self, data) -> np.ndarray:
    count_df = data.groupby("sid")["sid"].count()
    counts = np.zeros([self.num_items])
    counts[count_df.index] = count_df.values
    return counts

  def _get_buckets(self, num_buckets, item_frac=1.0) -> Sequence[_Bucket]:
    """Returns item buckets, based on the training data counts.

    Args:
      num_buckets: the number of buckets.
      item_frac: restricts the buckets to the most frequent items.
    """
    data_for_counts = self.matrix.history
    if data_for_counts is None:
      data_for_counts = self.matrix.data
    item_counts = self._item_count(data_for_counts)
    sorted_ids = np.argsort(item_counts)
    if item_frac < 1:
      to_keep = int(self.num_items*item_frac)
      sorted_ids = sorted_ids[-to_keep:]
    num_items = len(sorted_ids)
    bucket_sizes = [num_items // num_buckets]*num_buckets
    for i in range(num_items % num_buckets):
      bucket_sizes[i] += 1
    assert sum(bucket_sizes) == num_items
    buckets = []
    i = 0
    for s in bucket_sizes:
      buckets.append(sorted_ids[i:i+s])
      i += s
    return buckets

  def sliced_recall(self,
                    k: int,
                    num_buckets: int,
                    item_frac: float = 1.0) -> Sequence[float]:
    """Computes sliced Recall per item."""
    buckets = self._get_buckets(num_buckets, item_frac)
    preds = self.predictions()
    _, top_k_preds = tf.math.top_k(preds, k)
    return _sliced_recall(
        labels=self.matrix.data, preds=top_k_preds, k=k,
        buckets=buckets)

  def sliced_rmse(self,
                  num_buckets: int,
                  item_frac: float = 1.0,
                  user_avg: bool = True) -> Sequence[float]:
    """Returns sliced RMSE per item.

    Args:
      num_buckets: the number of buckets.
      item_frac: ignores rare items and only computes the metric on this top
        fraction.
      user_avg: whether to predict the user average for rare items.
    """
    mse = np.square(self.rmse_per_item(user_avg=user_avg))
    weights = self._item_count(self.matrix.data)
    # mse[i] may be nan if weights[i] is 0. We set it to 0 so that the product
    # is well defined.
    mse[weights == 0] = 0.0
    buckets = self._get_buckets(num_buckets, item_frac)
    sliced_rmse = [
        np.sqrt(np.sum(mse[bucket]*weights[bucket])/np.sum(weights[bucket]))
        for bucket in buckets]
    return sliced_rmse


class _TwoTowerModel(abc.ABC):
  """Abstract Two Tower model class."""

  def __init__(
      self,
      dataset: Dataset,
      embedding_dim: int,
      init_stddev: float = 0.1,
      binarize: bool = False,
      sanitizer: Optional[Sanitizer] = None,
      eval_batch_size: Optional[int] = None,
      row_reg_exponent: float = 0,
      col_reg_exponent: float = 0,
      recall_positions: Optional[list[int]] = None):
    """Creates a _TwoTowerModel.

    Provides a set of evaluation metrics (RMSE, Recall with and without slicing)
    for a two-tower model on the (train, validation, test) splits in `dataset`.
    The metrics are computed based on `row_embeddings` and `col_embeddings_freq`
    variables created by this class. Child classes are responsible for updating
    those variables.

    This also applies the sanitizer to the dataset to prepare the training data.
    This includes splitting the data into frequent/rare, and per-user sampling.

    Args:
      dataset: a Dataset object.
      embedding_dim: the embedding dimension.
      init_stddev: the initial standard deviation of the embeddings.
      binarize: Whether to use a binary value or the rating value as a target.
      sanitizer: a Sanitizer object, for privacy protection.
      eval_batch_size: batch size used in wals_projection for computing metrics.
      row_reg_exponent: The regularization coefficient of a row is scaled by its
        frequency raised to this power (see compute_weights). Defaults to 0,
        i.e. uniform weights.
      col_reg_exponent: see row_reg_exponent.
      recall_positions: computes the recall at these positions.
    """
    if binarize:
      print("Note: this model uses binarized ratings.", flush=True)
    else:
      print("Note: this model uses rating values.", flush=True)
    self.dataset = copy.deepcopy(dataset)
    self.embedding_dim = embedding_dim
    self.binarize = binarize
    self.sanitizer = sanitizer
    self.recall_positions = recall_positions or []
    # Prepare the data.
    self.dataset.preprocess(sanitizer, binarize=binarize)
    self.dataset.set_reg_weights(
        row_reg_exponent, col_reg_exponent, apply_in_eval=not binarize)
    assert self.dataset.noisy_frequent_item_counts is not None  # pytype hint
    self.num_users = self.dataset.num_users
    self.num_items = self.dataset.num_items
    self.num_frequent_items = self.dataset.num_frequent_items
    # Create variables.
    def make_var(shape, name, stddev=init_stddev):
      return tf.Variable(
          tf.random.normal(shape, stddev=stddev, dtype=tf.float32), name=name)
    self.row_embeddings = make_var(
        [self.num_users, embedding_dim], "row_embeddings")
    self.col_embeddings_freq = make_var(
        [self.num_frequent_items, embedding_dim], "col_embeddings")
    # Metrics.
    # Note: we disable recall computation on the train set, as it would require
    # computing the full prediction matrix.
    self.eval_train = EvalMetrics(
        "train", self, self.dataset.train, eval_batch_size, [])
    self.eval_valid = EvalMetrics(
        "valid", self, self.dataset.validation, eval_batch_size,
        self.recall_positions)
    self.eval_test = EvalMetrics(
        "test", self, self.dataset.test, eval_batch_size,
        self.recall_positions)
    # WALS projection parameters (can be overridden by subclasses).
    self._row_reg = 1
    self._col_reg = 1
    self._unobserved_weight = 0
    # History of metrics.
    self._step = 1
    self._iterations = []
    self._metrics_vals = None

  @property
  def col_embeddings(self) -> tf.Tensor:
    num_rare = self.num_items - self.num_frequent_items
    return tf.concat(
        [self.col_embeddings_freq, tf.zeros([num_rare, self.embedding_dim])],
        axis=0)

  @tf.function(reduce_retracing=True)
  def _wals_projection_batch(
      self, batch: InputBatch, col_gramian: tf.Tensor) -> tf.Tensor:
    """Compute the WALS projection for one batch."""
    lhs, rhs = _get_lhs_rhs(batch, self.col_embeddings)
    lhs = lhs + self._unobserved_weight*tf.expand_dims(col_gramian, 0)
    lhs = lhs + self._row_reg*_reg_tensor(batch, self.embedding_dim)
    return tf.squeeze(tf.linalg.solve(lhs, rhs), [2])

  def wals_projection(self, input_matrix: tf.data.Dataset) -> tf.Tensor:
    """Returns the WALS projection for a set of users.

    Args:
      input_matrix: The matrix of users to project.

    Returns:
      solution: a [num_users, embeddings_dim] tensor of the solution embeddings,
        where num_users is the number of rows in input_matrix.
    """
    col_gramian = tf.matmul(
        self.col_embeddings, self.col_embeddings, transpose_a=True)
    return tf.concat([self._wals_projection_batch(batch, col_gramian)
                      for batch in input_matrix], axis=0)

  def metrics_dict(
      self,
      train_metrics: bool = False,
      validation_metrics: bool = True,
      test_metrics: bool = True) -> Dict[str, tf.Tensor]:
    """Returns a dictionary of metrics."""
    metrics = collections.OrderedDict()
    if train_metrics:
      metrics.update(self.eval_train.metrics())
    if validation_metrics:
      metrics.update(self.eval_valid.metrics())
    if test_metrics:
      metrics.update(self.eval_test.metrics())
    return metrics

  @property
  def metrics(self) -> Dict[str, float]:
    return {k: v.numpy() for k, v in self.metrics_dict().items()}

  def _update_metrics_and_plots(self, plot_metrics, metrics_axs):
    """Updates the metrics and plots."""
    self._iterations.append(self._step)
    metrics = self.metrics
    if self._metrics_vals is None:
      self._metrics_vals = collections.OrderedDict((k, []) for k in metrics)
    for k, v in metrics.items():
      self._metrics_vals[k].append(v)
    display.clear_output(wait=True)
    if metrics_axs:
      all_keys = self._metrics_vals.keys()
      for plot_keys, ax in zip(plot_metrics, metrics_axs):
        ax.clear()
        for key in plot_keys:
          if key not in self._metrics_vals:
            raise ValueError(f"unknown key {key}. Available keys: {all_keys}")
          ax.plot(self._iterations, self._metrics_vals[key], label=key)
        ax.grid()
        ax.legend()
      display.display(plt.gcf())
    # Print metrics.
    print(f"step {self._step}:")
    table = [(key, float(vals[-1]))
             for key, vals in self._metrics_vals.items()]
    for k, v in table:
      print(f"{k}: {v:.4f}")

  @abc.abstractmethod
  def train_step(self, **kwargs):
    """Runs one training step."""
    raise NotImplementedError()

  def train(self,
            num_steps: int,
            compute_metrics: bool = True,
            plot_metrics: Optional[Sequence[str]] = None,
            num_eval_points: Optional[int] = None,
            **kwargs):
    """Trains the model.

    Args:
      num_steps: number of iterations to run.
      compute_metrics: whether to compute the metrics.
      plot_metrics: list of lists of the metrics to plot. Each list is plotted
        in a separate figure. Ignored if compute metrics is False.
      num_eval_points: the number of times we evaluate the model during the
        training.
      **kwargs: passed to the train_step() method.
    """
    num_eval_points = num_eval_points or num_steps
    # Create a figure.
    metric_axs = None
    if plot_metrics:
      num_cols = len(plot_metrics)
      fig = plt.figure(1)
      fig.set_size_inches(num_cols*6, 5)
      metric_axs = [fig.add_subplot(1, num_cols, i+1)
                    for i in range(num_cols)]

    # Train and display results.
    train_times = []
    eval_times = []
    eval_freq = max(1, num_steps // num_eval_points)
    while self._step <= num_steps:
      print(f"\rStep {self._step}/{num_steps}", end="", flush=True)
      t0 = time.time()
      self.train_step(**kwargs)
      t1 = time.time()
      train_times.append(t1-t0)
      if compute_metrics and self._step % eval_freq == 0:
        self._update_metrics_and_plots(plot_metrics, metric_axs)
        t2 = time.time()
        eval_times.append(t2 - t1)
      self._step += 1
    print(f"\nTraining took {np.mean(train_times):.2f}s per sweep, "
          f"{np.mean(eval_times):.2f}s per eval")
    if plot_metrics:
      plt.close(1)  # to avoid displaying a duplicate figure.


class DPALSModel(_TwoTowerModel):
  """Differentially Private Alternating Least Squares (DPALS)."""

  def __init__(
      self,
      dataset: Dataset,
      embedding_dim: int,
      unobserved_weight: float,
      init_stddev: float,
      binarize: bool = False,
      regularization_weight: float = 1.,
      row_reg_exponent: float = 0,
      col_reg_exponent: float = 0,
      sanitizer: Optional[Sanitizer] = None,
      optimizer: Optional[DPALSOptimizer] = None,
      batch_size: Optional[int] = None,
      recall_positions: Optional[Sequence[int]] = None,
      random_seed: Optional[int] = None):
    """Initializes a DPALSModel.

    Args:
      dataset: a Dataset object.
      embedding_dim: The dimension of the embedding space.
      unobserved_weight: The gravity regularization coefficient.
      init_stddev: the initial standard deviation of the embeddings.
      binarize: Whether to use a binary value or the rating value as a target.
      regularization_weight: The regularization coefficient.
      row_reg_exponent: The regularization coefficient of a row is scaled by its
        frequency raised to this power (see compute_weights). Defaults to 0,
        i.e. uniform weights.
      col_reg_exponent: see row_reg_exponent.
      sanitizer: a Sanitizer object, for privacy protection.
      optimizer: a DPALSOptimizer object, for optimizing the item embeddings.
      batch_size: process this many rows (resp. columns) in each solve.
      recall_positions: computes the recall at these positions.
      random_seed: the random seed to use.
    """
    sanitizer = sanitizer or Sanitizer()  # Defaults to no sampling.
    optimizer = optimizer or DPALSOptimizer()  # Defaults to non-private WALS.
    super().__init__(
        dataset,
        embedding_dim=embedding_dim,
        init_stddev=init_stddev,
        binarize=binarize,
        sanitizer=sanitizer,
        eval_batch_size=batch_size,
        row_reg_exponent=row_reg_exponent,
        col_reg_exponent=col_reg_exponent,
        recall_positions=recall_positions)
    if random_seed:
      tf.random.set_seed(random_seed)
    self.optimizer = optimizer
    self._unobserved_weight = unobserved_weight
    self._batch_size = batch_size
    def gram_var(name):
      return tf.Variable(
          tf.zeros([embedding_dim, embedding_dim], dtype=tf.float32),
          name=name)
    self._col_gramian = gram_var("col_gramian")
    self._row_gramian = gram_var("row_gramian")
    # Regularization.
    self._row_reg = regularization_weight
    self._col_reg = regularization_weight
    # Use the full (not sampled) data for the row solve.
    self._row_data = self.dataset.train.sp_data.batch_ls(self._batch_size)
    self._col_data = (self.dataset.train.sp_data_sampled_transposed
                      .batch_ls(self._batch_size))

  def train_step(self):
    """Runs one outer step of training."""
    # User update ==============================================================
    col_gram = tf.matmul(
        self.col_embeddings_freq, self.col_embeddings_freq, transpose_a=True)
    self._col_gramian.assign(col_gram)
    for batch in self._row_data:
      self.optimizer.solve(
          batch, self.row_embeddings, self.col_embeddings_freq,
          self._col_gramian, self._row_reg, self._unobserved_weight,
          private=False)
    # Item update ==============================================================
    # Clip row embeddings.
    self.row_embeddings.assign(self.optimizer.clip(self.row_embeddings))
    # Update row Gramian (with noise).
    row_gram = tf.matmul(
        self.row_embeddings, self.row_embeddings, transpose_a=True)
    self._row_gramian.assign(self.optimizer.apply_noise_gramian(row_gram))
    # Solve column embeddings (with noise).
    for batch in self._col_data:
      self.optimizer.solve(
          batch, self.col_embeddings_freq, self.row_embeddings,
          self._row_gramian, self._col_reg, self._unobserved_weight,
          private=True)


class _ItemTower(tf.keras.Model):
  """Item tower helper class used in the implementation of AMModel."""

  def __init__(
      self,
      row_embeddings: tf.Variable,
      encoder: tf.keras.Model):
    """Initializes an _ItemTower.

    The output of the tower (result of the call() method) is the dot product
    between the user encoding (looked up from `row_embeddings`) and the item
    encoding (result of calling `encoder`).

    Args:
      row_embeddings: used to look up the row embeddings, using the "uid"
        feature.
      encoder: The item encoder. See AMModel for assumptions that it needs to
        satisfy.
    """
    super().__init__()
    self._row_embeddings = row_embeddings
    self._encoder = encoder

  def call(self, features: InputBatch) -> tf.Tensor:
    """Returns the predictions."""
    row_emb = tf.nn.embedding_lookup(self._row_embeddings, features["uid"])
    col_emb = self._encoder.call(features)
    preds = tf.reduce_sum(row_emb*col_emb, axis=1)
    return preds

  @tf.function(reduce_retracing=True)
  def train_step(self, data: tuple[InputBatch, tf.Tensor, tf.Tensor]):
    features, labels, weights = data
    with tf.GradientTape() as tape:
      predictions = self.call(features)
      labels = tf.cast(labels, tf.float32)
      loss_vector = self._encoder.loss_vector(
          predictions, labels, weights=weights, features=features)
    self.optimizer.minimize(
        loss=loss_vector,
        var_list=self._encoder.trainable_variables,
        tape=tape)
    return {}


class AMModel(_TwoTowerModel):
  """A two tower model optimized using Alternating Minimization.

  The user tower is assumed to be an embedding, and is optimized using least
  squares, applied to the (user, item, rating) data only.
  The item tower is given by the `encoder`, and is optimized using an arbitrary
  optimizer.
  One optimization round (implemented in train_step) includes the following:
  - Update the user embeddings (while freezing the item tower): compute the item
    Gramian (based on the cached embeddings), then apply WALS projection using
    the (user, item, rating) data.
  - Optimize the item tower (while freezing the user embeddings): uses the
    provided `optimizer`.
  - Cache the item embeddings (while freezing all parameters).
  """

  def __init__(
      self,
      dataset: Dataset,
      embedding_dim: int,
      unobserved_weight: float,
      init_stddev: float,
      binarize: bool = False,
      regularization_weight: float = 1.,
      row_reg_exponent: Optional[float] = None,
      sanitizer: Optional[Sanitizer] = None,
      recall_positions: Optional[Sequence[int]] = None,
      random_seed: Optional[int] = None):
    """Initializes an AMModel.

    Args:
      dataset: a Dataset object.
      embedding_dim: The dimension of the embedding space.
      unobserved_weight: The gravity regularization coefficient.
      init_stddev: the initial standard deviation of the embeddings.
      binarize: Whether to use a binary value or the rating value as a target.
      regularization_weight: The regularization coefficient.
      row_reg_exponent: The regularization coefficient of a row (i.e. user) is
        scaled by its frequency raised to this power (see compute_weights).
        Defaults to 0, i.e. uniform weights. A value between 0 and 1 can
        improve quality (this means frequent items are regularized more).
      sanitizer: a Sanitizer object, for pre-processing the data.
      recall_positions: computes the recall at these positions.
      random_seed: the random seed to use.
    """
    super().__init__(
        dataset, embedding_dim=embedding_dim, binarize=binarize,
        row_reg_exponent=row_reg_exponent, sanitizer=sanitizer,
        recall_positions=recall_positions)
    if random_seed:
      tf.random.set_seed(random_seed)
    self._unobserved_weight = unobserved_weight
    self._row_reg = regularization_weight
    # TF ops
    self._col_gramian = tf.Variable(
        tf.zeros([embedding_dim, embedding_dim], dtype=tf.float32),
        name="col_gramian")
    self._row_gramian = tf.Variable(
        tf.zeros([embedding_dim, embedding_dim], dtype=tf.float32),
        name="row_gramian")
    # Data
    # No batching for now
    self._row_data = self.dataset.train.sp_data.batch_ls(None)

  def compile_item_tower(
      self,
      item_encoder: Optional[tf.keras.Model] = None,
      item_tower: Optional[_ItemTower] = None,
      item_batch_size: Optional[int] = None,
      num_users_per_batch: Optional[int] = None,
      num_examples_per_user: Optional[int] = None,
      optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
      ):
    """Configures and compiles the item tower.

    Args:
      item_encoder: a keras.Model representing the item encoder. It must satisfy
        the following assumptions:
        * encoder.call(features) returns a batch of item embeddings.
        * it provides a `loss_vector` method, which takes as input (predictions,
          labels, weights, features) and outputs a loss vector. Taking the
          features as input is useful for example for implementing L2
          regularization in which only the active embeddings are regularized
          (i.e. frequency-based regularization).
      item_tower: if present, use it as item_tower, exactly one of
        item_encoder and item_tower must be defined.
      item_batch_size: used for batching the item data (without user grouping).
      num_users_per_batch: used for batching the item data (with user grouping).
      num_examples_per_user: used for batching the item data (with user
        grouping).
      optimizer: used to optimize the item tower.
    """
    if not (item_tower is None) ^ (item_encoder is None):
      raise ValueError("Exactly One of item_encoder and item_tower must be "
                       "defined.")
    if item_tower is None:
      item_tower = _ItemTower(self.row_embeddings, item_encoder)
    user_batching = (num_users_per_batch is not None
                     and num_examples_per_user is not None)
    normal_batching = item_batch_size is not None
    if not normal_batching ^ user_batching:
      raise ValueError(
          "Must either specify item_batch_size, or (num_users_per_batch, "
          "num_items_per_user)")
    if user_batching:
      self._col_dataset = (
          self.dataset.train.sp_data_sampled_transposed.batch_gd_by_user(
              user_axis=1,
              num_examples_per_user=num_examples_per_user,
              num_users_per_batch=num_users_per_batch))
    else:
      self._col_dataset = (
          self.dataset.train.sp_data_sampled_transposed.batch_gd(
              user_axis=1,
              batch_size=item_batch_size))
    item_tower.compile(optimizer=optimizer)
    self._item_tower = item_tower

  # TODO(walidk): this currently only works for a shallow encoder. For general
  # encoders, we need to iterate through a different dataset.
  def cache_item_embeddings(self):
    """Caches the item embeddings."""
    encoder = self._item_tower._encoder  # pylint: disable=protected-access
    if not encoder.built:
      encoder.build(input_shape={"sid": [None]})
    col_emb = encoder.trainable_variables[0]
    self.col_embeddings_freq.assign(col_emb)

  def train_step(self, inner_steps: Optional[int] = None):
    """Runs one outer step of training.

    Args:
      inner_steps: number of steps to run for the item tower.
    """
    if inner_steps is None:
      raise ValueError("AMModel.train_step requires `inner_steps`")
    # User update ==============================================================
    if self._step == 0:
      self.cache_item_embeddings()
    self._col_gramian.assign(
        tf.matmul(self.col_embeddings_freq, self.col_embeddings_freq,
                  transpose_a=True))
    for batch in self._row_data:
      self._row_solve(batch)
    # Item update ==============================================================
    # Update the row Gramian so we can use it in the gravity loss.
    self._row_gramian.assign(
        tf.matmul(self.row_embeddings, self.row_embeddings, transpose_a=True))
    self._item_tower.fit(self._col_dataset, steps_per_epoch=inner_steps)
    self.cache_item_embeddings()

  @tf.function(reduce_retracing=True)
  def _row_solve(self, input_matrix: InputBatch):
    """Creates the ops for updating the user embeddings."""
    gramian = self._col_gramian
    # Compute LHS and RHS.
    lhs, rhs = _get_lhs_rhs(input_matrix, self.col_embeddings_freq)
    # Add Regularization and Gravity terms then solve.
    lhs = lhs + self._unobserved_weight*tf.expand_dims(gramian, 0)
    lhs = lhs + self._row_reg*_reg_tensor(input_matrix, self.embedding_dim)
    solution = tf.squeeze(tf.linalg.solve(lhs, rhs), [2])
    # Apply the update.
    update_indices = input_matrix["update_indices"]
    return self.row_embeddings.scatter_update(
        tf.IndexedSlices(solution, update_indices))


def _compute_adaptive_weights(sp_data: tf.SparseTensor,
                              item_counts: np.ndarray,
                              exponent: float,
                              budget: float) -> tf.Tensor:
  """Computes weights for the adaptive clipping method given a per-user budget.

  If the user has items indexed by i, then this computes weights cᵢ such that cᵢ
  is proportional to item_counts[i]^exponent and Σᵢcᵢ² = budget.

  Args:
    sp_data: SparseTensor of item data.
    item_counts: dense vector of item counts.
    exponent: must be non-positive.
    budget: total budget per user.
  Returns:
    A vector of weights, aligned with sp_data.values.
  """
  user_indices = sp_data.indices[:, 1]
  item_indices = sp_data.indices[:, 0]
  sq_weights = tf.pow(
      tf.gather(tf.cast(item_counts, tf.float32), item_indices),
      2.0*exponent)  # 2*exponent to have the squared weights.
  user_denom = tf.sparse.reduce_sum(sp_data.with_values(sq_weights), axis=0)
  sq_weights = budget * sq_weights / tf.gather(user_denom, user_indices)
  return tf.sqrt(sq_weights)


def _sliced_recall(labels: pd.DataFrame,
                   preds: tf.Tensor,
                   k: int,
                   buckets: Sequence[_Bucket]) -> Sequence[float]:
  """Returns the recall per bucket.

  Args:
    labels: a DataFrame of the labels, containing the columns "uid" and "sid".
    preds: a Tensor of predictions, of shape [num_users, num_items].
    k: computes the Recall@k.
    buckets: a list of lists of sid, used to partition the items into buckets.
  """
  labels = labels.sort_values(["uid", "sid"])
  num_items = max(np.max(labels["sid"]), np.max(preds)) + 1
  num_users = preds.shape[0]
  preds = preds[:, :k]
  recalls = []
  for bucket in buckets:
    l = labels[labels["sid"].isin(bucket)]
    sp_l = tf.SparseTensor(
        indices=l[["uid", "sid"]],
        values=tf.cast(l["sid"], tf.int32),
        dense_shape=[num_users, num_items])
    true_pos = tf.sets.size(tf.sets.intersection(preds, sp_l))
    all_pos = tf.minimum(k, tf.sets.size(sp_l))
    recall = true_pos/all_pos
    recalls.append(np.nanmean(recall.numpy()))
  return recalls


def _rmse(labels: tf.SparseTensor,
          pred_vals: tf.Tensor,
          axis: Optional[int] = None) -> tf.Tensor:
  """Returns the RMSE.

  Args:
    labels: SparseTensor of the labels.
    pred_vals: a vector of the predictions corresponding to the label indices.
    axis: when provided, computes the mean along this axis. For example, if
      axis=1, this returns the RMSE per row.
  """
  losses = labels.with_values(tf.pow(pred_vals - labels.values, 2))
  indicator = losses.with_values(tf.ones_like(losses.values))
  counts = tf.sparse.reduce_sum(indicator, axis)
  return tf.sqrt(tf.sparse.reduce_sum(losses, axis)/counts)


def _recall(labels: tf.SparseTensor,
            preds: tf.Tensor,
            k: int,
            per_user: bool = False) -> tf.Tensor:
  """Returns the recall at k given predictions and sparse labels.

  Args:
    labels: a SparseTensor of the positive ids.
    preds: a dense tensor of the predicted scores.
    k: compute Recall@k.
    per_user: when True, returns a vector of recalls, one per user. Otherwise,
      returns a scalar of the overall recall (in this case every positive will
      have an equal weight, so users with more positives contribute more to the
      metric).
  """
  top_k = tf.cast(tf.math.top_k(preds, k)[1], tf.int64)
  true_pos = tf.sets.size(tf.sets.intersection(top_k, labels))
  all_pos = tf.minimum(k, tf.sets.size(labels))
  if per_user:
    return true_pos/all_pos
  def nan_mean(xs):
    mask = tf.reshape(tf.logical_not(tf.math.is_nan(xs)), [-1])
    return tf.reduce_mean(tf.boolean_mask(xs, mask))
  return nan_mean(true_pos/all_pos)


def _compute_weights(counts: np.ndarray, exponent: float) -> np.ndarray:
  """Computes weights from counts.

  Args:
    counts: a vector of counts.
    exponent: raises the counts to this exponent.
  Returns:
    Weights proportional to counts^exponent. The weights are normalized so that
    sum(counts*weights) = sum(counts).
  """
  counts = np.array(counts) + 1
  weights = np.power(counts, exponent)
  mean = np.sum(counts*weights) / np.sum(counts)
  return weights / mean


def _project_psd(x: tf.Tensor, rank: int) -> tf.Tensor:
  """Projects a tensor on the cone of PSD matrices.

  Args:
    x: the tensor to project.
    rank: the rank of the tensor. When the rank is 3, the first dimension is
      interpreted as a batch dimension.
  Returns:
    The projected tensor.
  """
  if rank == 2:
    indices = [1, 0]
  elif rank == 3:
    indices = [0, 2, 1]
  else:
    raise ValueError("rank must be 2 or 3")
  def transpose(x):
    return tf.transpose(x, indices)
  x = (x + transpose(x))/2
  e, v = tf.linalg.eigh(x)
  e = tf.maximum(e, 0)
  return tf.matmul(v, tf.expand_dims(e, -1)*transpose(v))


def _get_lhs_rhs(
    input_batch: InputBatch,
    factors: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
  """Computes noiseless sufficient statistics."""
  with tf.device("/device:cpu:0"):
    # Use shifted indices to [0, batch_size) to use less memory.
    # The LHS is [batch_size, dim, dim] and RHS is [batch_size, dim].
    indices = input_batch["shifted_indices"]
    values = input_batch["values"]
    weights = input_batch.get("weights")
    # The weights are used to scale x (the matrix of factors) and y (the
    # vector of labels)
    rowids = indices[:, 0]
    ragged_indices = tf.RaggedTensor.from_value_rowids(indices[:, 1], rowids)
    ragged_values = tf.RaggedTensor.from_value_rowids(
        weights*values if weights is not None else values, rowids)
    x = tf.gather(factors, ragged_indices)  # [num_rows, (row_length), dim]
    y = tf.expand_dims(ragged_values, 2)  # [num_rows, (row length), 1]
    if weights is not None:
      ragged_weights = tf.RaggedTensor.from_value_rowids(weights, rowids)
      w_x = tf.expand_dims(ragged_weights, 2)*x
    else:
      w_x = x
    # [num_rows, dim, dim]
    lhs = tf.matmul(w_x, x, transpose_a=True).to_tensor()
    # [num_rows, dim, 1]
    rhs = tf.matmul(x, y, transpose_a=True).to_tensor()
    return lhs, rhs


def _reg_tensor(
    input_batch: InputBatch,
    embedding_dim: int,
) -> tf.Tensor:
  """Returns a regularization tensor, to be added to the LHS or RHS.

  Args:
    input_batch: a slice of the input matrix.
    embedding_dim: the embedding dimension.
  Returns:
    Regularization tensor of shape [num_rows, embedding_dim, embedding_dim],
    where R[i, :, :] is an identity matrix scaled by input_batch.row_reg[i].
  """
  row_reg = input_batch.get("row_reg")
  if row_reg is None:
    row_reg = np.array([1], dtype=np.float32)
  eye = tf.linalg.eye(embedding_dim, embedding_dim)[None, :, :]
  return eye*row_reg[:, None, None]
