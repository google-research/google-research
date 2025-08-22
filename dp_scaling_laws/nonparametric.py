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

"""Fitting + interpolating scaling law data.

This file loads in accounting data and cross entropy loss data from CNS, and
uses interpolation to create two efficient functions:

1) Estimate the noise multiplier required for DP-SGD to achieve a given
 Epsilon, Iterations, and Sampling Probability.  This function is exact
 for the grid points it was evaluated on, and approximated in between those
 grid points.

2) Estimate the cross entropy loss of a model of a given size trained for a
 given number of Model Size, number of Iterations and Noise Batch Ratio.

Both of these fitted functions are well defined within the range of the data
that was collected.

The primary user-facing function this library exposes is `scaling_law_query`,
which uses the fitted functions to estimate the cross entropy loss of a variety
of training configurations that use the given { user, privacy, compute} budget.
"""

import functools
from typing import Callable

import numpy as np
import pandas as pd
import scipy

from dp_scaling_laws import semiparametric
from dp_scaling_laws import smoothing

_BASE_PATH = 'data'


_EXPERIMENT_PATH = '%s/results.csv' % _BASE_PATH
_AMPLIFIED_ACCOUNTING_PATH = '%s/amplified_dpsgd_sigmas.csv' % _BASE_PATH
_UNAMPLIFIED_ACCOUNTING_PATH = '%s/unamplified_dpsgd_sigmas.csv' % _BASE_PATH

_ACCOUNTING_COLUMNS = [
    'Iterations',
    'Sampling Probability',
    'Epsilon',
    'Delta',
    'Noise Multiplier',
]

_MODEL_SIZES = {
    'BertTiny': 4.5e6,
    'BertMini': 11.4e6,
    'BertSmall': 29e6,
    'BertMedium': 41e6,
    'BertBase': 109e6,
    'BertLarge': 335e6,
    'BertMega': 778e6,
}
_SEQUENCE_LENGTH = 512
_MAX_ITERATIONS = 8192000


def _smooth(
    df,
    smooth_by_iters = smoothing.isotonic,
    smooth_by_noise = smoothing.isotonic,
    smoothing_steps = 1,
):
  """Smooth the data along the Iterations and Noise Batch Ratio dimensions.

  This function takes a rolling average across the last 10 values of Iterations,
  and applies Isotonic regression along the Iterations and Noise Batch Ratio
  dimensions to ensure that cross entropy is monotonically decreasing with
  Iterations and monotonically increasing with Noise Batch Ratio.

  Args:
    df: A dataframe with columns for Iterations, Noise Batch Ratio, and Cross
      Entropy.
    smooth_by_iters: A function that takes an array and returns a smoothed
      array. e.g., so that values are monotonically decreasing.
    smooth_by_noise: A function that takes an array and returns a smoothed
      array. e.g., so that values are monotonically increasing.
    smoothing_steps: The number of times to apply the smoothing process.

  Returns:
    A dataframe with the same columns as the input, where the Cross Entropy
    values are appropriately smoothed.
  """

  def apply_smoothing(data):
    for _ in range(smoothing_steps):
      data = data.apply(smooth_by_iters).T[::-1].apply(smooth_by_noise)[::-1].T
    return data

  pd.DataFrame.custom_smoothing = apply_smoothing  # allows chaining below.

  return (
      df.set_index(['Iterations', 'Noise Batch Ratio'])[
          ['Cross Entropy']
      ].unstack()
      # discard learning rate warm up measurements.
      .loc[1000:]
      # Ensure xent is monotonically increasing w/ sigma and decreasing w/ T.
      .custom_smoothing()
      # Flatten the data to a single row per (Iterations, Noise Batch Ratio).
      .stack(future_stack=True)
  )


@functools.cache
def load_accounting_data(
    amplified = True,
    unamplified = False,
):
  """Loads the accounting data from CNS.

  By default, this loads both the amplified and unamplified data, and returns
  the minimum noise multiplier of the two.

  Args:
    amplified: If true, load the data from the amplified DP-SGD file.
    unamplified: If true, load the data from the unamplified DP-SGD file.

  Returns:
    A dataframe with the following columns:
      Iterations: The number of iterations.
      Sampling Probability: The sampling probability.
      Epsilon: The privacy parameter.
      Delta: The delta (fixed to 1e-8).
      Noise Multiplier: The noise multiplier required for DP-SGD to achieve
        (epsilon, delta)-DP when run for the given number of iterations and
        minibatch sampling probability.
  """
  amplified_sigmas = pd.read_csv(
      _AMPLIFIED_ACCOUNTING_PATH, names=_ACCOUNTING_COLUMNS
  )

  unamplified_sigmas = pd.read_csv(
      _UNAMPLIFIED_ACCOUNTING_PATH, names=_ACCOUNTING_COLUMNS
  )

  if amplified and unamplified:
    index = [x for x in _ACCOUNTING_COLUMNS if x != 'NOISE_MULTIPLIER']
    noise1 = unamplified_sigmas.set_index(index)
    noise2 = amplified_sigmas.set_index(index).reindex(noise1.index)
    sigmas = np.minimum(noise1, noise2.fillna(np.inf)).reset_index()
  elif amplified:
    sigmas = amplified_sigmas
  elif unamplified:
    sigmas = unamplified_sigmas
  else:
    raise ValueError('Must specify at least one of amplified or unamplified.')

  return sigmas.sort_values(by=_ACCOUNTING_COLUMNS)


def _fill_missing_nbrs(df):
  """Fills in missing noise batch ratios with an interpoloated value."""
  index = ['Model', 'Model Size', 'Iterations', 'Noise Batch Ratio']
  return (
      df.set_index(index)[['Cross Entropy']]
      .unstack(level='Noise Batch Ratio')
      .interpolate(axis=1)
      .stack()
      .reset_index()
  )


@functools.cache
def load_experimental_data(
    smooth_fn = _smooth,
):
  """Loads the experimental data from CNS.

  Args:
    smooth_fn: A function that takes a dataframe indexed by Iterations with
      columns for Noise Batch Ratio and returns a smoothed dataframe of the same
      form.

  Returns:
    A dataframe with the following columns:
      Model: The model name.
      Model Size: The number of parameters in the model.
      Iterations: The number of iterations.
      Noise Batch Ratio: The noise batch ratio.
      Cross Entropy: The (smoothed) cross entropy loss.
  """
  df = (
      pd.read_csv(_EXPERIMENT_PATH)
      # Apply smoothing independently to each (Model, Learning Rate)
      .groupby(['Model', 'Learning Rate'])
      .apply(smooth_fn)
      # Minimize over Learning Rate
      .groupby(['Iterations', 'Model', 'Noise Batch Ratio'])[['Cross Entropy']]
      .min()
      .reset_index()
  )
  df['Model Size'] = df.Model.map(_MODEL_SIZES)
  df = semiparametric.extrapolate_data(df)
  df = _fill_missing_nbrs(df)
  return df.sort_values(by=['Iterations', 'Model Size', 'Noise Batch Ratio'])


def interpolated_function(
    df, x_cols, y_col
):
  """Creates a function that interpolates the points in the dataframe."""
  df = df.sort_values(by=x_cols)
  xs = tuple(df[col].unique() for col in x_cols)
  y = df[y_col].values.reshape(tuple(x.size for x in xs))
  return scipy.interpolate.RegularGridInterpolator(
      # Note: cubic may be more precise in some cases, but can cause weird
      # behavior in others.
      xs,
      y,
      method='linear',
      bounds_error=False,
  )


@functools.cache
def construct_accounting_fn():
  """Construct a function for computing/estimating the DP-SGD noise multiplier.

  We use an interpolated function to estimate noise multiplier for efficiency.
  The noise multiplier is exact when it is evaluated one one of the following
  grid points:
    epsilon = 2^{-7}, ..., 2^{7}
    T = 100, 200, 300, ...., 128000
    q: 2^{-20}, ..., 1

  Returns:
    A (vectorized) function `sigma_sgd`with the following signature:

    def sigma_sgd(epsilon: float, T: int, q: float) -> float:
      ...

    The function returns the noise multiplier required for DP-SGD to achieve
    (epsilon, 1e-8)-DP when run for T iterations with sampling probability q.
  """
  dpsgd_sigmas = load_accounting_data()

  inputs = ['Epsilon', 'Iterations', 'Sampling Probability']

  log_sigma_sgd = interpolated_function(
      dpsgd_sigmas.apply(np.log),
      inputs,
      'Noise Multiplier',
  )

  return lambda data: np.exp(log_sigma_sgd(np.log(data[inputs].values)))


@functools.cache
def construct_xent_fn(
    private = True,
):
  """Construct a function for computing/estimating the cross entropy loss.

  This returned function is defined within the grid of data points collected
  experimentally, that is:
    2.4M <= model size <= 335M
    100 <= iterations <= 128000
    0.5^{23} <= noise batch ratio <= 0.5^6

  Args:
    private: If true, return a function that estimates the cross entropy loss
      for a private model with varying amounts of noise. Otherwise, return a
      function that estimates the cross entropy loss for a non-private model
      without noise.

  Returns:
    A function `xent` with the following signature:

    def xent(model_size: float, iters: int, noise_batch_ratio: float) -> float:
      ...

    This function returns the estimated cross entropy loss for a model of given
    size trained for the given number of iterations and noise batch ratio.
  """
  xents = load_experimental_data()

  if private:
    mask = xents['Noise Batch Ratio'] > 0
    inputs = ['Model Size', 'Iterations', 'Noise Batch Ratio']
  else:
    mask = xents['Noise Batch Ratio'] == 0
    inputs = ['Model Size', 'Iterations']

  log_xent_fn = interpolated_function(
      xents[mask][inputs + ['Cross Entropy']].apply(np.log),
      inputs,
      'Cross Entropy',
  )

  return lambda df: np.exp(log_xent_fn(np.log(df[inputs].values)))


def make_constant_compute_configs(
    compute_budget, model_sizes = None
):
  """Creates a grid of training configs that use a constant compute budget.

  Args:
    compute_budget: The compute budget, as measured by 6 * (Model Size) *
      (Iterations) * (Batch Size) * (Sequence Length)
    model_sizes: The optional model sizes to use.

  Returns:
    A dataframe with the following columns:
      Model Size: The number of model parameters.
      Batch Size: The (hypothetical) batch size.
      Iterations: The number of training steps.

    The product of each row is equal to the given compute budget, and rows
    are filtered to be within the range of model sizes and iterations that
    we have experimental data for.
  """
  if model_sizes is None:
    model_sizes = np.logspace(
        np.log2(1.01 * min(_MODEL_SIZES.values())),
        np.log2(0.99 * max(_MODEL_SIZES.values())),
        num=100,
        base=2,
    )
  batch_sizes = np.logspace(0, 25, base=2, num=101)
  model_sizes, batch_sizes = np.meshgrid(model_sizes, batch_sizes)

  df = pd.DataFrame({
      'Model Size': model_sizes.flatten(),
      'Batch Size': batch_sizes.flatten(),
  })
  df['Iterations'] = compute_budget / (
      6 * _SEQUENCE_LENGTH * df['Model Size'] * df['Batch Size']
  )
  mask = (df.Iterations >= 1000) & (df.Iterations <= _MAX_ITERATIONS)
  return df[mask].reset_index(drop=True)


def nonprivate_scaling_law_query(compute_budget):
  """Computes estimated loss under various configs for fixed compute budget."""
  xent_fn = construct_xent_fn(private=False)
  ans = make_constant_compute_configs(compute_budget)
  ans = ans[ans['Batch Size'] == 1024].copy()
  ans['Noise Batch Ratio'] = 0
  ans['Cross Entropy'] = xent_fn(ans)
  return ans.dropna().sort_values(by='Cross Entropy').reset_index(drop=True)


def batched_scaling_law_query(
    compute_configs,
):
  """Computes estimated losses under given compute configs.

  Args:
    compute_configs: A dataframe with columns for Model Size, Batch Size, and
      Iterations.

  Returns:
    A dataframe with columns for different training configurations and their
    estimated cross entropy loss.
  """
  sigma_sgd_fn = construct_accounting_fn()
  xent_fn = construct_xent_fn(private=True)
  ans = compute_configs.copy()
  ans['Epsilon'] = ans['Privacy Budget']
  ans['Compute Budget'] = (
      6
      * ans['Batch Size']
      * ans['Model Size']
      * ans['Iterations']
      * _SEQUENCE_LENGTH
  )
  ans['Sampling Probability'] = ans['Batch Size'] / ans['Data Budget']
  ans['Noise Multiplier'] = sigma_sgd_fn(ans)
  ans['Noise Batch Ratio'] = ans['Noise Multiplier'] / ans['Batch Size']
  ans['Cross Entropy'] = xent_fn(ans)
  del ans['Epsilon']
  return ans


def scaling_law_query(
    user_budget,
    privacy_budget,
    compute_budget,
    model_sizes = None,
):
  """Computes possible training configs and estimated losses under given budgets.

  Args:
    user_budget: The number of users.
    privacy_budget: The privacy parameter (epsilon).
    compute_budget: The compute budget, as measured by 6 * (Model Size) *
      (Iterations) * (Batch Size) * (Sequence Length)
    model_sizes: The optional model sizes to use.

  Returns:
    A dataframe with columns for different training configurations and their
    estimated cross entropy loss.
  """
  sigma_sgd_fn = construct_accounting_fn()
  xent_fn = construct_xent_fn(private=True)
  ans = make_constant_compute_configs(compute_budget, model_sizes)
  ans['Epsilon'] = privacy_budget
  ans['Compute Budget'] = compute_budget
  ans['Data Budget'] = user_budget
  ans['Sampling Probability'] = ans['Batch Size'] / user_budget
  ans['Noise Multiplier'] = sigma_sgd_fn(ans)
  ans['Noise Batch Ratio'] = ans['Noise Multiplier'] / ans['Batch Size']
  ans['Cross Entropy'] = xent_fn(ans)
  ans['Tokens'] = ans['Iterations'] * ans['Batch Size'] * _SEQUENCE_LENGTH
  ans['Token Model Ratio'] = ans['Tokens'] / ans['Model Size']
  ans = ans.rename(columns={'Epsilon': 'Privacy Budget'})
  ans = ans.dropna().sort_values(by='Cross Entropy').reset_index(drop=True)
  return ans
