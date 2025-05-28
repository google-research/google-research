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

"""Utilities for semiparametric scaling law data anslysis."""

import numpy as np
import pandas as pd
import scipy


def _find_convergence_point(data_to_fit):
  """Find the convergence point of the series under a parametric assumption.

  L = exp(m log(T) + b) + E

  Args:
    data_to_fit: A Series indexed by iteration with values for Cross Entropy.

  Returns:
    The predicted point where the series converges.
  """
  f = lambda t, m, b, E: np.exp(m * np.log(t) + b) + E
  p0 = (-0.3, 1.94, data_to_fit.min() * 0.9)
  pstar = scipy.optimize.curve_fit(
      f, data_to_fit.index.values, data_to_fit.values, p0=p0, maxfev=1000000
  )[0]
  return pstar[-1]


def _extrapolate_iters(
    data_to_fit, convergence_xent
):
  """Extrapolate the cross entropy for future iterations.

  Args:
    data_to_fit: A Series indexed by iteration with values for Cross Entropy.
    convergence_xent: The cross entropy at the convergence point.

  Returns:
    A new series in the same format as data_to_fit, but with extrapolated values
    for the predictd cross entropy of future iterations.
  """
  f = lambda t, m, b: np.exp(m * np.log(t) + b) + convergence_xent
  p0 = (-0.3, 1.94)
  slope, intercept = scipy.optimize.curve_fit(
      f, data_to_fit.index.values, data_to_fit.values, p0=p0, maxfev=1000000
  )[0]
  ts = np.array([16000 * 2**k for k in range(10)])
  ys = f(ts, slope, intercept)
  return pd.Series(index=ts, data=ys)


def extrapolate_data(
    data,
):
  """Extrapolate the cross entropy for future iterations.

  Args:
    data: A dataframe with columns for Model, Model Size, Iterations, Noise
      Batch Ratio, and Cross Entropy.

  Returns:
    A dataframe with the same columns as the input, but with extrapolated values
    for the predicted cross entropy of future iterations.
  """
  # We begin by estimating E*, the minimum cross entropy achievable by a model
  # when trained without noise.
  convergence_xents = (
      data[data['Noise Batch Ratio'] == 0]
      .groupby('Model')
      .apply(
          lambda d: _find_convergence_point(
              d.set_index('Iterations')['Cross Entropy']
          )
      )
      .sort_values()
  )

  # Next we extrapolate the cross entropy for a given model and noise batch
  # ratio, assuming the same convergence point as the model without noise.
  # If trained for long enough, this should be the case.
  def extrapolate(group):
    model = group.name[0]
    xent_star = convergence_xents[model]
    data_to_fit = group.set_index('Iterations')['Cross Entropy']
    # We only fit the model to the suffix of cross entropies from 16K - 128K
    # iterations, since we will use the (smoothed) raw data when we have it.
    extrapolated_data = _extrapolate_iters(data_to_fit.loc[16000:], xent_star)
    extrapolated_data.index.name = 'Iterations'
    # We concatenate the raw data (which is defined up to T=128K) with the
    # extrapolated data, from T=256K onwards.
    result = pd.concat([data_to_fit, extrapolated_data.loc[256000:]])
    return result

  return (
      data.groupby(['Model', 'Model Size', 'Noise Batch Ratio'])
      .apply(extrapolate)
      .stack()
      .reset_index()
      .rename(columns={0: 'Cross Entropy'})
  )
