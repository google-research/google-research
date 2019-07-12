# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Utils related to AR and ARMA models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import sm_exceptions
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.tsatools import lagmat


def fit_ar(outputs, inputs, guessed_dim):
  """Fits an AR model of order p = guessed_dim.

  Args:
    outputs: Array with the output values from the LDS.
    inputs: Array with exogenous inputs values.
    guessed_dim: Guessed hidden dimension.

  Returns:
    - Fitted AR coefficients.
  """
  if outputs.shape[1] > 1:
    # If there are multiple output dimensions, fit autoregressive params on
    # each dimension separately and average.
    params_list = [
        fit_ar(outputs[:, j:j+1], inputs, guessed_dim) \
        for j in xrange(outputs.shape[1])]
    return np.mean(
        np.concatenate([a.reshape(1, -1) for a in params_list]), axis=0)
  if inputs is None:
    model = AR(outputs).fit(ic='bic', trend='c', maxlag=guessed_dim, disp=0)
    arparams = np.zeros(guessed_dim)
    arparams[:model.k_ar] = model.params[model.k_trend:]
    return arparams
  else:
    model = ARMA(outputs, order=(guessed_dim, 0), exog=inputs)
    try:
      arma_model = model.fit(start_ar_lags=guessed_dim, trend='c', disp=0)
      return arma_model.arparams
    except (ValueError, np.linalg.LinAlgError) as e:
      warnings.warn(str(e), sm_exceptions.ConvergenceWarning)
      return np.zeros(guessed_dim)


def _fit_arma_iter(outputs, inputs, p, q, r, l2_reg=0.0):
  """Iterative regression for estimating AR params in ARMAX(p, q, r) model.

  The iterative AR regression process provides consistent estimates for the
  AR parameters of an ARMAX(p, q, r) model after q iterative steps.

  It first fits an ARMAX(p, 0, r) model with least squares regression, then
  ARMAX(p, 1, r), and so on, ..., til ARMAX(p, q, r). At the i-th step, it
  fits an ARMAX(p, i, r) model, according to estimated error terms from the
  previous step.

  For description of the iterative regression method, see Section 2 of
  `Consistent Estimates of Autoregressive Parameters and Extended Sample
  Autocorrelation Function for Stationary and Nonstationary ARMA Models` at
  https://www.jstor.org/stable/2288340.

  The implementation here is a generalization of the method mentioned in the
  paper. We adapt the method for multidimensional outputs, exogenous inputs, nan
  handling, and also add regularization on the MA parameters.

  Args:
    outputs: Array with the output values from the LDS, nans allowed.
    inputs: Array with exogenous inputs values, nans allowed. Could be None.
    p: AR order, i.e. max lag of the autoregressive part.
    q: MA order, i.e. max lag of the error terms.
    r: Max lag of the exogenous inputs.
    l2_reg: L2 regularization coefficient, to be applied on MA coefficients.

  Returns:
    Fitted AR coefficients.
  """
  if outputs.shape[1] > 1:
    # If there are multiple output dimensions, fit autoregressive params on
    # each dimension separately and average.
    params_list = [
        _fit_arma_iter(outputs[:, j:j+1], inputs, p, q, r, l2_reg=l2_reg) \
        for j in xrange(outputs.shape[1])]
    return np.mean(
        np.concatenate([a.reshape(1, -1) for a in params_list]), axis=0)
  # We include a constant term in regression.
  k_const = 1
  # Input dim. If inputs is None, then in_dim = 0.
  in_dim = 0
  if inputs is not None:
    in_dim = inputs.shape[1]
    # Lag the inputs to obtain [?, r], column j means series x_{t-j}.
    # Use trim to drop rows with unknown values both at beginning and end.
    lagged_in = np.concatenate(
        [lagmat(inputs[:, i], maxlag=r, trim='both') for i in xrange(in_dim)],
        axis=1)
    # Since we trim in beginning, the offset is r.
    lagged_in_offset = r
  # Lag the series itself to p-th order.
  lagged_out = lagmat(outputs, maxlag=p, trim='both')
  lagged_out_offset = p
  y = outputs
  y_offset = 0
  # Estimated residuals, initialized to 0.
  res = np.zeros_like(outputs)
  for i in xrange(q + 1):
    # Lag the residuals to i-th order in i-th iteration.
    lagged_res = lagmat(res, maxlag=i, trim='both')
    lagged_res_offset = y_offset + i
    # Compute offset in regression, since lagged_in, lagged_out, and lagged_res
    # have different offsets. Align them.
    if inputs is None:
      y_offset = max(lagged_out_offset, lagged_res_offset)
    else:
      y_offset = max(lagged_out_offset, lagged_res_offset, lagged_in_offset)
    y = outputs[y_offset:, :]
    # Concatenate all variables in regression.
    x = np.concatenate([
        lagged_out[y_offset - lagged_out_offset:, :],
        lagged_res[y_offset - lagged_res_offset:, :]
    ],
                       axis=1)
    if inputs is not None:
      x = np.concatenate([lagged_in[y_offset - lagged_in_offset:, :], x],
                         axis=1)
    # Add constant term as the first variable.
    x = add_constant(x, prepend=True)
    if x.shape[1] < k_const + in_dim * r + p + i:
      raise ValueError('Insufficient sequence length for model fitting.')
    # Drop rows with nans.
    arr = np.concatenate([y, x], axis=1)
    arr = arr[~np.isnan(arr).any(axis=1)]
    y_dropped_na = arr[:, 0:1]
    x_dropped_na = arr[:, 1:]
    # Only regularize the MA part.
    alpha = np.concatenate(
        [np.zeros(k_const + in_dim * r + p), l2_reg * np.ones(i)], axis=0)
    # When L1_wt = 0, it's ridge regression.
    olsfit = OLS(y_dropped_na, x_dropped_na).fit_regularized(
        alpha=alpha, L1_wt=0.0)
    # Update estimated residuals.
    res = y - np.matmul(x, olsfit.params.reshape(-1, 1))
  if len(olsfit.params) != k_const + in_dim * r + p + q:
    raise ValueError('Expected param len %d, got %d.' %
                     (k_const + in_dim * r + p + q, len(olsfit.params)))
  if q == 0:
    return olsfit.params[-p:]
  return olsfit.params[-(p + q):-q]


def fit_arma_iter(outputs, inputs, guessed_dim, l2_reg=0.0):
  """Iterative regression for ARMAX(p, q, r) model, where p=q=r=guessed_dim.

  Args:
    outputs: Array with the output values from the LDS.
    inputs: Array with exogenous inputs values.
    guessed_dim: Guessed hidden dimension.
    l2_reg: L2 regularization coefficient.

  Returns:
    Fitted AR coefficients.
  """
  return _fit_arma_iter(
      outputs,
      inputs,
      p=guessed_dim,
      q=guessed_dim - 1,
      r=guessed_dim - 1,
      l2_reg=l2_reg)


def fit_arma_mle(outputs, inputs, guessed_dim, method='css-mle'):
  """Fits an ARMA model of order (p=guessed_dim, q=gussed_dim).

  First try using the statsmodel.ARMA default initialization of start params.

  If the start params are unstable (roots outside unit circle), try start params
  with only AR params but no MA.

  If the AR only start params result in SVD failure in optimizer, returns the AR
  only params as the fitted params and returns no model.

  Args:
    outputs: Array with the output values from the LDS.
    inputs: Array of exogenous inputs.
    guessed_dim: Guessed hidden dimension.
    method: 'ccs-mle' or 'ccs' or 'mle', fit method in statsmodel package.

  Returns:
    - Fitted AR coefficients.
  """
  p = guessed_dim
  q = guessed_dim
  model = ARMA(outputs, order=(p, q), exog=inputs)
  try:
    arma_model = model.fit(start_ar_lags=None, trend='c', method=method, disp=0)
    return arma_model.arparams
  except (ValueError, np.linalg.LinAlgError) as e:
    warnings.warn(str(e), sm_exceptions.ConvergenceWarning)
    return np.zeros(p)


def get_eig_from_arparams(arparams):
  eigs = np.roots(np.r_[1, -arparams])
  return eigs[np.argsort(eigs.real)[::-1]]
