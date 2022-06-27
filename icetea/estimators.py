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

"""Implementation of treatment effect estimators.

Estimator Functions
tau, mse, bias = estimator_oaxaca()
tau, mse, bias = estimator_aipw()

source https://keras.io/examples/keras_recipes/tfrecord/
"""

import numpy as np
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection


def default_param_method():
  param_method = {}
  param_method['base_model'] = linear_model.LinearRegression()
  param_method['metric'] = metrics.mean_squared_error
  param_method['prop_score'] = linear_model.LogisticRegression()
  param_method['image'] = False
  return param_method


def fit_model(x_z, y_z, model, x, metric, param_grid=None):
  """Function to fit the base-learner, m_z model.

  m_z is trained in x_z, and returns predictions of x.

  Args:
    x_z: subset of covariates where Z = z
    y_z: subset of targets where Z = z
    model: base learner
    x: full covariates dataset
    metric: evaluation metric
    param_grid: dict with parameters for grid search
  Returns:
    counterfactual: np.array, x predicted on m_z
    mse: float,
    bias: float
  """
  if param_grid is not None:
    param_grid = update_param_grid(param_grid, x_z.shape)
    gscv = model_selection.GridSearchCV(model, param_grid=param_grid,
                                        scoring='neg_mean_squared_error')
    gscv.fit(X=x_z, y=y_z)
    mse = metric(y_z, gscv.predict(x_z))
    bias = (gscv.predict(x_z)- y_z).mean()
    counterfactual = gscv.predict(x)
    return counterfactual, mse, bias
  else:
    model.fit(x_z, y_z)
    mse = metric(y_z, model.predict(x_z))
    error = np.subtract(model.predict(x_z), y_z).mean()
    counterfactual = model.predict(x)
    return counterfactual, mse, error


def update_param_grid(param_grid, sizes):
  """Update parameters of the grid search.

  Args:
    param_grid: current grid search dictionary
    sizes: dataset.shape
  Returns:
    param_grid: updated grid search dictionary
  """
  if 'h_units' in param_grid.keys():
    # NN and simple_linear data
    if sizes[1] > 1:
      param_grid['h_units'] = [i for i in param_grid['h_units'] if i < sizes[1]]
    else:
      param_grid['h_units'] = [1]

    if sizes[0] > 10001:
      param_grid['batch_size'] = [600]
  return param_grid


def estimator_oaxaca(data,
                     param_method=None,
                     param_grid=None):
  """Estimate treatment effect with oaxaca estimator.

  Args:
    data: DataSimulation Class
    param_method: dict with method's parameters
    param_grid: dict with grid search parameters
  Returns:
    t_estimated: estimated treatment effect using the oaxaca-blinder method
    metric: list, metric on control and treated groups
    bias: list, bias on control and treated group
    var: float, estimator variance.
  """
  # Defining basic parameters.
  if param_method is None:
    param_method = default_param_method()

  if param_method.get('image', False):
    # Fitting two models, one per treatment value.
    pred_control, mse_control, e_control, treated, y0_c = fit_base_image_models(
        data.dataset_control,
        param_method['base_model'],
        data.dataset_all,
        quick=False)
    pred_treated, mse_treated, e_treated, treated, y1_c = fit_base_image_models(
        data.dataset_treated,
        param_method['base_model'],
        data.dataset_all,
        quick=False)
    # Fill-in unobserved // counterfactual.
    y0_c[treated] = pred_control[treated]
    y1_c[~treated] = pred_treated[~treated]

    # Variance.
    n0 = len(treated) - treated.sum()
    n1 = treated.sum()
    var = mse_control / n0 + mse_treated / n1
  else:
    # Subset treatment and control group.
    y0, x0 = data.outcome_control, data.covariates_control
    y1, x1 = data.outcome_treated, data.covariates_treated

    # Fitting two models, one per treatment value.
    pred_control, mse_control, e_control = fit_model(
        x0, y0, param_method['base_model'], data.covariates,
        param_method['metric'], param_grid)
    pred_treated, mse_treated, e_treated = fit_model(
        x1, y1, param_method['base_model'], data.covariates,
        param_method['metric'], param_grid)

    # Fill-in unobserved // counterfactual.
    treated = data.treatment == 1
    y0_c = data.outcome.copy()
    y1_c = data.outcome.copy()
    y0_c[treated] = pred_control[treated]
    y1_c[~treated] = pred_treated[~treated]

    # Variance.
    var = mse_control / len(y0) + mse_treated / len(y1)

  # Estimate treatment effect.
  t_estimated = (y1_c - y0_c).mean()

  # Bias.
  bias = [e_control, e_treated]

  # Metrics.
  metric = [mse_control, mse_treated]

  return t_estimated, metric, bias, var


def estimator_aipw(data, param_method=None, param_grid=None):
  """Estimate treatment effect with aipw estimator.

  Args:
    data: DataSimulation Class
    param_method: dict with method's parameters
    param_grid: dict with grid search parameters
  Returns:
    t_estimated: estimated treatment effect using the aipw method
    metric: list, metric on control and treated groups
    bias: list, bias on control and treated group
    var: float, estimator variance.
  """
  if param_method is None:
    param_method = default_param_method()

  if param_method.get('image', False):
    # Fitting two models, one per treatment value.
    pred_control, mse_control, e_control, t, y0 = fit_base_image_models(
        data.dataset_control,
        param_method['base_model'],
        data.dataset_all,
        quick=False)
    pred_treated, mse_treated, e_treated, t, y1 = fit_base_image_models(
        data.dataset_treated,
        param_method['base_model'],
        data.dataset_all,
        quick=False)
    x = data.dataset_all_ps
  else:
    # Subset treatment group.
    data.split()
    y0, x0 = data.outcome_control, data.covariates_control
    y1, x1 = data.outcome_treated, data.covariates_treated

    # Fitting two models, one per treatment value.
    pred_control, mse_control, e_control = fit_model(
        x0, y0, param_method['base_model'], data.covariates,
        param_method['metric'], param_grid)
    pred_treated, mse_treated, e_treated = fit_model(
        x1, y1, param_method['base_model'], data.covariates,
        param_method['metric'], param_grid)
    x = data.covariates

  # Propensity score using a logistic regression.
  prop_score = param_method['prop_score']
  prop_score.fit(x)
  e = prop_score.predict_proba(x)

  # Estimating the treatment effect.
  pred_dif = (pred_treated - pred_control)
  try:
    z = data.treatment.ravel()
    y = data.outcome.ravel()
  except AttributeError:
    z = t*1
    y = y0

  sample_size = len(y)

  residual_treated = (z * (y - pred_treated))
  residual_treated = (residual_treated / e[:, 1].ravel())
  residual_control = ((1 - z) * (y - pred_control))
  residual_control = (residual_control / e[:, 0].ravel())

  residual_dif = (residual_treated - residual_control)
  t_estimated = np.mean(np.add(pred_dif, residual_dif))

  # Variance.
  var = np.add(pred_dif, residual_dif)
  var = np.subtract(var, t_estimated)
  var = var**2
  var = var.sum() / (sample_size - 1)
  var = var / sample_size

  # Bias.
  bias = [e_control, e_treated]

  return t_estimated, [mse_control, mse_treated], bias, var


def _prediction_image_models(data, model, quick=False):
  """Predicts the outcome on the full data.

  Args:
    data: tf.data.Dataset.
    model: fitted model.
    quick: predict in a subset of data.
  Returns:
    arrays with predicted outcome, observed outcome, and treat. assignment.
  """
  y_pred = []
  y_sim = []
  t = []
  for i, (batch_x, batch_y, batch_t) in enumerate(data):
    y_pred.append(model.predict_on_batch(batch_x))
    y_sim.append(batch_y.numpy())
    t.append(batch_t.numpy())
    if quick and i > 2000:
      break

  y_pred_flat = np.concatenate(y_pred).ravel()
  y_sim_flat = np.concatenate(y_sim).ravel()
  t_flat = np.concatenate(t).ravel()

  return np.array(y_pred_flat), np.array(y_sim_flat), np.array(t_flat)


def fit_base_image_models(data, model, dataset_all, quick=False):
  """Predicts the outcome on the full data.

  Args:
    data: tf.data.Dataset (control or treated).
    model: fitted model.
    dataset_all: tf.data.Dataset (all, for prediction).
    quick: predict in a subset of data_all.
  Returns:
    y_pred: predictions on data_all
    mse: array with mse on the control and treated group
    bias: array with bias on the control and treated group
    t: treatment assigment on data_all
    y_sim: observed outcome on data_all
  """
  epochs = 40
  steps = 60
  quick = True
  history = model.fit(data, steps_per_epoch=steps, epochs=epochs, verbose=2)
  try:
    mse = history.history['mean_squared_error'][-1]
  except KeyError:
    mse = history.history['mse'][-1]

  bias = history.history['mae'][-1]
  y_pred, y_sim, t = _prediction_image_models(dataset_all, model, quick=quick)
  return y_pred, mse, bias, t, y_sim

