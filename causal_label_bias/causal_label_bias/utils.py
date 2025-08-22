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

"""Utility functions.

Miscellaneous functions.
"""

from typing import Any, Optional, Sequence

from causal_label_bias import classifiers
from causal_label_bias import hparams
from causal_label_bias import types
import numpy as np
import pandas as pd
import sklearn
import sklearn.calibration
import sklearn.ensemble
import sklearn.linear_model
import sklearn.neural_network
import sklearn.pipeline


def logit(x):
  """Computes the logit function.

  Args:
    x: The numeric input to the function.

  Returns:
    The computed logit of x.
  """
  return np.log(x / (1 - x))


def get_sample_weight_kwargs(
    sample_weight = None,
):
  """Generates a dict that can be used for kwarg expansion to sample_weight calls for Scikit-learn models.

  Args:
    sample_weight: A numpy array of sample weights.

  Returns:
    A dict containing a kwargs and their values. For example: {'sample_weight':
    None}
  """

  return {'sample_weight': sample_weight}


def fit_models_df(
    source_df,
    target_df,
    outcome_key = 'y',
    outcome_key_target = 'y',
    features_keys = 'x',
    group_key = 'a',
    weight_key = None,
    stratified = False,
    model_type = 'logistic',
    model_cross_val = False,
    model_kwarg_dict = None,
    model_cross_val_param_grid = None,
    fit_calibration_model = True,
    calibration_model_type = 'logistic',
    calibration_model_cross_val = False,
    calibration_model_kwarg_dict = None,
    calibration_model_cross_val_param_grid = None,
):
  """Fits and evaluates a collection of models on data stored in dataframes.

  Arguments:
    source_df: A pd.DataFrame containing data used for model training. The
      dataframe must minimally contain numeric columns corresponding to features
      (feature_keys), outcomes (outcome_key), and group membership (group_key).
    target_df: A pd.DataFrame containing data used for model evaluation. The
      data follows the same format as source_df.
    outcome_key: A string indicating the name of the column used as the label
      that is the target of prediction. Currently, only binary outcomes coded as
      0 and 1 are supported.
    outcome_key_target: A string indicating the name of the column used as the
      label for model evaluation.
    features_keys: A string or iterable of strings indicating the name(s) of the
      columns used as features.
    group_key: A string indicating the column indicating the group variable.
      Currently, only binary groups coded as 0 and 1 are supported.
    weight_key: An optional string indicating the column that defines sample
      weights used during model training.
    stratified: If true, a separate model is trained for each group. Defaults to
      False.
    model_type: A string name of member of types.ModelType that determines the
      class of model used to predict the label.
    model_cross_val: If true, fit the model for the outcome using cross
      validation.
    model_kwarg_dict: A dict of keyword arguments passed to the model
      constructor.
    model_cross_val_param_grid: A dict or list of dicts with lists of parameters
      as values, matching the 'param_grid' input of
      sklearn.model_selection.GridSearchCV. If None, uses
      hparams.get_classifier_default_hparams to define default grid.
    fit_calibration_model: If true, a calibration curve is fit to the data in
      target_df for each group.
    calibration_model_type: A string name of member of types.ModelType that
      determines the model class used to fit calibration curves.
    calibration_model_cross_val: If true, fit the model for the calibration
      curve using cross validation.
    calibration_model_kwarg_dict: A dict of keyword arguments passed to the
      calibration model constructor.
    calibration_model_cross_val_param_grid: A dict or list of dicts with lists
      of parameters as values, matching the 'param_grid' input of
      sklearn.model_selection.GridSearchCV. If None, uses
      hparams.get_classifier_default_hparams to define default grid.

  Returns:
    A dict keyed by group with dict values.
    The inner dicts contain, for each group, the fitted model and numpy arrays
    corresponding to the features, labels, and predictions.
    If fit_calibration_model is true, then a fitted calibration curve will be
    present.
    For example:
    {
      0: {'model': ...,
          'features': ...,
          'outcomes': ...,
          'pred_probs': ...,
          'group': ...,
          'calibration_model': ...,
          'calibration_curve': ...
          },
      1: {...}
    }
    '''
  """

  if isinstance(features_keys, str):
    features_keys = [features_keys]

  model_kwarg_dict = model_kwarg_dict or {}

  # Fit population-level models
  features_population = source_df[features_keys].values
  model = None
  if not stratified:
    outcomes_population = source_df[outcome_key]
    model = classifiers.get_classifier(model_type, **model_kwarg_dict)
    if not model_cross_val:
      model.fit(
          features_population,
          outcomes_population,
          sample_weight=None
          if weight_key is None
          else source_df[weight_key].values,
      )
    else:
      params = (
          hparams.get_classifier_default_hparams(model_type, cross_val=True)
          if model_cross_val_param_grid is None
          else model_cross_val_param_grid
      )
      grid = sklearn.model_selection.GridSearchCV(
          model,
          param_grid=params,
          scoring='neg_log_loss',
      )
      grid.fit(
          features_population,
          outcomes_population,
          **get_sample_weight_kwargs(
              sample_weight=None
              if weight_key is None
              else source_df[weight_key].values,
          ),
      )
      model = grid.best_estimator_

  # Create a dict containing models for each group
  model_dict = {}
  for group_id in source_df[group_key].unique():
    model_dict[group_id] = {
        'model': classifiers.get_classifier(model_type, **model_kwarg_dict),
    }
    group_df = source_df.query(f'{group_key} == @group_id')
    features = group_df[features_keys].values
    outcomes = group_df[outcome_key].values
    if stratified:
      if not model_cross_val:
        model_dict[group_id]['model'].fit(
            features,
            outcomes,
            **get_sample_weight_kwargs(
                sample_weight=None
                if weight_key is None
                else group_df[weight_key].values,
            ),
        )
      else:
        grid = sklearn.model_selection.GridSearchCV(
            model_dict[group_id]['model'],
            param_grid=hparams.get_classifier_default_hparams(  # pylint: disable=g-long-ternary
                model_type, cross_val=True
            )
            if model_cross_val_param_grid is None
            else model_cross_val_param_grid,
            scoring='neg_log_loss',
        )
        grid.fit(
            features,
            outcomes,
            **get_sample_weight_kwargs(
                sample_weight=None
                if weight_key is None
                else group_df[weight_key].value,
            ),
        )
        model_dict[group_id]['model'] = grid.best_estimator_
    else:
      model_dict[group_id]['model'] = model

    # Extract target data
    group_df_target = target_df.query(f'{group_key} == @group_id')
    features_target = group_df_target[features_keys].values
    outcomes_target = group_df_target[outcome_key_target].values
    if model_dict[group_id]['model'] is None:
      raise ValueError(
          f'cannot evaluate model for group {group_id} because model is None'
      )
    pred_probs_target = model_dict[group_id]['model'].predict_proba(
        features_target
    )[:, -1]
    group_target = group_df_target[group_key].values

    # Sort by pred_probs_target
    sort_ids = np.argsort(pred_probs_target)
    pred_probs_target = pred_probs_target[sort_ids].reshape(-1, 1)
    outcomes_target = outcomes_target[sort_ids]
    features_target = features_target[sort_ids]
    group_target = group_target[sort_ids]

    # Pack data to model_dict
    model_dict[group_id]['pred_probs'] = pred_probs_target
    model_dict[group_id]['outcomes'] = outcomes_target
    model_dict[group_id]['features'] = features_target
    model_dict[group_id]['group'] = group_target

    # Fit a calibration curve to the data
    if fit_calibration_model:
      if calibration_model_type == 'sklearn':
        (
            model_dict[group_id]['calibration_curve_y'],
            model_dict[group_id]['calibration_curve_x'],
        ) = sklearn.calibration.calibration_curve(
            outcomes_target, pred_probs_target, n_bins=25, strategy='quantile'
        )
      else:
        model_dict[group_id]['calibration_model'] = classifiers.get_classifier(
            calibration_model_type, **calibration_model_kwarg_dict
        )

        if not calibration_model_cross_val:
          model_dict[group_id]['calibration_model'].fit(
              logit(pred_probs_target), outcomes_target
          )
        else:
          if calibration_model_cross_val_param_grid is None:
            calibration_model_cross_val_param_grid = (
                hparams.get_classifier_default_hparams(
                    calibration_model_type, cross_val=True
                )
            )
          grid = sklearn.model_selection.GridSearchCV(
              model_dict[group_id]['calibration_model'],
              param_grid=calibration_model_cross_val_param_grid,
              scoring='neg_log_loss',
          )
          grid.fit(logit(pred_probs_target), outcomes_target)
          model_dict[group_id]['calibration_model'] = grid.best_estimator_
        model_dict[group_id]['calibration_curve_y'] = model_dict[group_id][
            'calibration_model'
        ].predict_proba(logit(pred_probs_target))[:, -1]
        model_dict[group_id]['calibration_curve_x'] = model_dict[group_id][
            'pred_probs'
        ]

  return model_dict
