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

from collections.abc import Iterable
from typing import Any, Optional, Protocol, Tuple

from causal_evaluation.causal_evaluation import types
from causal_evaluation.causal_evaluation.experiments import hparams
from causal_evaluation.causal_evaluation.models import classifiers
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy
import sklearn
import sklearn.calibration
import sklearn.ensemble
import sklearn.linear_model
import sklearn.neural_network
import sklearn.pipeline
import statsmodels.stats.proportion


def get_sample_weight_kwargs(
    model_type = 'logistic',
    sample_weight = None,
):
  """Generates a dict that can be used for kwarg expansion to sample_weight calls for Scikit-learn models.

  This function solves for an issue where sklearn.pipeline.Pipeline objects
  cannot take sample_weight arguments in their fit method. For the model_types
  implemented in this version of the package, this function returns a dict
  containing a single key 'sample_weight' with value None.

  Args:
    model_type: A string name of a member of types.ModelType.
    sample_weight: A numpy array of sample weights.

  Returns:
    A dict containing a kwargs and their values. For example: {'sample_weight':
    None}
  """
  if model_type == types.ModelType.LOGISTIC_REGRESSION:
    kwarg_keys = ['sample_weight']
  elif model_type == types.ModelType.RIDGE:
    kwarg_keys = ['sample_weight']
  elif model_type == types.ModelType.GRADIENT_BOOSTING:
    kwarg_keys = ['sample_weight']
  else:
    kwarg_keys = ['sample_weight']
  return {key: sample_weight for key in kwarg_keys}


def get_squeezed_df(data_dict):
  """Converts a dict of numpy arrays into a DataFrame, extracting columns of arrays into separate DataFrame columns.

  Args:
    data_dict: A dict of numpy arrays.

  Returns:
    A DataFrame with the same data as the dict, but with columns of arrays
    extracted into separate DataFrame columns.
  """
  temp = {}
  for key, value in data_dict.items():
    squeezed_array = np.squeeze(value)
    if len(squeezed_array.shape) == 1:
      temp[key] = squeezed_array
    elif len(squeezed_array.shape) > 1:
      for i in range(value.shape[1]):
        temp[f'{key}_{i}'] = np.squeeze(value[:, i])
  return pd.DataFrame(temp)


def fit_model(
    features,
    labels,
    sample_weight = None,
    model_type = 'logistic',
    model_cross_val = False,
    model_kwarg_dict = None,
    model_cross_val_param_grid = None,
    cv_kwarg_dict = None,
):
  """Fits a model, optionally using cross validation.

  Args:
    features: A numpy array containing the features.
    labels: A numpy array containing the labels.
    sample_weight: An optional numpy array containing the sample weights.
    model_type: A string name of member of types.ModelType that determines the
      class of model used to predict the label.
    model_cross_val: If true, fit the model using cross validation.
    model_kwarg_dict: A dict of keyword arguments passed to the model
      constructor.
    model_cross_val_param_grid: A dict or list of dicts, matching the
      'param_grid' input of sklearn.model_selection.GridSearchCV. If None, uses
      hparams.get_classifier_default_hparams to define default grid.
    cv_kwarg_dict: A dict of keyword arguments passed to GridSearchCV.

  Returns:
    A sklearn model.
  """

  model_kwarg_dict = model_kwarg_dict or {}
  cv_kwarg_dict = cv_kwarg_dict or {}
  model = classifiers.get_classifier(model_type, **model_kwarg_dict)
  if not model_cross_val:
    model.fit(
        features,
        labels,
        sample_weight=sample_weight,
    )
    return model
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
        **cv_kwarg_dict,
    )
    grid.fit(
        features,
        labels,
        **get_sample_weight_kwargs(
            model_type,
            sample_weight=sample_weight,
        ),
    )
    return grid.best_estimator_


def fit_cross_val_predict(
    features,
    labels,
    sample_weight = None,
    model_type = 'logistic',
    model_cross_val = False,
    model_kwarg_dict = None,
    model_cross_val_param_grid = None,
    inner_cv_kwarg_dict = None,
    outer_cv_kwarg_dict = None,
):
  """Fits a model using cross validation and returns the predictions for each held-out fold.

  This function is a wrapper around sklearn.model_selection.cross_val_predict.

  Args:
    features: A numpy array containing the features.
    labels: A numpy array containing the labels.
    sample_weight: An optional numpy array containing the sample weights.
    model_type: A string name of member of types.ModelType that determines the
      class of model used to predict the label.
    model_cross_val: If true, fit the model using cross validation.
    model_kwarg_dict: A dict of keyword arguments passed to the model
      constructor.
    model_cross_val_param_grid: A dict or list of dicts, matching the
      'param_grid' input of sklearn.model_selection.GridSearchCV. If None, uses
      hparams.get_classifier_default_hparams to define default grid.
    inner_cv_kwarg_dict: A dict of keyword arguments passed to the inner cross
      validation.
    outer_cv_kwarg_dict: A dict of keyword arguments passed to the outer cross
      validation.

  Returns:
    A numpy array containing the predicted probabilities.
  """

  model_kwarg_dict = model_kwarg_dict or {}
  inner_cv_kwarg_dict = inner_cv_kwarg_dict or {}
  outer_cv_kwarg_dict = outer_cv_kwarg_dict or {}

  model = classifiers.get_classifier(model_type, **model_kwarg_dict)
  if not model_cross_val:
    grid = model
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
        **inner_cv_kwarg_dict,
    )
  return sklearn.model_selection.cross_val_predict(
      grid,
      features,
      labels,
      method='predict_proba',
      **outer_cv_kwarg_dict,
      params={
          **get_sample_weight_kwargs(
              model_type,
              sample_weight=sample_weight,
          )
      },
  )


def fit_model_stratified(
    features,
    labels,
    group = None,
    sample_weight = None,
    **kwargs,
):
  """Fits a model in a stratified manner, with one model per group.

  Args:
    features: A numpy array containing the features.
    labels: A numpy array containing the labels.
    group: A numpy array containing the group membership.
    sample_weight: An optional numpy array containing the sample weights.
    **kwargs: Keyword arguments passed to fit_model.

  Returns:
    A dict keyed by group containing the fitted models.
  """
  features_series = pd.Series(np.arange(features.shape[0])).map(
      lambda x: features[x].reshape(1, -1)
  )
  data_df = pd.DataFrame(
      {'features': features_series, 'labels': labels, 'group': group}
  )
  if sample_weight is not None:
    data_df['sample_weight'] = sample_weight

  model_dict = {}
  for group_name, group_df in data_df.groupby('group'):
    model_dict[group_name] = fit_model(
        np.concatenate(group_df['features'].values, axis=0),
        group_df['labels'].values,
        sample_weight=group_df['sample_weight'].values
        if 'sample_weight' in group_df
        else None,
        **kwargs,
    )
  return model_dict


def array_to_series(
    the_array, add_dim = False
):
  """Converts a numpy array to a Pandas series.

  This is designed to convert 2-D numpy arrays to 1-D Pandas series. For an NxM
  array, this will return a series of length N, where each element is an
  M-length
  array.

  Args:
    the_array: A numpy array.
    add_dim: A bool indicating whether to add an extra dimension to the array.
      If True, the array will be converted to a 2-D array. If False, the array
      will be returned as a 1-D array.

  Returns:
    A pandas series containing the same data as the numpy array.
  """
  result = pd.Series(np.arange(the_array.shape[0]))
  if add_dim:
    result = result.map(lambda x: the_array[x].reshape(1, -1))
  else:
    result = result.map(lambda x: the_array[x])
  return result


def predict_proba_stratified(
    features,
    model_dict,
    group = None,
    label_size = 2,
):
  """Predicts the probability of the label with stratified models.

  Args:
    features: A numpy array containing the features.
    model_dict: A dict keyed by group containing the fitted models.
    group: A numpy array containing the group membership.
    label_size: The number of labels in the output space of the model.

  Returns:
    A numpy array containing the predicted probabilities.
  """
  pred_probs_stratified = np.zeros((features.shape[0], label_size))
  for group_id in np.unique(group):
    group_filter = group == group_id
    pred_probs_group = model_dict[group_id].predict_proba(
        features[group_filter, :]
    )
    pred_probs_stratified[group_filter, :] = pred_probs_group

  return pred_probs_stratified


def fit_cross_val_predict_stratified(
    features,
    labels,
    group,
    sample_weight = None,
    **kwargs,
):
  """Fits a model using cross validation and returns the predictions for each held-out fold.

  Args:
    features: A numpy array containing the features.
    labels: A numpy array containing the labels.
    group: A numpy array containing the group membership.
    sample_weight: An optional numpy array containing the sample weights.
    **kwargs: Keyword arguments passed to fit_cross_val_predict.

  Returns:
    A numpy array containing the predicted probabilities.
  """

  pred_probs_stratified = np.zeros((features.shape[0], len(np.unique(labels))))
  for group_id in np.unique(group):
    group_filter = group == group_id
    pred_probs_stratified[group_filter] = fit_cross_val_predict(
        features[group_filter],
        labels[group_filter],
        sample_weight=sample_weight[group_filter]
        if sample_weight is not None
        else None,
        **kwargs,
    )
  return pred_probs_stratified


class MetricFn(Protocol):

  def __call__(
      self,
      args_0,
      args_1,
      sample_weight = None,
      **kwargs,
  ):
    Ellipsis


def multi_metric_wrapper(
    *args,
    metrics,
    weighted = False,
    pairwise = False,
    weighted_population_on_group_comparison = False,
    **kwargs,
):
  """Wraps a list of metrics and returns a tuple containing the results of each metric.

  This function is intended to be used to simplify the creation of a function
  that satisfies the interface required for the `statistic` argument to
  scipy.stats.bootstrap. When used with scipy.stats.bootstrap, this function
  enables the evaluation of multiple metrics on the same set of bootstrap
  samples.

  Args:
    *args: Additional arguments passed to the metrics.
    metrics: An iterable of metrics to evaluate.
    weighted: If True, the metrics are weighted, the third arg is expected to be
      the sample weights, and the metrics are expected to take a kwarg called
      sample_weight.
    pairwise: If True, the metrics are pairwise, the third and fourth arg are
      expected to be the reference data.
    weighted_population_on_group_comparison: If True, the metrics are weighted
      on the population and compared to the same metric on a group. The third
      arg is expected to be a sample_weight of size N x K (num_samples x
      num_groups), and the fourth arg is expected to be a group_filter of size
      N.
    **kwargs: Keyword arguments passed to the metrics.

  Returns:
    A tuple of results from the metrics.
  """
  result = list()
  for metric in metrics:
    if weighted_population_on_group_comparison:
      result.append(
          weighted_population_on_group_comparison_metric(
              *args,
              base_metric=metric,
              normalize=kwargs.get(
                  'normalize_population_on_group_comparison', False
              ),
          )
      )
    else:
      if not pairwise and not weighted:
        result.append(metric(*args, **kwargs))
      elif not pairwise and weighted:
        result.append(metric(args[0], args[1], sample_weight=args[2], **kwargs))
      elif pairwise and not weighted:
        result.append(pairwise_metric(*args, base_metric=metric))
      else:  # pairwise and weighted
        result.append(
            pairwise_metric(
                args[0],
                args[1],
                args[2],
                args[3],
                base_metric=metric,
                sample_weight_comparator=args[4],
                sample_weight_reference=args[4],
            )
        )

  return tuple(result)


EVAL_FUNCTIONS_THRESHOLD_FREE = {
    'log_loss': sklearn.metrics.log_loss,
    'roc_auc': sklearn.metrics.roc_auc_score,
    'label_rate': (
        lambda x, y, sample_weight=None: x.mean()
        if sample_weight is None
        else (x * sample_weight).sum() / sample_weight.sum()
    ),
}


def net_benefit(
    labels,
    pred_probs,
    sample_weight = None,
    pred_prob_threshold = 0.5,
    optimal_threshold = 0.5,
    verbose = False,
):
  """Computes the net benefit of a classifier at a given threshold.

    This metric was first proposed by Vickers and Elkin (2006), "Decision curve
    analysis: a novel method for evaluating prediction models".
    The formulation used here is described in Pfohl et al (2023), "Net benefit,
    calibration, threshold selection, and training objectives for algorithmic
    fairness in healthcare".


  Args:
    labels: A numpy array containing the labels.
    pred_probs: A numpy array containing the predicted probabilities of the
      label.
    sample_weight: An optional numpy array containing the sample weights. If
      None, no sample weights are used.
    pred_prob_threshold: The threshold at which the predictions are made.
    optimal_threshold: The decision-theoretically optimal threshold for a
      calibrated classifier. This threshold encodes context-specific preferences
      regarding the trade-off between false-positive and false-negative errors.
      This value should NOT be set based on the data.
    verbose: If True, prints verbose output.

  Returns:
    The net benefit of the classifier at the given threshold.
  """

  tpr = sklearn.metrics.recall_score(
      labels, pred_probs >= pred_prob_threshold, sample_weight=sample_weight
  )
  fpr = 1 - sklearn.metrics.recall_score(
      labels,
      pred_probs >= pred_prob_threshold,
      sample_weight=sample_weight,
      pos_label=0,
  )
  p_y_1 = np.average(labels, weights=sample_weight)
  nb = tpr * p_y_1 - fpr * (1 - p_y_1) * (
      optimal_threshold / (1 - optimal_threshold)
  )
  if verbose:
    print(
        f'threshold: {pred_prob_threshold}, tpr: {tpr}, fpr: {fpr}, p_y_1:'
        f' {p_y_1}, nb: {nb}'
    )
  return nb


def get_eval_functions_threshold(
    thresholds = None,
    net_benefit_optimal_threshold = 0.5,
):
  """Defines evaluation functions at a threshold.

  Args:
    thresholds: A list of thresholds to evaluate at.
    net_benefit_optimal_threshold: The optimal threshold for the net benefit
      metric.

  Returns:
    A dict of evaluation functions at the given thresholds.
  """
  if thresholds is None:
    thresholds = [0.5]

  result = {}
  for threshold in thresholds:
    result[f'recall_{threshold}'] = (
        lambda labels, pred_probs, sample_weight=None, threshold=threshold: sklearn.metrics.recall_score(
            labels, pred_probs >= threshold, sample_weight=sample_weight
        )
    )
    result[f'specificity_{threshold}'] = (
        lambda labels, pred_probs, sample_weight=None, threshold=threshold: sklearn.metrics.recall_score(
            labels,
            pred_probs >= threshold,
            sample_weight=sample_weight,
            pos_label=0,
        )
    )
    result[f'precision_{threshold}'] = (
        lambda labels, pred_probs, sample_weight=None, threshold=threshold: sklearn.metrics.precision_score(
            labels,
            pred_probs >= threshold,
            sample_weight=sample_weight,
            zero_division=0.0,
        )
    )
    result[f'net_benefit_{threshold}_{net_benefit_optimal_threshold}'] = (
        lambda labels, pred_probs, sample_weight=None, threshold=threshold: net_benefit(
            labels,
            pred_probs,
            sample_weight=sample_weight,
            pred_prob_threshold=threshold,
            optimal_threshold=net_benefit_optimal_threshold,
        )
    )
  return result


def pairwise_metric(
    args_comparator_0,
    args_comparator_1,
    args_reference_0,
    args_reference_1,
    base_metric,
    sample_weight_comparator = None,
    sample_weight_reference = None,
):
  """Computes the difference the value of a performance metric computed on two sets of data.

    This is evaluated as base_metric(args_comparator_0, args_comparator_1,
    sample_weight=sample_weight_comparator) -
    base_metric(args_reference_0, args_reference_1,
    sample_weight=sample_weight_reference). For scikit-learn metrics,
    the first argument is the labels, the second is the predictions.

  Args:
    args_comparator_0: The first set of arguments to pass to base_metric for the
      comparator data.
    args_comparator_1: The second set of arguments to pass to base_metric for
      the reference data.
    args_reference_0: The first set of arguments to pass to base_metric for the
      comparator data.
    args_reference_1: The second set of arguments to pass to base_metric for the
      reference data.
    base_metric: The metric to compute.
    sample_weight_comparator: The sample weights to pass to base_metric for the
      comparator data.
    sample_weight_reference: The sample

  Returns:
    The difference between the values of the metric computed on the comparator
    and reference data.
  """
  result_comparator = base_metric(
      args_comparator_0,
      args_comparator_1,
      sample_weight=sample_weight_comparator,
  )
  result_reference = base_metric(
      args_reference_0, args_reference_1, sample_weight=sample_weight_reference
  )
  return result_comparator - result_reference


def weighted_population_on_group_comparison_metric(
    labels,
    pred_probs,
    sample_weight,
    group_filter,
    base_metric,
    normalize = False,
    verbose = False,
    tol = 1e-4,
):
  """Computes the difference between a weighted metric on the population and the same metric unweighted on a group.

  Args:
    labels: A numpy array containing the labels.
    pred_probs: A numpy array containing the predicted probabilities of the
      label.
    sample_weight: A numpy array containing the sample weights. These are
      intended to be weights that map the population to the group. These weights
      are not optional. For example, if the goal is to match the distribution of
      X, then these weights should be proportional to P(A= 1| X).
    group_filter: A binary-valued numpy array indicating whether each sample is
      in the group or not.
    base_metric: The metric to compute.
    normalize: A bool indicating whether to normalize the metric.
    verbose: If True, prints verbose output.
    tol: A float indicating the tolerance for the denominator in the
      normalization step. If the absoluate value of the denominator is less than
      this value, the function returns np.nan.

  Returns:
    The difference between the weighted metric on the population and the same
    metric unweighted on the group. If normalize is True, the difference is
    normalized by the difference between the unweighted metric on the population
    and the same metric unweighted on the group.
  """
  group_filter = group_filter.astype(bool)
  metric_group = base_metric(labels[group_filter], pred_probs[group_filter])
  weighted_metric_population = base_metric(
      labels, pred_probs, sample_weight=sample_weight
  )
  if not normalize:
    if verbose:
      print(
          f'metric_group: {metric_group}, weighted_metric_population:'
          f' {weighted_metric_population}'
      )
    return metric_group - weighted_metric_population
  else:
    metric_population = base_metric(labels, pred_probs)
    numerator = metric_group - weighted_metric_population
    denominator = metric_group - metric_population
    if np.abs(denominator) < tol:
      return 0
    if verbose:
      print(
          f'metric_group: {metric_group}, weighted_metric_population:'
          f' {weighted_metric_population}, metric_population:'
          f' {metric_population}'
      )
    return numerator / (denominator)


def evaluate(
    labels,
    pred_probs,
    pred_probs_reference = None,
    group = None,
    sample_weight = None,
    metric_fn_dict = None,
    verbose = False,
    weighted_population_on_group_comparison = False,
    normalize_population_on_group_comparison = False,
):
  """Computes performance metrics on groups.

  Contains functionality for bootstrapping and computing pairwise metrics
  relative to a reference.

  Args:
    labels: A numpy array containing the labels.
    pred_probs: A numpy array containing the predicted probabilities for the
      model to be evaluated.
    pred_probs_reference: A numpy array containing the predicted probabilities
      for the reference model that is compared to in the computation of pairwise
      metrics. If None, there is no reference and the absolute performance with
      respect to pred_probs is computed.
    group: A numpy array containing the group membership. If None, the
      performance is computed overall.
    sample_weight: A numpy array containing the sample weights. If None, no
      sample weights are used. If weighted_population_on_group_comparison is
      False, this is a length N array, where N is the length of the labels
      array. If weighted_population_on_group_comparison is True, this must be an
      array of size N x K, where K is the number of groups.
    metric_fn_dict: A dict of metric functions to evaluate. If None, a default
      set of metrics are used.
    verbose: If True, prints verbose output.
    weighted_population_on_group_comparison: If True, computes the difference
      between the weighted metric on the population and the same metric
      unweighted on a group.
    normalize_population_on_group_comparison: If True, when using
      weighted_population_on_group_comparison, normalizes the difference through
      dividing by the difference between the unweighted metric on the population
      and the same metric unweighted on the group.

  Returns:
    A dict containing the results.
  """
  if metric_fn_dict is None:
    metric_fn_dict = {
        **EVAL_FUNCTIONS_THRESHOLD_FREE,
        **get_eval_functions_threshold(),
    }

  eval_dict = {}
  for metric_key, metric in metric_fn_dict.items():
    if verbose:
      print(metric_key)
    if weighted_population_on_group_comparison:
      if group is not None:
        eval_dict[metric_key] = pd.concat(
            objs=[
                pd.Series(
                    weighted_population_on_group_comparison_metric(
                        labels,
                        pred_probs,
                        sample_weight=sample_weight[:, the_group]
                        if sample_weight is not None
                        else None,
                        group_filter=group == the_group,
                        base_metric=metric,
                        normalize=normalize_population_on_group_comparison,
                        verbose=verbose,
                    ),
                    index=[the_group],
                ).to_frame()
                for the_group in np.unique(group)
            ]
        )
      else:
        raise ValueError(
            'Group is required for weighted_population_on_group_comparison.'
        )
    else:  # Standard evaluation (metrics computed overall or on groups)
      if group is None:  # Compute metrics overall
        if pred_probs_reference is None:
          # Absolute performance metrics
          eval_dict[metric_key] = pd.Series(
              metric(labels, pred_probs, sample_weight=sample_weight),
              index=['overall'],
          ).to_frame()
        else:
          # Pairwise performance metrics
          eval_dict[metric_key] = pd.Series(
              pairwise_metric(
                  labels,
                  pred_probs,
                  labels,
                  pred_probs_reference,
                  metric,
                  sample_weight_comparator=sample_weight,
                  sample_weight_reference=sample_weight,
              ),
              index=['overall'],
          ).to_frame()

      else:  # Grouped evaluation
        df = pd.DataFrame(
            {'labels': labels, 'pred_probs': pred_probs, 'group': group}
        )
        if pred_probs_reference is not None:
          df['pred_probs_reference'] = pred_probs_reference
        if sample_weight is not None:
          df['sample_weight'] = sample_weight

        if (
            pred_probs_reference is None
        ):  # Absolute performance metrics by group
          eval_dict[metric_key] = (
              df.groupby('group', observed=True)
              .apply(
                  lambda x: metric(  # pylint: disable=cell-var-from-loop
                      x['labels'].values,
                      x['pred_probs'].values,
                      sample_weight=x['sample_weight']
                      if sample_weight is not None
                      else None,
                  ),
                  include_groups=False,
              )
              .to_frame()
          )
        else:  # Pairwise performance by group
          eval_dict[metric_key] = (
              df.groupby('group', observed=True)
              .apply(
                  lambda x: pairwise_metric(
                      x['labels'],
                      x['pred_probs'],
                      x['labels'],
                      x['pred_probs_reference'],
                      metric,  # pylint: disable=cell-var-from-loop
                      sample_weight_comparator=x['sample_weight']
                      if sample_weight is not None
                      else None,
                      sample_weight_reference=x['sample_weight']
                      if sample_weight is not None
                      else None,
                  ),
                  include_groups=False,
              )
              .to_frame()
          )
  eval_df = (
      pd.concat(eval_dict)
      .rename_axis(['metric', 'group'])
      .reset_index()
      .rename(columns={0: 'performance'})
  )
  return eval_df


def evaluate_bootstrap(
    labels,
    pred_probs,
    pred_probs_reference = None,
    group = None,
    sample_weight = None,
    n_resamples = 100,
    metric_fn_dict = None,
    verbose = False,
    weighted_population_on_group_comparison = False,
    normalize_population_on_group_comparison = False,
):
  """Computes performance metrics on groups.

  Contains functionality for bootstrapping and computing pairwise metrics
  relative to a reference.

  Args:
    labels: A numpy array containing the labels.
    pred_probs: A numpy array containing the predicted probabilities for the
      model to be evaluated.
    pred_probs_reference: A numpy array containing the predicted probabilities
      for the reference model that is compared to in the computation of pairwise
      metrics. If None, there is no reference and the absolute performance with
      respect to pred_probs is computed.
    group: A numpy array containing the group membership. If None, the
      performance is computed overall.
    sample_weight: A numpy array containing the sample weights. If None, no
      sample weights are used.
    n_resamples: An int indicating the number of bootstrap iterations to run. If
      None, no bootstrapping is performed and the point estimates are returned.
    metric_fn_dict: A dict of metric functions to evaluate. If None, a default
      set of metrics are used.
    verbose: If True, prints verbose output.
    weighted_population_on_group_comparison: If True, computes the difference
      between the weighted metric on the population and the same metric
      unweighted on a group.
    normalize_population_on_group_comparison: If True, when using
      weighted_population_on_group_comparison, normalizes the difference through
      dividing by the difference between the unweighted metric on the population
      and the same metric unweighted on the group.

  Returns:
    A dict containing the results.
  """
  if metric_fn_dict is None:
    metric_fn_dict = {
        **EVAL_FUNCTIONS_THRESHOLD_FREE,
        **get_eval_functions_threshold(),
    }
  bootstrap_result_dict = {}
  result_df_dict = {}
  statistic_kwargs = {}
  if group is None:
    group = np.full(labels.shape, 'overall')

  for group_id in np.unique(group):
    if verbose:
      print(f'Group: {group_id}')
    group_filter = group_id == group

    labels_group = labels[group_filter]
    pred_probs_group = pred_probs[group_filter]
    if weighted_population_on_group_comparison:
      if sample_weight is None:
        raise ValueError(
            'Sample weight is required for'
            ' weighted_population_on_group_comparison.'
        )
      statistic_kwargs['normalize_population_on_group_comparison'] = (
          normalize_population_on_group_comparison
      )
      statistic_args = (
          labels,
          pred_probs,
          sample_weight[:, group_id],
          group_filter,
      )
    else:  # Standard evaluation (metrics computed overall or on groups)
      if pred_probs_reference is None:
        if sample_weight is None:
          statistic_args = (labels_group, pred_probs_group)
        else:
          statistic_args = (
              labels_group,
              pred_probs_group,
              sample_weight[group_filter],
          )
      else:
        if sample_weight is None:
          statistic_args = (
              labels_group,
              pred_probs_group,
              labels_group,
              pred_probs_reference[group_filter],
          )
        else:
          statistic_args = (
              labels_group,
              pred_probs_group,
              labels_group,
              pred_probs_reference[group_filter],
              sample_weight[group_filter],
          )

    bootstrap_result_dict[group_id] = scipy.stats.bootstrap(
        statistic_args,
        statistic=lambda *args: multi_metric_wrapper(
            *args,
            metrics=metric_fn_dict.values(),
            weighted=sample_weight is not None,
            pairwise=pred_probs_reference is not None,
            weighted_population_on_group_comparison=weighted_population_on_group_comparison,
            **statistic_kwargs,
        ),
        vectorized=False,
        paired=True,
        method='percentile',
        n_resamples=n_resamples,
    )
    result_df_dict[group_id] = pd.DataFrame({
        'metric': metric_fn_dict.keys(),
        'ci_low': bootstrap_result_dict[group_id].confidence_interval.low,
        'ci_high': bootstrap_result_dict[group_id].confidence_interval.high,
    })
    result_df_dict[group_id] = result_df_dict[group_id].assign(
        ci_str=lambda x: x.apply(
            lambda y: f'({y["ci_low"]:.3}, {y["ci_high"]:.3})', axis=1
        )
    )
  result = (
      pd.concat(result_df_dict)
      .reset_index(level=-1, drop=True)
      .rename_axis('group')
      .reset_index()
  )
  return result


def run_evaluation(*args, n_resamples, **kwargs):
  """A helper function to run both evaluate and evaluate_bootstrap and compile the results into a dataframe.

  Args:
    *args: The arguments to pass to evaluate and evaluate_bootstrap.
    n_resamples: The number of bootstrap iterations to run.
    **kwargs: The keyword arguments to pass to evaluate and evaluate_bootstrap.

  Returns:
    A dataframe containing the results of the evaluation.
  """

  performance_mean = evaluate(*args, **kwargs)
  performance_ci = evaluate_bootstrap(*args, n_resamples=n_resamples, **kwargs)
  performance_df = performance_mean.merge(performance_ci)
  performance_df['performance_str'] = performance_df.apply(
      lambda x: f'{x["performance"]:.3} {x["ci_str"]}', axis=1
  )
  return performance_df


def compute_balancing_weights(
    group_codes,
    pred_probs_group,
    normalize = False,
    weight_type = 'stable',
    group_frequency_array = None,
):
  """Computes balancing weights to enable controlled comparisons across groups.

  Args:
    group_codes: An N-length array containing indices indicating group
      membership. For example, if there are K groups, then this is an array with
      values in {0, 1, ..., K-1}.
    pred_probs_group: A N-by-K numpy array containing the predicted
      probabilities for each group.
    normalize: A bool indicating whether to normalize the weights to sum to 1.
    weight_type: A string indicating the type of weight to compute. Currently
      supported values are 'one-vs-rest' and 'stable'. `one-vs-rest` weights are
      proportional to P(~A|S) / P(A|S) and can be used to map expectations with
      respect to P(S|A) to P(S|~A); `stable` implements the weighting scheme
      proposed by Cai 2023 (https://arxiv.org/abs/2303.02011) and maps both
      P(S|A) and P(S|~A) to a region of overlapping support.
    group_frequency_array: An optional numpy array containing the frequency of
      each group. If None, the frequency is computed from the data.

  Returns:
    An N-length array containing the balancing weights.
  """
  num_groups = pred_probs_group.shape[1]
  if weight_type == 'one-vs-rest':
    weights = (
        1 - pred_probs_group[np.arange(pred_probs_group.shape[0]), group_codes]
    ) / pred_probs_group[np.arange(pred_probs_group.shape[0]), group_codes]
  elif weight_type == 'stable':
    if group_frequency_array is None:
      group_frequency_array = np.array(
          [(group_codes == group_id).mean() for group_id in range(num_groups)]
      )
    group_frequencies = group_frequency_array[group_codes]
    weight_numerator = (
        1 - pred_probs_group[np.arange(pred_probs_group.shape[0]), group_codes]
    )
    weight_denominator = (
        group_frequencies
        * (
            1
            - pred_probs_group[
                np.arange(pred_probs_group.shape[0]), group_codes
            ]
        )
    ) + (
        (1 - group_frequencies)
        * pred_probs_group[np.arange(pred_probs_group.shape[0]), group_codes]
    )
    weights = weight_numerator / weight_denominator
  else:
    raise ValueError('Invalid weight_type specified')

  if normalize:
    return weights / weights.sum()
  else:
    return weights


def calibration_curve_ci(
    y_true,
    y_score,
    n_bins = 10,
    alpha = 0.05,
    method = 'wilson',
):
  """Computes a calibration curve with confidence intervals.

  Args:
    y_true: The true outcomes.
    y_score: The scores.
    n_bins: The number of bins to use for the calibration curve.
    alpha: The significance level for the confidence intervals.
    method: The method to use for the confidence intervals. See
      statsmodels.stats.proportion.proportion_confint for options.

  Returns:
    A tuple containing the calibration curve y values, the calibration curve x
    values, the lower bound of the confidence interval, and the upper bound of
    the confidence interval.
  """
  bin_id = pd.qcut(y_score, q=n_bins, labels=False, duplicates='drop')
  y_sum = pd.Series(y_true).groupby(bin_id).agg('sum').values
  x_sum = pd.Series(y_score).groupby(bin_id).agg('sum').values
  counts = np.bincount(bin_id)
  calibration_curve_y = y_sum / counts
  calibration_curve_x = x_sum / counts

  ci = statsmodels.stats.proportion.proportion_confint(
      count=y_sum, nobs=counts, alpha=alpha, method=method
  )
  return calibration_curve_y, calibration_curve_x, ci[0], ci[1]
