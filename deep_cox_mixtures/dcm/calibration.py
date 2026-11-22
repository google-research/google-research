# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Utilitites to evaluate calibration of survival models at various horizons.

This module has utility functions to compute the reliability curves and the
Expected Calibration Error of the survival models at a specified horizon of
time. The module allows two ways to compute calibration in the presence of
right censoring:

1) Default: The calibration curve is computed only on the instances where the
            event occured.

2) IPCW: The calibration curves are adjusted by reweighting using the invese
         propensity of censoring weighting scheme. Module allows the IPCW
         estimates to be evaluated on a subgroup or a population level.

TODO:

3) l2-ECE (yadlowsky): Interface with the Non-Parametric l2-ECE from
                       Yadlowsky et. al (MLHC 2019)

Not designed to be called directly, would be called when running a function
from dcm.deep_cox_mixtures

"""


from lifelines import KaplanMeierFitter
import numpy as np


def _calibration_curve_ipcw(out,
                            e,
                            t,
                            a,
                            group,
                            eval_time,
                            ret_bins=True,
                            strat='quantile',
                            n_bins=10):

  """Returns the Calibration curve and the bins given some risk scores.

  Accepts the output of a trained survival model at a certain evaluation time,
  the event indicators and protected group membership and outputs an IPCW
  adjusted calibration curve.

  Args:
    out:
      risk scores P(T>t) issued by a trained survival analysis model
      (output of deep_cox_mixtures.models.predict_survival).
    e:
      a numpy vector of indicators specifying is event or censoring occured.
    t:
      a numpy vector of times at which the events or censoring occured.
    a:
      a numpy vector of protected attributes.
    group:
      string indicating the demogrpahic to evaluate calibration for.
    eval_time:
      float/int of the event time at which calibration is to be evaluated. Must
      be same as the time at which the Risk Scores were issues.
    ret_bins:
      Boolean that specifies if the bins of the calibration curve are to be
      returned.
    strat:
      Specifies how the bins are computed. One of:
      "quantile": Equal sized bins.
      "uniform": Uniformly stratified.
    n_bins:
      int specifying the number of bins to use to compute the ece.
  Returns:
    Calibration Curve: A tuple of True Probality, Estimated Probability in
    each bin and the estimated Expected Calibration Error.

  """

  out_ = out.copy()

  if group is not None:
      e = e[a == group]
      t = t[a == group]
      out = out[a == group]

  y = t > eval_time

  if strat == 'quantile':

    quantiles = [(1. / n_bins) * i for i in range(n_bins + 1)]
    outbins = np.quantile(out, quantiles)

  if strat == 'uniform':

    binlen = (out.max() - out.min()) / n_bins
    outbins = [out.min() + i * binlen for i in range(n_bins + 1)]

  prob_true = []
  prob_pred = []

  ece = 0

  for n_bin in range(n_bins):

    binmin = outbins[n_bin]
    binmax = outbins[n_bin + 1]

    scorebin = (out >= binmin) & (out <= binmax)

    weight = float(scorebin.sum()) / len(out)

    out_ = out[scorebin]
    y_ = y[scorebin]
    
    kmf = KaplanMeierFitter().fit(t[scorebin], 1 - e[scorebin])
    y_ = y_ / kmf.predict(eval_time)
    pred = y_.mean()
    
    prob_true.append(pred)
    prob_pred.append(out_.mean())

    gap = abs(prob_pred[-1] - prob_true[-1])

    ece += weight * gap

  if ret_bins:
    return prob_true, prob_pred, outbins, ece

  else:
    return prob_true, prob_pred, ece

def _calibration_curve_km(out,
                          e,
                          t,
                          a,
                          group,
                          eval_time,
                          ret_bins=True,
                          strat='quantile',
                          n_bins=10):

  """Returns the Calibration curve and the bins given some risk scores.

  Accepts the output of a trained survival model at a certain evaluation time,
  the event indicators and protected group membership and outputs an KM
  adjusted calibration curve.

  Args:
    out:
      risk scores P(T>t) issued by a trained survival analysis model
      (output of deep_cox_mixtures.models.predict_survival).
    e:
      a numpy vector of indicators specifying is event or censoring occured.
    t:
      a numpy vector of times at which the events or censoring occured.
    a:
      a numpy vector of protected attributes.
    group:
      string indicating the demogrpahic to evaluate calibration for.
    eval_time:
      float/int of the event time at which calibration is to be evaluated. Must
      be same as the time at which the Risk Scores were issues.
    ret_bins:
      Boolean that specifies if the bins of the calibration curve are to be
      returned.
    strat:
      Specifies how the bins are computed. One of:
      "quantile": Equal sized bins.
      "uniform": Uniformly stratified.
    n_bins:
      int specifying the number of bins to use to compute the ece.
  Returns:
    Calibration Curve: A tuple of True Probality, Estimated Probability in
    each bin and the estimated Expected Calibration Error.

  """

  out_ = out.copy()

  if group is not None:
      e = e[a == group]
      t = t[a == group]
      out = out[a == group]

  y = t > eval_time

  if strat == 'quantile':

    quantiles = [(1. / n_bins) * i for i in range(n_bins + 1)]
    outbins = np.quantile(out, quantiles)

  if strat == 'uniform':

    binlen = (out.max() - out.min()) / n_bins
    outbins = [out.min() + i * binlen for i in range(n_bins + 1)]

  prob_true = []
  prob_pred = []

  ece = 0

  for n_bin in range(n_bins):

    binmin = outbins[n_bin]
    binmax = outbins[n_bin + 1]

    scorebin = (out >= binmin) & (out <= binmax)

    weight = float(scorebin.sum()) / len(out)

    out_ = out[scorebin]
    y_ = y[scorebin]
    
    
    pred = KaplanMeierFitter().fit(t[scorebin], e[scorebin]).predict(eval_time)
    prob_true.append(pred)
    prob_pred.append(out_.mean())

    gap = abs(prob_pred[-1] - prob_true[-1])

    ece += weight * gap

  if ret_bins:
    return prob_true, prob_pred, outbins, ece

  else:
    return prob_true, prob_pred, ece


def calibration_curve(out,
                      e,
                      t,
                      a,
                      group,
                      eval_time,
                      typ='KM',
                      ret_bins=False,
                      strat='quantile',
                      n_bins=10):
  """Returns the Calibration curve and the bins given some risk scores.

  Accepts the output of a trained survival model at a certain evaluation time,
  the event indicators and protected group membership and outputs a calibration
  curve

  Args:
    out:
      risk scores P(T>t) issued by a trained survival analysis model
      (output of deep_cox_mixtures.models.predict_survival).
    e:
      a numpy vector of indicators specifying is event or censoring occured.
    t:
      a numpy vector of times at which the events or censoring occured.
    a:
      a numpy vector of protected attributes.
    group:
      string indicating the demogrpahic to evaluate calibration for.
      use None for entire population.
    eval_time:
      float/int of the event time at which calibration is to be evaluated. Must
      be same as the time at which the Risk Scores were issued.
    typ:
      Determines if the calibration curves are to be computed on the individuals
      that experienced the event or adjusted estimates for individuals that are
      censored using IPCW estimator on a population or subgroup level
    ret_bins:
      Boolean that specifies if the bins of the calibration curve are to be
      returned.
    strat:
      Specifies how the bins are computed. One of:
      "quantile": Equal sized bins.
      "uniform": Uniformly stratified.
    n_bins:
      int specifying the number of bins to use to compute the ece.
  Returns:
    Calibration Curve: A tuple of True Probality and Estimated Probability in
    each bin.

  """

  if typ == 'IPCW':
      return _calibration_curve_ipcw(
          out,
          e,
          t,
          a,
          group,
          eval_time,
          ret_bins=ret_bins,
          strat=strat,
          n_bins=n_bins)

  else:
      return _calibration_curve_km(
          out,
          e,
          t,
          a,
          group,
          eval_time,
          ret_bins=ret_bins,
          strat=strat,
          n_bins=n_bins)     
        
    
      
