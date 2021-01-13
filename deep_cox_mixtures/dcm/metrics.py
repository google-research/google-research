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

"""Utilitites to plot the ROC and Calibration for survival models.

This module has utility functions to generate ROC and Calibration plots for
survival models at given horizons of time. Note that ideally both the ROC and
Calibration curves require to be adjusted for censoring using IPCW estimates.


Not designed to be called directly, would be called when running a function from
dcm.deep_cox_mixtures

"""
from dcm import baseline_models
from dcm import models
from dcm.calibration import calibration_curve

import matplotlib as mpl
from matplotlib import pyplot as plt

from dcm.skmetrics import brier_score
from dcm.skmetrics import cumulative_dynamic_auc
from dcm.skmetrics import concordance_index_ipcw

import numpy as np

from sklearn.metrics import auc


def compute_ece(scores,
                e,
                t,
                a,
                group,
                eval_time,
                strat='quantile',
                adj='KM'):
                

  """Function to plot Calibration Curve at a specified time horizon.

  Accepts a matplotlib figure instance, risk scores from a trained survival
  analysis model, and quantiles of event interest and generates an IPCW
  adjusted calibration curve.

  Args
    scores:
      risk scores P(T>t) issued by a trained survival analysis model
      (output of deep_cox_mixtures.models.predict_survival)
    e:
      a numpy array of input features.
    t:
      a numpy array of input features.
    a:
      a numpy vector of protected attributes.
    group:
      string indicating the demographic to evaluate calibration for.  
      use None for entire population.
    eval_time:
      float/int of the event time at which calibration is to be evaluated. Must
      be same as the time at which the Risk Scores were issued.
    strat:
      Specifies how the bins are computed. One of:
      "quantile": Equal sized bins.
      "uniform": Uniformly stratified.
    adj (str):
      Determines if KM/IPCW adjustment is carried out.
      One of "KM" (Default) or "IPCW".

  Returns:
    A plotted matplotlib calibration curve.

  """
  _, _, _, ece = calibration_curve(
      scores,
      e,
      t,
      a,
      group,
      eval_time,
      typ=adj,
      ret_bins=True,
      strat=strat,
      n_bins=20)  

  return ece


def compute_metrics(scores,
                    e,
                    t,
                    a,
                    groups,
                    eval_time,
                    al=True):

  """Function to compute censoring adjusted discriminative metrics.

  Accepts risk scores from a trained survival analysis model, and evaluation
  time at which models are to be evaluated and outputs the IPCW adjusted
  Area under ROC, Time dependent Concordance Index and the Brier Score.

  Args:
    scores:
      risk scores P(T>t) issued by a trained survival analysis model
      (output of deep_cox_mixtures.models.predict_survival)
    e:
      a numpy array of input features.
    t:
      a numpy array of input features.
    a:
      a numpy vector of protected attributes.
    groups:
      list of strings indicating the demographic to evaluate calibration for.  
    eval_time:
      float/int of the event time at which calibration is to be evaluated. Must
      be same as the time at which the Risk Scores were issued.
    al:
      Binary flag indicating if metrics for entire population are also to be
      computed. Default: True
  Returns:
    tuple of python dictionaries

  """

  ctds, brss, aucs = {}, {}, {}
    
  str_train = baseline_models.structure_for_eval_(t, e)
  str_test = baseline_models.structure_for_eval_(t, e)
  
  scores_f = scores
    
  for group in groups:
        
    te_protg = (a == group)
    tr_protg = (a == group)
    
    try:
      roc_m = cumulative_dynamic_auc(str_train[tr_protg], str_test[te_protg],
                                     -scores_f[te_protg], [eval_time])
      brs_m = brier_score(str_train[tr_protg], str_test[te_protg],
                          scores_f[te_protg], eval_time)
      ctd_m = concordance_index_ipcw(str_train[tr_protg], str_test[te_protg],
                                     -scores_f[te_protg], eval_time)[0]
        
    except:
      roc_m = cumulative_dynamic_auc(str_train, str_test[te_protg],
                                     -scores_f[te_protg], [eval_time])
      brs_m = brier_score(str_train, str_test[te_protg],
                          scores_f[te_protg], eval_time)
      ctd_m = concordance_index_ipcw(str_train, str_test[te_protg],
                                     -scores_f[te_protg], eval_time)[0]
        
    ctds[group] = ctd_m
    brss[group] = brs_m[1][0]
    aucs[group] = roc_m[1][0]
  
  if al:
      roc_m = cumulative_dynamic_auc(str_train, str_test, -scores_f, [eval_time])
      ctd_m = concordance_index_ipcw(str_train, str_test, -scores_f, eval_time)[0]
      brs_m = brier_score(str_train, str_test, scores_f, eval_time)

      aucs['all'] = roc_m[1][0]
      ctds['all'] = ctd_m
      brss['all'] = brs_m[1][0]

  return aucs, ctds, brss
       
  