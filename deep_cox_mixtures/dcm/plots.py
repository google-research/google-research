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

import logging
logging.getLogger("matplotlib").setLevel(logging.CRITICAL)

from sklearn.metrics import auc


def plot_calibration_curve(ax,
                           scores,
                           e,
                           t,
                           a,
                           folds,
                           group,
                           quant,
                           strat='quantile',
                           adj='IPCW', 
                           plot=True):

  """Function to plot Calibration Curve at a specified time horizon.

  Accepts a matplotlib figure instance, risk scores from a trained survival
  analysis model, and quantiles of event interest and generates an IPCW
  adjusted calibration curve.

  Args:
    ax:
      a matplotlib subfigure object.
    scores:
      risk scores P(T>t) issued by a trained survival analysis model
      (output of deep_cox_mixtures.models.predict_survival).
    e:
      a numpy array of event indicators.
    t:
      a numpy array of event/censoring times.
    a:
      a numpy vector of protected attributes.
    folds:
      a numpy vector of cv folds.
    group:
      List of the demogrpahics to adjust for.
    quant:
      a list of event time quantiles at which the models are to be evaluated.
    strat:
      Specifies how the bins are computed. One of:
      "quantile": Equal sized bins.
      "uniform": Uniformly stratified.
    adj (str):
      Determines if IPCW adjustment is carried out on a population or subgroup
      level.
      One of "IPCWpop", "IPCWcon" (not implemented).

  Returns:
    A plotted matplotlib calibration curve.

  """

  allscores = np.ones_like(t).astype('float')

  for fold in set(folds):
    allscores[folds == fold] = scores[fold]

  scores = allscores

  b_fc = (0, 0, 1, .4)
  r_fc = (1, 0, 0, .2)

  b_ec = (0, 0, 1, .8)
  r_ec = (1, 0, 0, .8)

  n_bins = 20

  hatch = '//'

  fs = 16

  prob_true_n, _, outbins, ece = calibration_curve(
      scores,
      e,
      t,
      a,
      group,
      quant,
      typ=adj,
      ret_bins=True,
      strat=strat,
      n_bins=n_bins)
  
  for d in range(len(prob_true_n)):

    binsize = outbins[d + 1] - outbins[d]
    binloc = (outbins[d + 1] + outbins[d]) / 2

    gap = (prob_true_n[d] - binloc)

    if gap < 0:
      bottom = prob_true_n[d]
    else:
      bottom = prob_true_n[d] - abs(gap)

    if d == len(prob_true_n) - 1:
      lbl1 = 'Score'
      lbl2 = 'Gap'
    else:
      lbl1 = None
      lbl2 = None
        
    if plot:
        ax.bar(
            binloc,
            prob_true_n[d],
            width=binsize,
            facecolor=b_fc,
            edgecolor=b_ec,
            linewidth=2.5,
            label=lbl1)
        ax.bar(
            binloc,
            abs(gap),
            bottom=bottom,
            width=binsize,
            facecolor=r_fc,
            edgecolor=r_ec,
            linewidth=2.5,
            hatch=hatch,
            label=lbl2)

    d += 1
    
    if plot:
        
      ax.plot([0, 1], [0, 1], c='k', ls='--', lw=2, zorder=100)

      ax.set_xlabel('Predicted Score', fontsize=fs)
      ax.set_ylabel('True Score', fontsize=fs)

      ax.legend(fontsize=fs)
      ax.set_title(str(group), fontsize=fs)
      ax.set_xlim(0, 1)
      ax.set_ylim(0, 1)

      ax.grid(ls=':', lw=2, zorder=-100, color='grey')
      ax.set_axisbelow(True)

      ax.text(
          x=0.030,
          y=.7,
          s='ECE=' + str(round(ece, 3)),
          size=fs,
          bbox=dict(boxstyle='round', fc='white', ec='grey', pad=0.2))

  return ece


def plot_roc_curve(ax,
                   scores,
                   e,
                   t,
                   a,
                   folds,
                   groups,
                   quant,
                   plot=True):

  """Function to plot ROC at a specified time horizon.

  Accepts a matplotlib figure instance, risk scores from a trained survival
  analysis model, and quantiles of event interest and generates an IPCW
  adjusted ROC curve.

  Args:
    ax:
      a matplotlib subfigure object.
    scores:
      choice of model. One of "coupled_deep_cph", "coupled_deep_cph_vae".
    e:
      a numpy array of input features.
    t:
      a numpy array of input features.
    a:
      a numpy vector of protected attributes.
    folds:
      a numpy vector of cv fold.
    groups:
      List of the demogrpahics to adjust for.
    quant:
      a list of event time quantiles at which the models are to be evaluated.

  Returns:
    A plotted matplotlib ROC curve.

  """

  fs = 16

  fprs, tprs, tprs_std, ctds, brss = {}, {}, {}, {}, {}

  fprs['all'] = {}
  tprs['all'] = {}
  ctds['all'] = {}
  brss['all'] = {}
    
  for group in groups:

    fprs[group] = {}
    tprs[group] = {}
    ctds[group] = {}
    brss[group] = {}
    
  for fold in set(folds):
    
    ate = a[folds == fold]
    str_test = baseline_models.structure_for_eval_(t[folds == fold],
                                                   e[folds == fold])
    
    if len(set(folds)) == 1:
        
        atr = ate
        str_train = str_test
        
    else:
        atr = a[folds != fold]
        str_train = baseline_models.structure_for_eval_(t[folds != fold],
                                                        e[folds != fold])

    t_tr_max = np.max([t_[1] for t_ in str_train])
    t_ = np.array([t_[1] for t_ in str_test])
    
    clean = (t_<=t_tr_max)
    
    str_test = str_test[t_<=t_tr_max]
    ate      = ate[t_<=t_tr_max]
    
    scores_f = scores[fold][clean]
    
    for group in groups:
        
      te_protg = (ate == group)
      tr_protg = (atr == group)
    
      try:
        roc_m = cumulative_dynamic_auc(str_train[tr_protg], str_test[te_protg],
                                       -scores_f[te_protg], [quant])
        brs_m = brier_score(str_train[tr_protg], str_test[te_protg],
                            scores_f[te_protg], quant)
        ctd_m = concordance_index_ipcw(str_train[tr_protg], str_test[te_protg],
                                       -scores_f[te_protg], quant)[0]
        
      except:
        roc_m = cumulative_dynamic_auc(str_train, str_test[te_protg],
                                       -scores_f[te_protg], [quant])
        brs_m = brier_score(str_train, str_test[te_protg],
                            scores_f[te_protg], quant)
        ctd_m = concordance_index_ipcw(str_train, str_test[te_protg],
                                       -scores_f[te_protg], quant)[0]
        
      fprs[group][fold] = roc_m[0][0][1] 
      tprs[group][fold] = roc_m[0][0][0] 
      ctds[group][fold] = ctd_m
      brss[group][fold] = brs_m[1][0]
        
    roc_m = cumulative_dynamic_auc(str_train, str_test, -scores_f, [quant])
    ctd_m = concordance_index_ipcw(str_train, str_test, -scores_f, quant)[0]
    brs_m = brier_score(str_train, str_test, scores_f, quant)
        
    fprs['all'][fold], tprs['all'][fold] = roc_m[0][0][1], roc_m[0][0][0]
    ctds['all'][fold] = ctd_m
    brss['all'][fold] = brs_m[1][0]
    
  cols = ['b', 'r', 'g']

  roc_auc = {}
  ctds_mean = {}
  brss_mean = {}
    
  j = 0

  for group in list(groups) + ['all']:

    all_fpr = np.unique(np.concatenate([fprs[group][i] for i in set(folds)]))

    # The ROC curves are interpolated at these points.
    mean_tprs = []
    for i in set(folds):
      mean_tprs.append(np.interp(all_fpr, fprs[group][i], tprs[group][i]))

    # Finally the interpolated curves are averaged over to compute AUC.
    mean_tpr = np.mean(mean_tprs, axis=0)
    std_tpr = 1.96 * np.std(mean_tprs, axis=0) / np.sqrt(10)

    fprs[group]['macro'] = all_fpr
    tprs[group]['macro'] = mean_tpr
    tprs_std[group] = std_tpr

    roc_auc[group] = auc(fprs[group]['macro'], tprs[group]['macro'])

    ctds_mean[group] =  np.mean([ctds[group][fold] for fold in folds])
    brss_mean[group] =  np.mean([brss[group][fold] for fold in folds])
    
    lbl = str(group)
    lbl += ' AUC:' + str(round(roc_auc[group], 3))
    lbl += ' Ctd:'+  str(round(ctds_mean[group], 3))
    lbl += ' BS:'+  str(round(brss_mean[group], 3))
    
    if plot:
        ax.plot(
            all_fpr,
            mean_tpr,
            c=cols[j],
            label=lbl)

        ax.fill_between(
            all_fpr,
            mean_tpr - std_tpr,
            mean_tpr + std_tpr,
            color=cols[j],
            alpha=0.25)

    j += 1
  
  if plot:
      ax.set_xlabel('False Positive Rate', fontsize=fs)
      ax.set_ylabel('True Positive Rate', fontsize=fs)
      ax.legend(fontsize=fs)
      ax.set_xscale('log')

  return roc_auc, ctds_mean, brss_mean
       
    
def plot_results(outputs, x, e, t, a, folds, groups,
                 quantiles, strat='quantile', adj='KM', plot=True):

  """Function to plot the ROC and Calibration curves from a survival model.

  Accepts a trained survival analysis model, features and horizon of interest
  and generates the IPCW adjusted ROC curve and Calibration curve at
  pre-specified horizons of time.

  Args:
    outputs:
      a python dict with survival probabilities for each fold
    x:
      a numpy array of input features.
    e:
      a numpy array of input features.
    t:
      a numpy array of input features.
    a:
      a numpy vector of protected attributes.
    folds:
      a numpy vector of cv fold.
    groups:
      List of the demogrpahics to adjust for.
    quantiles:
      a list of event time quantiles at which the models are to be evaluated.
    strat:
      Specifies how the bins are computed. One of:
      "quantile": Equal sized bins.
      "uniform": Uniformly stratified.
    adj:
      Adjustment strategy for the Expected Calibration Error. One of:
      "KM": Kaplan-Meier (Default)
      "IPCW": Inverse Propensity of Censoring

  Returns:
    a numpy vector of estimated risks P(T>t|X) at the horizon "quant".

  """
  if plot:
      mpl.rcParams['hatch.linewidth'] = 2.0

      fig, big_axes = plt.subplots(
          figsize=(8 * (len(groups) + 2), 6 * len(quantiles)),
          nrows=len(quantiles),
          ncols=1)

      plt.subplots_adjust(hspace=0.4)

      i = 0
      for _, big_ax in enumerate(big_axes, start=1):
        big_ax.set_title(
            'Receiver Operator Characteristic and Calibration at t=' +
            str(quantiles[i]) + '\n',
            fontsize=16)
        big_ax.tick_params(
            labelcolor=(1., 1., 1., 0.0),
            top='off',
            bottom='off',
            left='off',
            right='off')
        i += 1
  
  eces = {}
  metrics = {}

  for quant in quantiles:
    eces[quant] = {}
        
  for i in range(len(quantiles)):

    scores = outputs[quantiles[i]]
    for j in range(len(groups) + 2):

      pt = (i * (len(groups) + 2) + j + 1)
      if plot:
          ax = fig.add_subplot(len(quantiles), len(groups) + 2, pt)
      else:
          ax = None
            
      if (j==1):
        eces[quantiles[i]]['all'] = plot_calibration_curve(ax,
                                                    scores,
                                                    e,
                                                    t,
                                                    a,
                                                    folds,
                                                    None,
                                                    quantiles[i],
                                                    strat=strat,
                                                    adj=adj,
                                                    plot=plot)        
        
      if (j>1):
        eces[quantiles[i]][groups[j - 2]] = plot_calibration_curve(ax,
                                                            scores,
                                                            e,
                                                            t,
                                                            a,
                                                            folds,
                                                            groups[j - 2],
                                                            quantiles[i],
                                                            strat=strat,
                                                            adj=adj,
                                                            plot=plot)
        
      if (j==0):
        metrics[quantiles[i]] = plot_roc_curve(ax,
                                        scores,
                                        e,
                                        t,
                                        a,
                                        folds,
                                        groups,
                                        quantiles[i],
                                        plot=plot)

  for quant in quantiles:
    metrics[quant] = metrics[quant] + (eces[quant], )
  
  if plot:      
      plt.show()
  return metrics
