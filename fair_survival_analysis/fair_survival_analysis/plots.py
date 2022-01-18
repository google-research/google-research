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

"""Utilitites to plot the ROC and Calibration for survival models.

This module has utility functions to generate ROC and Calibration plots for
survival models at given horizons of time. Note that ideally both the ROC and
Calibration curves require to be adjusted for censoring using IPCW estimates.


Not designed to be called directly, would be called when running a function from
fair_survival_analysis.fair_survival_analysis

"""
from fair_survival_analysis import baseline_models
from fair_survival_analysis import models
from fair_survival_analysis.utils import calibration_curve

import matplotlib as mpl
from matplotlib import pyplot as plt

from metrics import cumulative_dynamic_auc

import numpy as np

from sklearn.metrics import auc


def plot_calibration_curve(plot,
                           scores,
                           e,
                           t,
                           a,
                           folds,
                           group,
                           quant,
                           strat='quantile',
                           adj='IPCWpop'):

  """Function to plot Calibration Curve at a specified time horizon.

  Accepts a matplotlib figure instance, risk scores from a trained survival
  analysis model, and quantiles of event interest and generates an IPCW
  adjusted calibration curve.

  Args:
    plot:
      a trained survival analysis model
      (output of fair_survival_analysis.models.train_model).
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

    plot.bar(
        binloc,
        prob_true_n[d],
        width=binsize,
        facecolor=b_fc,
        edgecolor=b_ec,
        linewidth=2.5,
        label=lbl1)
    plot.bar(
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

  plot.plot([0, 1], [0, 1], c='k', ls='--', lw=2, zorder=100)

  plot.set_xlabel('Predicted Score', fontsize=fs)
  plot.set_ylabel('True Score', fontsize=fs)

  plot.legend(fontsize=fs)

  plot.set_title(str(group), fontsize=fs)

  plot.set_xlim(0, 1)
  plot.set_ylim(0, 1)

  plot.grid(ls=':', lw=2, zorder=-100, color='grey')
  plot.set_axisbelow(True)

  plot.text(
      x=0.030,
      y=.7,
      s='ECE=' + str(round(ece, 3)),
      size=fs,
      bbox=dict(boxstyle='round', fc='white', ec='grey', pad=0.2))


def plot_roc_curve(plot,
                   scores,
                   e,
                   t,
                   a,
                   folds,
                   groups,
                   quant):

  """Function to plot ROC at a specified time horizon.

  Accepts a matplotlib figure instance, risk scores from a trained survival
  analysis model, and quantiles of event interest and generates an IPCW
  adjusted ROC curve.

  Args:
    plot:
      a trained survival analysis model
      (output of fair_survival_analysis.models.train_model).
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

  fprs, tprs, tprs_std = {}, {}, {}

  fprs['all'] = {}
  tprs['all'] = {}

  for group in groups:

    fprs[group] = {}
    tprs[group] = {}

  for fold in set(folds):

    str_train = baseline_models.structureForEval_(t[folds != fold],
                                                  e[folds != fold])
    str_test = baseline_models.structureForEval_(t[folds == fold],
                                                 e[folds == fold])

    atr = a[folds != fold]
    ate = a[folds == fold]

    for group in groups:

      te_protg = (ate == group)
      tr_protg = (atr == group)

      roc_m = cumulative_dynamic_auc(str_train[tr_protg], str_test[te_protg],
                                     -scores[fold][te_protg], [quant])

      fprs[group][fold], tprs[group][fold] = roc_m[0][0][1], roc_m[0][0][0]

    roc_m = cumulative_dynamic_auc(str_train, str_test, -scores[fold], [quant])

    fprs['all'][fold], tprs['all'][fold] = roc_m[0][0][1], roc_m[0][0][0]

  cols = ['b', 'r', 'g']

  roc_auc = {}

  j = 0

  for group in groups + ['all']:

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

    plot.plot(
        all_fpr,
        mean_tpr,
        c=cols[j],
        label=group + ' AUC:' + str(round(roc_auc[group], 3)))
    plot.fill_between(
        all_fpr,
        mean_tpr - std_tpr,
        mean_tpr + std_tpr,
        color=cols[j],
        alpha=0.25)

    j += 1

  plot.set_xlabel('False Positive Rate', fontsize=fs)
  plot.set_ylabel('True Positive Rate', fontsize=fs)
  plot.legend(fontsize=fs)
  plot.set_xscale('log')


def plot_results(trained_model, model, fair_strategy, x, e, t, a,\
                 folds, groups, quantiles, strat='quantile', adj='IPCWcon'):

  """Function to plot the ROC and Calibration curves from a survival model.

  Accepts a trained survival analysis model, features and horizon of interest
  and generates the IPCW adjusted ROC curve and Calibration curve at
  pre-specified horizons of time.

  Args:
    trained_model:
      a trained survival analysis model
      (output of fair_survival_analysis.models.train_model).
    model:
      choice of model. One of "coupled_deep_cph", "coupled_deep_cph_vae".
    fair_strategy:
      List of the demogrpahics to adjust for. Must be same as what was used to
      originally train the model.
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

  Returns:
    a numpy vector of risks P(T>t) at the horizon "quant".

  """

  mpl.rcParams['hatch.linewidth'] = 2.0

  fig, big_axes = plt.subplots(
      figsize=(8 * (len(groups) + 1), 6 * len(quantiles)),
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

  for i in range(len(quantiles)):

    if model in ['coupled_deep_cph', 'coupled_deep_cph_vae']:

      scores = models.predict_scores(trained_model, groups, x, a, folds,
                                     quantiles[i])

    else:

      scores = baseline_models.predict_scores(trained_model, model,
                                              fair_strategy, x, a, folds,
                                              quantiles[i])
    for j in range(len(groups) + 1):

      pt = (i * (len(groups) + 1) + j + 1)

      ax = fig.add_subplot(len(quantiles), len(groups) + 1, pt)

      if j:

        plot_calibration_curve(
            ax,
            scores,
            e,
            t,
            a,
            folds,
            groups[j - 1],
            quantiles[i],
            strat=strat,
            adj=adj)

      else:

        plot_roc_curve(
            ax,
            scores,
            e,
            t,
            a,
            folds,
            groups,
            quantiles[i])

  plt.show()
