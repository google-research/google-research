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

"""Module containing Dataset class for empirical data."""
import collections
import os
import pickle

import numpy as np
from scipy.optimize import brute
from scipy.special import loggamma
from scipy.special import softmax
import scipy.stats as sps
import tensorflow as tf

from caltrain import calibration_metrics


class Dataset:
  """Class for empirical dataset."""

  @staticmethod
  def get_dataset(model, fold, data_dir=None):
    """Get a dataset object for a particular model and fold."""

    if data_dir is None:
      src_file = None
    else:
      src_file = os.path.join(data_dir,
                              'probs_{model}_logits.p'.format(model=model))

    return Dataset(model=model, path=src_file, fold=fold)

  def __init__(self,
               logits=None,
               y=None,
               model=None,
               resampled=False,
               path=None,
               fold=None):

    self._logits = logits
    self._y = y
    self.model = model
    self.resampled = resampled
    self.path = path
    self.fold = fold

  @property
  def logits(self):
    if self._logits is None:
      self.initialize()
    return self._logits

  @property
  def y(self):
    if self._y is None:
      self.initialize()
    return self._y

  def initialize(self):
    y = {}
    logits = {}
    fold = self.fold
    with tf.io.gfile.GFile(self.path, 'rb') as f:
      (logits[fold.val], y[fold.val]), (logits[fold.test],
                                        y[fold.test]) = pickle.load(f)
      self._y = np.squeeze(y[fold])
      self._logits = logits[fold]

  @property
  def y_hat(self):
    return np.argmax(self.logits, axis=1)

  @property
  def accuracy(self):
    return float((self.y == self.y_hat).sum()) / len(self.y)

  @property
  def scores(self):
    return softmax(self.logits, axis=1)

  def compute_error(self, **kwargs):
    calibration_metric = calibration_metrics.CalibrationMetric(**kwargs)
    raw_labels = np.eye(self.num_classes)[self.y]
    return calibration_metric.compute_error(self.scores, raw_labels)

  @property
  def num_classes(self):
    return self.logits.shape[1]

  def apply_mask(self, mask):
    return Dataset(
        self.logits[mask, :],
        y=self.y[mask],
        model=self.model,
        resampled=self.resampled)

  def __len__(self):
    return self.logits.shape[0]

  def bootstrap_resample(self, num_samples=None, seed=None):
    """Generate a new dataset, with resampled logits and values."""

    assert not self.resampled
    if num_samples is None:
      num_samples = len(self)
    if seed is None:
      rng = np.random.RandomState()
    else:
      rng = np.random.RandomState(seed)

    bootstrap_inds = rng.choice(np.arange(num_samples), num_samples)
    logits_bootstrapped = self.logits[bootstrap_inds, :]
    y_bootstrapped = self.y[bootstrap_inds]

    return Dataset(
        logits_bootstrapped, y=y_bootstrapped, model=self.model, resampled=True)

  @property
  def top_scores(self):
    return self.scores.max(axis=1)

  def beta_shift_fit(self, arange=(0, 50), brange=(0, 50), n_s=11, shift=1e-16):
    """Fit beta distribution, with a absolute shift in the data."""

    top_scores = self.top_scores.astype(np.float64) - shift

    beta_fit_p1_dict = {'loc': 0, 'scale': 1, 'p1': 0}

    def f(a_b, top_scores=top_scores):
      a, b = a_b
      nll = 0
      nll -= loggamma(a + b)
      nll -= -(loggamma(a) + loggamma(b))
      nll -= (a - 1) * np.log(top_scores)
      nll -= (b - 1) * np.log(1 - top_scores)
      nll = nll.sum()
      fout = nll

      if not np.isfinite(fout):
        fout = float('inf')
      return fout

    result, nll, _, _ = brute(
        f, ranges=[arange, brange], Ns=n_s, full_output=True, finish=None)
    beta_fit_p1_dict['a'] = result[0]
    beta_fit_p1_dict['b'] = result[1]
    beta_fit_p1_dict['nll'] = nll
    beta_fit_p1_dict['AIC'] = 2 * 2 + 2 * nll

    if result[0] in arange or result[1] in brange:
      return beta_fit_p1_dict, False
    return beta_fit_p1_dict, True

  def fit_glm(self, glm_model, remove_ones=True, **kwargs):
    if remove_ones:
      ds = self.apply_mask(self.top_scores != 1)
    else:
      ds = self
    return glm_model.fit(ds.top_scores, ds.y == ds.y_hat, **kwargs)

  def fit_glm_bootstrap(self, glm_model, n=20):
    """Fit GLM, using bootstrap resamplig to estimate parameter distributions."""

    data_dict = collections.defaultdict(list)
    for _ in range(n):
      beta_hat_poly, nll, aic = self.bootstrap_resample().fit_glm(glm_model)
      data_dict['beta_hat_poly'].append(beta_hat_poly)
      data_dict['nll'].append(nll)
      data_dict['AIC'].append(aic)

    stat_dict = {'AIC': {}, 'nll': {}}
    for key in ['AIC', 'nll']:
      fit = sps.bayes_mvs(np.array(data_dict[key]))
      stat_dict[key]['mean'] = {
          'statistic': fit[0].statistic,
          'minmax': list(fit[0].minmax)
      }
      stat_dict[key]['std'] = {
          'statistic': fit[2].statistic,
          'minmax': list(fit[2].minmax)
      }

    for bi, beta_data in enumerate(zip(*data_dict['beta_hat_poly'])):
      key = f'b{bi}'
      if np.abs(beta_data).sum() == 0:
        continue
      fit = sps.bayes_mvs(beta_data)
      stat_dict[key] = {}
      stat_dict[key]['mean'] = {
          'statistic': fit[0].statistic,
          'minmax': list(fit[0].minmax)
      }
      stat_dict[key]['std'] = {
          'statistic': fit[2].statistic,
          'minmax': list(fit[2].minmax)
      }

    return stat_dict

  def plot_emperical_accuracy(
      self,
      ax,
      nbins=35,
      transform_x=lambda x: x,
      transform_y=lambda y: y,
      plot_yx=True,
      how='star',
      xlim=None,
      ylim=None,
      fontsize=12,
      ylabel_formatter='${var}$',
      xlabel_formatter='${var}$',
      extra_lineplot_list=None,
      show_legend=False,
      emp_acc_label=None,
  ):
    """Plot the emirical accuracy function for the dataset."""

    if extra_lineplot_list is None:
      extra_lineplot_list = []

    dataset = self.apply_mask(self.top_scores != 1)
    x_data = transform_x(dataset.top_scores)
    y_data = dataset.y == dataset.y_hat

    x_min, x_max = x_data.min(), x_data.max()
    if xlim is None:
      xlim = x_min, x_max
    xbins = np.linspace(x_min, x_max, nbins + 1)
    xbinsc, emp_acc = [], []
    for lbin, rbin in zip(xbins[:-1], xbins[1:]):
      cbin = np.logical_and(lbin <= x_data, x_data < rbin)
      xbinsc.append((lbin + rbin) / 2)
      c_eq = np.logical_and(cbin, y_data).sum()
      c_neq = np.logical_and(cbin, np.logical_not(y_data)).sum()
      emp_acc.append(float(c_eq) / (c_eq + c_neq))
    xbinsc = np.array(xbinsc)
    emp_acc = transform_y(np.array(emp_acc))
    if ylim is None:
      emp_acc_finite = emp_acc[np.isfinite(emp_acc)]
      ylim = emp_acc_finite.min(), emp_acc_finite.max()
    if plot_yx:
      ax.plot(xlim, xlim, 'r--')
    if how == 'star':
      ax.plot(xbinsc, emp_acc, 'r*', label=emp_acc_label)
    elif how == 'bar':
      ax.bar(xbinsc, emp_acc, width=(xbinsc[1] - xbins[0]) * .5)

    for extra_lineplot_data in extra_lineplot_list:
      xfunc = extra_lineplot_data['xfunc']
      yfunc = extra_lineplot_data['yfunc']
      style = extra_lineplot_data.get('style', 'b-')
      label = extra_lineplot_data.get('label', None)
      xdata = xfunc(xbinsc)
      ydata = yfunc(xdata)
      ax.plot(xdata, ydata, style, label=label)

    ax.set_xlabel(xlabel_formatter.format(var='x'))
    ax.set_ylabel(ylabel_formatter.format(var=r'\mu'))
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    if show_legend:
      ax.legend(loc=0, prop={'size': fontsize})
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
      item.set_fontsize(fontsize)

  def plot_top_score_histogram(self,
                               ax,
                               fontsize=12,
                               nbins=25,
                               log=True,
                               bar_width_frac=.5,
                               bin_range=None,
                               x_transform_fcn=None,
                               legend=True):
    """Generate a histogram of top-scores for the dataset."""

    data_neq = self.apply_mask(self.y != self.y_hat).top_scores
    data_eq = self.apply_mask(self.y == self.y_hat).top_scores

    if x_transform_fcn is None and bin_range is None:
      bin_range = (0, 1)
    elif bin_range is None:
      data_neq_pre = x_transform_fcn(data_neq)
      data_neq = data_neq_pre[np.isfinite(data_neq_pre)]
      data_eq_pre = x_transform_fcn(data_eq)
      data_eq = data_eq_pre[np.isfinite(data_eq_pre)]
      bin_range_min = min(data_neq.min(), data_eq.min())
      bin_range_max = min(data_neq.max(), data_eq.max())
      bin_range = bin_range_min, bin_range_max

    bins = np.linspace(bin_range[0], bin_range[1], nbins + 1)
    count_eq, xbins_eq = np.histogram(data_neq, range=bin_range, bins=bins)
    count_neq, xbins_neq = np.histogram(data_eq, range=bin_range, bins=bins)
    np.testing.assert_allclose(xbins_neq, xbins_eq)
    xbinsc = xbins_neq[:-1] / 2 + xbins_neq[1:] / 2

    ax.bar(
        xbinsc,
        count_eq,
        width=(xbinsc[1] - xbinsc[0]) * bar_width_frac,
        log=log,
        label='Hit',
        color='b')
    ax.bar(
        xbinsc,
        count_neq,
        width=(xbinsc[1] - xbinsc[0]) * bar_width_frac,
        log=log,
        bottom=count_eq,
        label='Miss',
        color='r')

    ax.set_title('P[f(x)]')
    ax.set_xlabel('Top-1 Scores')
    ax.set_ylabel('Count')
    ax.set_xlim(bin_range)
    if legend:
      ax.legend(loc='best', prop={'size': fontsize})

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
      item.set_fontsize(fontsize)

    return
