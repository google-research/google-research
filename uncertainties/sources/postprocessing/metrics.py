# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

# Lint as: python2, python3
"""Metrics for classification."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import scipy.stats as spstats
import six.moves.cPickle as pickle
import tensorflow.compat.v1 as tf

import uncertainties.sources.utils.util as util


class Metrics(object):
  """Metrics for classification.

  Compute:
  - Brier score
  - entropy
  - accuracy
  - mutual information
  - q probability and its entropy
  - calibration (reliability diagram, maximum calibration error,
                 expected calibration error)
  - mean of the output predicted probabilities
  - std of the output predicted probabilities.
  - AURC and related metrics.
  """

  def __init__(self, y, workingdir):
    """Initialization and launch the computation of the metrics."""
    self.y = y
    self.workingdir = workingdir
    self.compute()
    self.save_results()

  def compute(self):
    """Computation of the metrics."""
    # Initialization
    p_mean_list = []
    p_std_list = []
    q_tab_list = []
    mi_list = []
    path = os.path.join(self.workingdir, 'proba_*.npy')
    files_list = tf.gfile.Glob(path)
    n = len(files_list)
    for i in np.arange(n):
      path = os.path.join(self.workingdir, 'proba_' + str(i) + '.npy')
      with tf.gfile.Open(path, 'rb') as f:
        p_tab = np.load(f)
        mi_list.append(entropy(np.mean(p_tab, axis=2))
                       - np.mean(entropy(p_tab), axis=1))
        p_mean_list.append(np.mean(p_tab, axis=2))
        p_std_list.append(np.std(p_tab, axis=2))
        q_tab_list.append(q_probability(p_tab))
        num_items = p_tab.shape[0]
        y = self.y[i*num_items:(i+1)*num_items, :]
        if i == 0:
          neglog = (1./n) * negloglikelihood(y, p_tab)
          acc = (1./n) * accuracy(y, p_tab)
          bs = (1./n) * brier_score(y, p_tab)
        else:
          neglog += (1./n) * negloglikelihood(y, p_tab)
          acc += (1./n) * accuracy(y, p_tab)
          bs += (1./n) * brier_score(y, p_tab)
    p_mean = np.vstack(tuple(p_mean_list))
    p_std = np.vstack(tuple(p_std_list))
    q_tab = np.vstack(tuple(q_tab_list))
    mi = np.concatenate(tuple(mi_list))
    del p_mean_list
    del p_std_list
    del q_tab_list
    del mi_list
    self.ent = entropy(p_mean)
    self.cal = calibration(self.y, p_mean)
    # Saving the results
    self.neglog = neglog
    self.acc = acc
    self.bs = bs
    self.mi = mi
    self.p_mean = p_mean
    self.p_std = p_std
    self.q_tab = q_tab
    self.ent_q = entropy(self.q_tab)
    # Compute AURC
    self.aurc()

  def aurc(self):
    """Compute the AURC, and other related metrics.

    Pairs of (classifier, confidence):
      - (argmax p_mean, - p_std(argmax p_mean))
      - (argmax p_mean, max p_mean)
      - (argmax q, -entropy(q))
    """
    # Classifier = max p probability
    # Confidence = - std of the max probability along the samples
    y_pred = np.argmax(self.p_mean, axis=1)
    argmax_y = np.argmax(self.y, axis=1)
    conf = - self.p_std[np.arange(self.p_std.shape[0]), y_pred]
    self.risk_cov_std = sec_classification(argmax_y, y_pred, conf)
    # Confidence = softmax response
    conf = np.max(self.p_mean, axis=1)
    self.risk_cov_softmax = sec_classification(argmax_y, y_pred, conf)
    # Classifier = max q probability
    # Confidence = - entropy of q
    y_pred = np.argmax(self.q_tab, axis=1)
    conf = - entropy(self.q_tab)
    self.risk_cov_q = sec_classification(argmax_y, y_pred, conf)

  def save_results(self):
    """Save the results."""
    if tf.gfile.IsDirectory(os.path.join(self.workingdir, 'metrics')):
      tf.gfile.DeleteRecursively(os.path.join(self.workingdir, 'metrics'))
    tf.gfile.MakeDirs(os.path.join(self.workingdir, 'metrics'))
    result_dic = {'acc': self.acc,
                  'bs': self.bs,
                  'p_mean': self.p_mean,
                  'p_std': self.p_std,
                  'neglog': self.neglog,
                  'ent': self.ent,
                  'cal': self.cal,
                  'q_tab': self.q_tab,
                  'ent_q': self.ent_q,
                  'mi': self.mi,
                  'risk_cov_std': self.risk_cov_std,
                  'risk_cov_softmax': self.risk_cov_softmax,
                  'risk_cov_q': self.risk_cov_q
                 }
    with tf.gfile.Open(os.path.join(
        self.workingdir, 'metrics', 'metrics.pkl'), 'wb') as f:
      pickle.dump(result_dic, f, protocol=2)


def sec_classification(y_true, y_pred, conf):
  """Compute the AURC.

  Args:
    y_true: true labels, vector of size n_test
    y_pred: predicted labels by the classifier, vector of size n_test
    conf: confidence associated to y_pred, vector of size n_test
  Returns:
    conf: confidence sorted (in decreasing order)
    risk_cov: risk vs coverage (increasing coverage from 0 to 1)
    aurc: AURC
    eaurc: Excess AURC
  """
  n = len(y_true)
  ind = np.argsort(conf)
  y_true, y_pred, conf = y_true[ind][::-1], y_pred[ind][::-1], conf[ind][::-1]
  risk_cov = np.divide(np.cumsum(y_true != y_pred).astype(np.float),
                       np.arange(1, n+1))
  nrisk = np.sum(y_true != y_pred)
  aurc = np.mean(risk_cov)
  opt_aurc = (1./n) * np.sum(np.divide(np.arange(1, nrisk + 1).astype(np.float),
                                       n - nrisk + np.arange(1, nrisk + 1)))
  eaurc = aurc - opt_aurc
  return (conf, risk_cov, aurc, eaurc)


def q_probability(p_tab):
  """Compute the q probability.

  Args:
    p_tab: numpy array, size (?, num_classes, num_samples)
           containing the output predicted probabilities
  Returns:
    q_tab: the probability obtained by averaging the prediction of the ensemble
           of classifiers
  """
  q_tab = np.zeros_like(p_tab)
  d1, _, d2 = p_tab.shape
  q_tab[np.arange(d1).repeat(d2),
        np.argmax(p_tab, axis=1).flatten(), np.tile(np.arange(d2), d1)] = 1.
  q_tab = np.mean(q_tab, axis=2)
  return q_tab


def negloglikelihood(y, p_tab):
  """Compute the negative log-likelihood.

  Args:
    y: one-hot encoding of the true classes, size (?, num_classes)
    p_tab: numpy array, size (?, num_classes, num_samples)
           containing the output predicted probabilities
  Returns:
    neglog: negative log likelihood, along the iterations
            numpy vector of size num_samples
  """
  p_mean = util.cummean(p_tab[y.astype(np.bool), :], axis=1)
  neglog = - np.mean(np.log(p_mean), axis=0)
  return neglog


def accuracy(y, p_tab):
  """Compute the accuracy.

  Args:
    y: one-hot encoding of the true classes, size (?, num_classes)
    p_tab: numpy array, size (?, num_classes, num_samples)
           containing the output predicted probabilities
  Returns:
    acc: accuracy along the iterations, numpy vector of size num_samples
  """
  class_pred = np.argmax(util.cummean(p_tab, axis=2), axis=1)
  argmax_y = np.argmax(y, axis=1)
  acc = np.apply_along_axis(lambda x: np.mean(x == argmax_y),
                            axis=0, arr=class_pred)
  return acc


def brier_score(y, p_tab):
  """Compute the Brier score.

  Brier Score: see
  https://www.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf,
  page 363, Example 1

  Args:
    y: one-hot encoding of the true classes, size (?, num_classes)
    p_tab: numpy array, size (?, num_classes, num_samples)
           containing the output predicted probabilities
  Returns:
    bs: Brier score along the iteration, vector of size num_samples.
  """
  p_cummean = util.cummean(p_tab, axis=2)
  y_repeated = np.repeat(y[:, :, np.newaxis], p_tab.shape[2], axis=2)
  bs = np.mean(np.power(p_cummean - y_repeated, 2), axis=(0, 1))
  return bs


def entropy(p_mean):
  """Compute the entropy.

  Args:
    p_mean: numpy array, size (?, num_classes, ?)
           containing the (possibly mean) output predicted probabilities
  Returns:
    ent: entropy along the iterations, numpy vector of size (?, ?)
  """
  ent = np.apply_along_axis(spstats.entropy, axis=1, arr=p_mean)
  return ent


def calibration(y, p_mean, num_bins=10):
  """Compute the calibration.

  References:
  https://arxiv.org/abs/1706.04599
  https://arxiv.org/abs/1807.00263

  Args:
    y: one-hot encoding of the true classes, size (?, num_classes)
    p_mean: numpy array, size (?, num_classes)
           containing the mean output predicted probabilities
    num_bins: number of bins

  Returns:
    cal: a dictionary
      {reliability_diag: realibility diagram
       ece: Expected Calibration Error
       mce: Maximum Calibration Error
      }
  """
  # Compute for every test sample x, the predicted class.
  class_pred = np.argmax(p_mean, axis=1)
  # and the confidence (probability) associated with it.
  conf = np.max(p_mean, axis=1)
  # Convert y from one-hot encoding to the number of the class
  y = np.argmax(y, axis=1)
  # Storage
  acc_tab = np.zeros(num_bins)  # empirical (true) confidence
  mean_conf = np.zeros(num_bins)  # predicted confidence
  nb_items_bin = np.zeros(num_bins)  # number of items in the bins
  tau_tab = np.linspace(0, 1, num_bins+1)  # confidence bins
  for i in np.arange(num_bins):  # iterate over the bins
    # select the items where the predicted max probability falls in the bin
    # [tau_tab[i], tau_tab[i + 1)]
    sec = (tau_tab[i + 1] > conf) & (conf >= tau_tab[i])
    nb_items_bin[i] = np.sum(sec)  # Number of items in the bin
    # select the predicted classes, and the true classes
    class_pred_sec, y_sec = class_pred[sec], y[sec]
    # average of the predicted max probabilities
    mean_conf[i] = np.mean(conf[sec]) if nb_items_bin[i] > 0 else np.nan
    # compute the empirical confidence
    acc_tab[i] = np.mean(
        class_pred_sec == y_sec) if nb_items_bin[i] > 0 else np.nan

  # Cleaning
  mean_conf = mean_conf[nb_items_bin > 0]
  acc_tab = acc_tab[nb_items_bin > 0]
  nb_items_bin = nb_items_bin[nb_items_bin > 0]

  # Reliability diagram
  reliability_diag = (mean_conf, acc_tab)
  # Expected Calibration Error
  ece = np.average(
      np.absolute(mean_conf - acc_tab),
      weights=nb_items_bin.astype(np.float) / np.sum(nb_items_bin))
  # Maximum Calibration Error
  mce = np.max(np.absolute(mean_conf - acc_tab))
  # Saving
  cal = {'reliability_diag': reliability_diag,
         'ece': ece,
         'mce': mce}
  return cal
