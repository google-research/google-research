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

r"""Evaluation on detecting key events using a RNN.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl import logging

import concurrent.futures as cf

import numpy as np
import sklearn

import tensorflow.compat.v2 as tf
from tcc.config import CONFIG
from tcc.dataset_splits import DATASET_TO_NUM_CLASSES
from tcc.evaluation.task import Task
from tcc.evaluation.task_utils import get_targets_from_labels
from tcc.evaluation.task_utils import unnormalize

FLAGS = flags.FLAGS
layers = tf.keras.layers


class VectorRegression(sklearn.base.BaseEstimator):
  """Class to perform regression on multiple outputs."""

  def __init__(self, estimator):
    self.estimator = estimator

  def fit(self, x, y):
    _, m = y.shape
    # Fit a separate regressor for each column of y
    self.estimators_ = [sklearn.base.clone(self.estimator).fit(x, y[:, i])
                        for i in range(m)]
    return self

  def predict(self, x):
    # Join regressors' predictions
    res = [est.predict(x)[:, np.newaxis] for est in self.estimators_]
    return np.hstack(res)

  def score(self, x, y):
    # Join regressors' scores
    res = [est.score(x, y[:, i]) for i, est in enumerate(self.estimators_)]
    return np.mean(res)


def get_error(predictions, labels, seq_lens, global_step, num_classes, prefix):
  """Get error based on predictions."""
  errs = []
  for i in xrange(num_classes - 1):
    abs_err = 0
    for j in xrange(len(predictions)):
      # Choose last seq_len steps as our preprocessing pads sequences in
      # front with zeros.
      unnorm_preds = unnormalize(predictions[j][:, i])
      unnorm_labels = unnormalize(labels[j][:, i])

      abs_err += abs(unnorm_labels - unnorm_preds) / seq_lens[j]

    err = abs_err / len(predictions)
    logging.info('[Global step: {}] {} {} Fraction Error: '
                 '{:.3f},'.format(global_step.numpy(), prefix, i, err))
    tf.summary.scalar('event_completion/%s_%d_error' % (prefix, i),
                      err, step=global_step)
    errs.append(err)

  avg_err = np.mean(errs)

  logging.info('[Global step: {}] {} Fraction Error: '
               '{:.3f},'.format(global_step.numpy(), prefix, avg_err))
  tf.summary.scalar('event_completion/avg_error_%s' % prefix,
                    avg_err, step=global_step)

  return avg_err


def fit_model(train_embs, train_labels, val_embs, val_labels,
              global_step, num_classes, prefix, report_error=False):
  """Linear Regression to regress to fraction completed."""

  train_seq_lens = [len(x) for x in train_labels]
  val_seq_lens = [len(x) for x in val_labels]

  train_embs = np.concatenate(train_embs, axis=0)
  train_labels = np.concatenate(train_labels, axis=0)
  val_embs = np.concatenate(val_embs, axis=0)
  val_labels = np.concatenate(val_labels, axis=0)

  lin_model = VectorRegression(sklearn.linear_model.LinearRegression())
  lin_model.fit(train_embs, train_labels)

  train_score = lin_model.score(train_embs, train_labels)
  val_score = lin_model.score(val_embs, val_labels)

  # Not used for evaluation right now.
  if report_error:
    val_predictions = lin_model.predict(val_embs)
    train_predictions = lin_model.predict(train_embs)

    train_labels = np.array_split(train_labels,
                                  np.cumsum(train_seq_lens))[:-1]
    train_predictions = np.array_split(train_predictions,
                                       np.cumsum(train_seq_lens))[:-1]
    val_labels = np.array_split(val_labels,
                                np.cumsum(val_seq_lens))[:-1]
    val_predictions = np.array_split(val_predictions,
                                     np.cumsum(val_seq_lens))[:-1]

    get_error(train_predictions, train_labels, train_seq_lens,
              global_step, num_classes, 'train_' + prefix)
    get_error(val_predictions, val_labels, val_seq_lens,
              global_step, num_classes, 'val_' + prefix)

  return train_score, val_score


class EventCompletion(Task):
  """Predict event completion using linear regression."""

  def __init__(self):
    super(EventCompletion, self).__init__(downstream_task=True)

  def evaluate_embeddings(self, algo, global_step, datasets):
    """Labeled evaluation."""
    fractions = CONFIG.EVAL.CLASSIFICATION_FRACTIONS

    train_embs = datasets['train_dataset']['embs']
    val_embs = datasets['val_dataset']['embs']
    num_classes = DATASET_TO_NUM_CLASSES[datasets['name']]

    if not train_embs or not val_embs:
      logging.warn('All embeddings are NAN. Something is wrong with model.')
      return 1.0

    val_labels = get_targets_from_labels(datasets['val_dataset']['labels'],
                                         num_classes)

    num_samples = len(datasets['train_dataset']['embs'])

    def worker(fraction_used):
      num_samples_used = max(1, int(fraction_used * num_samples))
      train_embs = datasets['train_dataset']['embs'][:num_samples_used]
      train_labels = get_targets_from_labels(
          datasets['train_dataset']['labels'][:num_samples_used], num_classes)
      return fit_model(train_embs, train_labels, val_embs, val_labels,
                       global_step, num_classes, '%s_%s' % (datasets['name'],
                                                            str(fraction_used)))
    val_scores = []
    with cf.ThreadPoolExecutor(max_workers=len(fractions)) as executor:
      results = executor.map(worker, fractions)
      for (fraction, (train_score, val_score)) in zip(fractions, results):
        prefix = '%s_%s' % (datasets['name'], str(fraction))
        logging.info('[Global step: {}] Event Completion {} Fraction Train '
                     'Score: {:.3f},'.format(global_step.numpy(), prefix,
                                             train_score))
        logging.info('[Global step: {}] Event Completion {} Fraction Val '
                     'Score: {:.3f},'.format(global_step.numpy(), prefix,
                                             val_score))
        tf.summary.scalar('event_completion/train_%s_score' % prefix,
                          train_score, step=global_step)
        tf.summary.scalar('event_completion/val_%s_score' % prefix,
                          val_score, step=global_step)
        val_scores.append(val_score)

    return val_scores[-1]
