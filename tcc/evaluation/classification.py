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

r"""Evaluation on per-frame labels for classification.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl import logging

import concurrent.futures as cf

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import tensorflow.compat.v2 as tf

from tcc.config import CONFIG
from tcc.evaluation.task import Task
from tcc.utils import gen_plot

FLAGS = flags.FLAGS


def fit_linear_model(train_embs, train_labels,
                     val_embs, val_labels):
  """Fit a linear classifier."""
  lin_model = LogisticRegression(max_iter=100000, solver='lbfgs',
                                 multi_class='multinomial', verbose=2)
  lin_model.fit(train_embs, train_labels)
  train_acc = lin_model.score(train_embs, train_labels)
  val_acc = lin_model.score(val_embs, val_labels)
  return lin_model, train_acc, val_acc


def fit_svm_model(train_embs, train_labels,
                  val_embs, val_labels):
  """Fit a SVM classifier."""
  svm_model = SVC(decision_function_shape='ovo', verbose=2)
  svm_model.fit(train_embs, train_labels)
  train_acc = svm_model.score(train_embs, train_labels)
  val_acc = svm_model.score(val_embs, val_labels)
  return svm_model, train_acc, val_acc


def fit_linear_models(train_embs, train_labels, val_embs, val_labels,
                      model_type='svm'):
  """Fit Log Regression and SVM Models."""
  if model_type == 'linear':
    _, train_acc, val_acc = fit_linear_model(train_embs, train_labels,
                                             val_embs, val_labels)
  elif model_type == 'svm':
    _, train_acc, val_acc = fit_svm_model(train_embs, train_labels,
                                          val_embs, val_labels)
  else:
    raise ValueError('%s model type not supported' % model_type)
  return train_acc, val_acc


class Classification(Task):
  """Classification using small linear models."""

  def __init__(self):
    super(Classification, self).__init__(downstream_task=True)

  def evaluate_embeddings(self, algo, global_step, datasets):
    """Labeled evaluation."""
    fractions = CONFIG.EVAL.CLASSIFICATION_FRACTIONS

    train_embs = np.concatenate(datasets['train_dataset']['embs'])
    val_embs = np.concatenate(datasets['val_dataset']['embs'])

    if train_embs.shape[0] == 0 or val_embs.shape[0] == 0:
      logging.warn('All embeddings are NAN. Something is wrong with model.')
      return 0.0

    val_labels = np.concatenate(datasets['val_dataset']['labels'])

    val_accs = []
    train_dataset = datasets['train_dataset']
    num_samples = len(train_dataset['embs'])

    def worker(fraction_used):
      num_samples_used = max(1, int(fraction_used * num_samples))
      train_embs = np.concatenate(train_dataset['embs'][:num_samples_used])
      train_labels = np.concatenate(train_dataset['labels'][:num_samples_used])
      return fit_linear_models(train_embs, train_labels, val_embs, val_labels)

    with cf.ThreadPoolExecutor(max_workers=len(fractions)) as executor:
      results = executor.map(worker, fractions)
      for (fraction, (train_acc, val_acc)) in zip(fractions, results):
        prefix = '%s_%s' % (datasets['name'], str(fraction))
        logging.info('[Global step: {}] Classification {} Fraction'
                     'Train Accuracy: {:.3f},'.format(global_step.numpy(),
                                                      prefix, train_acc))
        logging.info('[Global step: {}] Classification {} Fraction'
                     'Val Accuracy: {:.3f},'.format(global_step.numpy(),
                                                    prefix, val_acc))
        tf.summary.scalar('classification/train_%s_accuracy' % prefix,
                          train_acc, step=global_step)
        tf.summary.scalar('classification/val_%s_accuracy' % prefix,
                          val_acc, step=global_step)
        val_accs.append(val_acc)

    if len(fractions) > 1 and FLAGS.visualize:
      image = gen_plot(fractions, val_accs)
      tf.summary.image('val_accuracy', image, step=global_step)

    return val_accs[-1]
