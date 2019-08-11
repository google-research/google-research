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

r"""Evaluation on per-frame labels for few-shot classification.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl import logging

import concurrent.futures as cf

import numpy as np
import scipy.stats as st
import tensorflow.compat.v2 as tf

from tcc.config import CONFIG
from tcc.evaluation.classification import fit_linear_models
from tcc.evaluation.task import Task

FLAGS = flags.FLAGS


class FewShotClassification(Task):
  """Classification using small linear models."""

  def __init__(self):
    super(FewShotClassification, self).__init__(downstream_task=True)

  def evaluate_embeddings(self, algo, global_step, datasets):
    """Labeled evaluation."""
    num_labeled_list = [int(x) for x in CONFIG.EVAL.FEW_SHOT_NUM_LABELED]
    num_episodes = int(CONFIG.EVAL.FEW_SHOT_NUM_EPISODES)

    # Set random seed to ensure same samples are generated for each evaluation.
    np.random.seed(seed=42)

    train_embs = np.concatenate(datasets['train_dataset']['embs'])
    val_embs = np.concatenate(datasets['val_dataset']['embs'])

    if train_embs.shape[0] == 0 or val_embs.shape[0] == 0:
      logging.warn('All embeddings are NAN. Something is wrong with model.')
      return 0.0

    val_labels = np.concatenate(datasets['val_dataset']['labels'])

    report_val_accs = []
    train_dataset = datasets['train_dataset']
    num_samples = len(train_dataset['embs'])

    # Create episode list.
    episodes_list = []
    for num_labeled in num_labeled_list:
      episodes = []
      for _ in range(num_episodes):
        episodes.append(np.random.permutation(num_samples)[:num_labeled])
      episodes_list.append(episodes)

    def indi_worker(episode):
      """Executes single epsiode for a particular k-shot task."""
      train_embs = np.concatenate(np.take(train_dataset['embs'], episode))
      train_labels = np.concatenate(np.take(train_dataset['labels'], episode))
      train_acc, val_acc = fit_linear_models(train_embs, train_labels,
                                             val_embs, val_labels)
      return train_acc, val_acc

    def worker(episodes):
      """Executes all epsiodes for a particular k-shot task."""
      with cf.ThreadPoolExecutor() as executor:
        results = executor.map(indi_worker, episodes)
      results = list(zip(*results))
      train_accs = results[0]
      val_accs = results[1]
      return train_accs, val_accs

    with cf.ThreadPoolExecutor() as executor:
      results = executor.map(worker, episodes_list)

      for (num_labeled, (train_accs, val_accs)) in zip(num_labeled_list,
                                                       results):
        prefix = '%s_%s' % (datasets['name'], str(num_labeled))

        # Get average accuracy over all episodes.
        train_acc = np.mean(np.mean(train_accs))
        val_acc = np.mean(np.mean(val_accs))

        # Get 95% Confidence Intervals.
        train_ci = st.t.interval(0.95, len(train_accs) - 1, loc=train_acc,
                                 scale=st.sem(train_accs))[1] - train_acc
        val_ci = st.t.interval(0.95, len(val_accs) - 1, loc=val_acc,
                               scale=st.sem(val_accs))[1] - val_acc

        logging.info('[Global step: {}] Classification {} Shot '
                     'Train Accuracy: {:.4f},'.format(global_step.numpy(),
                                                      prefix,
                                                      train_acc))
        logging.info('[Global step: {}] Classification {} Shot '
                     'Val Accuracy: {:.4f},'.format(global_step.numpy(),
                                                    prefix,
                                                    val_acc))

        logging.info('[Global step: {}] Classification {} Shot '
                     'Train Confidence Interval: {:.4f},'.format(
                         global_step.numpy(), prefix, train_ci))
        logging.info('[Global step: {}] Classification {} Shot '
                     'Val Confidence Interval: {:.4f},'.format(
                         global_step.numpy(), prefix, val_ci))

        tf.summary.scalar('few_shot_cxn/train_%s_accuracy' % prefix,
                          train_acc, step=global_step)
        tf.summary.scalar('few_shot_cxn/val_%s_accuracy' % prefix,
                          val_acc, step=global_step)
        tf.summary.scalar('few_shot_cxn/train_%s_ci' % prefix,
                          train_ci, step=global_step)
        tf.summary.scalar('few_shot_cxn/val_%s_ci' % prefix,
                          val_ci, step=global_step)

        report_val_accs.append(val_acc)

    return report_val_accs[-1]
