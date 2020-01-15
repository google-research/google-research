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

r"""Evaluation train and val loss using the algo.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl import logging

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import kendalltau
import tensorflow.compat.v2 as tf

from tcc.config import CONFIG
from tcc.evaluation.task import Task
from tcc.utils import softmax

FLAGS = flags.FLAGS


def _get_kendalls_tau(embs_list, stride, global_step, split):
  """Get nearest neighbours in embedding space and calculate Kendall's Tau."""
  num_seqs = len(embs_list)
  taus = np.zeros((num_seqs * (num_seqs - 1)))
  idx = 0
  for i in range(num_seqs):
    query_feats = embs_list[i][::stride]
    for j in range(num_seqs):
      if i == j:
        continue
      candidate_feats = embs_list[j][::stride]
      dists = cdist(query_feats, candidate_feats,
                    CONFIG.EVAL.KENDALLS_TAU_DISTANCE)
      if FLAGS.visualize:
        if i == 0 and j == 1:
          sim_matrix = []
          for k in range(len(query_feats)):
            sim_matrix.append(softmax(-dists[k]))
          sim_matrix = np.array(sim_matrix, dtype=np.float32)
          # Convert to format expected by tf.summary .
          sim_matrix = sim_matrix[None, :, :, None]
          tf.summary.image('%s/sim_matrix' % split, sim_matrix,
                           step=global_step)
      nns = np.argmin(dists, axis=1)
      taus[idx] = kendalltau(np.arange(len(nns)), nns).correlation
      idx += 1
  # Remove NaNs.
  taus = taus[~np.isnan(taus)]
  tau = np.mean(taus)

  logging.info('Iter[{}/{}] {} set alignment tau: {:.4f}'.format(
      global_step.numpy(), CONFIG.TRAIN.MAX_ITERS, split, tau))

  tf.summary.scalar('kendalls_tau/%s_align_tau' % split, tau, step=global_step)
  return tau


class KendallsTau(Task):
  """Calculate Kendall's Tau."""

  def __init__(self):
    super(KendallsTau, self).__init__(downstream_task=True)

  def evaluate_embeddings(self, algo, global_step, datasets):
    """Labeled evaluation."""
    train_embs = datasets['train_dataset']['embs']

    _get_kendalls_tau(
        train_embs,
        CONFIG.EVAL.KENDALLS_TAU_STRIDE,
        global_step,
        split='%s_train' % datasets['name'])

    val_embs = datasets['val_dataset']['embs']

    tau = _get_kendalls_tau(val_embs, CONFIG.EVAL.KENDALLS_TAU_STRIDE,
                            global_step, '%s_val' % datasets['name'])
    return tau
