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

r"""Evaluation train and val loss using the algo.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl import logging

import tensorflow.compat.v2 as tf
from tcc.config import CONFIG
from tcc.evaluation.task import Task
from tcc.utils import get_data
from tcc.utils import visualize_nearest_neighbours

FLAGS = flags.FLAGS


class AlgoLoss(Task):
  """Computes loss of algorithm in eval mode."""

  def __init__(self):
    super(AlgoLoss, self).__init__(downstream_task=False)

  def get_loss(self, algo, iterators, global_step, split):
    iterator = iterators['%s_iterator' % split]

    avg_loss = 0.0
    for _ in range(CONFIG.EVAL.VAL_ITERS):
      data, steps, seq_lens = get_data(iterator)
      embs = algo.call(data, steps, seq_lens, training=False)
      avg_loss += algo.compute_loss(embs, steps, seq_lens, global_step,
                                    training=False,
                                    frame_labels=data['frame_labels'],
                                    seq_labels=data['seq_labels'])

    avg_loss /= CONFIG.EVAL.VAL_ITERS
    logging.info('Iter[{}/{}] {} loss: {:.4f}'.format(
        global_step.numpy(), CONFIG.TRAIN.MAX_ITERS, split, avg_loss))

    tf.summary.scalar('algo_loss/%s_loss' % split, avg_loss, step=global_step)

    if FLAGS.visualize:
      # Visualize nearest neighbors.
      visualize_nearest_neighbours(
          algo.model,
          data,
          global_step,
          CONFIG.EVAL.BATCH_SIZE,
          CONFIG.EVAL.NUM_FRAMES,
          CONFIG.DATA.NUM_STEPS,
          split=split)

    return avg_loss

  def evaluate_iterators(self, algo, global_step, iterators):
    self.get_loss(algo, iterators, global_step, split='train')
    val_loss = self.get_loss(algo, iterators, global_step, split='val')
    return val_loss
