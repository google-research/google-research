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

"""Alignment + SaL+ TCN loss for unsupervised training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tcc.algos.alignment import Alignment
from tcc.algos.sal import SaL
from tcc.algos.tcn import TCN
from tcc.config import CONFIG


class AlignmentSaLTCN(TCN):
  """Network trained with combination losses."""

  def __init__(self, model=None):
    super(AlignmentSaLTCN, self).__init__(model)
    algo_config = CONFIG.ALIGNMENT_SAL_TCN
    self.alignment_loss_weight = algo_config.ALIGNMENT_LOSS_WEIGHT
    self.sal_loss_weight = algo_config.SAL_LOSS_WEIGHT
    self.tcn_loss_weight = (1.0 - self.alignment_loss_weight -
                            self.sal_loss_weight)
    if self.alignment_loss_weight + self.sal_loss_weight > 1.0:
      raise ValueError('Sum of weights > 1 Not allowed.')
    if self.alignment_loss_weight < 0 or self.sal_loss_weight < 0:
      raise ValueError('Negative weights not allowed.')

    self.algos = []
    if self.alignment_loss_weight > 0:
      self.alignment_algo = Alignment(self.model)
      self.algos.append(self.alignment_algo)
    if self.sal_loss_weight > 0:
      self.sal_algo = SaL(self.model)
      self.algos.append(self.sal_algo)
    if self.tcn_loss_weight > 0:
      self.tcn_algo = TCN(self.model)
      self.algos.append(self.tcn_algo)

  def get_algo_variables(self):
    algo_variables = []
    for algo in self.algos:
      algo_variables.extend(algo.get_algo_variables())
    return algo_variables

  def compute_loss(self, embs, steps, seq_lens, global_step, training,
                   frame_labels, seq_labels):

    if self.tcn_loss_weight != 0.0:
      tcn_loss = self.tcn_algo.compute_loss(embs, steps, seq_lens, global_step,
                                            training, frame_labels, seq_labels)
      if training:
        tf.summary.scalar('alignment_sal_tcn/tcn_loss', tcn_loss,
                          step=global_step)
    else:
      tcn_loss = 0.0

    if self.alignment_loss_weight != 0.0 or self.sal_loss_weight != 0.0:
      if training:
        batch_size = CONFIG.TRAIN.BATCH_SIZE
        num_steps = CONFIG.TRAIN.NUM_FRAMES
      else:
        batch_size = CONFIG.EVAL.BATCH_SIZE
        num_steps = CONFIG.EVAL.NUM_FRAMES

      embs_list = []
      steps_list = []
      seq_lens_list = []

      for i in xrange(int(batch_size)):
        # Randomly sample half of TCN frames as in datasets.py we already
        # sample double the number of frames because it requires positives for
        # training.
        chosen_steps = tf.cond(tf.random.uniform(()) < 0.5,
                               lambda: tf.range(0, 2 * num_steps, 2),
                               lambda: tf.range(1, 2 * num_steps, 2))

        embs_ = tf.gather(embs[i], chosen_steps)
        steps_ = tf.gather(steps[i], chosen_steps)

        embs_list.append(embs_)
        steps_list.append(steps_)
        seq_lens_list.append(seq_lens[i])

      embs = tf.stack(embs_list)
      steps = tf.stack(steps_list)
      seq_lens = tf.stack(seq_lens_list)

    if self.alignment_loss_weight != 0:
      alignment_loss = self.alignment_algo.compute_loss(embs, steps, seq_lens,
                                                        num_steps, batch_size,
                                                        global_step, training)
      if training:
        tf.summary.scalar('alignment_sal_tcn/alignment_loss',
                          alignment_loss, step=global_step)
    else:
      alignment_loss = 0.0

    if self.sal_loss_weight != 0:
      sal_loss = self.sal_algo.compute_loss(embs, steps, seq_lens, global_step,
                                            training, frame_labels, seq_labels)

      if training:
        tf.summary.scalar('alignment_sal_tcn/sal_loss', sal_loss,
                          step=global_step)
    else:
      sal_loss = 0.0

    return (self.alignment_loss_weight * alignment_loss +
            self.sal_loss_weight * sal_loss +
            self.tcn_loss_weight * tcn_loss)
