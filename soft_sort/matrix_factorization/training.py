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

# Lint as: python3
"""Puts everything together to train any factorization on any dataset."""

import os.path
from typing import Optional

from absl import logging
import gin
import tensorflow.compat.v2 as tf


@gin.configurable
class TrainingLoop(object):
  """A class to run and configure the training of a matrix factorization."""

  def __init__(self,
               workdir: Optional[str] = None,
               data_loader=gin.REQUIRED,
               factorizer=gin.REQUIRED,
               steps: int = 100,
               max_checkpoints: int = 5):
    self._workdir = workdir
    self._data_loader = data_loader
    self.factorizer = factorizer
    self._steps = steps
    self._inputs = self._data_loader.make()
    self.factorizer.reset(self._inputs)

    self._summary_writer = None
    self._ckpt = None
    self._ckpt_manager = None
    if self._workdir is not None:
      self._summary_writer = tf.summary.create_file_writer(self._workdir)
      variables = dict(step=tf.Variable(1))
      variables.update(self.factorizer.ckpt_variables)
      self._ckpt = tf.train.Checkpoint(**variables)
      ckpt_folder = os.path.join(self._workdir, 'ckpts')
      self._ckpt_manager = tf.train.CheckpointManager(
          self._ckpt, ckpt_folder, max_to_keep=max_checkpoints)

  def may_restore(self):
    """May restore from an old checkpoint."""
    if self._ckpt_manager is not None:
      self._ckpt.restore(self._ckpt_manager.latest_checkpoint)
      if self._ckpt_manager.latest_checkpoint:
        logging.info('Restore from %s', self._ckpt_manager.latest_checkpoint)

  def run(self, steps=None):
    """Runs the training loop taking care of checkpointing."""
    self.may_restore()
    steps = self._steps if steps is None else steps
    start = 0 if self._ckpt is None else int(self._ckpt.step)
    for step in range(start, steps):
      if self._ckpt is not None:
        self._ckpt.step.assign_add(1)

      loss = self.factorizer.update()
      print('Step {}: loss={:.3f}'.format(step, loss), end='\r')

      if self._workdir is not None:
        self._ckpt_manager.save()
        with self._summary_writer.as_default():
          tf.summary.scalar('loss', loss, step=step)
