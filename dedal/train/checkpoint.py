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

"""Handles model checkpointing."""

import os
import time
from typing import Optional

from absl import logging
import gin
import tensorflow as tf


@gin.configurable
class Checkpointer:
  """A class to encapsulate checkpoints."""

  def __init__(self,
               workdir,
               strategy,
               model,
               step,
               optimizer=None,
               max_to_keep = 5,
               save_every = 1000,
               **kwargs):
    self._workdir = os.path.join(workdir, 'checkpoints')
    self._strategy = strategy
    self._ckpt = None
    self._manager = None
    self._model = model
    self._step = step
    self._kwargs = kwargs
    if optimizer is not None and optimizer not in kwargs:
      self._kwargs.update(optimizer=optimizer)

    self._save_every = save_every
    self._max_to_keep = max_to_keep

  def restore(self, expect_partial = False):
    """Tries to restore the latest checkpoint in the directory."""
    if self._workdir is None:
      return

    with self._strategy.scope():
      ckpt = tf.train.Checkpoint(
          model=self._model, step=self._step, **self._kwargs)
      self._manager = tf.train.CheckpointManager(
          ckpt, directory=self._workdir, max_to_keep=self._max_to_keep)
      if not expect_partial:
        latest = self._manager.restore_or_initialize()
      else:
        latest = ckpt.restore(self._manager.latest_checkpoint).expect_partial()

    if latest is not None:
      logging.info('Successfully loaded checkpoint: %i from %s',
                   self._step.numpy(), self._workdir)
    else:
      logging.info('No checkpoint found in %s', self._workdir)
    return latest

  def restore_after(self, step, retries_after = 30):
    while True:
      latest = self.restore()
      if latest is not None and self._step.numpy() > step:
        return
      logging.info('Cannot restore newer model (> %i) in %s. Trying again',
                   step, self._workdir)
      time.sleep(retries_after)

  def may_save(self, last = False):
    if last or (self._step.numpy() % self._save_every == 0):
      with self._strategy.scope():
        self._manager.save(self._step)

  def delete(self):
    """Removes all the checkpoints under the workdir."""
    logging.warning('Deleting all checkpoint under %s', self._workdir)
    tf.io.gfile.rmtree(self._workdir)
    tf.io.gfile.mkdir(self._workdir)

  def set_model(self, model):
    self._model = model
