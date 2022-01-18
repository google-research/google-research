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

# Lint as: python3
"""Training of a Keras model."""

import os.path
import time

import gin
import numpy as np
import tensorflow.compat.v2 as tf

from perturbations.experiments import data
from perturbations.experiments import metrics as perturbed_metrics
from perturbations.experiments import models


@gin.configurable
class TrainingLoop(object):
  """Hand made training loop."""

  TRAIN = 'train'
  TEST = 'test'
  TAGS = [TRAIN, TEST]

  def __init__(self,
               workdir: str,
               data_loader=None,
               model_fn=models.vanilla_cnn,
               loss_fn: tf.keras.losses.Loss = None,
               learning_rate: float = 1e-4,
               learning_rate_schedule=None,
               epochs: int = 10,
               batch_size: int = 8,
               optimizer: str = 'adam',
               metrics=()):

    self.data = data.DataLoader() if data_loader is None else data_loader
    self.model = model_fn(input_shape=self.data.input_shape,
                          output_shape=self.data.output_shape,
                          dtype=self.data.dtype)
    self._loss_fn = loss_fn
    self._learning_rate = learning_rate
    self.learning_rate_schedule = dict()
    if learning_rate_schedule is not None:
      self.learning_rate_schedule = dict(learning_rate_schedule)
    self._optimizer = tf.keras.optimizers.get(optimizer)
    self._optimizer.learning_rate = learning_rate
    self._epochs = epochs
    self._batch_size = batch_size
    self._checkpoint = None
    if workdir is not None:
      self._checkpoint_directory = os.path.join(workdir, 'checkpoints')
      self._checkpoint_prefix = os.path.join(self._checkpoint_directory, 'ckpt')
      self._checkpoint = tf.train.Checkpoint(
          optimizer=self._optimizer, model=self.model)

    self._metrics = {}
    self._summary_writer = {}
    for tag in self.TAGS:
      self._metrics[tag] = [tf.keras.metrics.Mean(name='loss')]
      for cls in metrics:
        self._metrics[tag].append(cls())
      if workdir is not None:
        self._summary_writer[tag] = tf.summary.create_file_writer(
            os.path.join(workdir, tag))

  def train_step(self, inputs, y_true, true_info=None, training=True):
    """Training step: predict and apply gradients."""
    if training:
      with tf.GradientTape() as tape:
        y_pred = self.model(inputs)
        loss = self._loss_fn(y_true, y_pred)
      gradients = tape.gradient(loss, self.model.trainable_variables)
      self._optimizer.apply_gradients(
          zip(gradients, self.model.trainable_variables))
    else:
      y_pred = self.model(inputs)
      loss = self._loss_fn(y_true, y_pred)

    tag = self.TRAIN if training else self.TEST
    self._metrics[tag][0].update_state(loss)
    for metric in self._metrics[tag][1:]:
      if isinstance(metric, perturbed_metrics.ShortestPathMetrics):
        metric.update_state(true_info, y_pred)
      else:
        metric.update_state(y_true, y_pred)

  def _metrics_debug_str(self, tag):
    return '{}: {}'.format(tag, ' '.join(
        ['{}: {:.3f}'.format(x.name, x.result())
         for x in self._metrics[tag]]))

  def on_epoch_start(self, epoch):
    lr = self.learning_rate_schedule.get(epoch, None)
    if lr is not None:
      self._optimizer.learning_rate = lr

    for metrics in self._metrics.values():
      for m in metrics:
        m.reset_states()

  def on_epoch_end(self, epoch):
    """What to do at the end of one epoch. Save metrics and checkpoints."""
    if self._checkpoint is not None:
      self._checkpoint.save(file_prefix=self._checkpoint_prefix)

    if self._summary_writer:
      for tag in self.TAGS:
        with self._summary_writer[tag].as_default():
          for metric in self._metrics[tag]:
            tf.summary.scalar(metric.name, metric.result(), step=epoch)

    dbg = 'Epoch {}: {}'.format(
        epoch + 1, ' || '.join([self._metrics_debug_str(t) for t in self.TAGS]))
    print(dbg)

  def run(self, epochs=None, batch_size=None):
    """Run the training loop."""
    shift = 0
    if self._checkpoint is not None:
      ckpt = tf.train.latest_checkpoint(self._checkpoint_directory)
      self._checkpoint.restore(ckpt)

    epochs = self._epochs if epochs is None else epochs
    batch_size = self._batch_size if batch_size is None else batch_size
    for epoch in range(shift, epochs):
      self.on_epoch_start(epoch)

      for tag in self.TAGS:
        ds = self.data.ds[tag].batch(batch_size)
        training = tag == self.TRAIN
        max_steps = None
        for step, inputs in enumerate(ds):

          # Do we have more than inputs and labels ?
          true_info = None
          if len(inputs) > 2:
            inputs, labels, true_info = inputs
          else:
            inputs, labels = inputs

          # Check the max step on the first batch.
          if max_steps is None:
            bs = tf.shape(inputs)[0]
            max_steps = np.ceil(self.data.num_examples[tag] / bs)

          if step >= max_steps:
            break

          start = time.time()
          self.train_step(inputs, labels, true_info, training=training)
          delta = (time.time() - start) * 1000

          if training:
            dbg = '{}/{} ({:.0f}ms/step) '.format(step + 1, max_steps, delta)
            dbg = dbg + self._metrics_debug_str(tag)
            print(dbg, end='\r')

      self.on_epoch_end(epoch)
