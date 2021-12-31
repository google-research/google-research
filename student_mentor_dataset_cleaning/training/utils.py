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

"""Utility classes and functions for student-mentor training."""

from absl import logging
import numpy as np
import tensorflow as tf

from student_mentor_dataset_cleaning.training.loss.triplet_loss import triplet_semihard_loss_fn


class CustomEarlyStopping(tf.keras.callbacks.EarlyStopping):
  """An early stopping callback that avoids resetting between calls to fit."""

  def __init__(self, *args, **kwargs):
    tf.keras.callbacks.EarlyStopping.__init__(self, *args, **kwargs)

    self.wait = 0
    self.stopped_epoch = 0
    if self.baseline is not None:
      self.best = self.baseline
    else:
      self.best = np.Inf if self.monitor_op == np.less else -np.Inf
    self.best_weights = None

  def on_train_begin(self, logs=None):
    pass


class CustomReduceLROnPlateau(tf.keras.callbacks.ReduceLROnPlateau):
  """A reduce LR callback that avoids resetting between executions of fit."""

  def __init__(self, *args, **kwargs):
    tf.keras.callbacks.ReduceLROnPlateau.__init__(self, *args, **kwargs)
    self._reset()

  def on_train_begin(self, logs=None):
    pass


class LearningRateLogger(tf.keras.callbacks.TensorBoard):
  """Logs the learning rate to tensorboard."""

  def __init__(self, log_dir, name='learning_rate', profile_batch=0, **kwargs):
    kwargs['profile_batch'] = profile_batch
    self.name = name
    super().__init__(log_dir=log_dir, **kwargs)

  def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}
    logs.update({self.name: self.model.optimizer.lr})
    super().on_epoch_end(epoch, logs)


def get_gradients_dataset_from_labelled_data(student, dataset):
  """Computes the gradients of the student when it processes a dataset.

  Args:
    student: The student model
    dataset: The training dataset

  Returns:
    The student's gradients on the given dataset.
  """

  loss_fn = student.loss

  def get_gradients(x, y):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(student.trainable_variables)
      student_output = student(x, training=True)
      loss_value = loss_fn(y, student_output)
    gradients = tape.gradient(loss_value, student.trainable_weights)
    flattened_gradients = tf.reshape(
        tf.concat([tf.reshape(g, [-1]) for g in gradients], axis=0), [-1])
    return flattened_gradients

  return dataset.map(get_gradients)


def get_gradients_dataset_from_triplet_data(student, dataset):
  """Computes the gradients of the student network for a dataset of triplets.

  Args:
    student: The student model
    dataset: The training dataset

  Returns:
    The student's gradients on the given dataset.
  """

  tf.keras.backend.clear_session()
  triplet_loss_fn = student.loss
  y_true = None
  y_pred = None
  batch_size = 1996
  for img, l in dataset.batch(batch_size).take(1):
    y_true = l
    y_pred = student(img)
    tf.stop_gradient(student)
  triplet_loss_fn.call(y_true, y_pred)

  def gradient_generator():
    logging.info('There are %d triplets.', len(triplet_loss_fn.triplets))
    for (anchor_id, positive_id, negative_id) in triplet_loss_fn.triplets:
      anchor_embedding = triplet_loss_fn.y_pred[anchor_id, :]
      positive_embedding = triplet_loss_fn.y_pred[positive_id, :]
      negative_embedding = triplet_loss_fn.y_pred[negative_id, :]

      with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(student.trainable_variables)
        loss_value = triplet_semihard_loss_fn(
            anchor_embedding, positive_embedding, negative_embedding,
            triplet_loss_fn.triplet_loss_margin)
      gradients = tape.gradient(loss_value, student.trainable_weights)
      flattened_gradients = tf.reshape(
          tf.concat([tf.reshape(g, [-1]) for g in gradients], axis=0), [-1])
      yield flattened_gradients

  return tf.data.Dataset.from_generator(
      gradient_generator,
      output_types=(tf.float32),
      output_shapes=(tf.TensorShape([104000])))
