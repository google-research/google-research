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

"""Implements the main training loop on the MNIST dataset."""

import os
import shutil

from absl import logging
import tensorflow as tf

from student_mentor_dataset_cleaning.training import utils
import student_mentor_dataset_cleaning.training.datasets as datasets
from student_mentor_dataset_cleaning.training.datasets import mnist


def _reinitilize_student(save_dir):
  return tf.keras.models.load_model(
      os.path.join(save_dir, 'student', 'init.hdf5'))


def _reinitilize_mentor(save_dir):
  return tf.keras.models.load_model(
      os.path.join(save_dir, 'mentor', 'init.hdf5'))


def _get_student_callbacks(log_dir, save_dir, current_iteration):
  """Creates callbacks to be used in student's training."""
  student_callbacks = []

  if log_dir:
    student_log_dir = os.path.join(log_dir, 'student',
                                   f'iteration_{current_iteration:04d}')
    os.makedirs(student_log_dir, exist_ok=True)

    student_callbacks.append(
        tf.keras.callbacks.TensorBoard(
            log_dir=student_log_dir, histogram_freq=1))
    student_callbacks.append(
        utils.LearningRateLogger(
            log_dir=student_log_dir, name='student_learning_rate'))

  student_callbacks.append(
      utils.CustomEarlyStopping(
          monitor='val_loss',
          min_delta=0,
          patience=60,
          verbose=1,
          mode='min',
          restore_best_weights=True))

  student_callbacks.append(
      utils.CustomReduceLROnPlateau(
          monitor='val_loss',
          factor=0.5,
          patience=20,
          verbose=1,
          mode='min',
          min_delta=0.0001,
          cooldown=0,
          min_lr=0.0000001))

  checkpoint_path = os.path.join(save_dir, 'student',
                                 f'iteration_{current_iteration:04d}')
  os.makedirs(checkpoint_path, exist_ok=True)
  student_callbacks.append(
      tf.keras.callbacks.ModelCheckpoint(
          filepath=os.path.join(checkpoint_path, 'weights.{epoch:04d}.hdf5'),
          monitor='val_loss',
          verbose=1,
          save_best_only=True,
          save_weights_only=False,
          mode='min',
          save_freq='epoch'))

  return student_callbacks


def _get_mentor_callbacks(log_dir, save_dir, current_iteration):
  """Creates callbacks to be used in mentor's training."""
  mentor_callbacks = []

  if log_dir:
    mentor_log_dir = os.path.join(log_dir, 'mentor',
                                  f'iteration_{current_iteration:04d}')
    os.makedirs(mentor_log_dir, exist_ok=True)

    mentor_callbacks.append(
        tf.keras.callbacks.TensorBoard(
            log_dir=mentor_log_dir, histogram_freq=1))
    mentor_callbacks.append(
        utils.LearningRateLogger(
            log_dir=mentor_log_dir, name='mentor_learning_rate'))

  mentor_callbacks.append(
      utils.CustomEarlyStopping(
          monitor='val_loss',
          min_delta=0,
          patience=100,
          verbose=1,
          mode='min',
          restore_best_weights=True))

  mentor_callbacks.append(
      utils.CustomReduceLROnPlateau(
          monitor='val_loss',
          factor=0.5,
          patience=20,
          verbose=1,
          mode='min',
          min_delta=0,
          cooldown=0,
          min_lr=0.0000001))

  checkpoint_path = os.path.join(save_dir, 'mentor',
                                 f'iteration_{current_iteration:04d}')
  os.makedirs(checkpoint_path, exist_ok=True)
  mentor_callbacks.append(
      tf.keras.callbacks.ModelCheckpoint(
          filepath=os.path.join(checkpoint_path, 'weights.{epoch:04d}.hdf5'),
          monitor='val_loss',
          verbose=1,
          save_best_only=True,
          save_weights_only=False,
          mode='min',
          save_freq='epoch'))

  return mentor_callbacks


def _get_weights_dataset(student, mentor, dataset, snapshot_fn):
  """Evaluates the confidence of the mentor in each data point.

  Args:
    student: The student model.
    mentor: The mentor model.
    dataset: The dataset the student is trained on.
    snapshot_fn: A function that returns the gradients of the student on a given
      dataset.

  Returns:
    The confidence of the mentor in each data point based on the student's
    gradients.
  """

  gradients_dataset = snapshot_fn(student, dataset)
  return gradients_dataset.batch(1).map(mentor)


def _create_mentor_dataset(student, dataset, snapshot_fn, noise_rate,
                           target_distribution_parameter):
  """Creates a dataset of gradients and mentor's ground truths.

  Adds label noise to `dataset`, and computes the student model's gradients wrt.
  the dataset. The labels of the output dataset are set to 1 for unmodified
  examples and to 0 for examples whose labels were randomized.

  Args:
    student: The student model.
    dataset: The dataset to create the mentor's training data from.
    snapshot_fn: A function that returns the gradients of a model for a given
      dataset.
    noise_rate: The probability of assigning an example from `dataset` a random
      label.
    target_distribution_parameter: The steepness of the exponential distrubtion
      used to resample the training datasets.

  Returns:
    The mentor dataset.
  """

  corrupted_dataset = datasets.corrupt_dataset(
      dataset,
      noise_rate=noise_rate,
      target_distribution_parameter=target_distribution_parameter,
      include_weights=True)

  gradients_dataset = snapshot_fn(
      student,
      corrupted_dataset.map(lambda x, y, w: (x, y)).batch(1))

  return tf.data.Dataset.zip(
      (gradients_dataset, corrupted_dataset)).map(lambda g, xyw: (g, xyw[2]))


def _train_student(student, train_dataset,
                   weights,
                   validation_dataset, mini_batch_size,
                   epochs, **kwargs):
  """Train the student model.

  Args:
    student: The student model to train.
    train_dataset: The dataset to train the model on.
    weights: List of weights of each training sample as estimated by the mentor
      model.
    validation_dataset: The validation dataset for student training.
    mini_batch_size: Number of examples per minibatch.
    epochs: The number of epochs to train the model for.
    **kwargs: Arguments to be passed through to keras.model.fit.

  Returns:
    The minimum validation loss seen during training.
  """

  train_dataset = train_dataset.batch(mini_batch_size)
  validation_dataset = validation_dataset.batch(mini_batch_size)

  weighted_dataset = tf.data.Dataset.zip(
      (train_dataset, weights)).map(lambda xy, z: (xy[0], xy[1], z))

  best_val_loss = float('inf')
  for epoch in range(epochs):
    history = student.fit(
        weighted_dataset,
        initial_epoch=epoch,
        epochs=epoch + 1,
        validation_data=validation_dataset,
        **kwargs)

    best_val_loss = min(best_val_loss, history.history['val_loss'][0])

    if student.stop_training:
      break

  return best_val_loss


def _train_mentor(student, mentor,
                  dataset, snapshot_fn, mini_batch_size,
                  epochs, noise_rate,
                  target_distribution_parameter, **kwargs):
  """Train the mentor.

  Args:
    student: The student model.
    mentor: The mentor model.
    dataset: The student's training dataset.
    snapshot_fn: A function that returns the gradients of a model given a
      dataset.
    mini_batch_size: The number of examples per training batch.
    epochs: The number of epochs to train the mentor for.
    noise_rate: The probability of assigning a random label to a training
      example.
    target_distribution_parameter: The steepness of the exponential distrubtion
      used to resample the training datasets.
    **kwargs: Arguments to be passed through to keras.model.fit.

  Returns:
    The minimum validation loss seen during training.
  """

  best_val_loss = float('inf')
  for epoch in range(epochs):
    mentor_dataset = _create_mentor_dataset(student, dataset, snapshot_fn,
                                            noise_rate,
                                            target_distribution_parameter)
    train_dataset, validation_dataset = datasets.dataset_split(
        mentor_dataset, 0.6)
    train_dataset = train_dataset.batch(mini_batch_size)
    validation_dataset = validation_dataset.batch(mini_batch_size)

    history = mentor.fit(
        train_dataset,
        initial_epoch=epoch,
        epochs=epoch + 1,
        validation_data=validation_dataset,
        **kwargs)

    best_val_loss = min(best_val_loss, history.history['val_loss'][0])

    if mentor.stop_training:
      break

  return best_val_loss


def _preserve_initial_models(save_dir, student,
                             mentor):
  mentor.save(
      os.path.join(save_dir, 'mentor', 'init.hdf5'), include_optimizer=True)
  student.save(
      os.path.join(save_dir, 'student', 'init.hdf5'), include_optimizer=True)


def _preserve_best_models(save_dir, student,
                          mentor):
  mentor.save(
      os.path.join(save_dir, 'mentor', 'best.hdf5'), include_optimizer=True)
  student.save(
      os.path.join(save_dir, 'student', 'best.hdf5'), include_optimizer=True)


def train(student,
          mentor,
          mini_batch_size,
          iteration_count,
          student_epoch_count,
          mentor_epoch_count,
          save_dir,
          log_dir = ''):
  """Trains a mentor-student pair.

  Args:
    student: The student model.
    mentor: The mentor model.
    mini_batch_size: The number of examples per training batch for both student
      and mentor.
    iteration_count: The number of iterations to train for. In each iteration,
      the student is trained for `student_epoch_count` epochs, then the mentor
      is trained for `mentor_epoch_count` on the student's gradients.
    student_epoch_count: The number of epochs the student is trained for in each
      iteration.
    mentor_epoch_count: The number of epochs the mentor is trained for in each
      iteration.
    save_dir: The path where the checkpoints of both models are saved to.
    log_dir: The path where logs are written to.

  Returns:
    The trained student and mentor models.
  """

  snapshot_fn = utils.get_gradients_dataset_from_labelled_data
  noise_rate = 0.1
  target_distribution_parameter = 0.01
  patience = 20

  shutil.rmtree(save_dir)
  if log_dir:
    shutil.rmtree(log_dir)

  # Preserve the initial models
  _preserve_initial_models(save_dir, student, mentor)

  # Create datasets
  (train_dataset_student, validation_dataset_student,
   train_dataset_mentor) = mnist.create_dataset(
       student_train_portion=0.4,
       student_validation_portion=0.1,
       mentor_portion=0.5,
       noise_rate=noise_rate,
       target_distribution_parameter=target_distribution_parameter)

  best_val_loss = float('inf')
  best_mentor = mentor
  best_student = student
  waiting = 0
  for current_iteration in range(iteration_count):
    # Train student
    logging.info('Training the student in iteration %d', current_iteration)

    weights = _get_weights_dataset(student, mentor,
                                   train_dataset_student.batch(mini_batch_size),
                                   snapshot_fn)

    student = _reinitilize_student(save_dir)

    _train_student(
        student=student,
        train_dataset=train_dataset_student,
        weights=weights,
        validation_dataset=validation_dataset_student,
        mini_batch_size=mini_batch_size,
        epochs=student_epoch_count,
        verbose=2,
        callbacks=_get_student_callbacks(log_dir, save_dir, current_iteration))

    # Train mentor
    logging.info('Training the mentor in iteration %d', current_iteration)

    mentor = _reinitilize_mentor(save_dir)

    val_loss = _train_mentor(
        student=student,
        mentor=mentor,
        dataset=train_dataset_mentor,
        snapshot_fn=snapshot_fn,
        mini_batch_size=mini_batch_size,
        epochs=mentor_epoch_count,
        noise_rate=noise_rate,
        target_distribution_parameter=target_distribution_parameter,
        class_weight={
            0: 1 - noise_rate,
            1: noise_rate
        },
        verbose=2,
        callbacks=_get_mentor_callbacks(log_dir, save_dir, current_iteration))

    if val_loss < best_val_loss:
      waiting = 0
      best_val_loss = val_loss
      best_student = student
      best_mentor = mentor
      _preserve_best_models(save_dir, student, mentor)
    else:
      waiting += 1
      if waiting > patience:
        break

  return best_student, best_mentor
