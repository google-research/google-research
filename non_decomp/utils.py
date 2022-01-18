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

"""Various dataset and model training/evaluation utilities."""
import dataclasses
import os
from typing import List, Tuple
import tensorflow as tf
import tensorflow_constrained_optimization as tfco


def build_loss_fn(loss_type, base_probs, tau=1.0):
  """Builds the loss function to be used for training.

  The loss function is either a standard, a re-weighted or a logit-adjusted
  cross-entropy loss. For the second and third option, the loss takes an
  optional class_weights argument, and uses class_weights / base_probs as the
  per-class costs; if the argument is not specified, the per-class costs are
  set to 1.0 / base_probs.

  Args:
    loss_type: 'standard', 'reweighted', or 'logit_adjusted'.
    base_probs: Base probabilities to use with weighted or logit-adjusted loss.
    tau: Temperature scaling parameter for the base probabilities.

  Returns:
    A loss function with signature loss(labels, logits).
  """
  def loss_fn(labels, logits, class_weights=None):
    """Compute loss from *one-hot* labels, logits, and class weights.

    Args:
      labels: one-hot labels of shape (num_examples, num_classes)
      logits: model scores of shape (num_examples, num_classes)
      class_weights: optional, per-class weights, (num_classes,)

    Returns:
      Scalar loss value.

    Raises:
      ValueError: if class_weights are specified for a 'standard' loss type.
    """
    # Outer weights and logit modifications.
    if loss_type == 'standard':
      if class_weights is not None:
        raise ValueError('Loss type "standard" does not take `class_weights`.')
      outer_weights = tf.ones((logits.shape[-1], 1))  # num_classes x 1
    else:
      scaled_class_weights = 1.0 / tf.reshape(
          tf.cast(base_probs ** tau + 1e-12, tf.float32),
          (1, -1))  # (1, num_classes)
      if class_weights is not None:
        scaled_class_weights *= tf.reshape(
            class_weights + 1e-12, (1, -1))  # (1, num_classes)
      if loss_type == 'logit_adjusted':
        logits -= tf.math.log(
            scaled_class_weights)  # (num_examples, num_classes)
        outer_weights = tf.ones((logits.shape[-1], 1))  # (num_classes, 1)
      else:
        outer_weights = tf.reshape(
            scaled_class_weights, (logits.shape[-1], 1))  # (num_classes, 1)

    # Final re-weighted or logit-adjusted loss.
    per_sample_weights = tf.reshape(tf.tensordot(
        labels, outer_weights, axes=1), (-1,))  # (num_examples,)
    cce_fn = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True,
        reduction=tf.keras.losses.Reduction.NONE)
    loss = cce_fn(labels, logits)  # (num_examples,)
    return tf.reduce_sum(loss * per_sample_weights) / tf.reduce_sum(
        per_sample_weights)

  return loss_fn


class Placeholder:
  """A nullary functor returning a settable placeholder value.

  For use with the TF constrained optimization (TFCO) library.
  """

  def __init__(self):
    self.value = None

  def __hash__(self):
    return id(self)

  def __call__(self):
    if self.value is None:
      raise ValueError('Placeholder was never assigned.')
    return self.value


class FalseNegativeRates:
  """Tracks aggregate false negative rates from per-batch stats.

  We will use the TF Constrained Optimization (TFCO) library, a general-purpose
  library that enables training models with constraints on classification rates.
  We'll be using the "rate" helpers from the library to keep track of false
  negative rates from per-batch labels and logits. To measure the false negative
  rate for a class, the library will divide the false positive counts for class
  i in the current batch (i.e. the numerator) by an estimate of the  total
  number of class i examplees (i.e. the denominator).
  """

  def __init__(self, num_classes):
    # Set up TFCO library objects. We will create a constrained optimization
    # problem object with dummy constraints on the false negative rates, so
    # when we read out the "constraint value", we get the false negative rates.
    self.labels_placeholder = Placeholder()
    self.logits_placeholder = Placeholder()
    context = tfco.multiclass_rate_context(
        num_classes, self.logits_placeholder, self.labels_placeholder)
    constraints = [
        tfco.false_negative_rate(context, ii) <= 0.0
        for ii in range(num_classes)]
    self.problem = tfco.RateMinimizationProblem(
        tfco.wrap_rate(0.0), constraints)

  def update_state(self, labels, logits):
    # Updates placeholders with batch *one-hot* labels and logits.
    self.labels_placeholder.value = labels
    self.logits_placeholder.value = logits
    self.problem.update_ops()

  def result(self):
    return self.problem.constraints()


class MinRecall(tf.keras.metrics.Metric):
  """Minimum of per-class recalls."""

  def __init__(
      self, recall_list=None, name='min_recall', **kwargs):
    super(MinRecall, self).__init__(name=name, **kwargs)
    self.recall_list = recall_list

  def update_state(self, recall_list):
    self.recall_list = list(recall_list)

  def reset_state(self):
    # No state to reset.
    pass

  def result(self):
    if not self.recall_list:
      raise ValueError('No Recall metrics specified.')
    results = [m.result() for m in self.recall_list]
    return tf.reduce_min(results)


class LearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Step learning rate schedule."""

  def __init__(self, schedule, steps_per_epoch, base_learning_rate):
    """Creates a step learning rate schedule.

    Args:
      schedule: List of (epoch_number, lr_multiplier) pairs. The base learning
        rate will be multiplied by the multiplier at the epoch number. The first
        entry says when to finish the linear warmup.
      steps_per_epoch: Number of steps per epoch.
      base_learning_rate: Base learning rate.
    """
    super().__init__()
    self._schedule = schedule
    self._steps_per_epoch = steps_per_epoch
    self._base_learning_rate = base_learning_rate

  def __call__(self, step):
    """See base class."""
    lr_epoch = tf.cast(step, tf.float32) / self._steps_per_epoch
    warmup_lr_multiplier, warmup_end_epoch = self._schedule[0]
    learning_rate = (
        self._base_learning_rate * warmup_lr_multiplier * lr_epoch /
        warmup_end_epoch)
    for mult, start_epoch in self._schedule:
      learning_rate = tf.where(lr_epoch >= start_epoch,
                               self._base_learning_rate * mult, learning_rate)
    return learning_rate

  def get_config(self):
    """See base class."""
    return {
        'schedule': self._schedule,
        'steps_per_epoch': self._steps_per_epoch,
        'base_learning_rate': self._base_learning_rate
    }


@dataclasses.dataclass
class Dataset:
  """Represents a dataset."""
  name: str
  num_classes: int
  train_file_name: str
  test_file_name: str
  num_train: int
  num_test: int
  num_epochs: int
  lr_schedule: List[Tuple[float, int]]


def dataset_mappings():
  """Returns dataset_name -> Dataset mappings."""
  return {
      'cifar10-lt':
          Dataset(
              'cifar10-lt',
              10,
              'cifar10-lt_train.tfrecord',
              'cifar10_test.tfrecord',
              12406,
              10000,
              1241,
              [  # (multiplier, epoch to start) tuples
                  (1.0, 20), (0.1, 604), (0.01, 926), (0.001, 1128)
              ]),
      'cifar100-lt':
          Dataset(
              'cifar100-lt',
              100,
              'cifar100-lt_train.tfrecord',
              'cifar100_test.tfrecord',
              10847,
              10000,
              1419,
              [  # (multiplier, epoch to start) tuples
                  (1.0, 22), (0.1, 691), (0.01, 1059), (0.001, 1290)
              ]),
      'test':
          Dataset('test', 10, 'test.tfrecord', 'test.tfrecord', 4, 4, 2,
                  [(1.0, 2)]),
  }


def _process_image(record, training):
  """Decodes the image and performs data augmentation if training."""
  image = tf.io.decode_raw(record, tf.uint8)
  image = tf.cast(image, tf.float32)
  image = tf.reshape(image, [32, 32, 3])
  image = image * (1. / 255) - 0.5
  if training:
    padding = 4
    image = tf.image.resize_with_crop_or_pad(image, 32 + padding, 32 + padding)
    image = tf.image.random_crop(image, [32, 32, 3])
    image = tf.image.random_flip_left_right(image)
  return image


def _parse(serialized_examples, training):
  """Parses the given protos and performs data augmentation if training."""
  feature_spec = {
      'image/encoded': tf.io.FixedLenFeature((), tf.string),
      'image/class/label': tf.io.FixedLenFeature((), tf.int64)
  }
  features = tf.io.parse_example(serialized_examples, feature_spec)
  images = tf.map_fn(
      lambda record: _process_image(record, training),
      features['image/encoded'],
      dtype=tf.float32)
  return images, features['image/class/label']


def create_tf_dataset(
    dataset, data_home, batch_size, split, test_size=5000):
  """Creates a Tensorflow Dataset instance for training/testing.

  Args:
    dataset:    Dataset definition.
    data_home:  Directory where the .tfrecord files are stored.
    batch_size: Batch size.
    split:      Either 'train', 'vali' or 'test'. Training datasets
      have data augmentation.
    test_size:  Size of test set. The remainder of the test set file will be
      used as a validation sample.

  Returns:
    A tf.data.Dataset instance.
  """
  training = split == 'train'
  filename = dataset.train_file_name if training else dataset.test_file_name

  tf_dataset = tf.data.TFRecordDataset(
      os.path.join(data_home, filename)).shuffle(10000)
  if split == 'test':
    tf_dataset = tf_dataset.take(test_size)
  elif split == 'vali':
    tf_dataset = tf_dataset.skip(test_size)

  tf_dataset = tf_dataset.batch(
      batch_size, drop_remainder=training
  ).map(
      lambda record: _parse(record, training)
  ).prefetch(
      tf.data.experimental.AUTOTUNE
  )
  if split == 'vali':
    # Repeat dataset indefinitely. The vali data stream will then be used to
    # perform interleaved updates on the per-class weights.
    return tf_dataset.repeat()
  else:
    return tf_dataset
