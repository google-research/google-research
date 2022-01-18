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

"""Functions for personalized evaluation."""

import collections
from typing import Optional, Sequence, Tuple

from absl import logging
import numpy as np
import tensorflow as tf


FINE_TUNE_LR = 1e-4


def _get_sample_weight(y_true, dtype, masked_classes, masked_class_range):
  """Returns a 0/1 weight tensor of the same shape as `y_true`."""
  sample_weight = tf.ones_like(y_true, dtype)
  for c in masked_classes:
    mask = tf.cast(tf.not_equal(y_true, c), dtype)
    sample_weight = tf.math.multiply(sample_weight, mask)
  lower = tf.math.greater_equal(y_true, masked_class_range[0])
  upper = tf.math.less_equal(y_true, masked_class_range[1])
  mask = tf.where(
      tf.math.logical_and(lower, upper), tf.zeros_like(y_true, dtype),
      tf.ones_like(y_true, dtype))
  sample_weight = tf.math.multiply(sample_weight, mask)
  return sample_weight


def _reduce_non_vocabulary_logits(y_pred, non_vocabulary_classes):
  """Replace logits in `non_vocabulary_classes` with a very small value."""
  y_pred_min = tf.math.reduce_min(y_pred)
  # Create a 0/1 tensor of same shape as `y_pred`, and its columns in
  # `non_vocabulary_classes` are ones.
  num_examples = tf.shape(y_pred)[0]
  indices = tf.reshape(non_vocabulary_classes, [-1, 1])
  updates = tf.ones([len(non_vocabulary_classes), num_examples])
  shape = tf.shape(tf.transpose(y_pred))
  mask = tf.scatter_nd(indices, updates, shape)
  mask = tf.transpose(tf.cast(mask, dtype=tf.bool))
  # Replace logits in `non_vocabulary_classes` with y_pred_min-1 (i.e., a value
  # smaller than the minimum of y_pred). This ensures that the largest logit
  # in y_pred will be from `vocabulary_classes`.
  return tf.where(mask, y_pred_min-1, y_pred)


class SubsetInVocabAccuracy(tf.keras.metrics.SparseCategoricalAccuracy):
  """An accuracy metric computed over a subset of classes.

  Note: The classes/labels are assumed to be represented as a contiguous set of
  integers.

  This metric allows you to specify a subset of masked classes (i.e., masked
  classes will not be used in the denominator when computing the accuracy). The
  masked classes can be defined in two ways:
  * `masked_classes`: A list of integers indicating the classes that will be
                      maskes out in calculation. The size of this list has to be
                      small, otherwise, it may incur OOM error.
  * `masked_class_range`: A tuple of two integers, e.g., (low, high), which
                          means that low <= classes <= high will be
                          masked out in calculation. If low > high, then
                          `masked_class_range` does not mask out any class.
  """

  def __init__(self,
               name = 'accuracy',
               non_vocabulary_classes = None,
               dtype = None,
               masked_classes = None,
               masked_class_range = None):
    self._non_vocabulary_classes = non_vocabulary_classes
    self._masked_classes = masked_classes if masked_classes else []
    if masked_class_range:
      if len(masked_class_range) != 2:
        raise ValueError('Expected `masked_class_range` to be a tuple of '
                         'length 2, found {}.'.format(masked_class_range))
      self._masked_class_range = masked_class_range
    else:
      self._masked_class_range = (2, 1)  # 2 > 1, so no masking is performed.
    super().__init__(name=name, dtype=dtype)

  def update_state(self, y_true, y_pred, sample_weight=None):
    sample_weight = _get_sample_weight(y_true, self._dtype,
                                       self._masked_classes,
                                       self._masked_class_range)
    num_classes = tf.shape(y_pred)[-1]
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1, num_classes])
    if self._non_vocabulary_classes:
      y_pred = _reduce_non_vocabulary_logits(y_pred,
                                             self._non_vocabulary_classes)
    sample_weight = tf.reshape(sample_weight, [-1])
    super().update_state(y_true, y_pred, sample_weight)

  def get_config(self):
    config = super().get_config()
    if self._non_vocabulary_classes:
      config['non_vocabulary_classes'] = self._non_vocabulary_classes
    if self._masked_classes:
      config['masked_classes'] = self._masked_classes
    if self._masked_class_range:
      config['masked_class_range'] = self._masked_class_range
    return config


def eval_per_acc(model, dataset, mask_vocab_id):
  """Evaluate personalized models on the test set which contains many clients.

  Group the accuracy by clients (one accuracy per client).
  Need to ignore mask vocab.

  Args:
    model: The global model for personalization.
    dataset: The test set of many clients.
    mask_vocab_id: the vocabuary to mask.

  Returns:
    A list of personalized accuracy per client.

  """
  per_acc_dict = collections.OrderedDict()

  for (x, id_list), yy in dataset:
    preds = model.predict([x, id_list])
    y_hat_list = np.argmax(preds, -1)
    id_list = id_list.numpy()
    y_list = yy.numpy()

    for y_hat, y, idx in zip(y_hat_list, y_list, id_list):
      valid_index = [i for i, yy in enumerate(y) if yy not in mask_vocab_id]
      y = y[valid_index]
      y_hat = y_hat[valid_index]

      if idx not in per_acc_dict:
        per_acc_dict[idx] = {'cnt': 0, 'correct': 0}
      per_acc_dict[idx]['cnt'] += len(y_hat)
      per_acc_dict[idx]['correct'] += int(np.sum(y_hat == y))

  per_acc_list = [d['correct'] / d['cnt'] for d in per_acc_dict.values()]
  logging.info(per_acc_list)
  logging.info(np.nanmean(per_acc_list))
  return per_acc_list


def log_acc(before_ft, after_ft):
  diff = np.array(after_ft) - np.array(before_ft)
  logging.info(after_ft)
  logging.info(np.nanmean(after_ft))
  logging.info('% of degraded clients')
  logging.info(np.nanmean(diff < 0))


def per_evaluation(basisnet,
                   fine_tuning_data_tuples,
                   global_model_builder,
                   model_builder,
                   mask_vocab_id,
                   optimizer_name='adam',
                   fix_basis=True,
                   fine_tune_epoch=20):
  """Personalized evaluation.

  Args:
    basisnet: The global model for fine-tuning.
    fine_tuning_data_tuples: A training dataset, a training dataset with smaller
      split, a test dataset.
    global_model_builder: The model builder for using global embedding.
    model_builder: A function for building the model.
    mask_vocab_id: A list of vocabulary to exclude from evaluation.
    optimizer_name: Optimizer name string for fine-tuning.
    fix_basis: Fix the bases or not.
    fine_tune_epoch: Number of epochs for fine-tuning.

  Returns:
    A training set for fine-tuning, a split training set with smaller size, a
    test set.
  """

  (ft_dataset, sp_ft_dataset, ft_val_dataset) = fine_tuning_data_tuples

  before_ft = eval_per_acc(basisnet, ft_val_dataset, mask_vocab_id)

  def finetuning(ft_dataset,
                 ft_dataset_test,
                 model_builder,
                 fine_tune_epoch=20,
                 optimizer_name='adam',
                 global_exp=False,
                 fix_basis=True):
    """Fine-tunes and evaluates personazlied accuracy.

    Args:
      ft_dataset: Training dataset for fine-tuning.
      ft_dataset_test: Test dataset for personalized evaluation.
      model_builder: A function to create the model.
      fine_tune_epoch: Number of epochs for fine-tuning.
      optimizer_name: Name of the optimzer of fine-tuning.
      global_exp: Learning the global embedding only.
      fix_basis: Fix the bases or not.

    Returns:
      A list of personalized accuracy per client.
    """
    local_basisnet = model_builder()

    if optimizer_name == 'rmsprop':
      optimizer = tf.keras.optimizers.RMSprop(learning_rate=FINE_TUNE_LR)
    elif optimizer_name == 'sgd':
      optimizer = tf.keras.optimizers.SGD(learning_rate=FINE_TUNE_LR * 10)
    elif optimizer_name == 'adam':
      optimizer = tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR)

    local_basisnet.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    local_basisnet.set_weights(basisnet.get_weights())

    if fix_basis:
      # only fine-tune the embedding
      logging.info('Fix basis')
      for layer in local_basisnet.layers:
        if layer.name != 'client_embedding':
          layer.trainable = False

    all_per_acc_list = []
    for ep in range(fine_tune_epoch):
      local_basisnet.fit(
          ft_dataset, epochs=1, verbose=2, validation_data=ft_dataset_test)
      logging.info('Fine-tune epoch %d', ep)
      all_per_acc_list.append(
          eval_per_acc(local_basisnet, ft_dataset_test, mask_vocab_id))

    if global_exp:
      global_embedding = local_basisnet.get_layer(
          'client_embedding').get_weights()[0][0]
      new_embedding = np.tile(global_embedding, (500000, 1))
      basisnet.get_layer('client_embedding').set_weights([new_embedding])

    # return the best epoch for every client.
    all_per_acc_list = np.array(all_per_acc_list)
    per_acc_list = np.max(np.array(all_per_acc_list), axis=0)
    return per_acc_list

  logging.info('Training the global embedding.')
  logging.info('==============')

  # Train the global embedding.
  after_ft = finetuning(
      ft_dataset=ft_dataset,
      ft_dataset_test=ft_val_dataset,
      model_builder=global_model_builder,
      global_exp=True,
      fix_basis=fix_basis,
      fine_tune_epoch=fine_tune_epoch,
      optimizer_name=optimizer_name)
  log_acc(before_ft, after_ft)

  logging.info('Fine-tuning the client embedding with 100% data.')
  # fix bases, fine-tune client embeddings
  after_ft = finetuning(
      ft_dataset=ft_dataset,
      ft_dataset_test=ft_val_dataset,
      model_builder=model_builder,
      fix_basis=fix_basis,
      fine_tune_epoch=fine_tune_epoch,
      optimizer_name=optimizer_name)
  log_acc(before_ft, after_ft)

  logging.info('Fine-tuning the client embedding with 50% data.')
  after_ft = finetuning(
      ft_dataset=sp_ft_dataset,
      ft_dataset_test=ft_val_dataset,
      model_builder=model_builder,
      fix_basis=fix_basis,
      optimizer_name=optimizer_name,
      fine_tune_epoch=fine_tune_epoch)
  log_acc(before_ft, after_ft)

