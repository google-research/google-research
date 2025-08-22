# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Base selective prediction method class."""

import abc
import os
import time
from typing import Any, Dict, Optional, Union

from active_selective_prediction.utils import model_util
from active_selective_prediction.utils import tf_util
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD


class SelectivePredictionMethod(object):
  """Base selective prediction method class.

  Attributes:
    model_arch_name: the model architecture name
    model_arch_kwargs: the arguments for the model architecture
    model_path: the path to the pre-trained model checkpoint
    source_train_ds: the source training dataset
    label_budget: the labeling budget
    batch_size: the batch size
    sampling_rounds: the number of sampling rounds
    max_epochs: the maximum number of training epochs
    patience_epochs: the number of training epochs to wait
      before stopping training when the loss is not decreased further
    min_epochs: the minimum number of training epochs
    optimizer_name: the name of optimization method
    optimizer_kargs: the arguments of the optimization method
    sampling_method: the name of the sampling method
    sampling_kwargs: the arguments of the sampling method
    finetune_method: the name of the finetuning method
    finetune_kwargs: the arguments of the finetuning method
    debug_info: whether to print debugging information
    print_freq: the frequency to print training statistics
  """
  __metaclass__ = abc.ABCMeta

  def __init__(
      self,
      model_arch_name,
      model_arch_kwargs,
      model_path,
      source_train_ds,
      label_budget,
      batch_size,
      sampling_rounds,
      max_epochs,
      patience_epochs,
      min_epochs,
      optimizer_name,
      optimizer_kargs,
      sampling_method,
      sampling_kwargs,
      finetune_method = 'joint_train',
      finetune_kwargs = None,
      debug_info = False,
      print_freq = 1,
  ):
    self.model_arch_name = model_arch_name
    self.model_arch_kwargs = model_arch_kwargs
    self.model_path = model_path
    self.source_train_ds = source_train_ds
    self.label_budget = label_budget
    self.batch_size = batch_size
    self.sampling_rounds = sampling_rounds
    self.max_epochs = max_epochs
    self.patience_epochs = patience_epochs
    self.min_epochs = min_epochs
    self.optimizer_name = optimizer_name
    self.optimizer_kargs = optimizer_kargs
    self.sampling_method = sampling_method
    self.sampling_kwargs = sampling_kwargs
    self.finetune_method = finetune_method
    if finetune_kwargs is None:
      self.finetune_kwargs = {'lambda': 1.0}
    else:
      self.finetune_kwargs = finetune_kwargs
    self.debug_info = debug_info
    self.print_freq = print_freq
    self.sampling_kwargs['debug_info'] = self.debug_info

  def load_pretrained_model(self, model_path):
    """Loads a pretrained model."""
    init_inputs, _ = next(iter(self.source_train_ds))
    if isinstance(init_inputs, dict):
      input_shape = tuple(init_inputs['input_ids'].shape[1:])
    else:
      input_shape = tuple(init_inputs.shape[1:])
    if self.model_arch_name == 'simple_convnet':
      model = model_util.get_simple_convnet(
          input_shape=input_shape,
          num_classes=self.model_arch_kwargs['num_classes'],
      )
    elif self.model_arch_name == 'cifar_resnet':
      model = model_util.get_cifar_resnet(
          input_shape=input_shape,
          num_classes=self.model_arch_kwargs['num_classes'],
      )
    elif self.model_arch_name == 'simple_mlp':
      model = model_util.get_simple_mlp(
          input_shape=input_shape,
          num_classes=self.model_arch_kwargs['num_classes'],
      )
    elif self.model_arch_name == 'densenet121':
      model = model_util.get_densenet121(
          input_shape=input_shape,
          num_classes=self.model_arch_kwargs['num_classes'],
          weights=self.model_arch_kwargs['backbone_weights'],
      )
    elif self.model_arch_name == 'resnet50':
      model = model_util.get_resnet50(
          input_shape=input_shape,
          num_classes=self.model_arch_kwargs['num_classes'],
          weights=self.model_arch_kwargs['backbone_weights'],
      )
    elif self.model_arch_name == 'roberta_mlp':
      model = model_util.get_roberta_mlp(
          input_shape=input_shape,
          num_classes=self.model_arch_kwargs['num_classes'],
      )
    else:
      raise ValueError(
          f'Not supported model architecture: {self.model_arch_name}'
      )
    # Makes an initial forward pass to create model Variables.
    model(init_inputs, training=False)
    model.load_weights(os.path.join(model_path, 'checkpoint')).expect_partial()
    return model

  def get_optimizer(self):
    """Gets optimizer for training models."""
    if self.optimizer_name == 'Adam':
      optimizer = Adam(learning_rate=self.optimizer_kargs['learning_rate'])
    elif self.optimizer_name == 'SGD':
      optimizer = SGD(
          learning_rate=self.optimizer_kargs['learning_rate'],
          momentum=self.optimizer_kargs['momentum'],
      )
    else:
      raise ValueError(f'Unsupported optimizer {self.optimizer_name}')
    return optimizer

  def train_model(
      self, model, target_ds
  ):
    """Trains the model on both source and target datasets."""

    @tf.function
    def train_step(
        batch_source_x,
        batch_source_y,
        batch_target_x,
        batch_target_y
    ):
      """Trains the model for one optimization step."""
      with tf.GradientTape() as tape:
        if self.finetune_method == 'joint_train':
          # Concatenates two batches as a single batch and then uses it
          # as the model input to ensure that the batchnorm parameters
          # are updated correctly.
          if isinstance(batch_source_x, dict):
            batch_x = {}
            for key in batch_source_x:
              batch_x[key] = tf.concat(
                  [batch_source_x[key], batch_target_x[key]], axis=0
              )
          else:
            batch_x = tf.concat([batch_source_x, batch_target_x], axis=0)
          batch_outputs = model(batch_x, training=True)
          batch_source_outputs = batch_outputs[:batch_source_y.shape[0]]
          batch_target_outputs = batch_outputs[batch_source_y.shape[0]:]
          source_loss = tf.math.reduce_mean(
              tf.keras.losses.sparse_categorical_crossentropy(
                  batch_source_y, batch_source_outputs
              )
          )
          target_loss = tf.math.reduce_mean(
              tf.keras.losses.sparse_categorical_crossentropy(
                  batch_target_y, batch_target_outputs
              )
          )
        elif self.finetune_method == 'target_only':
          batch_target_outputs = model(batch_target_x, training=True)
          source_loss = 0
          target_loss = tf.math.reduce_mean(
              tf.keras.losses.sparse_categorical_crossentropy(
                  batch_target_y, batch_target_outputs
              )
          )
        else:
          raise ValueError(
              f'Not supported finetuning method: {self.finetune_method}'
          )
        loss = self.finetune_kwargs['lambda'] * source_loss + target_loss
      grads = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))
      return loss

    t0 = time.time()
    optimizer = self.get_optimizer()
    source_train_ds_iter = iter(self.source_train_ds)
    count = 0
    min_target_loss = np.inf
    for epoch in range(1, self.max_epochs + 1):
      avg_loss = 0
      num_batches = 0
      for batch_target_x, batch_target_y in target_ds:
        try:
          batch_source_x, batch_source_y = next(source_train_ds_iter)
        except StopIteration:
          source_train_ds_iter = iter(self.source_train_ds)
          batch_source_x, batch_source_y = next(source_train_ds_iter)
        loss = train_step(
            batch_source_x, batch_source_y, batch_target_x, batch_target_y
        )
        avg_loss += loss
        num_batches += 1
      avg_loss /= num_batches
      target_loss = tf_util.evaluate_loss(model, target_ds)
      if self.debug_info and (
          epoch % self.print_freq == 0 or epoch == self.max_epochs
      ):
        print(
            f'Epoch: {epoch}, Loss: {avg_loss:.6f}, Target Loss:'
            f' {target_loss:.6f}, Time: {time.time()-t0:.2f}s'
        )
        t0 = time.time()
      # Stops the training early based on the loss on the target test data.
      if target_loss < min_target_loss:
        min_target_loss = target_loss
        count = 0
      elif epoch >= self.min_epochs:
        count += 1
      if count == self.patience_epochs:
        if self.debug_info:
          print(
              f'Epoch: {epoch}, Loss: {avg_loss:.6f}, Target Loss:'
              f' {target_loss:.6f}, Time: {time.time()-t0:.2f}s'
          )
        break
    if self.debug_info:
      print(f'Stop training at epoch {epoch}/{self.max_epochs}')
      acc = tf_util.evaluate_acc(model, target_ds)
      print(f'Accuracy on the selected test data: {acc:.2%}')

  def compute_metrics(
      self,
      adapted_model_preds,
      selection_scores,
      selected_indices,
      labels
  ):
    """Computes metrics for the selective prediction task.

    Args:
      adapted_model_preds: predictions given by the adapted model
      selection_scores: a score that can indicate the correctness of the
        predictions (a larger score means a larger probability of being
        correct)
      selected_indices: the indices of selected samples for labeling
      labels: the ground-truth labels

    Returns:

    """
    if selected_indices.shape[0] > 0:
      # Makes predictions correct on the selected labeled test examples.
      adapted_model_preds[selected_indices] = labels[selected_indices]
      # Assigns the highest score to the selected labeled test examples
      # so that they are always accepted,
      # since the predictions are correct on those examples.
      selection_scores[selected_indices] = np.max(selection_scores) + 1.0
    adapted_model_acc = np.mean(adapted_model_preds == labels)
    ed_true_labels = adapted_model_preds != labels
    threshold_set = np.unique(selection_scores)
    # Append a largest threshold such that the coverage is 0.
    threshold_set = np.append(threshold_set, [threshold_set[-1] + 1.0])
    accuracy_set = []
    coverage_set = []
    ed_acc_set = []
    selected_conds = np.ones(selection_scores.shape[0], dtype=np.int32)
    selected_conds[selected_indices] = 0
    for threshold in threshold_set:
      cover_conds = selection_scores >= threshold
      # Those selected labeled samples are not covered
      # since they need human labeling.
      cover_conds[selected_indices] = False
      coverage = np.mean(cover_conds) / np.mean(selected_conds)
      if coverage == 0:
        accuracy = 1.0
      else:
        accuracy = np.mean(
            cover_conds & (adapted_model_preds == labels)
        ) / np.mean(cover_conds)
      # By thresholding the selection score,
      # we can get the error detection predictions.
      ed_preds = selection_scores < threshold
      ed_acc = np.mean(ed_preds == ed_true_labels)
      accuracy_set.append(accuracy)
      coverage_set.append(coverage)
      ed_acc_set.append(ed_acc)
    accuracy_set = np.array(accuracy_set)
    coverage_set = np.array(coverage_set)
    ed_acc_set = np.array(ed_acc_set)
    sorted_index = np.argsort(coverage_set)
    accuracy_set = accuracy_set[sorted_index]
    coverage_set = coverage_set[sorted_index]
    ed_acc_set = ed_acc_set[sorted_index]
    metrics = {
        'adapted_model_preds': adapted_model_preds,
        'selection_scores': selection_scores,
        'selected_indices': selected_indices,
        'labels': labels,
        'adapted_model_acc': adapted_model_acc,
        'accuracy_set': accuracy_set,
        'coverage_set': coverage_set,
        'ed_acc_set': ed_acc_set,
    }
    return metrics

  @abc.abstractmethod
  def get_results(self, ds):
    """Gets results for the selective prediction task."""
    return {}
