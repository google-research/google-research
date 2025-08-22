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

"""Active Selective Prediction using Ensembles and Self-Training."""

import time
from typing import Any, Dict, List, Optional, Tuple, Union

from active_selective_prediction import sampling_methods
from active_selective_prediction.methods.base_sp import SelectivePredictionMethod
from active_selective_prediction.utils import data_util
from active_selective_prediction.utils import tf_util
import numpy as np
import tensorflow as tf


class ASPEST(SelectivePredictionMethod):
  """Active Selective Prediction using Ensembles and Self-Training."""

  def __init__(
      self,
      model_arch_name,
      model_arch_kwargs,
      source_train_ds,
      label_budget,
      batch_size,
      sampling_rounds,
      max_epochs,
      patience_epochs,
      min_epochs,
      optimizer_name,
      optimizer_kargs,
      model_path,
      num_models,
      self_train_kwargs,
      finetune_method,
      finetune_kwargs,
      debug_info = False,
      print_freq = 1,
  ):
    super().__init__(
        model_arch_name=model_arch_name,
        model_arch_kwargs=model_arch_kwargs,
        model_path=model_path,
        source_train_ds=source_train_ds,
        label_budget=label_budget,
        batch_size=batch_size,
        sampling_rounds=sampling_rounds,
        max_epochs=max_epochs,
        patience_epochs=patience_epochs,
        min_epochs=min_epochs,
        optimizer_name=optimizer_name,
        optimizer_kargs=optimizer_kargs,
        sampling_method='average_margin',
        sampling_kwargs={},
        finetune_method=finetune_method,
        finetune_kwargs=finetune_kwargs,
        debug_info=debug_info,
        print_freq=print_freq,
    )
    self.num_models = num_models
    self.self_train_kwargs = self_train_kwargs
    self.ensemble_method = 'soft'
    self.reset_ensemble_state()

  def reset_ensemble_state(
      self, target_test_ds = None
  ):
    """Resets the state of the checkpoint ensemble."""
    self.ensemble_model_outputs = None
    self.counts = 0
    self.target_test_ds = target_test_ds

  def train_init_model(self, model):
    """Trains initial models on the source dataset."""

    @tf.function
    def train_step(
        batch_x,
        batch_y
    ):
      """Trains the model for one optimization step."""
      with tf.GradientTape() as tape:
        batch_outputs = model(batch_x, training=True)
        loss = tf.math.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                batch_y, batch_outputs
            )
        )
      grads = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))
      return loss

    t0 = time.time()
    optimizer = self.get_optimizer()
    source_train_ds_iter = iter(self.source_train_ds)
    num_steps = self.finetune_kwargs['init_steps']
    total_loss = 0
    num_batches = 0
    for step in range(1, num_steps + 1):
      try:
        batch_x, batch_y = next(source_train_ds_iter)
      except StopIteration:
        source_train_ds_iter = iter(self.source_train_ds)
        batch_x, batch_y = next(source_train_ds_iter)
      loss = train_step(batch_x, batch_y)
      total_loss += loss
      num_batches += 1
      if (
          step % self.finetune_kwargs['init_ckpt_step'] == 0
          or step == num_steps
      ):
        self.update_ensemble_state(model)
      if self.debug_info and (
          step % self.print_freq == 0 or step == num_steps
      ):
        print(
            f'Step: {step}, Loss: {total_loss/num_batches:.6f}, Time:'
            f' {time.time()-t0:.2f}s'
        )
        t0 = time.time()

  def train_model(
      self,
      model,
      target_ds
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
      if (
          epoch % self.finetune_kwargs['ckpt_epoch'] == 0
          or epoch == self.max_epochs
      ):
        self.update_ensemble_state(model)
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

  def pseudo_train_model(
      self,
      model,
      target_ds
  ):
    """Trains the model on both source and target datasets with pseudo-labels."""

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
              kl_divergence_loss(
                  batch_target_y, batch_target_outputs
              )
          )
        elif self.finetune_method == 'target_only':
          batch_target_outputs = model(batch_target_x, training=True)
          source_loss = 0
          target_loss = tf.math.reduce_mean(
              kl_divergence_loss(
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
    kl_divergence_loss = tf.keras.losses.KLDivergence(
        reduction=tf.keras.losses.Reduction.NONE
    )
    source_train_ds_iter = iter(self.source_train_ds)
    num_epochs = self.self_train_kwargs['pseudo_train_epochs']
    for epoch in range(1, num_epochs + 1):
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
      if (
          epoch % self.self_train_kwargs['pseudo_ckpt_epoch'] == 0
          or epoch == num_epochs
      ):
        self.update_ensemble_state(model)
      avg_loss /= num_batches
      if self.debug_info and (
          epoch % self.print_freq == 0 or epoch == num_epochs
      ):
        print(
            f'Epoch: {epoch}, Loss: {avg_loss:.6f}, Time: {time.time()-t0:.2f}s'
        )
        t0 = time.time()

  def construct_pseudo_labeled_set(
      self,
      models,
      ds,
      test_data_dict,
  ):
    """Constructs pseudo labeled set."""
    pseudo_label_data_dict = {'inputs': test_data_dict['inputs']}
    if self.self_train_kwargs['use_checkpoint_ensemble']:
      ensemble_outputs = self.get_ensemble_output()
    else:
      ensemble_outputs = []
      for batch_x, _ in ds:
        batch_output = tf_util.get_ensemble_model_output(
            models, batch_x, self.ensemble_method
        )
        ensemble_outputs.extend(batch_output.numpy())
      ensemble_outputs = np.array(ensemble_outputs)
    confs = np.max(ensemble_outputs, axis=1)
    n = ensemble_outputs.shape[0]
    lower_threshold = self.self_train_kwargs['lower_threshold']
    upper_threshold = self.self_train_kwargs['upper_threshold']
    conds = (confs >= lower_threshold)&(confs < upper_threshold)
    pseudo_label_data_dict['labels'] = ensemble_outputs
    pseudo_label_indices = np.where(conds)[0]
    np.random.shuffle(pseudo_label_indices)
    frac = self.self_train_kwargs['frac']
    pseudo_label_indices = pseudo_label_indices[:int(n * frac)]
    if pseudo_label_indices.shape[0] > 0:
      pseudo_label_ds = data_util.construct_sub_dataset(
          data_dict=pseudo_label_data_dict,
          selected_indices=pseudo_label_indices,
          batch_size=self.batch_size,
          shuffle=True,
          include_label=True,
      )
    else:
      pseudo_label_ds = None
    if self.debug_info:
      print(f'Pseudo label set size: {pseudo_label_indices.shape[0]}')
      print(
          f'Size of conf>={upper_threshold}: {np.sum(confs>=upper_threshold)}'
      )
      print(
          f'Conf range: [{np.min(confs[pseudo_label_indices])},'
          f' {np.max(confs[pseudo_label_indices])}]'
      )
      labels = test_data_dict['labels']
      preds = np.argmax(ensemble_outputs, axis=1)
      correct = preds == labels
      print(
          f'Pseudo Label Accuracy: {np.mean(correct[pseudo_label_indices]):.2%}'
      )
    return pseudo_label_ds

  def active_learning(
      self,
      target_test_ds
  ):
    """Performs active learning to learn a model ensemble."""
    self.reset_ensemble_state(target_test_ds)
    adapted_models = []
    for j in range(self.num_models):
      model = self.load_pretrained_model(
          model_path=self.model_path,
      )
      adapted_models.append(model)
    test_data_dict = data_util.get_ds_data(target_test_ds)
    n = test_data_dict['labels'].shape[0]
    selected_indices = np.array([], dtype=np.int64)
    if self.finetune_kwargs['init_steps'] > 0:
      for j, model in enumerate(adapted_models):
        self.train_init_model(model)
        if self.debug_info:
          test_acc = tf_util.evaluate_acc(model, target_test_ds)
          print(f'Model {j}, Test Accuracy: {test_acc:.2%}')
    if self.debug_info:
      test_acc = tf_util.evaluate_ensemble_acc(adapted_models, target_test_ds)
      print(f'Round 0, Ensemble Test Accuracy: {test_acc:.2%}')
    if self.label_budget <= 0:
      return adapted_models, selected_indices
    sampling_module = sampling_methods.AverageMarginSampling(
        n=n, **self.sampling_kwargs
    )
    label_budget_per_round = self.label_budget // self.sampling_rounds
    label_budget_array = (
        np.ones(self.sampling_rounds, dtype=np.int32) * label_budget_per_round
    )
    label_budget_array[-1] += (
        self.label_budget - label_budget_per_round * self.sampling_rounds
    )
    for i in range(self.sampling_rounds):
      sampling_module.update_ensemble_outputs(self.get_ensemble_output())
      selected_indices = sampling_module.select_batch_to_label(
          selected_indices, label_budget_array[i]
      )
      self.reset_ensemble_state(target_test_ds)
      sub_test_ds = data_util.construct_sub_dataset(
          data_dict=test_data_dict,
          selected_indices=selected_indices,
          batch_size=self.batch_size,
          shuffle=True,
          include_label=True,
      )
      for j, model in enumerate(adapted_models):
        self.train_model(
            model=model,
            target_ds=sub_test_ds,
        )
        if self.debug_info:
          test_acc = tf_util.evaluate_acc(model, target_test_ds)
          print(f'Round {i+1}, Model {j}, Test Accuracy: {test_acc:.2%}')
      pseudo_label_ds = self.construct_pseudo_labeled_set(
          adapted_models,
          target_test_ds,
          test_data_dict,
      )
      if pseudo_label_ds is not None:
        for j, model in enumerate(adapted_models):
          self.pseudo_train_model(
              model=model,
              target_ds=pseudo_label_ds,
          )
          if self.debug_info:
            test_acc = tf_util.evaluate_acc(model, target_test_ds)
            print(f'Round {i+1}, Model {j}, Test Accuracy: {test_acc:.2%}')
      if self.debug_info:
        test_acc = tf_util.evaluate_ensemble_acc(
            adapted_models, target_test_ds
        )
        print(f'Round {i+1}, Ensemble Test Accuracy: {test_acc:.2%}')
    selected_size = selected_indices.shape[0]
    assert selected_size == self.label_budget, (
        "Size of selected samples doesn't match label budget"
        f' ({selected_size}!={self.label_budget})!'
    )
    return adapted_models, selected_indices

  def update_ensemble_state(self, model):
    """Updates the state of the checkpoint ensemble."""
    if self.self_train_kwargs['use_checkpoint_ensemble']:
      outputs = []
      for batch_x, _ in self.target_test_ds:
        batch_output = tf_util.get_model_output(model, batch_x)
        outputs.extend(batch_output.numpy())
      outputs = np.array(outputs)
      if self.ensemble_model_outputs is None:
        self.ensemble_model_outputs = outputs
      else:
        self.ensemble_model_outputs = (
            (self.ensemble_model_outputs * self.counts + outputs)
            / (self.counts + 1)
        )
      self.counts += 1

  def get_ensemble_output(self):
    """Gets the output of the checkpoint ensemble."""
    return self.ensemble_model_outputs

  def get_results(self, ds):
    """Gets results for the selective prediction task."""
    adapted_models, selected_indices = self.active_learning(ds)
    if self.self_train_kwargs['use_checkpoint_ensemble']:
      ensemble_outputs = self.get_ensemble_output()
    else:
      ensemble_outputs = []
    labels = []
    for batch_x, batch_y in ds:
      labels.extend(batch_y.numpy())
      if not self.self_train_kwargs['use_checkpoint_ensemble']:
        batch_ensemble_outputs = tf_util.get_ensemble_model_output(
            adapted_models, batch_x, self.ensemble_method
        )
        ensemble_outputs.extend(batch_ensemble_outputs.numpy())
    if not self.self_train_kwargs['use_checkpoint_ensemble']:
      ensemble_outputs = np.array(ensemble_outputs)
    labels = np.array(labels)
    adapted_model_preds = np.argmax(ensemble_outputs, axis=1)
    selection_scores = np.max(ensemble_outputs, axis=1)
    results = self.compute_metrics(
        adapted_model_preds, selection_scores, selected_indices, labels
    )
    return results
