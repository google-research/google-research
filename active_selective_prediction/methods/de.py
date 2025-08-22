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

"""Deep Ensembles for active selective prediction."""

import time
from typing import Any, Dict, List, Optional, Tuple, Union

from active_selective_prediction import sampling_methods
from active_selective_prediction.methods.base_sp import SelectivePredictionMethod
from active_selective_prediction.utils import data_util
from active_selective_prediction.utils import tf_util
import numpy as np
import tensorflow as tf


class DE(SelectivePredictionMethod):
  """Deep Ensembles for active selective prediction."""

  def __init__(
      self,
      model_arch_name,
      model_arch_kwargs,
      source_train_ds,
      model_path,
      num_models,
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
        sampling_method=sampling_method,
        sampling_kwargs=sampling_kwargs,
        finetune_method=finetune_method,
        finetune_kwargs=finetune_kwargs,
        debug_info=debug_info,
        print_freq=print_freq,
    )
    self.num_models = num_models
    self.ensemble_method = 'soft'

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
    num_steps = self.finetune_kwargs['init_steps']
    source_train_ds_iter = iter(self.source_train_ds)
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
      if self.debug_info and (
          step % self.print_freq == 0 or step == num_steps
      ):
        print(
            f'Step: {step}, Loss: {total_loss/num_batches:.6f}, Time:'
            f' {time.time()-t0:.2f}s'
        )
        t0 = time.time()

  def active_learning(
      self,
      target_test_ds
  ):
    """Performs active learning to learn a model ensemble."""
    adapted_models = []
    for _ in range(self.num_models):
      model = self.load_pretrained_model(self.model_path)
      adapted_models.append(model)
    if self.finetune_kwargs['init_steps'] > 0:
      for model in adapted_models:
        self.train_init_model(model)
    selected_indices = np.array([], dtype=np.int64)
    if self.label_budget <= 0:
      return adapted_models, selected_indices
    test_data_dict = data_util.get_ds_data(target_test_ds)
    n = test_data_dict['labels'].shape[0]
    if self.debug_info:
      for j, model in enumerate(adapted_models):
        test_acc = tf_util.evaluate_acc(model, target_test_ds)
        print(f'Model {j}, Test Accuracy: {test_acc:.2%}')
    if self.sampling_method == 'uniform':
      sampling_module = sampling_methods.UniformSampling(
          n=n,
          **self.sampling_kwargs,
      )
    elif self.sampling_method == 'confidence':
      sampling_module = sampling_methods.ConfidenceSampling(
          ensemble_models=adapted_models,
          n=n,
          target_test_ds=target_test_ds,
          ensemble_method=self.ensemble_method,
          **self.sampling_kwargs,
      )
    elif self.sampling_method == 'entropy':
      sampling_module = sampling_methods.EntropySampling(
          ensemble_models=adapted_models,
          n=n,
          target_test_ds=target_test_ds,
          ensemble_method=self.ensemble_method,
          **self.sampling_kwargs,
      )
    elif self.sampling_method == 'margin':
      sampling_module = sampling_methods.MarginSampling(
          ensemble_models=adapted_models,
          n=n,
          target_test_ds=target_test_ds,
          ensemble_method=self.ensemble_method,
          **self.sampling_kwargs,
      )
    elif self.sampling_method == 'average_kl_divergence':
      sampling_module = sampling_methods.AverageKLDivergenceSampling(
          ensemble_models=adapted_models,
          n=n,
          target_test_ds=target_test_ds,
          ensemble_method=self.ensemble_method,
          **self.sampling_kwargs,
      )
    elif self.sampling_method == 'kcenter_greedy':
      sampling_module = sampling_methods.KCenterGreedySampling(
          ensemble_models=adapted_models,
          n=n,
          target_test_ds=target_test_ds,
          **self.sampling_kwargs,
      )
    elif self.sampling_method == 'clue':
      sampling_module = sampling_methods.CLUESampling(
          ensemble_models=adapted_models,
          n=n,
          target_test_ds=target_test_ds,
          **self.sampling_kwargs,
      )
    elif self.sampling_method == 'badge':
      sampling_module = sampling_methods.BADGESampling(
          ensemble_models=adapted_models,
          n=n,
          target_test_ds=target_test_ds,
          **self.sampling_kwargs,
      )
    else:
      raise ValueError(f'Not supported sampling method {self.sampling_method}!')
    label_budget_per_round = self.label_budget // self.sampling_rounds
    label_budget_array = (
        np.ones(self.sampling_rounds, dtype=np.int32) * label_budget_per_round
    )
    label_budget_array[-1] += (
        self.label_budget - label_budget_per_round * self.sampling_rounds
    )
    for i in range(self.sampling_rounds):
      selected_indices = sampling_module.select_batch_to_label(
          selected_indices, label_budget_array[i]
      )
      sub_test_ds = data_util.construct_sub_dataset(
          test_data_dict,
          selected_indices,
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
          print(f'Model {j}, Test Accuracy: {test_acc:.2%}')
    selected_size = selected_indices.shape[0]
    assert selected_size == self.label_budget, (
        "Size of selected samples doesn't match label budget"
        f' ({selected_size}!={self.label_budget})!'
    )
    return adapted_models, selected_indices

  def get_results(self, ds):
    """Gets results for the selective prediction task."""
    adapted_models, selected_indices = self.active_learning(ds)
    ensemble_outputs = []
    labels = []
    for batch_x, batch_y in ds:
      labels.extend(batch_y.numpy())
      batch_ensemble_outputs = tf_util.get_ensemble_model_output(
          adapted_models, batch_x, self.ensemble_method
      )
      ensemble_outputs.extend(batch_ensemble_outputs.numpy())
    ensemble_outputs = np.array(ensemble_outputs)
    labels = np.array(labels)
    adapted_model_preds = np.argmax(ensemble_outputs, axis=1)
    selection_scores = np.max(ensemble_outputs, axis=1)
    results = self.compute_metrics(
        adapted_model_preds, selection_scores, selected_indices, labels
    )
    return results
