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

"""Softmax Response method."""

from typing import Any, Dict, Optional, Tuple

from active_selective_prediction import sampling_methods
from active_selective_prediction.methods.base_sp import SelectivePredictionMethod
from active_selective_prediction.utils import data_util
from active_selective_prediction.utils import tf_util
import numpy as np
import tensorflow as tf


class SR(SelectivePredictionMethod):
  """Softmax Response method."""

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

  def active_learning(
      self,
      target_test_ds
  ):
    """Performs active learning to fine-tune the pre-trained model."""
    adapted_model = self.load_pretrained_model(self.model_path)
    selected_indices = np.array([], dtype=np.int64)
    if self.label_budget <= 0:
      return adapted_model, selected_indices
    if self.debug_info:
      test_acc = tf_util.evaluate_acc(adapted_model, target_test_ds)
      print(f'Test Accuracy: {test_acc:.2%}')
    test_data_dict = data_util.get_ds_data(target_test_ds)
    n = test_data_dict['labels'].shape[0]
    if self.sampling_method == 'uniform':
      sampling_module = sampling_methods.UniformSampling(
          n=n,
          **self.sampling_kwargs,
      )
    elif self.sampling_method == 'confidence':
      sampling_module = sampling_methods.ConfidenceSampling(
          ensemble_models=[adapted_model],
          n=n,
          target_test_ds=target_test_ds,
          **self.sampling_kwargs,
      )
    elif self.sampling_method == 'entropy':
      sampling_module = sampling_methods.EntropySampling(
          ensemble_models=[adapted_model],
          n=n,
          target_test_ds=target_test_ds,
          **self.sampling_kwargs,
      )
    elif self.sampling_method == 'margin':
      sampling_module = sampling_methods.MarginSampling(
          ensemble_models=[adapted_model],
          n=n,
          target_test_ds=target_test_ds,
          **self.sampling_kwargs,
      )
    elif self.sampling_method == 'kcenter_greedy':
      sampling_module = sampling_methods.KCenterGreedySampling(
          ensemble_models=[adapted_model],
          n=n,
          target_test_ds=target_test_ds,
          **self.sampling_kwargs,
      )
    elif self.sampling_method == 'clue':
      sampling_module = sampling_methods.CLUESampling(
          ensemble_models=[adapted_model],
          n=n,
          target_test_ds=target_test_ds,
          **self.sampling_kwargs,
      )
    elif self.sampling_method == 'badge':
      sampling_module = sampling_methods.BADGESampling(
          ensemble_models=[adapted_model],
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
      )
      self.train_model(adapted_model, sub_test_ds)
      if self.debug_info:
        test_acc = tf_util.evaluate_acc(adapted_model, target_test_ds)
        print(f'Test Accuracy: {test_acc:.2%}')
    selected_size = selected_indices.shape[0]
    assert selected_size == self.label_budget, (
        "Size of selected samples doesn't match label budget"
        f' ({selected_size}!={self.label_budget})!'
    )
    return adapted_model, selected_indices

  def get_results(self, ds):
    """Gets results for the selective prediction task."""
    adapted_model, selected_indices = self.active_learning(ds)
    adapted_model_outputs = []
    labels = []
    for batch_x, batch_y in ds:
      labels.extend(batch_y.numpy())
      batch_adapted_model_outputs = tf_util.get_model_output(
          adapted_model, batch_x
      )
      adapted_model_outputs.extend(batch_adapted_model_outputs.numpy())
    adapted_model_outputs = np.array(adapted_model_outputs)
    labels = np.array(labels)
    adapted_model_preds = np.argmax(adapted_model_outputs, axis=1)
    selection_scores = np.max(adapted_model_outputs, axis=1)
    results = self.compute_metrics(
        adapted_model_preds, selection_scores, selected_indices, labels
    )
    return results
