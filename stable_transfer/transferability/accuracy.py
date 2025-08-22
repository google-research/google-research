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

"""Get Accuracy of an Experiment."""

import tensorflow as tf

from stable_transfer.transferability import transfer_experiment


@transfer_experiment.load_or_compute
def get_test_accuracy(experiment):
  """Train target model and evaluate test accuracy."""

  model = experiment.source_model('target_predictions')

  optimizer_name = experiment.config.experiment.accuracy.optimizer
  optimizer_config = experiment.config.experiment.accuracy[optimizer_name]
  # Remove hyper-parameters sweep options (defined as keys ending with '_range')
  optimizer_config = {k: v for k, v in optimizer_config.items()
                      if not k.endswith('_range')}
  model.compile(
      optimizer=tf.keras.optimizers.get(
          dict(class_name=optimizer_name, config=optimizer_config)),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
      )

  if experiment.config.experiment.accuracy.base_trainable:
    epochs = experiment.config.experiment.accuracy.base_trainable_epochs
  else:
    epochs = experiment.config.experiment.accuracy.base_frozen_epochs

  h = model.fit(
      experiment.target_train_dataset,
      epochs=epochs,
      validation_data=experiment.target_test_dataset,
  )

  return dict(test_loss=h.history['val_loss'],
              test_accuracy=h.history['val_sparse_categorical_accuracy'],
              train_loss=h.history['loss'],
              train_accuracy=h.history['sparse_categorical_accuracy'],
              )
