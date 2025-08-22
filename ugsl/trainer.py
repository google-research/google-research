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

"""Trainer of the UGSL framework."""
import csv
import os

from ml_collections import config_dict
import tensorflow as tf

from ugsl import input_layer
from ugsl import models


def get_tf_dataset(
    input_graph, split
):
  """Creates a dataset for the provided split."""
  features = tf.data.Dataset.from_tensors(
      input_graph.get_initial_node_features()
  )
  labels = input_graph.get_all_labels()
  labels = tf.gather(labels, split)
  y = tf.data.Dataset.from_tensors(labels)
  x = (features, tf.data.Dataset.from_tensors(split))
  return tf.data.Dataset.zip((x, y)).repeat()


def train(cfg):
  """The main function to train the UGSL model.

  Args:
    cfg: config dictionary containing hyperparameters
  """
  strategy = tf.distribute.MirroredStrategy()
  input_graph = input_layer.InputLayer(**cfg.dataset)
  splits = input_graph.get_node_split()
  train_ds = get_tf_dataset(input_graph, splits.train)
  validation_ds = get_tf_dataset(input_graph, splits.validation)
  test_ds = get_tf_dataset(input_graph, splits.test)

  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True,
      reduction=tf.keras.losses.Reduction.AUTO,
  )

  with strategy.scope():
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=cfg.run.learning_rate,
        weight_decay=cfg.run.weight_decay,
    )
    model = models.get_gsl_model(input_graph, cfg.model)
    model.compile(
        loss=loss_object,
        optimizer=optimizer,
        metrics=tf.keras.metrics.SparseCategoricalAccuracy("accuracy"),
    )

    best_model_file_path = os.path.join(cfg.run.model_dir, "best_model_weights")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            best_model_file_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(cfg.run.model_dir, "log.csv"),
            append=True,
            separator=";",
        ),
    ]
    history = model.fit(
        train_ds,
        epochs=cfg.run.num_epochs,
        steps_per_epoch=1,
        validation_data=validation_ds,
        validation_steps=1,
        callbacks=callbacks,
    )

    best_val_accuracy = max(history.history["val_accuracy"])
    model.load_weights(best_model_file_path)
    test_loss, test_accuracy = model.evaluate(
        test_ds, steps=1, callbacks=callbacks
    )

    with open(
        os.path.join(cfg.run.model_dir, "test_metrics.csv"), mode="w"
    ) as employee_file:
      employee_writer = csv.writer(
          employee_file, delimiter=";", quotechar='"', quoting=csv.QUOTE_MINIMAL
      )
      employee_writer.writerow(
          ["best val accuracy", "test accuracy", "test loss"]
      )
      employee_writer.writerow([best_val_accuracy, test_accuracy, test_loss])
