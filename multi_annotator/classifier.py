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

"""Trains a single task or multi task classifier and uses it for prediction."""

import collections
import os
import tempfile

from absl import logging
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import transformers

import utils

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("INFO")
tf.compat.v1.disable_eager_execution()
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
logging.set_verbosity(logging.ERROR)
logging.warning("should not print")
tf.get_logger().setLevel("ERROR")


class Classifier():
  """Classifier can be single-task, or multi-task."""

  def __init__(self, params, task_labels=("majority")):
    """Creates a Classifier instance for predicting task_labels.

    Args:
      params: a Params instance which includes the hyperparameters of the model
      task_labels: list of label names to be predicted from text.
    """
    self.params = params
    self.task_labels = task_labels
    self.cache_dir = tempfile.gettempdir()

  def create(self, labels=1):
    """Creates a single-task, multi-task or multi-label classifier.

    Args:
      labels: shows the number of labels for the output. If more than 1, the
        model is multi-label. The length of self.task_labels shows whether the
        model is a single-task or multi-task.
    """
    tf.compat.v1.reset_default_graph()
    config = transformers.BertConfig.from_pretrained(
        os.path.join(self.params.bert_path, "config.json"),
        cache_dir=self.cache_dir)

    model = transformers.TFBertModel.from_pretrained(
        os.path.join(self.params.bert_path, "tf_model.h5"),
        config=config,
        cache_dir=self.cache_dir)

    inputs = tf.keras.layers.Input(
        shape=(int(self.params.max_l),), dtype=tf.int64, name="inputs")
    atten = tf.keras.layers.Input(
        shape=(int(self.params.max_l),), dtype=tf.int64, name="atten")

    hidden = model([inputs, atten])[1]

    drop_hidden = tf.keras.layers.Dropout(.1)(
        hidden, training=(self.params.mc_dropout))
    logits = dict()

    for task_label in self.task_labels:
      logits[task_label] = tf.keras.layers.Dense(
          labels, activation="sigmoid", name=task_label)(
              drop_hidden)

    self.tf_model = tf.keras.Model({
        "inputs": inputs,
        "atten": atten,
    }, logits)

  def train_model(self, train_batches,
                  val_batches, loss_function,
                  weights):
    """Trains and validates a classifier on the input batches.

    Args:
      train_batches: a dictionary of inputs, attnetions and labels created by
        self.get_batches()
      val_batches: a dictionary of inputs, attnetions and labels created by
        self.get_batches()
      loss_function: a function for calculating the loss value during training
      weights: a dictionary of weights for each task.

    Returns:
      the number of training epochs before early stopping.
    """
    cb_early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=self.params.early_stopping_check,
        min_delta=self.params.min_epoch_change)

    cb_scheduler = tf.keras.callbacks.LearningRateScheduler(
        utils.reduce_learning_rate)

    self.tf_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=self.params.lr),
        loss=loss_function)

    # replacing the unavailable labels with -1
    # -1 will then be masked when calculating the loss
    # in the multi_task_loss function
    y = {
        task_label: train_batches[task_label].replace(np.nan, -1)
        for task_label in self.task_labels
    }
    val_y = {
        task_label: val_batches[task_label].replace(np.nan, -1)
        for task_label in self.task_labels
    }

    history = self.tf_model.fit(
        x={
            "inputs": train_batches["inputs"],
            "atten": train_batches["attentions"],
        },
        y=y,
        epochs=self.params.n_epoch,
        callbacks=[cb_early, cb_scheduler],
        validation_data=({
            "inputs": val_batches["inputs"],
            "atten": val_batches["attentions"]
        }, val_y),
        class_weight=weights,
    )

    return len(history.history["loss"])

  def predict(self, batches):
    """Predicts the outputs for each task_label.

    Args:
      batches: the input batches created through self.get_batches()

    Returns:
      A dataframe that includes predictions and labels for each task_label
    """

    results = collections.defaultdict(list)
    if "majority" in batches.keys():
      results["majority"] = batches["majority"]
    if "text_id" in batches.keys():
      results["text_id"] = batches["text_id"]

    logits = self.tf_model.predict(x={
        "inputs": batches["inputs"],
        "atten": batches["attentions"]
    })

    for i, task_label in enumerate(self.task_labels):

      predictions = utils.to_binary(
          logits[i] if len(self.task_labels) > 1 else logits)
      results[task_label + "_pred"] = predictions
      if task_label in batches.keys():
        results[task_label + "_label"] = batches[task_label]
      if len(self.task_labels) == 1:
        results[task_label + "_logit"] = logits.flatten()

    return pd.DataFrame(results)

  def mc_predict(self, batches):
    """Uses the trained models for mc_pass iterations to calculate uncertainty.

    Each iteration is performed with dropouts, so the predictions vary.
    Based on Gal and Ghahramani, 2016.

    Args:
      batches: the input batches created through self.get_batches()

    Returns:
      A dataframe that includes predictions and labels for each task_label
    """

    results = collections.defaultdict(list)
    dropout_predictions = np.empty((0, batches["inputs"].shape[0], 1))
    # for each task the mc dropout will be performed for mc_pass iterations
    for i, task_label in enumerate(self.task_labels):
      for _ in range(self.params.mc_passes):
        logits = self.tf_model.predict(x={
            "inputs": batches["inputs"],
            "atten": batches["attentions"]
        })
        mc_predictions = utils.to_binary(
            logits[i] if len(self.task_labels) > 1 else logits)

        dropout_predictions = np.vstack(
            (dropout_predictions, mc_predictions[np.newaxis, :, np.newaxis]))
      # the average of the predictions with different dropouts
      # is calculated as the uncertainty of the model predictions
      results[task_label + "_mean"] = list(
          np.squeeze(np.mean(dropout_predictions, axis=0)))
      results[task_label + "_variance"] = list(
          np.squeeze(np.var(dropout_predictions, axis=0)))

    return pd.DataFrame(results)
