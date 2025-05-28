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

"""Utilities for logging."""
import os
from absl import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import callbacks
from postproc_fairness.utils import utils


TARGET_NAME = "target"


class CustomLogger:
  """Helper class for logging model outputs.

  This class writes scalar results to multiple destinatons:
  - INFO logs for easy debugging.
  - Summary files in `model_dir`.
  - experiment measurements.
  """

  def __init__(self, model_dir):
    """Create a new logger object.

    Args:
      model_dir: Path to model directory.
    """
    self.summary_writer = tf.summary.create_file_writer(model_dir)
    self._measurement_series_cache = {}
    self.model_dir = model_dir
    self.work_unit = None  # Don't record any measurements if None

  def _log_to_measurement_series(self, name, value, epoch):
    """Log `value` to the measurement series with name `name`."""
    if not self.work_unit:
      return
    if name not in self._measurement_series_cache:
      self._measurement_series_cache[name] = (
          self.work_unit.get_measurement_series(name)
      )
    self._measurement_series_cache[name].create_measurement(value, step=epoch)

  def log_scalars(self, epoch, **kwargs):
    """Log scalars (given as keyword arguments)."""
    # step = self.global_step.numpy()
    print(kwargs.items())
    log_msg = ", ".join(["{} = {:.3f}".format(k, v) for k, v in kwargs.items()])
    logging.info("[%d] %s", epoch, log_msg)
    with self.summary_writer.as_default():
      for k, v in sorted(kwargs.items()):
        tf.summary.scalar(k, v, step=epoch)
        self._log_to_measurement_series(k, v, epoch=epoch)


class LogMetricsCallback(callbacks.Callback):
  """Callback for logging metrics."""

  def __init__(
      self,
      logger,
      train_df=None,
      train_y=None,
      valid_df=None,
      valid_y=None,
      test_df=None,
      test_y=None,
      mindiff_data_idxs=None,
      sensitive_attribute=None,
      batch_for_eval=256,
      log_every_n_epochs=1,
  ):
    """The argument `logger` should be of type CustomLogger."""
    self.logger = logger
    self.log_every_n_epochs = log_every_n_epochs
    self.sensitive_attribute = sensitive_attribute
    self.batch_for_eval = batch_for_eval
    self.train_df = train_df.reset_index()
    self.train_y = train_y
    self.train_ds = utils.df_to_dataset(
        self.train_df, self.train_y, shuffle=False
    ).batch(self.batch_for_eval)
    self.mask_for_training_fair_metrics = np.zeros(train_y.shape[0])
    if mindiff_data_idxs is not None:
      self.mask_for_training_fair_metrics[mindiff_data_idxs] = 1
      print(
          "Fairness metrics computed on"
          f" {self.mask_for_training_fair_metrics.sum()} samples."
      )
    self.valid_df = valid_df.reset_index()
    self.valid_y = valid_y
    self.valid_ds = utils.df_to_dataset(
        self.valid_df, self.valid_y, shuffle=False
    ).batch(self.batch_for_eval)
    self.test_df = test_df.reset_index()
    self.test_y = test_y
    self.test_ds = utils.df_to_dataset(
        self.test_df, self.test_y, shuffle=False
    ).batch(self.batch_for_eval)

  def on_train_begin(self, logs=None):
    # Compute metrics on test set.
    test_metric_names = [
        f"test_{metric}" for metric in self.model.metrics_names
    ]
    test_metric_values = self.model.evaluate(self.test_ds)
    print("Starting training now; pretrained model stats:")
    print(dict(zip(test_metric_names, test_metric_values)))

  def on_epoch_end(self, epoch, logs=None):
    if epoch % self.log_every_n_epochs != 0:
      return

    # Compute metrics on test set.
    test_metric_names = [
        f"test_{metric}" for metric in self.model.metrics_names
    ]
    test_metric_values = self.model.evaluate(self.test_ds)
    logs = logs | dict(zip(test_metric_names, test_metric_values))

    # Compute FPR and FNR on training, validation and test data.
    for prefix in ["", "val_", "test_"]:
      logs[f"{prefix}fpr"] = logs[f"{prefix}fp"] / (
          logs[f"{prefix}fp"] + logs[f"{prefix}tn"]
      )
      logs[f"{prefix}fnr"] = logs[f"{prefix}fn"] / (
          logs[f"{prefix}fn"] + logs[f"{prefix}tp"]
      )

    # pylint: disable=unused-argument
    # Compute FPRgap metrics.
    def get_fpr(data_ds, data_df, y, sample_weight=None):
      _ = self.model.predict(data_ds)
      # Note: placeholder for computing FPR gap between y_pred and y, according
      # to x[self.sensitive_attribute], and using sample_weight to select which
      # examples to consider (in particular for the training data).
      fpr_gap = 0
      return fpr_gap

    # Compute FNRgap metrics.
    def get_fnr(data_ds, data_df, y, sample_weight=None):
      _ = self.model.predict(data_ds)
      # Note: placeholder for computing FNR gap between y_pred and y, according
      # to x[self.sensitive_attribute], and using sample_weight to select which
      # examples to consider (in particular for the training data).
      fnr_gap = 0
      return fnr_gap
    # pylint: enable=unused-argument

    logs["fpr_gap"] = get_fpr(
        self.train_ds,
        self.train_df,
        self.train_y,
        self.mask_for_training_fair_metrics,
    )
    logs["val_fpr_gap"] = get_fpr(self.valid_ds, self.valid_df, self.valid_y)
    logs["test_fpr_gap"] = get_fpr(self.test_ds, self.test_df, self.test_y)

    logs["fnr_gap"] = get_fnr(
        self.train_ds,
        self.train_df,
        self.train_y,
        self.mask_for_training_fair_metrics,
    )
    logs["val_fnr_gap"] = get_fnr(self.valid_ds, self.valid_df, self.valid_y)
    logs["test_fnr_gap"] = get_fnr(self.test_ds, self.test_df, self.test_y)

    logs["MEO"] = (logs["fpr_gap"] + logs["fnr_gap"]) / 2
    logs["val_MEO"] = (logs["val_fpr_gap"] + logs["val_fnr_gap"]) / 2
    logs["test_MEO"] = (logs["test_fpr_gap"] + logs["test_fnr_gap"]) / 2

    self.logger.log_scalars(epoch, **logs)

  def on_train_end(self, logs=None):
    if not hasattr(self.model, "original_model"):
      print(
          "Not computing on_train_end statistics for model of type"
          f" {type(self.model)}."
      )
      return

    curr_model = self.model.original_model
    if "pp_multiplier" not in [l.name for l in curr_model.layers]:
      print("Only compute on_train_end statistics for post-processing.")
      return

    multiplier_model = tf.keras.Model(
        inputs=curr_model.input,
        outputs=curr_model.get_layer("pp_multiplier").output,
    )

    def save_multipliers(data_df, y, filename):
      ds = utils.df_to_dataset(data_df, y, shuffle=False).batch(
          self.batch_for_eval
      )
      multipliers = multiplier_model.predict(ds).reshape(-1, 1)
      multipliers_df = pd.DataFrame(
          np.concatenate(
              (
                  data_df.index.to_numpy().reshape(-1, 1),
                  multipliers,
              ),
              axis=1,
          ),
          columns=["index", "multiplier"],
      )
      csv_path = os.path.join(self.logger.model_dir, filename)
      multipliers_df.to_csv(csv_path, index=False)

    save_multipliers(self.train_df, self.train_y, "multiplier_values_train.csv")
    save_multipliers(self.valid_df, self.valid_y, "multiplier_values_valid.csv")
    save_multipliers(self.test_df, self.test_y, "multiplier_values_test.csv")


def print_eval(
    model, data_df, targets, sensitive_attribute="sex", batch_size=256
):
  """Copmpute and print evaluation results."""
  eval_results = model.evaluate(
      x=utils.df_to_dataset(data_df, targets, shuffle=False).batch(batch_size),
      verbose=0,
      return_dict=True,
  )
  if sensitive_attribute is not None:
    attr_counts = data_df.groupby(sensitive_attribute)[
        sensitive_attribute
    ].count()
    for attr in data_df[sensitive_attribute].unique():
      curr_results = model.evaluate(
          x=utils.df_to_dataset(
              data_df[data_df[sensitive_attribute] == attr],
              targets[data_df[sensitive_attribute] == attr],
              shuffle=False,
          ).batch(batch_size),
          batch_size=batch_size,
          verbose=0,
          return_dict=True,
      )
      curr_results["count"] = attr_counts[attr]
      curr_results["fpr"] = curr_results["fp"] / (
          curr_results["fp"] + curr_results["tn"]
      )
      curr_results["fnr"] = curr_results["fn"] / (
          curr_results["fn"] + curr_results["tp"]
      )
      eval_results.update(
          {
              f"{k}_{sensitive_attribute}{attr}": v
              for k, v in curr_results.items()
          }
      )

  for name, value in eval_results.items():
    print(f"{name}: {value}")
  print()
