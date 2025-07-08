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

"""Utilities for computing ECE, Brier, etc."""

import dataclasses
from typing import Sequence
import numpy as np
import pandas as pd
import scipy.optimize
import sklearn.metrics


def compute_ece(
    is_correct, confidence
):
  """Computes the ECE.

  Args:
    is_correct: Whether or not the prediction is correct.
    confidence: The confidence of each prediction.

  Returns:
    The ECE and a dataframe with the information used to calculate the ECE.
  """
  confidence_col_name = "confidence"
  is_correct_col_name = "is_correct"
  df = pd.DataFrame({
      confidence_col_name: list(confidence),
      is_correct_col_name: list(is_correct),
  })
  # Sainty check that the confidence values are in [0, 1]. We allow a small
  # margin because temperature scaling can return values slightly above 1.
  between_0_and_1 = df[confidence_col_name].apply(lambda x: 0 <= x <= 1 + 1e-1)
  count_not_between_0_and_1 = (~between_0_and_1).sum()
  if count_not_between_0_and_1 > 0:
    print(
        f"WARNING: {count_not_between_0_and_1} confidence values out of"
        f" {len(df)} are not in [0, 1]"
    )

  num_rows = len(df)
  df["bin"] = pd.cut(df[confidence_col_name], bins=11, labels=False)
  df = (
      df.groupby("bin")
      .agg({
          is_correct_col_name: ["mean", len],
          confidence_col_name: "mean",
      })
      .rename({is_correct_col_name: "pred_accuracy"}, axis=1)
  )
  df["diff"] = df[confidence_col_name]["mean"] - df["pred_accuracy"]["mean"]
  df["bin_size"] = df["pred_accuracy"]["len"] / num_rows
  ece = np.sum(df["bin_size"] * np.abs(df["diff"]))
  return ece, df


def compute_brier(
    is_correct, confidence
):
  confidence_col_name = "confidence"
  is_correct_col_name = "is_correct"
  df = pd.DataFrame({
      confidence_col_name: list(confidence),
      is_correct_col_name: list(is_correct),
  })
  brier_score = np.mean((df["is_correct"] - df[confidence_col_name]) ** 2)
  return brier_score, df


class TemperatureScaling:
  """Temperature scaling calibration method."""

  def __init__(self, temperature = 1):
    self.temperature = temperature

  def _loss(
      self,
      temperature,
      confidences,
      one_hot_labels,
  ):
    predicted = self.predict(confidences, temperature)
    return sklearn.metrics.log_loss(
        y_true=one_hot_labels,
        y_pred=predicted,
        # Pass the labels explicitly to avoid an error in the rare cases where
        # all the labels are the same.
        labels=[True, False],
    )

  def fit(self, confidences, one_hot_labels):
    opt = scipy.optimize.minimize(
        self._loss,
        x0=self.temperature,
        args=(confidences, one_hot_labels),
        options={"maxiter": 1000},
        jac="3-point",
    )
    self.temperature = opt.x[0]

    return opt

  def predict(
      self, confidences, temperature = None
  ):
    if temperature is None:
      temperature = self.temperature
    scaled_confidences = (np.array(confidences) * abs(temperature)).reshape(-1)
    return np.clip(scaled_confidences, 0, 1)


def temperature_scaling(is_correct, confidence):
  scalar = TemperatureScaling()
  confidence = confidence.reshape(-1, 1)
  labels = is_correct.reshape(-1, 1)
  scalar.fit(confidences=confidence, one_hot_labels=labels)
  output = np.array(scalar.predict(confidence)).reshape(-1)
  return output


@dataclasses.dataclass
class CalibrationMetrics:
  ece: float
  brier_score: float


def fit_and_calculate_calibration_metrics(
    data,
    confidence_col_name,
    only_first_trace = True,
):
  """Calculates the calibration metrics for a single confidence method.

  First the confidence is scaled by the temperature scaling method. Then, all
  the metrics are calculated on the scaled confidence.

  Args:
    data: Per trace data frame.
    confidence_col_name: The name of the confidence column in `data`.
    only_first_trace: If true, only the first trace is used for caclulating the
      calibration metrics.

  Returns:
    The calculated stats.
  """
  data = data.copy()
  data["confidence"] = data[confidence_col_name]
  if only_first_trace:
    data = data.groupby("question_id").agg({
        "is_correct": "first",
        "confidence": "first",
    })
  is_correct = data["is_correct"].to_numpy()
  confidence = data["confidence"].to_numpy()
  confidence = temperature_scaling(is_correct, confidence)

  ece, _ = compute_ece(is_correct, confidence)
  brier_score = compute_brier(is_correct, confidence)[0]

  return CalibrationMetrics(ece=ece, brier_score=brier_score)
