# coding=utf-8
# Copyright 2023 The Google Research Authors.
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

"""Metrics functions for Chart and Table related tasks."""

from collections.abc import Mapping, Sequence
from typing import Optional

import numpy as np
from pix2struct import metrics as pix2struct_metrics
from scipy import optimize


def _to_float(text):
  try:
    if text.endswith("%"):
      # Convert percentages to floats.
      return float(text.rstrip("%")) / 100.0
    else:
      return float(text)
  except ValueError:
    return None


def _get_relative_distance(
    target, prediction, theta = 1.0
):
  """Returns min(1, |target-prediction|/|target|)."""
  if not target:
    return int(not prediction)
  distance = min(abs((target - prediction) / target), 1)
  return distance if distance < theta else 1


def _table_numbers_match(target, prediction):
  """Calculates matching similarity between two tables following ChartQA."""

  target_numbers = _get_table_numbers(target)
  prediction_numbers = _get_table_numbers(prediction)
  if not target_numbers and not prediction_numbers:
    return 1
  if not target_numbers or not prediction_numbers:
    return 0
  max_len = max(len(target_numbers), len(prediction_numbers))
  distance = []
  for t in target_numbers:
    distance.append([_get_relative_distance(t, p) for p in prediction_numbers])
  cost_matrix = np.array(distance)
  row_ind, col_ind = optimize.linear_sum_assignment(cost_matrix)
  return 1 - cost_matrix[row_ind, col_ind].sum() / max_len


def _get_table_numbers(text):
  numbers = []
  for line in text.splitlines():
    for part in line.split(" | "):
      if part.strip():
        try:
          numbers.append(float(part))
        except ValueError:
          pass
  return numbers


def table_number_accuracy(
    targets,
    predictions,
):
  """Calculates matching similarity between two tables following ChartQA.

  Keeps only numbers and performas a linear matching using the relative error.

  Args:
    targets: ground truth text.
    predictions: predicted text.

  Returns:
    dictionary with metric names as keys and metric value as values.
  """
  return {
      "numbers_match": pix2struct_metrics.aggregate_metrics(
          targets=targets,
          predictions=predictions,
          metric_fn=_table_numbers_match,
          normalize_fn=lambda v: v,
      ),
  }


def _get_table_datapoints(
    text, transposed = False
):
  """Extracts a dict of datapoints from a table."""
  datapoints = {}
  lines = text.lower().splitlines()
  if not lines:
    return datapoints
  if lines[0].startswith("title |"):
    datapoints["title"] = lines[0][len("title |") :].strip()
    offset = 1
  else:
    offset = 0
  if len(lines) < offset + 1:
    return datapoints
  headers = lines[offset].split(" | ")
  if len(headers) == 1:
    return datapoints
  for line in lines[offset + 1 :]:
    parts = line.split(" | ")
    if len(parts) == 1:
      continue
    for header, part in zip(headers[1:], parts[1:]):
      key = (
          f"{header.strip()} {parts[0].strip()}"
          if transposed
          else f"{parts[0].strip()} {header.strip()}"
      )
      datapoints[key] = part.strip()
  return datapoints


def _get_datapoint_metric(
    target,
    prediction,
    text_theta=0.5,
    number_theta=0.1,
):
  """Computes a metric that scores how similar two datapoint pairs are."""
  key_metric = pix2struct_metrics.anls_metric(
      target[0], prediction[0], text_theta
  )
  pred_float = _to_float(prediction[1])
  target_float = _to_float(target[1])
  if pred_float is not None and target_float:
    return key_metric * (
        1 - _get_relative_distance(target_float, pred_float, number_theta)
    )
  elif target[1] == prediction[1]:
    return key_metric
  else:
    return key_metric * pix2struct_metrics.anls_metric(
        target[1], prediction[1], text_theta
    )


def _table_datapoints_precision_recall_f1(
    target,
    prediction,
    text_theta = 0.5,
    number_theta = 0.1,
    transposed = False,
):
  """Calculates matching similarity between two tables as dicts."""
  target_datapoints = list(_get_table_datapoints(target).items())
  prediction_datapoints = list(
      _get_table_datapoints(prediction, transposed=transposed).items()
  )
  if not target_datapoints and not prediction_datapoints:
    return 1, 1, 1
  if not target_datapoints:
    return 0, 1, 0
  if not prediction_datapoints:
    return 1, 0, 0
  distance = []
  for t, _ in target_datapoints:
    distance.append(
        [
            1 - pix2struct_metrics.anls_metric(t, p, text_theta)
            for p, _ in prediction_datapoints
        ]
    )
  cost_matrix = np.array(distance)
  row_ind, col_ind = optimize.linear_sum_assignment(cost_matrix)
  score = 0
  for r, c in zip(row_ind, col_ind):
    score += _get_datapoint_metric(
        target_datapoints[r], prediction_datapoints[c], text_theta, number_theta
    )
  if score == 0:
    return 0, 0, 0
  precision = score / len(prediction_datapoints)
  recall = score / len(target_datapoints)
  return precision, recall, 2 * precision * recall / (precision + recall)


def table_datapoints_precision_recall(
    targets,
    predictions,
    text_theta = 0.5,
    number_theta = 0.1,
):
  """Computes precisin recall and F1 metrics given two flattened tables.

  Parses each string into a dictionary of keys and values using row and column
  headers. Then we match keys between the two dicts as long as their relative
  levenshtein distance is below a threshold. Values are also compared with
  ANLS if strings or relative distance if they are numeric.

  Args:
    targets: list of list of strings.
    predictions: list of strings.
    text_theta: relative edit distance above this is set to the maximum of 1.
    number_theta: relative error rate above this is set to the maximum of 1.

  Returns:
    Mapping with precision, recall and F1
  """
  assert len(targets) == len(predictions)
  precision, recall, f1 = 0, 0, 0
  for pred, target in zip(predictions, targets):
    all_metrics = []
    for transposed in [True, False]:
      all_metrics.extend(
          [
              _table_datapoints_precision_recall_f1(
                  t, pred, text_theta, number_theta, transposed
              )
              for t in target
          ]
      )
    p, r, f = max(all_metrics, key=lambda x: x[-1])
    precision += p
    recall += r
    f1 += f
  return {
      "table_datapoints_precision": 100.0 * precision / len(targets),
      "table_datapoints_recall": 100.0 * recall / len(targets),
      "table_datapoints_f1": 100.0 * f1 / len(targets),
  }
