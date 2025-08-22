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

"""Metrics functions for Chart and Table related tasks."""

from collections.abc import Mapping, Sequence
import dataclasses
import itertools
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


def table_number_accuracy_per_point(
    targets,
    predictions,
):
  """Calculates matching similarity between two tables following ChartQA.

  Keeps only numbers and performas a linear matching using the relative error.

  Args:
    targets: ground truth text.
    predictions: predicted text.

  Returns:
    A list of float numbers.
  """
  all_points_scores = []
  for p, targets in zip(predictions, targets):
    all_points_scores.append(max(_table_numbers_match(t, p) for t in targets))
  return all_points_scores


def table_number_accuracy(
    targets,
    predictions,
):
  """Aggregated version of table_number_accuracy_per_point().

  Same as table_number_accuracy_per_point() but returning an aggregated score.

  Args:
    targets: ground truth text.
    predictions: predicted text.

  Returns:
    dictionary with metric names as keys and metric value as values.
  """
  scores = table_number_accuracy_per_point(targets, predictions)
  return {"numbers_match": (100.0 * sum(scores)) / len(targets)}


def _permute(values, indexes):
  return tuple(values[i] if i < len(values) else "" for i in indexes)


@dataclasses.dataclass(frozen=True)
class Table:
  """Helper class for the content of a markdown table."""

  title: Optional[str] = None
  headers: tuple[str, Ellipsis] = dataclasses.field(default_factory=tuple)
  rows: tuple[tuple[str, Ellipsis], Ellipsis] = dataclasses.field(default_factory=tuple)

  def permuted(self, indexes):
    """Builds a version of the table changing the column order."""
    return Table(
        title=self.title,
        headers=_permute(self.headers, indexes),
        rows=tuple(_permute(row, indexes) for row in self.rows),
    )

  def aligned(
      self, headers, text_theta = 0.5
  ):
    """Builds a column permutation with headers in the most correct order."""
    if len(headers) != len(self.headers):
      raise ValueError(f"Header length {headers} must match {self.headers}.")
    distance = []
    for h2 in self.headers:
      distance.append(
          [
              1 - pix2struct_metrics.anls_metric(h1, h2, text_theta)
              for h1 in headers
          ]
      )
    cost_matrix = np.array(distance)
    row_ind, col_ind = optimize.linear_sum_assignment(cost_matrix)
    permutation = [idx for _, idx in sorted(zip(col_ind, row_ind))]
    score = (1 - cost_matrix)[permutation[1:], range(1, len(row_ind))].prod()
    return self.permuted(permutation), score


def _parse_table(text, transposed = False):
  """Builds a table from a markdown representation."""
  lines = text.lower().splitlines()
  if not lines:
    return Table()
  if lines[0].startswith("title |"):
    title = lines[0][len("title |") :].strip()
    offset = 1
  else:
    title = None
    offset = 0
  if len(lines) < offset + 1:
    return Table(title=title)
  rows = []
  for line in lines[offset:]:
    rows.append(tuple(v.strip() for v in line.split(" | ")))
  if transposed:
    rows = [tuple(row) for row in itertools.zip_longest(*rows, fillvalue="")]
  return Table(title=title, headers=rows[0], rows=tuple(rows[1:]))


def _get_table_datapoints(table):
  """Extracts a dict of datapoints from a table."""
  datapoints = {}
  if table.title is not None:
    datapoints["title"] = table.title
  if not table.rows or len(table.headers) <= 1:
    return datapoints
  for row in table.rows:
    for header, cell in zip(table.headers[1:], row[1:]):
      datapoints[f"{row[0]} {header}"] = cell
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
    target_table,
    prediction_table,
    text_theta = 0.5,
    number_theta = 0.1,
):
  """Calculates matching similarity between two tables as dicts."""
  target_datapoints = list(_get_table_datapoints(target_table).items())
  prediction_datapoints = list(_get_table_datapoints(prediction_table).items())
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


def table_datapoints_precision_recall_per_point(
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
    Dictionary with per-point precision, recall and F1
  """
  assert len(targets) == len(predictions)
  per_point_scores = {"precision": [], "recall": [], "f1": []}
  for pred, target in zip(predictions, targets):
    all_metrics = []
    for transposed in [True, False]:
      pred_table = _parse_table(pred, transposed=transposed)
      # pylint:disable=g-complex-comprehension
      all_metrics.extend(
          [
              _table_datapoints_precision_recall_f1(
                  _parse_table(t),
                  pred_table,
                  text_theta,
                  number_theta,
              )
              for t in target
          ]
      )
      # pylint:enable=g-complex-comprehension
    p, r, f = max(all_metrics, key=lambda x: x[-1])
    per_point_scores["precision"].append(p)
    per_point_scores["recall"].append(r)
    per_point_scores["f1"].append(f)
  return per_point_scores


def table_datapoints_precision_recall(
    targets,
    predictions,
    text_theta = 0.5,
    number_theta = 0.1,
):
  """Aggregated version of table_datapoints_precision_recall_per_point().

  Same as table_datapoints_precision_recall_per_point() but returning aggregated
  scores instead of per-point scores.

  Args:
    targets: list of list of strings.
    predictions: list of strings.
    text_theta: relative edit distance above this is set to the maximum of 1.
    number_theta: relative error rate above this is set to the maximum of 1.

  Returns:
    Dictionary with aggregated precision, recall and F1
  """
  score_dict = table_datapoints_precision_recall_per_point(
      targets, predictions, text_theta, number_theta
  )
  return {
      "table_datapoints_precision": (
          100.0 * sum(score_dict["precision"]) / len(targets)
      ),
      "table_datapoints_recall": (
          100.0 * sum(score_dict["recall"]) / len(targets)
      ),
      "table_datapoints_f1": 100.0 * sum(score_dict["f1"]) / len(targets),
  }


def _get_row_datapoints(table):
  """Extracts a list of datapoints from a table as rows."""
  if table.title is None:
    return table.rows
  return table.rows + (("title", table.title),)


def _get_row_metric(
    target_parts,
    prediction_parts,
    text_theta=0.5,
    number_theta=0.1,
):
  """Computes a metric that scores how similar two datapoint pairs are."""
  if len(target_parts) != len(prediction_parts) or not target_parts:
    return 0.0
  result = []
  for target, prediction in zip(target_parts, prediction_parts):
    pred_float = _to_float(prediction)
    target_float = _to_float(target)
    if target == prediction:
      result.append(1.0)
    elif pred_float is not None and target_float:
      result.append(
          1 - _get_relative_distance(target_float, pred_float, number_theta)
      )
    elif target_float is not None:
      result.append(0.0)
    else:
      result.append(
          pix2struct_metrics.anls_metric(target, prediction, text_theta)
      )
  return np.prod(result)


def _row_datapoints_precision_recall_f1(
    target,
    prediction,
    text_theta = 0.5,
    number_theta = 0.1,
):
  """Calculates matching similarity between two tables as list of rows."""
  target_datapoints = _get_row_datapoints(target)
  aligned_prediction, aligned_score = prediction.aligned(
      target.headers, text_theta
  )
  prediction_datapoints = _get_row_datapoints(aligned_prediction)
  if not target_datapoints and not prediction_datapoints:
    return 1, 1, 1
  if not target_datapoints:
    return 0, 1, 0
  if not prediction_datapoints or not aligned_score:
    return 1, 0, 0
  metrics = []
  for t in target_datapoints:
    metrics.append(
        [
            aligned_score * _get_row_metric(t, p, text_theta, number_theta)
            for p in prediction_datapoints
        ]
    )
  metrics_matrix = np.array(metrics)
  row_ind, col_ind = optimize.linear_sum_assignment(1 - metrics_matrix)
  score = metrics_matrix[row_ind, col_ind].sum()
  if score == 0:
    return 0, 0, 0
  precision = score / len(prediction_datapoints)
  recall = score / len(target_datapoints)
  return precision, recall, 2 * precision * recall / (precision + recall)


def row_datapoints_precision_recall(
    targets,
    predictions,
    text_theta = 0.5,
    number_theta = 0.1,
):
  """Computes precisin recall and F1 metrics given two flattened tables.

  Parses each string into a list of rows using column headers. Then we match
  entries by their levenshtein / numeric relative distance is below a threshold.

  Args:
    targets: list of list of strings.
    predictions: list of strings.
    text_theta: relative edit distance above this is set to the maximum of 1.
    number_theta: relative error rate above this is set to the maximum of 1.

  Returns:
    Mapping with precision, recall and F1
  """
  if len(targets) != len(predictions):
    raise ValueError(
        f"Targets has length {len(targets)} and predictions has length "
        f"{len(predictions)}."
    )
  precision, recall, f1 = 0, 0, 0
  for pred, target in zip(predictions, targets):
    all_metrics = []
    prediction_tables = [
        _parse_table(pred, transposed=transposed)
        for transposed in [True, False]
    ]
    for t in target:
      for target_transposed in [True, False]:
        target_table = _parse_table(t, transposed=target_transposed)
        for prediction_table in prediction_tables:
          if len(target_table.headers) != len(prediction_table.headers):
            continue
          all_metrics.append(
              _row_datapoints_precision_recall_f1(
                  target_table,
                  prediction_table,
                  text_theta,
                  number_theta,
              )
          )
    p, r, f = max(all_metrics, key=lambda x: x[-1], default=(0, 0, 0))
    precision += p
    recall += r
    f1 += f
  return {
      "row_datapoints_precision": 100.0 * precision / len(targets),
      "row_datapoints_recall": 100.0 * recall / len(targets),
      "row_datapoints_f1": 100.0 * f1 / len(targets),
  }
