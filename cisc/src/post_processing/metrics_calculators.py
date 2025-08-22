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

"""Calculate metrics for the evaluation results.

For examples, we calculate the accuracy difference between two methods,
or the number of traces saved by a specific method.
"""

import dataclasses
import numpy as np


@dataclasses.dataclass(frozen=True)
class ResultMetrics:
  """Holds all the results of the methods evaluation.

  All the indexes in this class are one-based.
  """

  # The number of traces at which the accuracy is calculated.
  # For example running CISC with a compute budged of 5 or 10 traces
  evaluated_position: int
  # The basline score at `evaluated_position` in percentage.
  baseline_accuracy: float
  # The evaluated score at `evaluated_position` in percentage..
  evaluated_accuracy: float
  # The number of traces needed in the baseline to reach the accuracy of the
  # evaluated method. If the evaluated method is better, then we expect
  # `baseline_traces_needed` > `evaluated_position``. For example, in the paper
  # we show that self consistency (the baseline) sometimes needs more than 30
  # traces to reach the same performance as CISC (the evaluated) with 5 traces.
  comparable_baseline_traces_needed: int


def calculate_metrics_at_position(
    baseline,
    evaluated,
    evaluated_position,
):
  """Calculate metrics based on the baseline and evaluated lists.

  Args:
    baseline: Baseline metrics.
    evaluated: Evaluated metrics.
    baseline_position: The index for accuracy in the lists. This index is
      one-based. That is, if baseline_position = 3, then the accuracy at index 2
      in the lists will be used.

  Returns:
    A ResultMetrics with the calculated metrics.
  """
  if len(baseline) != len(evaluated):
    raise ValueError("Baseline and evaluated lists must have the same length.")
  total_traces = len(baseline)

  baseline_accuracy = baseline[evaluated_position - 1]
  evaluated_accuracy = evaluated[evaluated_position - 1]

  comparable_baseline_traces_needed = total_traces + 1
  for i, value in enumerate(baseline):
    if value >= evaluated_accuracy:
      comparable_baseline_traces_needed = i + 1
      break

  return ResultMetrics(
      evaluated_position=evaluated_position,
      baseline_accuracy=baseline_accuracy * 100,
      evaluated_accuracy=evaluated_accuracy * 100,
      comparable_baseline_traces_needed=comparable_baseline_traces_needed,
  )


def find_first_improvment_above_threshold(
    accuracies, threshold
):
  """Finds the index of the first improvement that is "good enough".

  When doing self consistency, increasing the number of traces can improve the
  accuracy. We want to find the first improvement that is "good enough", and
  then we can use this information to decide how many traces we want to use.

  For example, if we have the following accuracies:
  [40, 60, 70, 71, 72]

  and the threshold is 0.9, the it is enoguh to use 3 traces, because the
  accuracy does not improve much after using 3 traces.

  Args:
    accuracies: The list of self consistency accuracies per traces number.
    threshold: The minimal fraction of improvements we require.

  Returns:
    The one-based position of the first improvement that is "good enough".
  """
  if accuracies is None:
    raise ValueError("Accuracies list must not be None.")
  improvments = np.array(accuracies) - accuracies[0]
  improvments = improvments / max(improvments)
  for i, val in enumerate(improvments):
    if val >= threshold:
      return i + 1
  return accuracies[-1]
