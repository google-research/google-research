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

"""Evaluation script for video timeline modeling.

This script takes a single json file containing groundtruth and predictions, and
calculates the pre-defined evaluation metrics including:
  - Video-to-cluster prediction accuracy
  - Levenshtein distance
  - Edit distance
  - Relative order prediction accuracy
  - Clustering quality metrics

Sample input format:
  [
    {
      "timeline_url": "foo",
      "label": [0, 0, 1, 2, 3, 3],
      "pred": [0, 1, 1, 2, 3, 4]
    },
    ...
  ]

Sample usage:
  python3 -m vtm.eval --input_path=<input_path> --output_path=<output_path>
"""

from collections.abc import Sequence
import json
from typing import Union

from absl import app
from absl import flags
import Levenshtein
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import v_measure_score
import tensorflow as tf
import tqdm

_INPUT_PATH = flags.DEFINE_string(
    'input_path',
    None,
    'File pattern to the prediction and groundtruth json file.',
    required=True)
_OUTPUT_PATH = flags.DEFINE_string('output_path', None,
                                   'Path for saving the evaluation result.')

_TIMELINE_TOTAL = 'timeline_total'
_CLASSIFICATION_ACCURACY = 'classification_accuracy'
_CLASSIFICATION_CORRECT = 'classification_correct'
_CLASSIFICATION_TOTAL = 'classification_total'

_ORDER_PAIRS_ACCURACY = 'order_pairs_accuracy'
_ORDER_PAIRS_CORRECT = 'order_pairs_correct'
_ORDER_PAIRS_TOTAL = 'order_pairs_total'

_HOMOGENEITY_SCORE = 'homogeneity_score'
_COMPLETENESS_SCORE = 'completeness_score'
_ADJUSTED_RAND_SCORE = 'adjusted_rand_score'
_V_MEASURE_SCORE = 'v_measure_score'

_NORMALIZED_LEVENSHTEIN_DISTANCE = 'normalized_levenshtein_distance'
_NORMALIZED_CUSTOM_EDIT_DISTANCE = 'normalized_custom_edit_distance'
_NORMALIZED_CUSTOM_EDIT_DISTANCE_SUM = 'normalized_custom_edit_distance_sum'


def _generate_order_pairs(assignments):
  """Generates a list of relative orders given cluster assignment.

  Assume given assignment is [0, 2, 1, 1], then this function will generate a
  list of pairwise order comparison, with length of (N * (N-1) // 2).
    - 0 < 2: -1
    - 0 < 1: -1
    - 0 < 1: -1
    - 2 > 1:  1
    - 2 > 1:  1
    - 1 = 1:  0
  The generated sequence will be [-1, -1, -1, 1, 1, 0].
  If there is only one element in the assignment then the order will be empty.

  This aims at evaluating the algorithm with respect to the temporal orders.
  Specifically, since there might be multiple "reasonable" timelines but only
  a few groundtruth annotations, it is possible that number of nodes in the
  timeline may differ from groundtruth and predictions. However, any reasonable
  timelines should be consistent in the temporal order. We take this intuition
  as an additional evaluation metric.

  At this moment we define three relations regarding two arbitrary videos A & B:
    (1) video A belongs to a cluster that is before video B (-1);
    (2) video A and B belong to an identical cluster in timeline (0);
    (3) video A belongs to a cluster that is after video B (1).
  Currently we just apply equal penalty to all misclassifications, however,
  we may consider using different weights to penalize misclassifications
  differently. For example, misclassifying -1 to 0 might be less severe than
  misclassifying -1 to 1, as the former may follow a different granularity on
  clusters but the latter totally reversed temporal order.

  Args:
    assignments: list of integers representing video assignments.

  Returns:
    List of integers of relative order comparison.
  """
  pair_list = []
  for i, x in enumerate(assignments):
    for y in assignments[i + 1:]:
      if x == y:
        pair_list.append(0)  # Both videos belong to the same cluster.
      elif x < y:
        pair_list.append(-1)  # First video happens before second video.
      elif x > y:
        pair_list.append(1)  # First video happens after second video.
  return pair_list


def evaluate_timeline(groundtruths,
                      predictions):
  """Evaluates a single timeline."""
  output = {}

  # Video-to-cluster classification metrics.
  classification_correct = sum(
      p == g for p, g in zip(predictions, groundtruths))
  classification_total = len(predictions)
  output.update({
      _CLASSIFICATION_TOTAL: classification_total,
      _CLASSIFICATION_CORRECT: classification_correct,
      _CLASSIFICATION_ACCURACY: classification_correct / classification_total,
  })

  # Clustering quality metrics.
  output.update({
      _HOMOGENEITY_SCORE:
          homogeneity_score(labels_true=groundtruths, labels_pred=predictions),
      _COMPLETENESS_SCORE:
          completeness_score(labels_true=groundtruths, labels_pred=predictions),
      _ADJUSTED_RAND_SCORE:
          adjusted_rand_score(
              labels_true=groundtruths, labels_pred=predictions),
      _V_MEASURE_SCORE:
          v_measure_score(labels_true=groundtruths, labels_pred=predictions)
  })

  # Edit distance metrics.
  levenshtein = Levenshtein.distance(''.join(
      chr(int(x)) for x in groundtruths), ''.join(
          chr(int(x)) for x in predictions)) / len(groundtruths)
  custom_edit_distance_sum = sum(
      abs(x - y) / len(groundtruths) for x, y in zip(groundtruths, predictions))
  output.update({
      _NORMALIZED_LEVENSHTEIN_DISTANCE:
          levenshtein,
      _NORMALIZED_CUSTOM_EDIT_DISTANCE_SUM:
          custom_edit_distance_sum,
      _NORMALIZED_CUSTOM_EDIT_DISTANCE:
          custom_edit_distance_sum / len(groundtruths),
  })

  # Relative order classification metrics.
  prediction_order_pairs = _generate_order_pairs(predictions)
  groundtruth_order_pairs = _generate_order_pairs(groundtruths)

  order_pairs_correct = sum(
      p == g for (p, g) in zip(prediction_order_pairs, groundtruth_order_pairs))
  order_pairs_total = len(groundtruth_order_pairs)
  output.update({
      _ORDER_PAIRS_CORRECT: order_pairs_correct,
      _ORDER_PAIRS_TOTAL: order_pairs_total,
      _ORDER_PAIRS_ACCURACY: order_pairs_correct / order_pairs_total,
  })

  return output


def calculate_dataset_metric(
    dataset_summary
):
  """Aggregates individual timeline eval results to obtain dataset metric."""
  output = {'inputs': _INPUT_PATH.value}

  def _aggregate_stats(target, stats):
    return sum((x[target][stats] for x in dataset_summary))

  def _summarize_dataset(target):
    counters = {
        _TIMELINE_TOTAL:
            len(dataset_summary),
        _CLASSIFICATION_CORRECT:
            _aggregate_stats(target, _CLASSIFICATION_CORRECT),
        _CLASSIFICATION_TOTAL:
            _aggregate_stats(target, _CLASSIFICATION_TOTAL),
        _ORDER_PAIRS_CORRECT:
            _aggregate_stats(target, _ORDER_PAIRS_CORRECT),
        _ORDER_PAIRS_TOTAL:
            _aggregate_stats(target, _ORDER_PAIRS_TOTAL),
    }

    timeline_total = counters[_TIMELINE_TOTAL]
    timeline_level_average = {
        _CLASSIFICATION_ACCURACY:
            _aggregate_stats(target, _CLASSIFICATION_ACCURACY) / timeline_total,
        _HOMOGENEITY_SCORE:
            _aggregate_stats(target, _HOMOGENEITY_SCORE) / timeline_total,
        _COMPLETENESS_SCORE:
            _aggregate_stats(target, _COMPLETENESS_SCORE) / timeline_total,
        _ADJUSTED_RAND_SCORE:
            _aggregate_stats(target, _ADJUSTED_RAND_SCORE) / timeline_total,
        _V_MEASURE_SCORE:
            _aggregate_stats(target, _V_MEASURE_SCORE) / timeline_total,
        _NORMALIZED_LEVENSHTEIN_DISTANCE:
            _aggregate_stats(target, _NORMALIZED_LEVENSHTEIN_DISTANCE) /
            timeline_total,
        _NORMALIZED_CUSTOM_EDIT_DISTANCE:
            _aggregate_stats(target, _NORMALIZED_CUSTOM_EDIT_DISTANCE) /
            timeline_total,
        _ORDER_PAIRS_ACCURACY:
            _aggregate_stats(target, _ORDER_PAIRS_ACCURACY) / timeline_total,
    }
    video_level_average = {
        _CLASSIFICATION_ACCURACY:
            counters[_CLASSIFICATION_CORRECT] / counters[_CLASSIFICATION_TOTAL],
        _ORDER_PAIRS_ACCURACY:
            counters[_ORDER_PAIRS_CORRECT] / counters[_ORDER_PAIRS_TOTAL],
        _NORMALIZED_CUSTOM_EDIT_DISTANCE:
            _aggregate_stats(target, 'normalized_custom_edit_distance_sum') /
            counters[_CLASSIFICATION_TOTAL],
    }

    return {
        'counters': counters,
        'timeline_level_average': timeline_level_average,
        'video_level_average': video_level_average,
    }

  # Evaluation metrics with raw predictions.
  output.update({
      'raw': _summarize_dataset('raw'),
      'shrunk': _summarize_dataset('shrunk'),
  })
  return output


def main(argv):
  if len(argv) > 1:
    raise app.UsageError(f'Too many command-line arguments: {argv[1:]}')

  with tf.io.gfile.GFile(_INPUT_PATH.value, 'r') as fp:
    inputs = json.load(fp)

  dataset_summary = []

  for item in tqdm.tqdm(inputs):
    predictions = item['pred']
    groundtruths = item['label']

    shrink_mapping = {x: i for i, x in enumerate(sorted(set(predictions)))}
    shrunk_predictions = [shrink_mapping[x] for x in predictions]

    timeline_summary = {
        'raw': evaluate_timeline(groundtruths, predictions),
        'shrunk': evaluate_timeline(groundtruths, shrunk_predictions),
    }

    dataset_summary.append(timeline_summary)

  dataset_metric = calculate_dataset_metric(dataset_summary)

  with tf.io.gfile.GFile(_OUTPUT_PATH.value, 'w') as fp:
    json.dump(dataset_metric, fp, indent=2)


if __name__ == '__main__':
  app.run(main)
