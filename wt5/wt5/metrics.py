# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""WT5 metrics."""

import numpy as np
import sklearn.metrics
import t5.evaluation


def esnli_metric(targets, predictions):
  """Compute label accuracy and BLEU score for e-SNLI predictions.

  This function gets the label and explanation and computes accuracy and
  BLEU score on the explanation.

  Args:
    targets: list of dict of label and explanation
    predictions: list of dict of label and explanation
  Returns:
    a dict with accuracy and bleu score.
  """
  def get_label_and_explanation(answers):
    """Helper function to get lists of labels and explanations from a dict."""
    labels = []
    explanations_1 = []
    explanations_2 = []
    for answer in answers:
      for key, value in answer.items():
        if key == "label":
          labels.append(value)
        # In e-snli, the authors only use the first two explanations to compute
        # the BLEU score.
        elif key == "explanations":
          explanations_1.append("" if not value else value[0])
          if len(value) > 1:
            explanations_2.append(value[1])
        else:
          raise RuntimeError(
              "Unexpected key:%s provided. to metric fn." % (key))

    if explanations_2:
      return labels, [explanations_1, explanations_2]
    else:
      return labels, explanations_1

  def get_first_explanation_length(explanations):
    return len(explanations) if isinstance(explanations, str) else len(
        explanations[0])

  target_labels, target_explanations = get_label_and_explanation(targets)
  # The model can only predict one explanation
  for prediction in predictions:
    if prediction["explanations"]:
      prediction["explanations"] = [prediction["explanations"][0]]
  prediction_labels, prediction_explanations = get_label_and_explanation(
      predictions)

  return {
      "accuracy":
          t5.evaluation.metrics.accuracy(target_labels, prediction_labels)
          ["accuracy"],
      "bleu":
          t5.evaluation.metrics.bleu(target_explanations,
                                     prediction_explanations)["bleu"],
      "expln1_length":
          get_first_explanation_length(prediction_explanations)
  }


def extractive_explanations_metric(targets, predictions):
  """Compute label accuracy and macro F1 score for explanations."""

  def get_labels_spans_and_expls(answers):
    """Gets a list of labels and spans from a list of dicts."""
    labels = []
    spans = []
    span_arrays = []
    explanations = []

    for answer in answers:
      for key, value in answer.items():
        if key == "label":
          labels.append(value)
        elif key == "overlap_spans":
          spans.append(value)
        elif key == "span_array":
          span_arrays.append(value)
        elif key == "explanations":
          explanations.append(value)
        else:
          raise ValueError("Unexpected key found in answers dict: %s" % key)

    return labels, spans, span_arrays, explanations

  labels_t, spans_t, arrays_t, _ = get_labels_spans_and_expls(targets)
  labels_p, spans_p, arrays_p, explns_p = get_labels_spans_and_expls(
      predictions)

  # Compute f1 score for each example in the target prediction pair
  f1_scores = []
  for gt_span, pred_span in zip(spans_t, spans_p):
    elem_prec = len(set(gt_span)
                    & set(pred_span)) / len(pred_span) if pred_span else 0
    elem_rec = len(set(gt_span)
                   & set(pred_span)) / len(gt_span) if gt_span else 0

    if elem_prec == 0 or elem_rec == 0:
      elem_f1 = 0
    else:
      elem_f1 = 2 * elem_prec * elem_rec / (elem_prec + elem_rec)
    f1_scores.append(elem_f1)

  exact_match_f1 = np.mean(f1_scores) * 100

  partial_match_f1 = 100 * np.mean(
      [sklearn.metrics.f1_score(t, p) for t, p in zip(arrays_t, arrays_p)]
  )

  def get_avg_num_explanations(explanations):
    total_explns = 0
    for e in explanations:
      total_explns += len(e)
    return float(total_explns)/len(explanations) if explanations else 0.0

  return {
      "accuracy": 100 * sklearn.metrics.accuracy_score(labels_t, labels_p),
      "f1": exact_match_f1,
      "partial match f1": partial_match_f1,
      "avg_explanation_count": get_avg_num_explanations(explns_p),
  }
