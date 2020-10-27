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

"""Library to compute official NQ scores for a file with output predictions."""

import collections
import json
import pickle

from natural_questions import eval_utils
from natural_questions import nq_eval
import numpy as np
import tensorflow.compat.v1 as tf


class NQInference(object):
  """Computes official NQ predictions."""

  def __init__(self, max_short_answer_len=32):
    self.max_short_answer_len = max_short_answer_len

  def compute_predictions(self, predicted_file):
    """Computes span predictions maximizing start and end scores separately."""
    gz = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    reader = tf.python_io.tf_record_iterator(predicted_file, options=gz)
    pred_by_document = collections.defaultdict(
        lambda: collections.defaultdict(list))
    for record_idx, record in enumerate(reader):
      if (record_idx + 1) % 50000 == 0: print(record_idx + 1)
      x = tf.train.Example.FromString(record).features.feature
      unique_id = int(x["unique_ids"].int64_list.value[0])
      output_names = ["sa_start", "sa_end", "la_start", "la_end"]
      for output_name in output_names:
        true_idx = x[output_name + "_mapped"].int64_list.value
        pred_idx = x[output_name + "_pred_mapped"].int64_list.value
        pred_score = list(
            np.array(x[output_name + "_logit"].float_list.value) /
            x[output_name + "_logit0"].float_list.value[0])
        tups = [t for t in zip(pred_score, pred_idx) if t[1] >= 0]
        pred_by_document[unique_id][output_name + "_true"].append(true_idx)
        pred_by_document[unique_id][output_name].extend(tups)
      pred_by_document[unique_id]["answer_type_logits"] = (
          x["answer_type_logits"].float_list.value)
    return pred_by_document

  def compute_official(self,
                       pred_by_document,
                       span_selection_method="joint",
                       consider_answer_type=True):
    """Converts predictions into the official format for NQ."""
    official_output = {"predictions": []}
    for unique_id, pred in pred_by_document.items():
      official_prediction = {"example_id": unique_id, "yes_no_answer": "NONE"}
      if span_selection_method == "disjoint":
        disjoint_span_prediction(pred, official_prediction)

      elif span_selection_method == "joint":
        joint_span_prediction(pred, official_prediction,
                              self.max_short_answer_len)

      elif span_selection_method == "joint-exhaustive":
        joint_exhaustive_span_prediction(pred, official_prediction,
                                         self.max_short_answer_len)

      if consider_answer_type:
        modify_prediction_with_answer_type(pred, official_prediction)

      official_output["predictions"].append(official_prediction)
    return official_output


class NQEvaluator(object):
  """Computes official NQ metrics given a prediction file."""

  def __init__(self, gold_path,
               max_short_answer_len=32):
    self.nq_gold_dict = pickle.load(tf.gfile.Open(gold_path, "rb"))
    self.nq_inference = NQInference(max_short_answer_len=max_short_answer_len)

  def _compute_metrics(self, official_output):
    temp_path = "/tmp/official_output.json"
    json.dump(official_output, open(temp_path, "w"))
    nq_pred_dict = eval_utils.read_prediction_json(temp_path)
    long_answer_stats, short_answer_stats = nq_eval.score_answers(
        self.nq_gold_dict, nq_pred_dict)
    scores = nq_eval.get_metrics_with_answer_stats(long_answer_stats,
                                                   short_answer_stats)
    return scores

  def evaluate_predicted(self,
                         predicted_file,
                         span_selection_method="joint",
                         consider_answer_type=True):
    """Returns the official NQ metrics for given predicted output."""
    pred_by_document = self.nq_inference.compute_predictions(predicted_file)
    official_output = self.nq_inference.compute_official(
        pred_by_document,
        span_selection_method=span_selection_method,
        consider_answer_type=consider_answer_type)
    return self._compute_metrics(official_output)


def disjoint_span_prediction(pred, official_prediction):
  """Predict start/end points separately (original in chrisalberti's code)."""

  sa_start_score, sa_start_pred = sorted(pred["sa_start"], reverse=True)[0]
  sa_end_score, sa_end_pred = sorted(pred["sa_end"], reverse=True)[0]
  la_start_score, la_start_pred = sorted(pred["la_start"], reverse=True)[0]
  la_end_score, la_end_pred = sorted(pred["la_end"], reverse=True)[0]
  official_prediction["short_answers_score"] = sa_start_score + sa_end_score
  official_prediction["short_answers"] = [{
      "start_token": min(sa_start_pred, sa_end_pred),
      "end_token": max(sa_start_pred, sa_end_pred) + 1,
      "start_byte": -1,
      "end_byte": -1
  }]
  official_prediction["long_answer_score"] = la_start_score + la_end_score
  official_prediction["long_answer"] = {
      "start_token": min(la_start_pred, la_end_pred),
      "end_token": max(la_start_pred, la_end_pred)+1,
      "start_byte": -1,
      "end_byte": -1
  }


def joint_span_prediction(pred, official_prediction, max_short_answer_len):
  """Condition the best end, on the best start position."""

  if pred["sa_start"]:
    sa_start_score, sa_start_pred = sorted(pred["sa_start"], reverse=True)[0]
  else:
    sa_start_score = 0
    sa_start_pred = 0
  if pred["la_start"]:
    la_start_score, la_start_pred = sorted(pred["la_start"], reverse=True)[0]
  else:
    la_start_score = 0
    la_start_pred = 0

  la_end_score = None
  la_end_pred = None
  for score, token in pred["la_end"]:
    if token >= la_start_pred:
      if la_end_score is None or score > la_end_score:
        la_end_score = score
        la_end_pred = token
  if la_end_score is None:
    if pred["la_end"]:
      la_end_score, la_end_pred_tmp = sorted(pred["la_end"], reverse=True)[0]
      la_start_pred_tmp = la_start_pred
      la_start_pred = min(la_start_pred_tmp, la_end_pred_tmp)
      la_end_pred = max(la_start_pred_tmp, la_end_pred_tmp)
    else:
      la_start_score = 0
      la_start_pred = 0
      la_end_score = 0
      la_end_pred = 0

  sa_end_score = None
  sa_end_pred = None
  for score, token in pred["sa_end"]:
    if (token >= sa_start_pred and
        token-sa_start_pred < max_short_answer_len):
      if sa_end_score is None or score > sa_end_score:
        sa_end_score = score
        sa_end_pred = token

  # if we don't have any good prediction, default to independent predictions:
  if sa_end_score is None:
    if pred["sa_end"]:
      sa_end_score, sa_end_pred = sorted(pred["sa_end"], reverse=True)[0]
    else:
      sa_start_score = 0
      sa_start_pred = 0
      sa_end_score = 0
      sa_end_pred = 0

  official_prediction["short_answers_score"] = sa_start_score + sa_end_score
  official_prediction["long_answer_score"] = la_start_score + la_end_score
  official_prediction["short_answers"] = [{
      "start_token": min(sa_start_pred, sa_end_pred),
      "end_token": max(sa_start_pred, sa_end_pred) + 1,
      "start_byte": -1,
      "end_byte": -1
  }]
  official_prediction["long_answer"] = {
      "start_token": la_start_pred,
      "end_token": la_end_pred+1,
      "start_byte": -1,
      "end_byte": -1
  }


def joint_exhaustive_span_prediction(pred, official_prediction,
                                     max_short_answer_len):
  """Exhaustively try all combinations of start/end spans."""

  la_score = -float("inf")
  la_start_pred = -1
  la_end_pred = -1
  for score_start, token_start in pred["la_start"]:
    if score_start < 0:
      continue
    for score_end, token_end in pred["la_end"]:
      if score_end < 0:
        continue
      if token_end >= token_start:
        if score_start+score_end > la_score:
          la_score = score_start+score_end
          la_start_pred = token_start
          la_end_pred = token_end
  if la_end_pred == -1:
    la_start_score, la_start_pred_tmp = sorted(pred["la_start"],
                                               reverse=True)[0]
    la_end_score, la_end_pred_tmp = sorted(pred["la_end"], reverse=True)[0]
    la_score = la_start_score + la_end_score
    la_start_pred = min(la_start_pred_tmp, la_end_pred_tmp)
    la_end_pred = max(la_start_pred_tmp, la_end_pred_tmp)

  sa_score = -float("inf")
  sa_start_pred = -1
  sa_end_pred = -1
  for score_start, token_start in pred["sa_start"]:
    if (score_start < 0 or token_start < la_start_pred or
        token_start > la_end_pred):
      continue
    for score_end, token_end in pred["sa_end"]:
      if (score_end < 0 or token_end < la_start_pred or
          token_end > la_end_pred):
        continue
      if (token_end >= token_start and
          token_end-token_start < max_short_answer_len):
        if score_start+score_end > sa_score:
          sa_score = score_start+score_end
          sa_start_pred = token_start
          sa_end_pred = token_end

  if sa_end_pred == -1:
    sa_start_score, sa_start_pred = sorted(pred["sa_start"],
                                           reverse=True)[0]
    sa_end_score, sa_end_pred = sorted(pred["sa_end"], reverse=True)[0]
    sa_score = sa_start_score + sa_end_score

  official_prediction["short_answers_score"] = sa_score
  official_prediction["long_answer_score"] = la_score
  official_prediction["short_answers"] = [{
      "start_token": min(sa_start_pred, sa_end_pred),
      "end_token": max(sa_start_pred, sa_end_pred) + 1,
      "start_byte": -1,
      "end_byte": -1
  }]
  official_prediction["long_answer"] = {
      "start_token": la_start_pred,
      "end_token": la_end_pred+1,
      "start_byte": -1,
      "end_byte": -1
  }


def argmax(l):
  max_idx = -1
  for i in range(len(l)):
    if max_idx == -1 or l[i] > l[max_idx]:
      max_idx = i
  return max_idx


def modify_prediction_with_answer_type(pred, official_prediction):
  """If yes/no have the highest logits, overwrite the answer prediction."""

  answer_type_logits = pred["answer_type_logits"]
  max_idx = argmax(answer_type_logits)
  # Indexes are: NO-ANSWER, YES, NO, LONG, SHORT
  if max_idx == 1:
    # yes:
    official_prediction["short_answers"] = []
    official_prediction["yes_no_answer"] = "yes"

  elif max_idx == 2:
    # no:
    official_prediction["short_answers"] = []
    official_prediction["yes_no_answer"] = "no"

  elif max_idx == 3:
    # long:
    official_prediction["short_answer"] = []
