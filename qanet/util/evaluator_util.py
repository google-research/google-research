# coding=utf-8
# Copyright 2018 The Google Research Authors.
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

"""SQuAD evaluation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import re
import string

import numpy as np
import tensorflow as tf

from qanet.util import misc_util

__all__ = ["evaluate", "load_ground_truths", "load_predictions",
           "evaluate_scores", "compute_f1", "compute_exact",
           "get_start_end"]


def evaluate(groundtruth_file, prediction_file, data_format="squad"):
  """Evaluate SQUAD predictions given files."""
  ground_truths, _ = load_ground_truths(groundtruth_file)
  predictions, _, _ = load_predictions(prediction_file)
  scores, f1_scores, exact_scores = evaluate_scores(ground_truths, predictions)

  # SQuAD 2.0: get some more statistics over answers and no-answers
  if data_format == "squad2":
    has_ans_qids = [k for k, v in ground_truths.items() if v]
    no_ans_qids = [k for k, v in ground_truths.items() if not v]

    if has_ans_qids:
      has_ans_scores = _make_eval_dict(f1_scores, exact_scores,
                                       qid_list=has_ans_qids)
      _merge_eval(scores, has_ans_scores, "HasAns")
    if no_ans_qids:
      no_ans_scores = _make_eval_dict(f1_scores, exact_scores,
                                      qid_list=no_ans_qids)
      _merge_eval(scores, no_ans_scores, "NoAns")

  return scores


def load_ground_truths(groundtruth_file):
  """Load ground truth data."""
  print("# Loading ground truths from %s" % groundtruth_file)
  with tf.gfile.Open(groundtruth_file) as f:
    dataset_json = json.load(f)
    dataset = dataset_json["data"]

    # get ground truths
    ground_truths = {}
    questions = {}
    num_examples = 0
    num_paragraphs = 0
    for article in dataset:
      for paragraph in article["paragraphs"]:
        num_paragraphs += 1

        # Answers
        for qa in paragraph["qas"]:
          if qa["id"] in ground_truths:
            message = "Duplicated id " + qa["id"] + "."
            tf.logging.info(message)
            continue

          ground_truths[qa["id"]] = list(
              map(lambda x: x["text"], qa["answers"]))

          questions[qa["id"]] = qa["question"]
          num_examples += 1

    tf.logging.info("  Num ground truths: %d" % num_examples)
    tf.logging.info("  Num paragraphs: %d" % num_paragraphs)
    return ground_truths, questions


def load_predictions(prediction_file, load_prob=False):
  """Load predictions from a prediction file."""
  print("# Loading predictions from %s" % prediction_file)
  num_examples = 0
  predictions = {}
  if load_prob:
    start_prob, end_prob = {}, {}
  else:
    start_prob, end_prob = None, None

  with tf.gfile.GFile(prediction_file) as f:
    data = json.load(f)

    for q_id in data:
      if not isinstance(data[q_id], dict):
        predictions[q_id] = data[q_id]
      else:
        predictions[q_id] = data[q_id]["answer"]
        if load_prob:
          start_prob[q_id] = np.array(data[q_id]["start_prob"])
          end_prob[q_id] = np.array(data[q_id]["end_prob"])

      num_examples += 1
    tf.logging.info("  Num predictions: %d" % num_examples)
    return predictions, start_prob, end_prob


def _make_eval_dict(f1_scores, exact_scores, qid_list=None):
  """Compute aggregated F1 and exact match scores."""

  # Filter scores if qid_list is specified
  if qid_list:
    f1_scores_select = {}
    exact_scores_select = {}
    for qid in qid_list:
      if qid in f1_scores and exact_scores:
        f1_scores_select[qid] = f1_scores[qid]
        exact_scores_select[qid] = exact_scores[qid]
      else:
        tf.logging.info("missing qid %s" % qid)
  else:
    f1_scores_select = f1_scores
    exact_scores_select = exact_scores

  # Compute scores
  total = len(exact_scores_select)
  return collections.OrderedDict([
      ("exact_match", 100.0 * sum(exact_scores_select.values()) / total),
      ("f1", 100.0 * sum(f1_scores_select.values()) / total),
      ("total", total),
  ])


def _merge_eval(main_eval, new_eval, prefix):
  """Merge evaluation dicts."""
  for k in new_eval:
    main_eval["%s_%s" % (prefix, k)] = new_eval[k]


def _normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""

  def remove_articles(text):
    return re.sub(r"\b(a|an|the)\b", " ", text)

  def white_space_fix(text):
    return " ".join(text.split())

  def remove_punc(text):
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)

  def lower(text):
    return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))


def _handle_no_answer(prediction, ground_truths):
  """Check if there is no groundtruth answer and compute no-answer score."""
  score = None

  # Check for no-answer: ground_truths can look like ['', '', '']
  #   the reason there are multiple empty values because we operate at batch
  #   and append '' to the maximum number of answers.
  has_answer = False
  if ground_truths:
    for answer in ground_truths:
      if answer:
        has_answer = True
        break

  if not has_answer:  # No groundtruth answer
    if _normalize_answer(prediction):  #  predict answer
      score = 0.0
    else:
      score = 1.0

  return score


def _f1_score(prediction, ground_truth):
  """Compute F1 score."""
  prediction_tokens = _normalize_answer(prediction).split()
  ground_truth_tokens = _normalize_answer(ground_truth).split()
  common = (collections.Counter(prediction_tokens) &
            collections.Counter(ground_truth_tokens))
  num_same = sum(common.values())

  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(prediction_tokens)
  recall = 1.0 * num_same / len(ground_truth_tokens)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1


def _em_score(prediction, ground_truth):
  """Compute EM score (binary value)."""
  return int(_normalize_answer(prediction) == _normalize_answer(ground_truth))


def _compute_score(score_fn, prediction, ground_truths, answer_mask, is_byte):
  """Compute scores (EM, F1, etc.) given a score function."""
  # Whether we are dealing with a sequence of bytes or not
  if is_byte:
    prediction = prediction.decode("utf-8")
    ground_truths = [gt.decode("utf-8") for gt in ground_truths]

  # SQuAD 2.0
  score = _handle_no_answer(prediction, ground_truths)

  # Has-answer case
  # NOTE(thangluong): score can be 0.0 so we need to explicitly compare to None
  if score is None:
    # Execute over multiple answers
    scores = [score_fn(prediction, gt) for gt in ground_truths]
    if answer_mask is not None:  # answer_mask can be a tensor
      scores = scores * answer_mask
    tf.logging.info("prediction %s, ground_truths %s, scores %s" % (
        prediction, str(ground_truths), str(scores)))
    score = max(scores)
  else:
    tf.logging.info("prediction %s, ground_truths %s, score %s" % (
        prediction, str(ground_truths), str(score)))

  return score


def compute_f1(prediction, ground_truths, answer_mask=None, is_byte=False):
  """Compute F1 score over multiple ground truths."""
  return _compute_score(
      _f1_score, prediction, ground_truths, answer_mask, is_byte)


def compute_exact(prediction, ground_truths, answer_mask=None, is_byte=False):
  """Compute exact match (EM) score over multiple ground_truths."""
  return _compute_score(
      _em_score, prediction, ground_truths, answer_mask, is_byte)


def evaluate_scores(ground_truths, predictions, label="# Scores"):
  """Main evaluation."""
  f1_scores = {}
  exact_scores = {}
  for q_id in ground_truths:
    if q_id not in predictions:
      print("Unanswered question %s will receive score 0." % q_id)
      continue

    pred_answer = predictions[q_id]
    gold_answers = ground_truths[q_id]

    # Take max over all gold answers
    exact_scores[q_id] = compute_exact(pred_answer, gold_answers)
    f1_scores[q_id] = compute_f1(pred_answer, gold_answers)

  scores = _make_eval_dict(f1_scores, exact_scores)
  tf.logging.info("%s: %s" % (label, str(scores)))
  misc_util.print_out("%s: %s" % (label, str(scores)))

  return scores, f1_scores, exact_scores


def _compute_prob_matrix(start_prob, end_prob, max_ans_size=25):
  """Compute span prob matrix given start and end probabilities."""
  assert len(start_prob) == len(end_prob)
  context_len = len(start_prob)
  mask = np.triu(
      np.ones([context_len, context_len]) -
      np.triu(np.ones([context_len, context_len]), max_ans_size))
  prob_matrix = np.outer(start_prob, end_prob)
  prob_matrix *= mask
  return prob_matrix


def _compute_start_end(prob_matrix):
  """Given a span prob matrix, return the best span and its probability."""
  assert prob_matrix.shape[0] == prob_matrix.shape[1]
  context_len = prob_matrix.shape[0]
  argmax_id = np.argmax(prob_matrix)
  start = argmax_id // context_len
  end = argmax_id % context_len
  return start, end, prob_matrix[start, end]


def get_start_end(start_prob, end_prob):
  prob_matrix = _compute_prob_matrix(start_prob, end_prob)
  start, end, prob = _compute_start_end(prob_matrix)
  return start, end, prob
