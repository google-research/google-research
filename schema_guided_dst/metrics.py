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

"""Evaluation metrics for Schema-guided dialogue.

This library provides functions for calculating the evaluation metrics for a
single dialogue. The following metrics are defined:

(1) Active intent accuracy: The fraction of user turns for which the active
  intent has been correctly predicted.
(2) Slot tagging F1: The macro-averaged F1 score for tagging slot values for
  non-categorical slots. This metric is optional to report in the final paper
  if participants decide not to use slot tagging.
(3) Requested slots F1: The macro-averaged F1 score for requested slots over the
  turns. For a turn, if there are no requested slots in both the ground truth
  and the prediction, that turn is skipped. The reported number is the average
  F1 score for all un-skipped user turns. This metric is optional to report in
  the final paper.
(4) Average goal accuracy: For each turn, participants must predict a single
  value for each slot present in the dialogue state. The slots which have a
  non-empty assignment in the ground truth dialogue state are only considered.
  This is the average accuracy of predicting the value of a slot correctly. A
  fuzzy matching based score is used for non-categorical slots.
(5) Joint goal accuracy: This is the average accuracy of predicting all slot
  assignments for a turn correctly. A fuzzy matching based score is used for
  non-categorical slots. This is the primary evaluation metric used for ranking
  submissions. More details to follow with the evaluation script.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections
from fuzzywuzzy import fuzz
import numpy as np

F1Scores = collections.namedtuple("F1Scores", ["f1", "precision", "recall"])

# Evaluation and other relevant metrics for DSTC8 Schema-guided DST.
# (1) Active intent accuracy.
ACTIVE_INTENT_ACCURACY = "active_intent_accuracy"
# (2) Slot tagging F1.
SLOT_TAGGING_F1 = "slot_tagging_f1"
SLOT_TAGGING_PRECISION = "slot_tagging_precision"
SLOT_TAGGING_RECALL = "slot_tagging_recall"
# (3) Requested slots F1.
REQUESTED_SLOTS_F1 = "requested_slots_f1"
REQUESTED_SLOTS_PRECISION = "requested_slots_precision"
REQUESTED_SLOTS_RECALL = "requested_slots_recall"
# (4) Average goal accuracy.
AVERAGE_GOAL_ACCURACY = "average_goal_accuracy"
AVERAGE_CAT_ACCURACY = "average_cat_accuracy"
AVERAGE_NONCAT_ACCURACY = "average_noncat_accuracy"
# (5) Joint goal accuracy.
JOINT_GOAL_ACCURACY = "joint_goal_accuracy"
JOINT_CAT_ACCURACY = "joint_cat_accuracy"
JOINT_NONCAT_ACCURACY = "joint_noncat_accuracy"

NAN_VAL = "NA"


def compute_f1(list_ref, list_hyp):
  """Compute F1 score from reference (grouth truth) list and hypothesis list.

  Args:
    list_ref: List of true elements.
    list_hyp: List of postive (retrieved) elements.

  Returns:
    A F1Scores object containing F1, precision, and recall scores.
  """

  ref = collections.Counter(list_ref)
  hyp = collections.Counter(list_hyp)
  true = sum(ref.values())
  positive = sum(hyp.values())
  true_positive = sum((ref & hyp).values())
  precision = float(true_positive) / positive if positive else 1.0
  recall = float(true_positive) / true if true else 1.0
  if precision + recall > 0.0:
    f1 = 2.0 * precision * recall / (precision + recall)
  else:  # The F1-score is defined to be 0 if both precision and recall are 0.
    f1 = 0.0

  return F1Scores(f1=f1, precision=precision, recall=recall)


def fuzzy_string_match(str_ref, str_hyp):
  """Returns fuzzy string similarity score in range [0.0, 1.0]."""

  # The higher the score, the higher the similarity between the two strings.
  return fuzz.token_sort_ratio(str_ref, str_hyp) / 100.0


def noncat_slot_value_match(str_ref_list, str_hyp, use_fuzzy_match):
  """Calculate non-categorical slots correctness.

  Args:
    str_ref_list: a list of reference strings.
    str_hyp: the hypothesis string.
    use_fuzzy_match: whether to use fuzzy string matching.

  Returns:
    score: The highest fuzzy string match score of the references and hypotheis.
  """
  score = 0.0
  for str_ref in str_ref_list:
    if not use_fuzzy_match:
      match_score = float(str_ref == str_hyp)
    else:
      match_score = fuzzy_string_match(str_ref, str_hyp)
    score = max(score, match_score)
  return score


def compare_slot_values(slot_values_ref, slot_values_hyp, service,
                        use_fuzzy_match):
  """Compare and get correctness of goal state's slot_values.

  Args:
    slot_values_ref: goal state slot_values from reference (ground truth).
    slot_values_hyp: goal state slot_values from hypothesis (prediction).
    service: a service data structure in the schema. We use it to obtain the
      list of slots in the service and infer whether a slot is categorical.
    use_fuzzy_match: whether to use fuzzy string matching for non-categorical
      slot values.

  Returns:
    (list_cor, slot_active, slot_cat)
    list_cor: list of corectness scores, each corresponding to one slot in the
        service. The score is a float either 0.0 or 1.0 for categorical slot,
        and in range [0.0, 1.0] for non-categorical slot.
    slot_active: list indicating whether the element in list_cor corresponds to
        an active ground-truth slot.
    slot_cat: list indicating whether the element in list_cor corresponds to a
        categorical slot.
  """
  list_cor = []
  slot_active = []
  slot_cat = []

  for slot in service["slots"]:
    slot_name = slot["name"]
    slot_cat.append(slot["is_categorical"])

    if slot_name in slot_values_ref:  # REF=active
      slot_active.append(True)
      if slot_name in slot_values_hyp:  # HYP=active, apply matching
        value_ref_list = slot_values_ref[slot_name]
        value_hyp = slot_values_hyp[slot_name][0]
        if slot["is_categorical"]:
          cor = float(value_ref_list[0] == value_hyp)
        else:
          cor = noncat_slot_value_match(value_ref_list, value_hyp,
                                        use_fuzzy_match)

        list_cor.append(cor)
      else:  # HYP=off
        list_cor.append(0.0)
    else:  # REF=off
      slot_active.append(False)
      if slot_name in slot_values_hyp:  # HYP=active
        list_cor.append(0.0)
      else:  # HYP=off
        list_cor.append(1.0)

  assert len(list_cor) == len(service["slots"])
  assert len(slot_active) == len(service["slots"])
  assert len(slot_cat) == len(service["slots"])
  return list_cor, slot_active, slot_cat


def get_active_intent_accuracy(frame_ref, frame_hyp):
  """Get active intent accuracy of a frame.

  Args:
    frame_ref: single semantic frame from reference (ground truth) file.
    frame_hyp: single semantic frame from hypothesis (prediction) file.

  Returns:
    1.0 if the intent prediction is correct, otherwise 0.0.
  """
  return float(frame_ref["state"]["active_intent"] == frame_hyp["state"]
               ["active_intent"])


def get_slot_tagging_f1(frame_ref, frame_hyp, utt, service):
  """Get slot tagging (non-categorical slots only) F1 scores of a frame.

  Args:
    frame_ref: single semantic frame from reference (ground truth) file.
    frame_hyp: single semantic frame from hypothesis (prediction) file.
    utt: user utterance. Slot tagging annotations are the character positions in
      the utterance.
    service: a service data structure in the schema. We use it to infer whether
      a slot is non-categorical.

  Returns:
    A F1Scores object containing F1, precision, and recall scores.
  """

  list_noncat_slots = [
      s["name"] for s in service["slots"] if not s["is_categorical"]
  ]
  if "slots" not in frame_hyp:
    return None
  else:
    list_ref = [(s["slot"], utt[s["start"]:s["exclusive_end"]])
                for s in frame_ref["slots"]
                if s["slot"] in list_noncat_slots]
    list_hyp = [(s["slot"], utt[s["start"]:s["exclusive_end"]])
                for s in frame_hyp["slots"]
                if s["slot"] in list_noncat_slots]
    return compute_f1(list_ref, list_hyp)


def get_requested_slots_f1(frame_ref, frame_hyp):
  """Get requested slots F1 scores of a frame.

  Args:
    frame_ref: single semantic frame from reference (ground truth) file.
    frame_hyp: single semantic frame from hypothesis (prediction) file.

  Returns:
    A F1Scores object containing F1, precision, and recall scores.
  """
  return compute_f1(frame_ref["state"]["requested_slots"],
                    frame_hyp["state"]["requested_slots"])


def get_average_and_joint_goal_accuracy(frame_ref, frame_hyp, service,
                                        use_fuzzy_match):
  """Get average and joint goal accuracies of a frame.

  Args:
    frame_ref: single semantic frame from reference (ground truth) file.
    frame_hyp: single semantic frame from hypothesis (prediction) file.
    service: a service data structure in the schema. We use it to obtain the
      list of slots in the service and infer whether a slot is categorical.
    use_fuzzy_match: whether to use fuzzy string matching for comparing
      non-categorical slot values.

  Returns:
    goal_acc: a dict whose values are average / joint
        all-goal / categorical-goal / non-categorical-goal accuracies.
  """
  goal_acc = {}

  list_acc, slot_active, slot_cat = compare_slot_values(
      frame_ref["state"]["slot_values"], frame_hyp["state"]["slot_values"],
      service, use_fuzzy_match)

  # (4) Average goal accuracy.
  active_acc = [acc for acc, active in zip(list_acc, slot_active) if active]
  goal_acc[AVERAGE_GOAL_ACCURACY] = np.mean(
      active_acc) if active_acc else NAN_VAL
  # (4-a) categorical.
  active_cat_acc = [
      acc for acc, active, cat in zip(list_acc, slot_active, slot_cat)
      if active and cat
  ]
  goal_acc[AVERAGE_CAT_ACCURACY] = (
      np.mean(active_cat_acc) if active_cat_acc else NAN_VAL)
  # (4-b) non-categorical.
  active_noncat_acc = [
      acc for acc, active, cat in zip(list_acc, slot_active, slot_cat)
      if active and not cat
  ]
  goal_acc[AVERAGE_NONCAT_ACCURACY] = (
      np.mean(active_noncat_acc) if active_noncat_acc else NAN_VAL)

  # (5) Joint goal accuracy.
  goal_acc[JOINT_GOAL_ACCURACY] = np.prod(list_acc) if list_acc else NAN_VAL
  # (5-a) categorical.
  cat_acc = [acc for acc, cat in zip(list_acc, slot_cat) if cat]
  goal_acc[JOINT_CAT_ACCURACY] = np.prod(cat_acc) if cat_acc else NAN_VAL
  # (5-b) non-categorical.
  noncat_acc = [acc for acc, cat in zip(list_acc, slot_cat) if not cat]
  goal_acc[JOINT_NONCAT_ACCURACY] = np.prod(
      noncat_acc) if noncat_acc else NAN_VAL

  return goal_acc
