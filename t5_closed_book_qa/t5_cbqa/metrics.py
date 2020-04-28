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

"""T5 CBQA metrics."""
import t5.evaluation


def natural_questions(
    targets, predictions, non_null_threshold=2, normalize_answers=True):
  """Computes the recall based on the Natural Questions evaluation script.

  We first remove any targets that do not contain "gold answers", which are
  defined as having at least `non_null_threshold` annotations with non-empty
  answers (either short or yes/no). For examples with gold answers, we count the
  prediction as correct if the (unordered) set of answers matches those in any
  of the annotations.

  Args:
    targets: list of lists of answer group tuples
    predictions: list of lists each containing single answer group tuple
    non_null_threshold: int, the minimum number of non-null annotations.
    normalize_answers: bool, whether to normalize answer strings before
      comparing.
  Returns:
    a dict containing the recall
  """
  has_gold_answer = 0
  is_correct = 0
  if len(targets) != len(predictions):
    raise ValueError("Number of targets and predictions must match.")
  for targ_groups, pred_groups in zip(targets, predictions):
    if len(pred_groups) != 1:
      raise ValueError(
          "Predictions must have exactly 1 group each. Got %d." %
          len(pred_groups))
    if normalize_answers:
      def _normalize(groups):
        return [
            tuple(t5.evaluation.qa_utils.normalize_squad(a) for a in g)
            for g in groups
        ]
      targ_groups = _normalize(targ_groups)
      pred_groups = _normalize(pred_groups)
    # Convert to sets and remove null annotations.
    pred_set = set(pred_groups[0])
    targ_sets = [set(g) for g in targ_groups if g]
    if len(targ_sets) < non_null_threshold:
      continue
    has_gold_answer += 1
    for targ_set in targ_sets:
      if targ_set == pred_set:
        is_correct += 1
        break

  if not has_gold_answer:
    raise ValueError("No gold answers found.")

  return {
      "recall": is_correct / has_gold_answer * 100,
      "golden_answers": has_gold_answer
  }
