# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""T5 CBQA postprocessors."""

import numpy as np
import tensorflow.compat.v1 as tf


def natural_questions(output,
                      prefix="answer:",
                      example=None,
                      is_target=False):
  """Get answers from predictions and targets.

  The predictions will contain a single set of one or more answers. The example
  may contain multiple sets of answers from different annotators. We return
  an answer group for each annotation, even if its empty.

  Args:
    output: str, target or prediction in text format.
    prefix: str, prefix expected before each answer.
    example: dict, input example.
    is_target: bool, whether the input was ground truth (True) or
      prediction (False).
  Returns:
    a list of answer tuples.
  """
  if is_target:
    answer_groups = []
    short_answers = np.split(
        example["short_answers/values"],
        example["short_answers/row_starts"][1:])
    yes_no_answers = example["yes_no_answers"]
    if len(short_answers) != len(yes_no_answers):
      raise ValueError(
          "Number of annotations not consistent: %d vs %d" %
          (len(short_answers), len(yes_no_answers)))
    for short_ans_grp, y_n_ans in zip(short_answers, yes_no_answers):
      # Annotators cannot provide both y/n and short answers.
      if y_n_ans > -1 and short_ans_grp:
        raise ValueError(
            "Annotation cannot include both yes/no and short answers.")
      if y_n_ans == 0:
        answer_groups.append(("no",))
      elif y_n_ans == 1:
        answer_groups.append(("yes",))
      else:
        answer_groups.append(
            tuple(tf.compat.as_text(ans) for ans in short_ans_grp)
        )
  else:
    answer_groups = [
        tuple(s.strip() for s in output.split(prefix)[1:])
    ]

  return answer_groups
