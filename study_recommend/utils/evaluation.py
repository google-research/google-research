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

# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions for computing hits@n statistics for recommendations."""

import collections
from collections.abc import Sequence
from typing import Any, Optional

import pandas as pd
from pandas.core import groupby as pd_groupby
from study_recommend import types

EVAL_TYPES = types.EvalTypes


def hits_at_n(
    reference_titles,
    recommendations,
    n,
):
  """The fraction of recommendations that had a hit in its top n choices.

  Assumes n or more recommendations provided at each timestep
  Args:
    reference_titles: A mapping from a student identifier number to a sequence
      of title identifiers that the student interacted with.
    recommendations: A mapping from student identifiers to a sequence of
      recommendations. Each recommendation is a sequence of title identifiers.
      ordered by confidence of recommendation. If there N are recommendations
      where N < number of titles read then we assume we made recommendations for
      the last N interactions.
    n: The number of choices per recommendation to consider.

  Returns:
    score: The fraction of recommendations with a hits@n in [0, 1].  Returns
      None if no recommendations were made.
  """
  counts, hits = compute_per_student_hits(reference_titles, recommendations, n)
  return aggregate(counts["all"], hits["all"])


def compute_per_student_hits(
    reference_titles,
    recommendations,
    n,
    compute_non_continuation = False,
    compute_non_history = False,
):
  """Returns a pair of dictionaries with hits@n per n statics per student.

  Assumes n or more recommendations provided at each timestep
  Args:
    reference_titles: A mapping from student identifiers to a sequence of title
      identifiers that the student interacted with.
    recommendations: A mapping from student identifiers to a sequence of
      recommendations. Each recommendation is a sequence of title identifiers.
      ordered by confidence of recommendation. If there N recommendations where
      N < number of titles read then we assume we made recommendations for the
      last N interactions.
    n: The number of choices per recommendation to consider.
    compute_non_continuation: Also compute statistics for the subset of
      interactions that were not continuations of the previous interaction.
    compute_non_history: Also compute statistics for the subset of interactions
      that the user interacted with a title for the first time.

  Returns:
    recommendation_counts: A dictionary reporting how many recommendations
      were made per student in the class of recommendations under evaluation.
    hit_counts: A dictionary reporting how many recommendations
      per student were hits in the class of recommendations under evaluation.
  """

  all_recs_counts = collections.defaultdict(lambda: 0)
  all_recs_hit_counts = collections.defaultdict(lambda: 0)

  non_continuation_recs_counts = collections.defaultdict(lambda: 0)
  non_continuation_hits_counts = collections.defaultdict(lambda: 0)

  non_history_recs_counts = collections.defaultdict(lambda: 0)
  non_history_hits_counts = collections.defaultdict(lambda: 0)

  for student_id in reference_titles:
    user_titles_read = reference_titles[student_id]
    user_recommendations = recommendations.get(student_id, None)
    if user_recommendations is None:
      continue

    length_diff = len(user_titles_read) - len(user_recommendations)
    if length_diff > 0:
      user_recommendations = [None] * length_diff + list(user_recommendations)

    for i, (title_read, corresponding_rec) in enumerate(
        zip(user_titles_read, user_recommendations)
    ):
      if corresponding_rec is None:
        continue

      all_recs_counts[student_id] += 1
      is_hit = int(title_read in corresponding_rec[:n])
      all_recs_hit_counts[student_id] += is_hit

      if compute_non_continuation and (
          i == 0 or title_read != user_titles_read[i - 1]
      ):
        non_continuation_recs_counts[student_id] += 1
        non_continuation_hits_counts[student_id] += is_hit

      if compute_non_history and title_read not in user_titles_read[:i]:
        non_history_recs_counts[student_id] += 1
        non_history_hits_counts[student_id] += is_hit

  rec_counts = {EVAL_TYPES.ALL: dict(all_recs_counts)}
  hit_counts = {EVAL_TYPES.ALL: dict(all_recs_hit_counts)}

  if compute_non_continuation:
    rec_counts[EVAL_TYPES.NON_CONTINUATION] = dict(
        non_continuation_recs_counts
    )
    hit_counts[EVAL_TYPES.NON_CONTINUATION] = dict(non_continuation_hits_counts)

  if compute_non_history:
    rec_counts[EVAL_TYPES.NON_HISTORY] = dict(non_history_recs_counts)
    hit_counts[EVAL_TYPES.NON_HISTORY] = dict(non_history_hits_counts)

  return rec_counts, hit_counts


def compute_per_student_hits_at_n_dataframe(
    reference_titles,
    recommendations,
    n_values,
    compute_non_continuation = False,
    compute_non_history = False,
):
  """Returns a dataframe containing hits@n on a per student basis.

  Assumes n or more recommendations provided at each timestep
  Args:
    reference_titles: A mapping from student identifiers to a sequence of title
      identifiers that the student interacted with.
    recommendations: A mapping from student identifiers to a sequence of
      recommendations. Each recommendation is a sequence of title identifiers.
      ordered by confidence of recommendation. If there N recommendations where
      N < number of titles read then we assume we made recommendations for the
      last N interactions.
    n_values: List of values for number of choices per recommendation to
      consider.
    compute_non_continuation: Also compute statistics for the subset of
      interactions that were not continuations of the previous interaction.
    compute_non_history: Also compute statistics for the subset of interactions
      that the user interacted with a title for the first time.

  Returns:
    A dataframe with the hits@n at each value of n, and each evaluation subset,
      reported for each student.
  """
  data = []
  headers = [
      types.ResultsRecordFields.STUDENT_ID.value,
      types.ResultsRecordFields.EVAL_TYPE.value,
      types.ResultsRecordFields.N_RECOMMENDATIONS.value,
      types.ResultsRecordFields.HITS_AT_N.value,
  ]

  for n in n_values:
    recs, hits = compute_per_student_hits(
        reference_titles,
        recommendations,
        n,
        compute_non_history=compute_non_history,
        compute_non_continuation=compute_non_continuation,
    )

    for eval_subset in hits:
      for student_id in hits[eval_subset]:
        hits_fraction = (
            hits[eval_subset][student_id] / recs[eval_subset][student_id]
        )
        new_record = [student_id, eval_subset, n, hits_fraction]
        data.append(new_record)

  results = pd.DataFrame(data, columns=headers)

  return results


def aggregate(
    recommendations_per_student,
    hits_per_student,
):
  """Compute hits@n aggregate score from per student statistics."""
  count = len(hits_per_student)
  sum_ = 0
  for student_id in hits_per_student:
    sum_ += (
        hits_per_student[student_id] / recommendations_per_student[student_id]
    )
  if count == 0:
    return None
  return sum_ / count


def interaction_df_to_title_array(
    grouped_df,
):
  """Converts a grouped dataframe of student activity into a dictionary.

  Args:
    grouped_df:  A pandas dataframe of reading activity, grouped by student and
      sorted by activity date.

  Returns:
    result: A mapping from student identifier to a sequence of identifiers of
      titles the student read.
  """
  result = {}
  for student_id, student_data in grouped_df:
    assert isinstance(student_id, int)
    result[types.StudentID(student_id)] = student_data[
        types.StudentActivityFields.BOOK_ID
    ].values

  return result
