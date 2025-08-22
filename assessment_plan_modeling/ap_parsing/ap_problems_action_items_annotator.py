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

"""Annotates problem clusters in assessment and plan sections."""
import dataclasses
import re
from typing import Any, List, Optional, Tuple
from assessment_plan_modeling.ap_parsing import ap_parsing_lib

_CharStartEndTuple = Tuple[int, int]


@dataclasses.dataclass
class ProblemCluster:
  problem_title: _CharStartEndTuple
  problem_description: List[_CharStartEndTuple]
  action_items: List[_CharStartEndTuple]


def is_contained(part, whole):
  return part[0] >= whole[0] and part[1] <= whole[1]


def offset_relative_span(offset,
                         span):
  return (span[0] + offset, span[1] + offset)


def problem_cluster_to_labeled_char_spans(
    problem_cluster):
  """Convert regex annotator output to labeled char spans.

  Args:
    problem_cluster: ProblemCluster object containing character spans.

  Returns:
    A list of LabeledCharSpan corresponding to the spans in the cluster.
  """

  labeled_char_spans = []
  # Problem title
  labeled_char_spans.append(
      ap_parsing_lib.LabeledCharSpan(
          span_type=ap_parsing_lib.LabeledSpanType.PROBLEM_TITLE,
          start_char=problem_cluster.problem_title[0],
          end_char=problem_cluster.problem_title[1]))

  # Problem description
  for pd_start, pd_end in problem_cluster.problem_description:
    if pd_start > 0:
      labeled_char_spans.append(
          ap_parsing_lib.LabeledCharSpan(
              span_type=ap_parsing_lib.LabeledSpanType.PROBLEM_DESCRIPTION,
              start_char=pd_start,
              end_char=pd_end))

  # Action items
  for ai_start, ai_end in problem_cluster.action_items:
    labeled_char_spans.append(
        ap_parsing_lib.LabeledCharSpan(
            span_type=ap_parsing_lib.LabeledSpanType.ACTION_ITEM,
            start_char=ai_start,
            end_char=ai_end))
  return labeled_char_spans


_PROBLEM_SPLIT_REGEXP = re.compile(r"(.+?)(?:[:\n]|[ \t]+\-)\s*")


def _split_problem(
    problem_text
):
  """Splits problem text based on regex into title and description.

  For example:
    "# DM2: on insulin" ->
    "DM2" - problem title, "on insulin" - problem description
  Args:
    problem_text: str

  Returns:
    Tuple(
      problem_title_span - Tuple[int, int],
      problem_description_span - Tuple[int, int]
    ) - if doesn't match, problem_description is returned as None.
  """
  problem_title_match = _PROBLEM_SPLIT_REGEXP.match(problem_text)
  if problem_title_match and problem_title_match.end(0) != len(problem_text):
    problem_title_span = problem_title_match.span(1)
    problem_description_span = (problem_title_match.end(0), len(problem_text))
  else:
    problem_title_span = (0, len(problem_text))
    problem_description_span = None
  return (problem_title_span, problem_description_span)


class APProblemsActionItemsAnnotator():
  """Annotates assessment and plan sections (text) with problem clusters.

  These problem clusters (see data class) consist of
  problem title, problem description(s) and action item(s).
  Object is callable with text.
  """
  _problem_regexp = re.compile(
      r"(^)[ \t]*(?:[#>\*]+[\)\.]?|\d+[\)\.])[ \t]*(.+$)", re.M)
  _problem_title_inorganic = re.compile(r"(^)[ \t]*([A-Z]+[^a-z\n:]+)$", re.M)
  _action_item_regexp = re.compile(r"(^)[ \t]*\-[ \t]*(.+$)", re.M)

  # Finding any of the keywords below in the following regexp is considered
  # as a problem title. This is supposed to capture problem titles in a more
  # lenient way, especially for non-bulleted titles.
  _problem_keyword_regex_template = r"(?mi)(^)[\t\ ]*({}(?::\s?.*)?$)"
  _problem_keywords = [
      "access", "aki", "arf", "cad", "cardiovascular", "communication",
      "consults", "coronaries", "cvs", "dispo|disposition", "dm", "dvt",
      "endo|endocrine", "fen", "f / e / n", "fluids",
      "gastrointestinal|gastrointestinal / abdomen", "gi|gi / abd",
      "glycemic control", "gu", "heart rhythm", "hematology", "heme", "htn",
      "hypertention", "hypotension", "id", "imaging", "infection",
      "infectious disease|infectious diseases", "injuries", "ivf",
      "lines / tubes / drains", "neuro|neurologic", "nutrition", "pain",
      "pain control", "ppx", "prophylaxis", "pulm", "pulmonary", "pump",
      "renal", "resp|respiratory", "rhythm", "sepsis", "stress ulcer", "tld",
      "t / l / d", "wounds"
  ]

  def __init__(self):
    self._problem_keyword_regexps = []
    for keyword in self._problem_keywords:
      esc_keyword = re.escape(keyword)
      esc_keyword = esc_keyword.replace(" ", " ?")  # Spaces are optional.
      esc_keyword = esc_keyword.replace("/", "[\\/]")  # Treat slashes.

      # Treat OR
      esc_keyword = esc_keyword.replace(r"\|", "|")

      self._problem_keyword_regexps.append(
          re.compile(self._problem_keyword_regex_template.format(esc_keyword)))

  def _lookup_problems_by_keyword(
      self, assessment_plan_text):
    """Identifies specific problems by keywords and regexp.

    Keywords are noted in _problem_keywords.
    Regex as defined by _problem_keyword_regex_template.

    Args:
      assessment_plan_text: Text to lookup keywords in.

    Returns:
      List of identified problem keyword regex matches, not sorted.
    """

    problems = []
    for keyword_regexp in self._problem_keyword_regexps:
      problems.extend(keyword_regexp.finditer(assessment_plan_text))

    return problems

  def _find_problems(self, assessment_plan_text):
    """Find all problem matches using regular expressions.

    Args:
      assessment_plan_text: string to find problems in.

    Returns:
      List of re.Match objects.
    """

    # Run all regex matchers.
    problems = list(self._problem_regexp.finditer(assessment_plan_text))
    problems.extend(
        self._problem_title_inorganic.finditer(assessment_plan_text))
    problems.extend(self._lookup_problems_by_keyword(assessment_plan_text))

    # Filter dup matches by match start.
    problems = list({x.span()[0]: x for x in problems}.values())

    # Sort by start char.
    problems = sorted(problems, key=lambda x: x.span()[0])
    return problems

  def __call__(self, assessment_plan_text):
    problem_clusters = []

    # Identify rows containing problems or action items (by regex).
    problems = self._find_problems(assessment_plan_text)
    action_items = list(self._action_item_regexp.finditer(assessment_plan_text))

    # Clusters are defined by the begining of each problem.
    problem_cluster_indices = [x.span()[0] for x in problems
                              ] + [len(assessment_plan_text)]

    # Action item iteration is external to the loop to accout for
    # potential 1:many/zero mapping.
    i_ai = 0

    # Finding the first action item inside a problem cluster.
    while (i_ai < len(action_items) and action_items[i_ai] and
           action_items[i_ai].span(2)[0] < problem_cluster_indices[0]):
      i_ai += 1

    def start_next_or_end_text(i, matches):
      if i < len(matches) and matches[i]:
        return matches[i].span()[0]
      return len(assessment_plan_text)

    # Iterating over clusters to link problems with action items.
    for i, cur_problem in zip(range(1, len(problem_cluster_indices)), problems):
      cluster_span = problem_cluster_indices[i - 1], problem_cluster_indices[i]
      cur_problem_text_span = cur_problem.span(2)

      next_action_item = start_next_or_end_text(i_ai, action_items)
      problem_end = min(next_action_item, cluster_span[1])

      problem_title_span, problem_description_span = _split_problem(
          assessment_plan_text[cur_problem_text_span[0]:problem_end])

      cur_cluster = ProblemCluster(
          problem_title=offset_relative_span(cur_problem_text_span[0],
                                             problem_title_span),
          problem_description=[
              offset_relative_span(cur_problem_text_span[0],
                                   problem_description_span)
              if problem_description_span else (-1, -1)
          ],
          action_items=[])

      # Iterating over action items until exhausting those bounded by cluster.
      while (i_ai < len(action_items) and action_items[i_ai] and
             is_contained(action_items[i_ai].span(2), cluster_span)):
        next_action_item = start_next_or_end_text(i_ai + 1, action_items)
        action_item_end = min(next_action_item, cluster_span[1])

        cur_cluster.action_items.append(
            (action_items[i_ai].span(2)[0], action_item_end))
        i_ai += 1

      problem_clusters.append(cur_cluster)
    return problem_clusters
