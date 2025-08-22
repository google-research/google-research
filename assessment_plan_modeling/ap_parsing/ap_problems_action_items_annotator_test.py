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

"""Tests for ap_problems_action_items_annotator."""

from typing import Tuple
from absl.testing import absltest
from assessment_plan_modeling.ap_parsing import ap_problems_action_items_annotator


def _tag_span(text, offset, tag,
              span):
  text = "{0}<{1}>{2}</{1}>{3}".format(text[:span[0] + offset], tag,
                                       text[offset + span[0]:offset + span[1]],
                                       text[offset + span[1]:])
  offset += len(tag) * 2 + 5
  return text, offset


def _cluster_highlight(
    ap_text,
    problem_clusters):
  offset = 0

  for cluster in problem_clusters:
    ap_text, offset = _tag_span(ap_text, offset, "PT", cluster.problem_title)
    for problem_description in cluster.problem_description:
      if problem_description[0] > 0:
        ap_text, offset = _tag_span(ap_text, offset, "PD", problem_description)
    for action_item in cluster.action_items:
      ap_text, offset = _tag_span(ap_text, offset, "AI", action_item)
  print(ap_text + "\n===================\n")


class APProblemsActionItemsAnnotatorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.annotator = ap_problems_action_items_annotator.APProblemsActionItemsAnnotator(
    )

  def test_usage(self):
    ap_text = "\n".join([
        "50 yo m with hx of copd, dm2", "#. COPD ex: started on abx in ed.",
        "  - continue abx.", "  - nebs prn", "  - CIS", "#. DM2: on insulin",
        "  - RISS"
    ])

    problem_clusters = self.annotator(ap_text)
    _cluster_highlight(ap_text, problem_clusters)

    self.assertLen(problem_clusters, 2)
    self.assertEqual(
        problem_clusters[0],
        ap_problems_action_items_annotator.ProblemCluster(
            problem_title=(32, 39),
            problem_description=[(41, 63)],
            action_items=[(67, 81), (85, 94), (98, 102)]))
    self.assertEqual(
        problem_clusters[1],
        ap_problems_action_items_annotator.ProblemCluster(
            problem_title=(105, 108),
            problem_description=[(110, 121)],
            action_items=[(125, 129)]))

  def test_multiline(self):
    ap_text = "\n".join([
        "50 yo m with hx of copd, dm2", "#. COPD ex: no o2 at home",
        "started on abx in ed.", "  - continue abx.", "currenly on azenil",
        "  - nebs prn", "  - CIS", "#. DM2: on insulin", "  - RISS"
    ])

    problem_clusters = self.annotator(ap_text)
    _cluster_highlight(ap_text, problem_clusters)

    self.assertLen(problem_clusters, 2)
    self.assertEqual(
        problem_clusters[0],
        ap_problems_action_items_annotator.ProblemCluster(
            problem_title=(32, 39),
            problem_description=[(41, 77)],
            action_items=[(81, 114), (118, 127), (131, 135)]))
    self.assertEqual(
        problem_clusters[1],
        ap_problems_action_items_annotator.ProblemCluster(
            problem_title=(138, 141),
            problem_description=[(143, 154)],
            action_items=[(158, 162)]))

  def test_action_item_before_cluster(self):
    ap_text = "\n".join([
        "50 yo m with hx of copd, dm2", " - spurious action item",
        "#. COPD ex: started on abx in ed.", "  - continue abx.",
        "  - nebs prn", "  - CIS", "#. DM2: on insulin", "  - RISS"
    ])

    problem_clusters = self.annotator(ap_text)
    _cluster_highlight(ap_text, problem_clusters)

    self.assertLen(problem_clusters, 2)
    self.assertEqual(
        problem_clusters[0],
        ap_problems_action_items_annotator.ProblemCluster(
            problem_title=(56, 63),
            problem_description=[(65, 87)],
            action_items=[(91, 105), (109, 118), (122, 126)]))
    self.assertEqual(
        problem_clusters[1],
        ap_problems_action_items_annotator.ProblemCluster(
            problem_title=(129, 132),
            problem_description=[(134, 145)],
            action_items=[(149, 153)]))

  def test_problem_without_action_items(self):
    ap_text = "\n".join([
        "50 yo m with hx of copd, dm2", "#. COPD ex: started on abx in ed.",
        "#. DM2: on insulin", "  - RISS"
    ])

    problem_clusters = self.annotator(ap_text)
    _cluster_highlight(ap_text, problem_clusters)

    self.assertLen(problem_clusters, 2)
    self.assertEqual(
        problem_clusters[0],
        ap_problems_action_items_annotator.ProblemCluster(
            problem_title=(32, 39),
            problem_description=[(41, 63)],
            action_items=[]))
    self.assertEqual(
        problem_clusters[1],
        ap_problems_action_items_annotator.ProblemCluster(
            problem_title=(66, 69),
            problem_description=[(71, 82)],
            action_items=[(86, 90)]))

  def test_no_problems_no_ais(self):
    ap_text = "\n".join(["50 yo m with hx of copd, dm2"])

    problem_clusters = self.annotator(ap_text)
    self.assertEmpty(problem_clusters)

  def test_problems_without_descriptions(self):
    ap_text = "\n".join([
        "50 yo m with hx of copd, dm2", "#. COPD ex:", "  -continue abx.",
        "#. DM2", "  - RISS"
    ])

    problem_clusters = self.annotator(ap_text)
    _cluster_highlight(ap_text, problem_clusters)

    self.assertLen(problem_clusters, 2)
    self.assertEqual(
        problem_clusters[0],
        ap_problems_action_items_annotator.ProblemCluster(
            problem_title=(32, 41),
            problem_description=[(-1, -1)],
            action_items=[(44, 58)]))
    self.assertEqual(
        problem_clusters[1],
        ap_problems_action_items_annotator.ProblemCluster(
            problem_title=(61, 65),
            problem_description=[(-1, -1)],
            action_items=[(69, 73)]))

  def test_problem_keyword(self):
    ap_text = "\n".join([
        "50 yo m with hx of copd, dm2",
        "ID: started on abx in ed.",
        "  - continue abx.",
        "  - nebs prn",
        "  - CIS",
        "ppx",
        "  - Pneumoboots",
        "Type 2 diabetes: on insulin",  # Non keyword so unmatched.
        "  - RISS"
    ])

    problem_clusters = self.annotator(ap_text)
    _cluster_highlight(ap_text, problem_clusters)

    self.assertLen(problem_clusters, 2)
    self.assertEqual(
        problem_clusters[0],
        ap_problems_action_items_annotator.ProblemCluster(
            problem_title=(29, 31),
            problem_description=[(33, 55)],
            action_items=[(59, 73), (77, 86), (90, 94)]))
    self.assertEqual(
        problem_clusters[1],
        ap_problems_action_items_annotator.ProblemCluster(
            problem_title=(94, 98),
            problem_description=[(-1, -1)],
            action_items=[(102, 142), (146, 150)]))

  def test_inorganic_problem_title(self):
    ap_text = "\n".join([
        "50 yo m with hx of copd, dm2", "CHRONIC OBSTRUCTIVE PULMONARY DISEASE",
        "started on abx in ed.", "  - continue abx.", "  - nebs prn", "  - CIS",
        "DIABETES", "on insulin", "  - RISS", "SHOCK, SEPTIC",
        "ALTERED MENTAL STATUS (NOT DELIRIUM)"
    ])

    problem_clusters = self.annotator(ap_text)
    _cluster_highlight(ap_text, problem_clusters)

    self.assertLen(problem_clusters, 4)
    self.assertEqual(
        problem_clusters[0],
        ap_problems_action_items_annotator.ProblemCluster(
            problem_title=(29, 66),
            problem_description=[(67, 89)],
            action_items=[(93, 107), (111, 120), (124, 128)]))
    self.assertEqual(
        problem_clusters[1],
        ap_problems_action_items_annotator.ProblemCluster(
            problem_title=(128, 136),
            problem_description=[(137, 148)],
            action_items=[(152, 157)]))
    self.assertEqual(
        problem_clusters[2],
        ap_problems_action_items_annotator.ProblemCluster(
            problem_title=(157, 171),
            problem_description=[(-1, -1)],
            action_items=[]))
    self.assertEqual(
        problem_clusters[3],
        ap_problems_action_items_annotator.ProblemCluster(
            problem_title=(171, 207),
            problem_description=[(-1, -1)],
            action_items=[]))

  def test_inorganic_and_keyword(self):
    ap_text = "\n".join(["50 yo m with hx of copd, dm2", "HTN", "- metoprolol"])

    problem_clusters = self.annotator(ap_text)
    _cluster_highlight(ap_text, problem_clusters)

    self.assertLen(problem_clusters, 1)
    self.assertEqual(
        problem_clusters[0],
        ap_problems_action_items_annotator.ProblemCluster(
            problem_title=(29, 33),
            problem_description=[(-1, -1)],
            action_items=[(35, 45)]))


if __name__ == "__main__":
  absltest.main()
