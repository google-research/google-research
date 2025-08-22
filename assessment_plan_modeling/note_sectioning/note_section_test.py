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

"""Tests for note_section."""

from absl.testing import absltest
from assessment_plan_modeling.note_sectioning import note_section_lib


class NoteSectionTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.marker_dict = {
        "hpi": ["history of present illness"],
        "pmh": ["past medical history"],
        "a&p": ["assessment and plan"],
        "past medical and surgical history": [
            "past medical history", "past surgical history"
        ],
    }
    self.section_finder = note_section_lib.SectionFinder(self.marker_dict)

  def test_usage(self):
    text = "\n".join([
        "HPI: 50 yo m with hx of dm2, presents with chest pain",
        "PMH:",
        "-DM2 on insulin",
        "-s/p CABG 1999",
        "A&P:",
        "50 yo m with hx of dm2, presents with chest pain",
        "chest pain:",
        "- PCI today",
        "- DAPT, statin, bb",
    ])

    self.assertListEqual(
        self.section_finder.find_sections(text), [
            note_section_lib.Section(
                char_start=0,
                char_end=54,
                section_types=["history of present illness"]),
            note_section_lib.Section(
                char_start=54,
                char_end=90,
                section_types=["past medical history"]),
            note_section_lib.Section(
                char_start=90,
                char_end=186,
                section_types=["assessment and plan"]),
        ])

  def test_multiple_section_types(self):
    text = "\n".join([
        "HPI: 50 yo m with hx of dm2, presents with chest pain",
        "past medical and surgical history:",
        "-DM2 on insulin",
        "-s/p CABG 1999",
        "A&P:",
        "50 yo m with hx of dm2, presents with chest pain",
        "chest pain:",
        "- PCI today",
        "- DAPT, statin, bb",
    ])

    self.assertListEqual(
        self.section_finder.find_sections(text), [
            note_section_lib.Section(
                char_start=0,
                char_end=54,
                section_types=["history of present illness"]),
            note_section_lib.Section(
                char_start=54,
                char_end=120,
                section_types=["past medical history", "past surgical history"
                              ]),
            note_section_lib.Section(
                char_start=120,
                char_end=216,
                section_types=["assessment and plan"]),
        ])

  def test_multiple_matches(self):
    text = "\n".join([
        "HPI: 50 yo m with hx of dm2, presents with chest pain",
        "Assessment: Plan:",
        "50 yo m with hx of dm2, presents with chest pain",
        "chest pain:",
        "- PCI today",
        "- DAPT, statin, bb",
    ])

    # A delimiter matching the regex (:) is found inside the marker
    marker_dict = {
        "hpi": ["history of present illness"],
        "assessment": ["assessment and plan"],
        "assessment: plan": ["assessment and plan"],
    }
    section_finder = note_section_lib.SectionFinder(marker_dict)

    self.assertListEqual(
        section_finder.find_sections(text), [
            note_section_lib.Section(
                char_start=0,
                char_end=54,
                section_types=["history of present illness"]),
            note_section_lib.Section(
                char_start=54,
                char_end=163,
                section_types=["assessment and plan"]),
        ])

  def test_no_sections(self):
    text = "50 yo m with hx of dm2, presents with chest pain"

    # A delimiter matching the regex (:) is found inside the marker
    marker_dict = {
        "hpi": ["history of present illness"],
        "assessment": ["assessment and plan"],
        "assessment: plan": ["assessment and plan"],
    }
    section_finder = note_section_lib.SectionFinder(marker_dict)

    self.assertEmpty(section_finder.find_sections(text))


if __name__ == "__main__":
  absltest.main()
