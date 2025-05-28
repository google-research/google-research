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

"""Note sectioning library for MIMIC notes."""

import dataclasses
import json
import re
from typing import Any, Dict, List


@dataclasses.dataclass
class Section:
  """Container of character postings for a section.

  Sections are classified into one (or more) section types. Section types are
  strings and can be expended through adding section markers.

  Sections are classified into the following types:
  "addendum", "allergies", "assessment and plan", "billing diagnosis", "chief
  complaint", "discharge condition", "discharge diagnoses", "discharge
  instructions", "disposition", "events", "family history", "social history",
  "followup instructions", "health maintenance", "history of present illness",
  "hospital course", "icu care", "medications", "past medical history", "past
  surgical history", "physical examination", "procedures", "protected section",
  "review of systems", "service", "social history", "test results", "vital
  signs"
  """

  char_start: int
  char_end: int
  section_types: List[str]

  def __len__(self):
    if self.char_end <= self.char_start:
      raise ValueError("Section has invalid length")
    return self.char_end - self.char_start


def get_markers(path):
  """Loads marker dict from json file.

  Args:
    path: str, gfile path to marker dict.

  Returns:
    A dictionary mapping section headers (uncased) to section types.
  """
  return json.load(open(path, "r"))


# Regex pattern for matching section headers. Matches a section header in a
# new-line (potentially indented) followed by a colon and spaces or new-line.
_REGEX_SECTION_HEADER = r"(?mi)^[\t\ ]*{}(?::\s+|:?$)"


def _get_regexpr_from_marker(marker):
  return re.compile(_REGEX_SECTION_HEADER.format(re.escape(marker)))


class SectionFinder():
  """Finds sections in note text based on section header markers."""

  def __init__(self, marker_dict):
    self._marker_dict = marker_dict
    self._marker_regexpr = {
        marker: _get_regexpr_from_marker(marker) for marker in self._marker_dict
    }

  def find_sections(self, text):
    """Finds sections in text given markers (based on regex).

    Find sections matching the regex (REGEX_SECTION_HEADER).
    Unlike MR sectioning logic, treats section header as part of the section.

    Args:
      text: str, full note text

    Returns:
      List of Section dataclass containing char interval and
        types as list of strings.
    """
    sections = []
    for marker in self._marker_dict:
      for match in self._marker_regexpr[marker].finditer(text):
        sections.append(
            Section(
                char_start=match.start(),
                char_end=match.end(),
                section_types=self._marker_dict[marker]))

    # No sections found, return empty list.
    if not sections:
      return []

    # Sort.
    sections.sort(key=lambda x: x.char_start)

    # Remove overlapping sections - this should never happen as the regex should
    # match from start line to a delimiter, but as a fail safe:
    section_index = 1
    while section_index < len(sections):
      # If any overlap, delete shorter.
      if min(sections[section_index].char_end,
             sections[section_index - 1].char_end) > max(
                 sections[section_index].char_start,
                 sections[section_index - 1].char_start):
        if len(sections[section_index]) > len(sections[section_index - 1]):
          i_del = section_index
        else:
          i_del = section_index - 1
        del sections[i_del]
      else:
        section_index += 1

    # Adjust end to start of next section.
    for section_index in range(len(sections) - 1):
      sections[section_index].char_end = sections[section_index + 1].char_start
    sections[-1].char_end = len(text)

    return sections
