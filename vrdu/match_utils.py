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

"""Utilities for conducting fuzzy matching for each match type, Price, Date, etc."""

import datetime
import re
from typing import Any, Optional, TypeVar

import editdistance

# `Entity` defines the generic type for a entity pair: (entity_text, entity_box,
# entity_segments) where:
# 1) `entity_text` is a string showing the textual contents of the entity;
# 2) `entity_box` refers to a tuple of 5 values decribing the page index and the
# the x, y coordinates of the upper-left and down-right corner of the bounding
# box.
# 3) `entity_segments` is a list of two-value tuples, and each tuple indicates
# the start and end index of this entity in the reading order text. There might
# be multiple segments in case that this entity involves multiple text spans.
# e.g., ('Google Research', (0, 1.1, 2.2, 3.3, 4.4), [(0, 15)]).
#
# **The `entity_box` and `entity_segments` are used to map the entity back to
# the image or reading order sequence. They are useful when interpreting the
# bahavior of the models and converting the benchmark into the required format
# for the models.
Entity = TypeVar(
    'Entity',
    bound=tuple[str, tuple[int, float, float, float, float], list[tuple[int,
                                                                        int]]])


class Match:
  """The ancestor class for all specific {Type}Match, e.g., DateMatch.

  Also include the general fuzzy matching functions, e.g. match_by_substring().
  The {Type}Match will perform fuzzy matching for each match type by calling
  these general matching functions.
  """

  @classmethod
  def match(cls, extracted_entity,
            labeled_entities):
    """The template for any matching function of a specific {Type}Match.

    Args:
      extracted_entity: Extraction result, a tuple of two fields: (text, bbox)
      labeled_entities: A list of candidate entities: [(text, bbox), (text,
        bbox), ...], where `text` indicates the textual contents and `bbox`
        locates the entity uniquely in the page. Since the same entity may
        appear multiple times in the doc and the model only needs to extract one
        of them, a list of candidates are provided here. When is only one
        appearance, the list will have one element.

    Raises:
      NotImplementedError: This is just a template and should not be called.
      Instead the specific {Type}Match should be called. For example,
        DateMatch.match(('7/1/2022', box), [('07/02/2022', box)])
    """
    raise NotImplementedError

  @classmethod
  def is_entity(cls, obj):
    """Check whether the input obj is a type of entity."""
    if isinstance(obj, tuple) and len(obj) == 3:
      # 1) entity text
      if not isinstance(obj[0], str):
        return False
      # 2) entity box
      if not isinstance(obj[1], tuple) or len(obj[1]) != 5:
        return False
      if not isinstance(obj[1][0], int):
        return False
      for v in obj[1][1:]:
        if not isinstance(v, float):
          return False
      # 3) entity segments
      if not isinstance(obj[2], list):
        return False
      for segment in obj[2]:
        if not isinstance(segment, tuple) or len(segment) != 2:
          return False
        if not isinstance(segment[0], int) or not isinstance(segment[1], int):
          return False
    return True

  @classmethod
  def remove_redundant_whitespace(cls, string):
    r"""Removes the redunant whitespace in the input string.

    1. Remove the prefix/suffix whitespace.
    2. Replace the continuous whitespace with a single one.

    Args:
      string: Entity text string from extractions or annotations.

    Returns:
      Remove the prefix/suffix whitespace and replace the continuous whitespace
      with a single one, e.g., ' abc\ndef   ghi\t' => 'abc def ghi'.
    """

    substrs = string.strip().split()
    proc_str = ' '.join([substr.strip() for substr in substrs])
    return proc_str

  @classmethod
  def match_by_alpha_numeric_text(cls, str_a, str_b):
    """Strings match if they are the same after removing all non-alpha-numeric contents.

    Args:
      str_a: String A
      str_b: String B

    Returns:
      If A and B are equivalent after removing all non-alpha-numeric contents.
      e.g. "Xy_Z1 2@3" == "XyZ123"

    """
    proc_a = re.sub(r'[^0-9a-zA-Z]', '', str_a)
    proc_b = re.sub(r'[^0-9a-zA-Z]', '', str_b)
    return proc_a == proc_b

  @classmethod
  def match_by_non_whitespace_text(cls, str_a, str_b):
    r"""Strings match if they are the same after removing all whitespaces.

    Args:
      str_a: String A
      str_b: String B

    Returns:
      If A and B are equivalent after removing all whitespaces.
      e.g. "X y Z 1\t 2 3" == "XyZ123".
    """
    proc_a = re.sub(r'[\s]', '', str_a)
    proc_b = re.sub(r'[\s]', '', str_b)
    return proc_a == proc_b

  @classmethod
  def match_by_substring(cls, str_a, str_b):
    """Strings match if the one is the substring of the other.

    Args:
      str_a: String A
      str_b: String B

    Returns:
      A and B are equivalent if the one is the substring of the other.
      e.g. "yZ1" == "XyZ123".
    """
    proc_a = str_a.strip()
    proc_b = str_b.strip()

    if not proc_a and not proc_b:
      return True
    if not proc_a or not proc_b:
      return False
    if proc_a in proc_b or proc_b in proc_a:
      return True
    return False

  @classmethod
  def match_by_value(cls, str_a, str_b, diff = 0.01):
    """Strings match if the absolute difference is smaller than `diff`.

    Args:
      str_a: String A
      str_b: String B
      diff: The tolerable difference

    Returns:
      First remove any special characters and only keep numbers & `.`, and then
      A and B are equivalent if the absolute difference is smaller than `diff`.
      e.g. "$3.14" == "3.1415926".
    """
    to_num = lambda x: re.sub(r'[^0-9.]', '', x)
    proc_a = to_num(str_a)
    proc_b = to_num(str_b)

    try:
      num_a = float(proc_a)
      num_b = float(proc_b)
      if abs(num_a - num_b) <= diff:
        return True
    except Exception:  # pylint: disable=broad-except
      pass

    return False

  @classmethod
  def match_by_numeric_text(cls, str_a, str_b):
    """Strings match if they are the same after removing all non-numeric contents.

    Args:
      str_a: String A
      str_b: String B

    Returns:
      If A and B are equivalent after removing all non-numeric contents.
      e.g. "Xy_Z1 2@3" == "1xx2yy3zz"
    """
    proc_a = re.sub(r'[^0-9]', '', str_a)
    proc_b = re.sub(r'[^0-9]', '', str_b)
    return proc_a == proc_b

  @classmethod
  def match_by_edit_distance(cls,
                             str_a,
                             str_b,
                             threshold = 3):
    """Strings match if the edit_distance is not larger than the threshold.

    The threshold is 3 as default, since we allow some small mistakes but do not
    want the mistakes change the content meanings.

    Args:
      str_a: String A.
      str_b: String B.
      threshold: The minimum distance that can be tolerated.

    Returns:
      If the distance is not larger than the threshold, then two strings match.
    """
    distance = editdistance.eval(str_a, str_b)
    return distance <= threshold


class DateMatch(Match):
  """Match class for match type `date`.

  Two dates match if they have same year/month/day, OR they have the same
  year, but the month and day are swapped (because different date format:
  MM/DD/YY vs DD/MM/YY).
  """

  @classmethod
  def decode_date(cls, date_string):
    """Extracts the year, month, day fields from the date string."""

    proc_date_string = re.sub(r'[^0-9a-zA-Z/\-,]', '', date_string)

    def match_pattern(date_string,
                      pattern):
      """Returns the date dictionary if the date string satisfies the pattern."""
      try:
        date = datetime.datetime.strptime(date_string, pattern).date()
        month = date.month
        day = date.day
        year = date.year
        return {'year': year, 'month': month, 'day': day}
      except ValueError:
        return None

    patterns = [
        '%m/%d/%y',  # 07/01/2022
        '%m/%d/%Y',  # 07/01/22
        '%m/%d',  # 07/01, year will be set as 1900 by default
        '%b%d/%y',  # Jul01/22
        '%m-%d-%Y',  # 07-01-2022
        '%m-%d-%y',  # 07-01-22
        '%B%d,%Y',  # July01,2022
        '%Y/%m/%d',  # 2022/07/01
    ]

    for pattern in patterns:
      match_result = match_pattern(proc_date_string, pattern)
      if match_result:
        return match_result

    return None

  @classmethod
  def match(cls, extracted_entity,
            labeled_entities):
    for labeled_entity in labeled_entities:
      extracted_text = cls.remove_redundant_whitespace(extracted_entity[0])
      labeled_text = cls.remove_redundant_whitespace(labeled_entity[0])

      extracted_date = cls.decode_date(extracted_text)
      labeled_date = cls.decode_date(labeled_text)

      if not extracted_date or not labeled_date:
        if cls.match_by_alpha_numeric_text(extracted_text, labeled_text):
          return True
      else:
        if (extracted_date['year'] == labeled_date['year'] and
            extracted_date['month'] == labeled_date['month'] and
            extracted_date['day'] == labeled_date['day']):
          return True

        # Some date format is ambiguous. For example, 01/02/2022 can be
        # interpreted as Feb 1, 2022 or Jan 2, 2022. It is hard to distinguish
        # without contexts. Therefore, if the day and month are swapped, we also
        # consider the result as correct.
        if (extracted_date['year'] == labeled_date['year'] and
            extracted_date['day'] == labeled_date['month'] and
            extracted_date['month'] == labeled_date['day']):
          return True

    return False


class PriceMatch(Match):
  """Match class for match type `price`.

  First convert the string into float/int numbers and do Match.match_by_value().
  """

  @classmethod
  def match(cls, extracted_entity,
            labeled_entities):
    for labeled_entity in labeled_entities:
      extracted_text = cls.remove_redundant_whitespace(extracted_entity[0])
      labeled_text = cls.remove_redundant_whitespace(labeled_entity[0])
      if cls.match_by_value(extracted_text, labeled_text):
        return True
    return False


class AddressMatch(Match):
  """Match class for match type `address`.

  Two strings match if the edit distance is smaller than a threshold.
  Use Match.match_by_edit_distance().
  """

  @classmethod
  def match(cls, extracted_entity,
            labeled_entities):
    for labeled_entity in labeled_entities:
      extracted_text = cls.remove_redundant_whitespace(extracted_entity[0])
      labeled_text = cls.remove_redundant_whitespace(labeled_entity[0])
      if cls.match_by_edit_distance(extracted_text, labeled_text):
        return True
    return False


class NumericalStringMatch(Match):
  """Match class for match type `numerical_string`.

  Two strings match if they are equivalent after removing all non-numerical
  contents. Use Match.match_by_numeric_text().
  """

  @classmethod
  def match(cls, extracted_entity,
            labeled_entities):
    for labeled_entity in labeled_entities:
      extracted_text = cls.remove_redundant_whitespace(extracted_entity[0])
      labeled_text = cls.remove_redundant_whitespace(labeled_entity[0])
      if cls.match_by_numeric_text(extracted_text, labeled_text):
        return True
    return False


class GeneralStringMatch(Match):
  """Match class for match type `general_string`.

  Two strings match if they are equivalent after removing all non-alpha-numeric
  contents. Use Match.match_by_alpha_numeric_text().
  """

  @classmethod
  def match(cls, extracted_entity,
            labeled_entities):
    for labeled_entity in labeled_entities:
      extracted_text = cls.remove_redundant_whitespace(extracted_entity[0])
      labeled_text = cls.remove_redundant_whitespace(labeled_entity[0])
      if cls.match_by_alpha_numeric_text(extracted_text, labeled_text):
        return True
    return False


class NameMatch(Match):
  """Match class for match type `name`.

  Two strings match if they are equivalent after removing all whitespaces. Use
  Match.match_by_non_whitespace_text().
  """

  @classmethod
  def match(cls, extracted_entity,
            labeled_entities):
    for labeled_entity in labeled_entities:
      extracted_text = cls.remove_redundant_whitespace(extracted_entity[0])
      labeled_text = cls.remove_redundant_whitespace(labeled_entity[0])
      if cls.match_by_non_whitespace_text(extracted_text, labeled_text):
        return True
    return False


class DefaultMatch(Match):
  """Default matching class.

  Used when no matching class is specified.
  Do the strict string matching.
  """

  @classmethod
  def match(cls, extracted_entity,
            labeled_entities):
    print('The default matching function is used.')
    for labeled_entity in labeled_entities:
      extracted_text = cls.remove_redundant_whitespace(extracted_entity[0])
      labeled_text = cls.remove_redundant_whitespace(labeled_entity[0])
      if extracted_text == labeled_text:
        return True
    return False
