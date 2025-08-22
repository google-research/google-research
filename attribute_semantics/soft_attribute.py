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

"""Library to extracts pairwise preferences from soft attribute data collection.

This library takes as input three-way bucketed data about soft attributes (with
items classified as about the same, more or less relative to an anchor item, see
https://github.com/google-research-datasets/soft-attributes). It processes this
data into more useful forms, using the
SoftAttributeJudgment class as the basic representation.
"""

import collections
import csv
import re
from typing import FrozenSet, Mapping, MutableMapping, Set, Text

import attr
from tensorflow.io import gfile


# eq=False since equality by value is meaningless (each entry is one rating).
@attr.s(auto_attribs=True, frozen=True)
class SoftAttributeJudgment:
  """Class to represent judgments made on soft attributes per https://github.com/google-research-datasets/soft-attributes.

  Each judgments consists of:
    - An attribute name
    - A rater id
    - A reference item name
    - Item names placed into each of the three buckets (less/same/more).
  """

  attribute: Text
  rater_id: int
  reference_item: Text
  less_items: FrozenSet[Text]
  same_items: FrozenSet[Text]
  more_items: FrozenSet[Text]


@attr.s(auto_attribs=True, frozen=True)
class PairwisePreference:
  """Class representing a pairwise preference between two items.

  If preference_strength is 0, it should be interpreted as:
    "Item <smaller_item> is ABOUT THE SAME <attribute> as <larger item>
  If preference_strength is 1, it should be interpreted as:
    "Item <smaller_item> is LESS <attribute> than <larger_item>.".
  If preference_strength is 2, it should be interpreted as:
    "Item <smaller_item> is MUCH LESS <attribute> than <larger_item>."
  """

  attribute: Text
  rater_id: int
  smaller_item: Text
  larger_item: Text
  preference_strength: int


def load_judgments(
    filename,
):
  """Loads ratings from filename, returns a dictionary of judgments.

  Args:
    filename: The name of CSV file with user ratings.

  Returns:
    A dictionary from attribute name to all judgments for that attribute.
  """

  def items_str_to_sets(s):
    # This is an awkward format sometimes in the data file:
    #  ["item,with,commas","item2","item3"]
    if s.startswith("[") and s.endswith("]"):
      return frozenset(match for match in re.findall(str_split_regex, s))
    return frozenset(filter(None, s.split(sep=",")))

  judgments: MutableMapping[Text, Set[SoftAttributeJudgment]] = (
      collections.defaultdict(set)
  )
  with gfile.GFile(filename, "r") as csv_file:
    str_split_regex = re.compile('"([^"]+)"')

    reader = csv.DictReader(csv_file, delimiter=",", quotechar='"')
    for row in reader:
      attribute = row["soft_attribute"]
      judgment = SoftAttributeJudgment(
          attribute=attribute,
          rater_id=int(row["rater_id"]),
          reference_item=row["reference_title"],
          less_items=items_str_to_sets(row["less_than"]),
          same_items=items_str_to_sets(row["about_as"]),
          more_items=items_str_to_sets(row["more_than"]),
      )
      judgments[attribute].add(judgment)
  return {
      attr: frozenset(attr_judgments)
      for attr, attr_judgments in judgments.items()
  }


def convert_to_pairwise_preferences(
    judgment,
):
  """Convert a single rater's judgment to a set of pairwise preferences.

  This includes generating pairwise preferences between the reference item and
  all three sets, as well as between items in each of the sets.

  Args:
    judgment: A single raters judgment.

  Returns:
    A set of pairwise preferences between individual items.
  """

  def make_pref(smaller, larger, strength):
    if smaller == larger:
      raise ValueError("An item cannot have a preference relative to itself.")
    return PairwisePreference(
        attribute=judgment.attribute,
        rater_id=judgment.rater_id,
        smaller_item=smaller,
        larger_item=larger,
        preference_strength=strength,
    )

  def make_pref_set(
      smaller_set, larger_set, strength
  ):
    """Returns prefs for all items in smaller_set vs all items in larger_set."""
    tmp_preferences: Set[PairwisePreference] = set()
    for smaller in smaller_set:
      tmp_preferences.update(
          make_pref(smaller, larger, strength) for larger in larger_set
      )
    return tmp_preferences

  preferences: Set[PairwisePreference] = set()

  preferences.update(
      make_pref_set(
          judgment.less_items, frozenset({judgment.reference_item}), 1
      )
  )
  preferences.update(
      make_pref_set(
          frozenset({judgment.reference_item}), judgment.more_items, 1
      )
  )
  preferences.update(make_pref_set(judgment.less_items, judgment.same_items, 1))
  preferences.update(make_pref_set(judgment.same_items, judgment.more_items, 1))
  preferences.update(make_pref_set(judgment.less_items, judgment.more_items, 2))

  # Equality, only within the "same items", using lexical order for items to
  # ensure we don't duplicate preferences. Also, we assume that all items in the
  # same-as set are transitively same. This might not be the case if the
  # reference item is in the middle -- and items at the extreme of "about the
  # same" are not about the same as each other.
  for item in judgment.same_items:
    if item < judgment.reference_item:
      preferences.add(make_pref(item, judgment.reference_item, 0))
    else:
      preferences.add(make_pref(judgment.reference_item, item, 0))

    for item2 in judgment.same_items:
      if item < item2:
        preferences.add(make_pref(item, item2, 0))

  return preferences


def make_title_presentable(title, convert_a = True):
  """Makes titles more presentable, cleaning "X, The (year)" and "X, A (year)".

  This converts "X, The" to "The X" and (optionally) "X, A"/"X, An" to "A X"/"An
  X". It is necessary when data files come from different sources, with
  different formats.

  Args:
    title: Title to convert.
    convert_a: If set, also convert "X, A"/"X, An" to "A X"/"An X".

  Returns:
    Presentable format of title.
  """
  if convert_a:
    regex = re.compile(r"^([A-Za-z0-9:\.,& ']+), ((The)|(An)|(A))(.*)")
  else:
    regex = re.compile(r"^([A-Za-z0-9:\.,& ']+), (The)(.*)")
  match = regex.search(title)
  if match is None:
    return title
  return f"{match.group(2)} {match.group(1)}{match.group(regex.groups)}"
