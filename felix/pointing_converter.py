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

"""Conversion from training target text into target points.

The conversion algorithm from (source, target) pairs to (source, target_points)

We construct the target sequence by re-arraning the source tokens. As such
each source token points to the location (within the source sequence) of the
next target token.

In the case that a source token is not used to construct the target sequence,
this source token points to the start of the sequence.

If there is a target token which does not appear within the source, then the
previouse source token which does appear, stores this target token.

See pointing_converter_test.py for examples.
"""

import collections

from typing import Iterable, List, Text
from felix import pointing
from felix import utils


class PointingConverter(object):
  """Converter from training target texts into pointing format."""

  def __init__(self,
               phrase_vocabulary,
               do_lower_case = True):
    """Initializes an instance of PointingConverter.

    Args:
      phrase_vocabulary: Iterable of phrase vocabulary items (strings), if empty
        we assume an unlimited vocabulary.
      do_lower_case: Should the phrase vocabulary be lower cased.
    """
    self._do_lower_case = do_lower_case
    self._phrase_vocabulary = set()
    for phrase in phrase_vocabulary:
      if do_lower_case:
        phrase = phrase.lower()
      # Remove the KEEP/DELETE flags for vocabulary phrases.
      if "|" in phrase:
        self._phrase_vocabulary.add(phrase.split("|")[1])
      else:
        self._phrase_vocabulary.add(phrase)

  def compute_points(self, source_tokens,
                     target):
    """Computes points needed for converting the source into the target.

    Args:
      source_tokens: Source tokens.
      target: Target text.

    Returns:
      List of pointing.Point objects. If the source couldn't be converted into
      the target via pointing, returns an empty list.
    """
    if self._do_lower_case:
      target = target.lower()
      source_tokens = [x.lower() for x in source_tokens]
    target_tokens = utils.get_token_list(target)

    points = self._compute_points(source_tokens, target_tokens)
    return points

  def _compute_points(self, source_tokens, target_tokens):
    """Computes points needed for converting the source into the target.

    Args:
      source_tokens: List of source tokens.
      target_tokens: List of target tokens.

    Returns:
      List of pointing.Pointing objects. If the source couldn't be converted
      into the target via pointing, returns an empty list.
    """
    source_tokens_indexes = collections.defaultdict(set)
    for i, source_token in enumerate(source_tokens):
      source_tokens_indexes[source_token].add(i)

    target_points = {}
    last = 0
    token_buffer = ""

    def find_nearest(indexes, index):
      # In the case that two indexes are equally far apart
      # the lowest index is returned.
      return min(indexes, key=lambda x: abs(x - index))

    for target_token in target_tokens[1:]:
      # Is the target token in the source tokens and is buffer in the vocabulary
      # " ##" converts word pieces into words
      if (source_tokens_indexes[target_token] and
          (not token_buffer or not self._phrase_vocabulary or
           token_buffer in self._phrase_vocabulary)):
        # Maximum length expected of source_tokens_indexes[target_token] is 512,
        # median length is 1.
        src_indx = find_nearest(source_tokens_indexes[target_token], last)
        # We can only point to a token once.
        source_tokens_indexes[target_token].remove(src_indx)
        target_points[last] = pointing.Point(src_indx, token_buffer)
        last = src_indx
        token_buffer = ""

      else:
        token_buffer = (token_buffer + " " + target_token).strip()

    ## Buffer needs to be empty at the end.
    if token_buffer.strip():
      return []

    points = []
    for i in range(len(source_tokens)):
      ## If a source token is not pointed to,
      ## then it should point to the start of the sequence.
      if i not in target_points:
        points.append(pointing.Point(0))
      else:
        points.append(target_points[i])

    return points
