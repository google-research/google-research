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

"""Helper functions to attribute expansions to abbreviations."""

import collections
from collections.abc import Mapping, MutableMapping, Sequence
import dataclasses
import enum
from typing import Optional
from deciphering_clinical_abbreviations import text_alignment as text_alignment_lib
from deciphering_clinical_abbreviations import tokenizer as tokenizer_lib


# Type aliases.
AbbrevExpansionDict = Mapping[str, Sequence[str]]


# Text attribution enums.
class TextAttributorDirection(enum.Enum):
  """Attributor direction."""
  LEFT_TO_RIGHT = enum.auto()
  RIGHT_TO_LEFT = enum.auto()


@dataclasses.dataclass
class TextAttributor:
  """Class for attributing expansion text to original tokens using heuristics.

  Attributes:
    orig_tokens: The tokens from the original abbreviated text.
    expanded_tokens: The tokens from the expanded text.
    tokenizer: The tokenizer used to break apart and join strings to and from
      tokens.
    abbreviation_expansions_dict: A dictionary mapping known abbreviations to
      their valid expansions.
  """
  orig_tokens: Sequence[str]
  expanded_tokens: Sequence[str]
  tokenizer: tokenizer_lib.Tokenizer
  abbreviation_expansions_dict: Optional[AbbrevExpansionDict] = None

  _attributed_orig_idxs: set[int] = dataclasses.field(
      init=False, default_factory=set)
  _attributed_expanded_idxs: set[int] = dataclasses.field(
      init=False, default_factory=set)
  _orig_idx_to_expanded_idxs: collections.defaultdict[int, Sequence[int]] = (
      dataclasses.field(
          init=False, default_factory=lambda: collections.defaultdict(list)))
  _expanded_idx_to_orig_idx: MutableMapping[int, int] = dataclasses.field(
      init=False, default_factory=dict)

  def _link_orig_and_expansion_indices(
      self,
      orig_idx,
      expansion_idx):
    """Links original and expanded indices, overwriting preexisting linkages.

    To create a link, two dictionaries must be updated:
      - A mapping from expanded index to the original index (1 to 1)
      - A mapping from original index to expanded indices (1 to many)
    When the expanded index to original index mapping is updated, the
    preexisting value is overwritten. To reflect this overwrite in the reverse
    mapping, the expanded index must also be removed from the list of indices
    for the original index it was previously linked to.

    Args:
      orig_idx: The index of the token in the original text to link.
      expansion_idx: The index of the token in the expanded text to connect.
    """
    self._orig_idx_to_expanded_idxs[orig_idx].append(expansion_idx)
    if expansion_idx in self._expanded_idx_to_orig_idx:
      prior_orig_idx = self._expanded_idx_to_orig_idx[expansion_idx]
      prior_orig_expansion_idxs = (
          self._orig_idx_to_expanded_idxs[prior_orig_idx])
      del prior_orig_expansion_idxs[
          prior_orig_expansion_idxs.index(expansion_idx)]
      self._orig_idx_to_expanded_idxs[prior_orig_idx] = (
          prior_orig_expansion_idxs)
    self._expanded_idx_to_orig_idx[expansion_idx] = orig_idx

  def _process_match_candidates(
      self,
      text_to_match,
      match_candidates,
      ):
    """Checks if any match candidates are a match or submatch with text.

    Args:
      text_to_match: The text which we are cross-referencing for matches.
      match_candidates: The list of candidate strings which may be partial or
        full matches with the text_to_match.
    Returns:
      A tuple of two elements:
        1) a boolean indicating whether a full match was found
        2) the remaining candidates which partially match the text_to_match but
          do not fully match.
    """
    remaining_candidates = []
    match_found = False
    for match_candidate in match_candidates:
      if not (match_candidate.startswith(text_to_match)
              or match_candidate.endswith(text_to_match)):
        continue
      if match_candidate == text_to_match:
        match_found = True
        continue
      remaining_candidates.append(match_candidate)
    return match_found, remaining_candidates

  def _search_for_match_in_expansion(
      self,
      expansion_idx,
      valid_match_candidates,
      direction):
    """Search expansion text for the longest phrase that matches the dictionary.

    Args:
      expansion_idx: The start index of the expansion tokens from which to
        iteratively build larger expansion strings to search for a dictionary
        expansion match.
      valid_match_candidates: The initial expansion candidates within which
        a match will be searched for.
      direction: Which end of the sequences to start at and which direction to
        iterate towards.
    Returns:
      The number of expanded tokens required to construct the expansion
        string which matches. If no match is found, this will be None.
    """
    left_idx, right_idx = expansion_idx, expansion_idx + 1
    match_token_count = None
    while (left_idx >= 0
           and right_idx <= len(self.expanded_tokens)
           and valid_match_candidates):
      token_count = right_idx - left_idx
      expansion_text = (
          self.tokenizer.detokenize(self.expanded_tokens[left_idx:right_idx]))
      match_found, valid_match_candidates = self._process_match_candidates(
          expansion_text, valid_match_candidates)
      if match_found: match_token_count = token_count
      if direction == TextAttributorDirection.LEFT_TO_RIGHT: right_idx += 1
      if direction == TextAttributorDirection.RIGHT_TO_LEFT: left_idx -= 1
    return match_token_count

  def _attribute_text_based_on_alignment(
      self,
      orig_idx,
      expansion_idx,
      direction):
    """Attribute tokens based on how they are aligned.

    The expanded tokens attributed to an original token based on alignment
    should be those whose first token is directly aligned to the original and
    whose subsequent tokens have been aligned to consecutive empty tokens.
    Consider the below example:
                  Abbreviated: A - B - '' - '' - C
                               |   |    |    |   |
                  Expanded   : 1 - 2 -- 3 -- 4 - 5
    The expanded tokens that would be attributed to token B are [2,3,4]. If
    the algorithm is moving left to right and tokens preceding token 2 have not
    yet been attributed, they will be included in B's attribution. Similarly,
    if the algorithm is moving right to left and tokens after token 4 have not
    yet been attributed, they will be included in B's attribution.

    Args:
      orig_idx: The index of the original token that is being attributed to.
      expansion_idx: The starting index of the expanded tokens that will be
        attributed to the original token index.
      direction: Which end of the sequences to start at and which direction to
        iterate towards.
    Returns:
      The new expansion index after iterating through all expanded tokens to
        attribute.
    """
    if direction == TextAttributorDirection.LEFT_TO_RIGHT:
      while (expansion_idx < len(self.expanded_tokens)
             and (expansion_idx <= orig_idx
                  or not self.orig_tokens[expansion_idx])):
        if expansion_idx not in self._attributed_expanded_idxs:
          self._link_orig_and_expansion_indices(orig_idx, expansion_idx)
        expansion_idx += 1
    else:
      while expansion_idx >= orig_idx:
        if expansion_idx not in self._attributed_expanded_idxs:
          self._link_orig_and_expansion_indices(orig_idx, expansion_idx)
        expansion_idx -= 1
    return expansion_idx

  def _attribute_text_one_direction(self, direction):
    """Moves in one direction and attributes expansion text using heuristics.

    The first heuristic is check if an expansion substring can be found that
    matches either the original token itself or, if the token is an
    abbreviation, any known expansions for that abbreviation. If such a match is
    found, the matched text is attributed to the original token and the token is
    omitted from further attribution. If no such match is found, the second
    heuristic is to carry out the attribution such that it reflects the way the
    text is aligned by the Needleman-Wunsch algorithm.

    Args:
      direction: Which end of the sequences to start at and which direction to
        iterate towards.
    """
    if direction == TextAttributorDirection.LEFT_TO_RIGHT:
      orig_idx = 0
      expansion_idx = 0
      idx_iter_fn = lambda idx: idx + 1
    else:
      orig_idx = len(self.orig_tokens) - 1
      expansion_idx = len(self.expanded_tokens) - 1
      idx_iter_fn = lambda idx: idx - 1
    while orig_idx >= 0 and orig_idx < len(self.orig_tokens):
      orig_token = self.orig_tokens[orig_idx]
      if orig_token and orig_idx not in self._attributed_orig_idxs:
        valid_match_candidates = [orig_token]
        if (self.abbreviation_expansions_dict
            and orig_token in self.abbreviation_expansions_dict):
          valid_match_candidates.extend(
              self.abbreviation_expansions_dict[orig_token])
        num_matched_expansion_tokens = (
            self._search_for_match_in_expansion(
                expansion_idx=expansion_idx,
                valid_match_candidates=valid_match_candidates,
                direction=direction))
        if num_matched_expansion_tokens is not None:
          for _ in range(num_matched_expansion_tokens):
            self._link_orig_and_expansion_indices(orig_idx, expansion_idx)
            self._attributed_expanded_idxs.add(expansion_idx)
            expansion_idx = idx_iter_fn(expansion_idx)
          self._attributed_orig_idxs.add(orig_idx)
        else:
          expansion_idx = self._attribute_text_based_on_alignment(
              orig_idx,
              expansion_idx,
              direction=direction)
      orig_idx = idx_iter_fn(orig_idx)

  def _attribute_text(self):
    """Performs all original-expanded text attributions bi-directionally.

    Attributions must be carried out in both directions in order to allow
    valid expansion matches from either side to be greedily attributed.
    Otherwise, the frame of attribution may be shifted by a bad alignment and
    valid expansions may not be attributed. Consider the following alignment:
                hbp   ''      qam    ''    ''
                  |    |       |      |     |
                high blood pressure every morning
    And the corresponding abbreviation-expansion dictionary:
          {"hbp": ["high-blood pressure"], "qam": ["every morning"]}
    Notice that the actual output is missing a hyphen, which means it does not
    match a valid expansion in the dictionary. Therefore, the alignment is used,
    which results in the mapping:
          {"hbp": ["high blood"], "qam": "pressure every morning"}
    If the attribution had been carried out in reverse, "every morning" would
    have been correctly identified as a valid expansion, resulting in the
    correct mapping:
          {"hbp": ["high blood pressure"], "qam": "every morning"}
    """
    self._attribute_text_one_direction(
        direction=TextAttributorDirection.LEFT_TO_RIGHT)
    self._attribute_text_one_direction(
        direction=TextAttributorDirection.RIGHT_TO_LEFT)

  def _get_text_pairs_for_attribution(self):
    """Generates the pairs of attributed text.

    Returns:
      A list of tuples containing two elements:
        1) the original token
        2) the expanded text that has been attributed to that token
    """
    attributed_pairs = []
    for orig_idx in range(len(self.orig_tokens)):
      orig_token = self.orig_tokens[orig_idx]
      if not orig_token: continue
      if orig_idx in self._orig_idx_to_expanded_idxs:
        expanded_idxs = self._orig_idx_to_expanded_idxs[orig_idx]
        expanded_tokens = [
            self.expanded_tokens[i] for i in sorted(expanded_idxs)]
        expanded_text = self.tokenizer.detokenize(expanded_tokens)
      else:
        expanded_text = ""
      attributed_pairs.append((orig_token, expanded_text))
    return attributed_pairs

  def generate_attribution_pairs(self):
    """Carries out attribution and returns the attribution pairs."""
    self._attribute_text()
    return self._get_text_pairs_for_attribution()


def _attribute_expansions_to_original_text(
    alignment_pairs,
    tokenizer,
    abbreviation_expansions_dict = None
    ):
  """Attributes expansion text to the original tokens using NW alignment pairs.

  Alignment pairs consisting of two matching tokens are automatically
  paired with each other. Regions of original and expanded text in which no
  pairs are exact matches, called ambiguous regions, are attributed using a
  number of heuristics (see TextAttributor).

  Args:
    alignment_pairs: A list of pairs which align the original and expanded text,
      consisting of tokens from the aligned texts along with empty strings for
      insertions or deletions.
    tokenizer: The tokenizer used to break apart and join strings to and from
      tokens.
    abbreviation_expansions_dict: A dictionary mapping each
      abbreviation to all known expansions.
  Returns:
    A list of tuple pairs in which each tuple contains 2 elements:
      1) a token from the original text
      2) the text from the expansion that is being attributed to that token
  """
  attribution_pairs = []
  ambiguous_orig_tokens, ambiguous_expanded_tokens = [], []
  for orig_token, expanded_token, score_type in alignment_pairs:
    if (score_type == text_alignment_lib.TokenPairScoreType.MATCH
        or not expanded_token):
      if ambiguous_orig_tokens or ambiguous_expanded_tokens:
        attribution_pairs.extend(
            TextAttributor(
                orig_tokens=ambiguous_orig_tokens,
                expanded_tokens=ambiguous_expanded_tokens,
                tokenizer=tokenizer,
                abbreviation_expansions_dict=abbreviation_expansions_dict
                ).generate_attribution_pairs())
        ambiguous_orig_tokens, ambiguous_expanded_tokens = [], []
      attribution_pairs.append((orig_token, expanded_token))
    else:
      ambiguous_orig_tokens.append(orig_token)
      ambiguous_expanded_tokens.append(expanded_token)
  if ambiguous_orig_tokens or ambiguous_expanded_tokens:
    attribution_pairs.extend(
        TextAttributor(
            orig_tokens=ambiguous_orig_tokens,
            expanded_tokens=ambiguous_expanded_tokens,
            tokenizer=tokenizer,
            abbreviation_expansions_dict=abbreviation_expansions_dict
            ).generate_attribution_pairs())
  return attribution_pairs


def _combine_key_value_pairs_to_mapping(
    key_value_pairs
    ):
  """Combines key-value pair tuples to a mapping from key to list of values."""
  mapping = collections.defaultdict(list)
  for key, value in key_value_pairs:
    mapping[key].append(value)
  return dict(mapping)


def create_expansion_mapping(
    orig_text,
    expanded_text,
    match_score = 0.0,
    mismatch_score = 1.0,
    indel_score = 1.0,
    tokenizer = None,
    abbreviation_expansions_dict = None,
    label_abbreviation_expansions = None,
    same_first_char_score = 0.5,
    expansion_gap_score = 0.6
    ):
  """Creates a token mapping from the abbreviated text to the expanded text.

  Tokens are aligned between the original and expanded text using a modified
  Needleman-Wunsch algorithm. The aligned token pairs are then processed to
  attribute text in the expansion to the correct tokens in the original
  text. These attributions are returned as a dictionary mapping. For example,
  given the original text:
    "the pt is in pt for lbp"
  and the expanded text:
    "the patient is in physical therapy for lower back pain"
  the resulting mapping would look like the following:
    {"the": ["the"], "pt": ["patient", "physical therapy"], "is": ["is"],
     "in": ["in"], "for": ["for"], "lbp": ["lower back pain"]}

  Args:
    orig_text: The original text that potentially contains abbreviations.
    expanded_text: The text that represents an attempted expansion of the
      potential abbreviations contained in the original text.
    match_score: The score attributed to a pair of matching tokens that have
      been aligned.
    mismatch_score: The score attributed to a pair of mismatching tokens that
      have been aligned.
    indel_score: The score attributed to an insertion/deletion of a token
      between the two text sequences.
    tokenizer: The tokenizer used to break apart and join strings to and from
      tokens.
    abbreviation_expansions_dict: A dictionary mapping known abbreviations to
      their valid expansions.
    label_abbreviation_expansions: An optional mapping from each ground truth
      abbreviation in the original text to a sequence of the ground truth
      expansions for that abbreviation. This dictionary is used for scoring
      abbreviation expansion gaps by determining whether an Nth insertion after
      an abbreviation still falls within the expected expansion length. If an
      abbreviation maps to a single expansion, this expansion will be assumed to
      apply to all instances of the abbreviation. If it maps to multiple
      expansions, it should match the number of abbreviations in the original
      text.
    same_first_char_score: The optional score attributed to a pair of
      mismatching tokens that share a first character that have been aligned.
    expansion_gap_score: The optional score attributed to all insertions that
      fall within the expected gap of an abbreviation expansion. For instance,
      if an abbreviation is expected to have a 3-token expansion, the 1st token
      should be paired with the abbreviation and the following 2 token
      insertions represent the expansion gap and should be scored less harshly
      that a typical insertion.

  Returns:
    A dictionary in which each key is a unique token in the original text, and
      each value is a list of phrases from the expanded text that have been
      attributed to that unique token.
  """
  if tokenizer is None:
    abbreviation_expansions_for_tokenizer = None
    if label_abbreviation_expansions:
      abbreviation_expansions_for_tokenizer = label_abbreviation_expansions
    tokenizer = tokenizer_lib.Tokenizer(
        abbreviation_expansions_for_tokenizer)
  orig_tokens = tokenizer.tokenize(
      orig_text, string_type=tokenizer_lib.TokenizationStringType.ORIGINAL)
  expanded_tokens = tokenizer.tokenize(
      expanded_text,
      string_type=tokenizer_lib.TokenizationStringType.EXPANDED)
  alignment_pairs = text_alignment_lib.NeedlemanWunschMatrix(
      tokenizer=tokenizer,
      orig_tokens=orig_tokens,
      expanded_tokens=expanded_tokens,
      match_score=match_score,
      mismatch_score=mismatch_score,
      indel_score=indel_score,
      label_abbreviation_expansions=label_abbreviation_expansions,
      same_first_char_score=same_first_char_score,
      expansion_gap_score=expansion_gap_score).get_aligned_pairs()
  attribution_pairs = _attribute_expansions_to_original_text(
      alignment_pairs, tokenizer, abbreviation_expansions_dict)
  return _combine_key_value_pairs_to_mapping(attribution_pairs)
