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

"""Helper functions for aligning original abbreviated text and expanded text."""

import collections
from collections.abc import Mapping, Sequence
import dataclasses
import enum
from typing import Optional
import numpy as np
from deciphering_clinical_abbreviations import tokenizer as tokenizer_lib

# Type aliases.
AbbrevExpansionDict = Mapping[str, Sequence[str]]


# Needleman-Wusnch enums.
class NWMatrixReverseDir(enum.Enum):
  """Directions to take when reversing through a Needleman-Wunsch matrix."""
  DIAG = enum.auto()
  UP = enum.auto()
  LEFT = enum.auto()
  NOWHERE = enum.auto()


class TokenPairScoreType(enum.Enum):
  """Types of scores that can be assigned to a pair of aligned tokens."""
  MATCH = enum.auto()
  SAME_FIRST_CHAR = enum.auto()
  MISMATCH = enum.auto()
  INSERTION = enum.auto()
  EXPANSION_INSERTION = enum.auto()
  DELETION = enum.auto()


@dataclasses.dataclass
class NWMatrixCell:
  """A single Needleman-Wunsch matrix cell.

  Attributes:
    root_cell: Whether the cell is the root cell at the top left of the matrix.
    up_score: The score resulting from reaching the current token pair from
      directly above this cell, which represents a deletion of the token
      represented by the row index of the current cell.
    diag_score: The score resulting from reaching the current token pair from
      the up-left diagonal cell, which represents a pairing of the tokens
      represented by the row and column of the current cell.
    left_score: The score resulting from reaching the current token pair from
      the left, which represents an insertion of the token represented by the
      column index of the current cell.
    num_lefts: The number of consecutive lefts that must be taken to reach the
      current cell in order to produce the score stored in the 'left_score'
      attribute. For basic alignment in which insertions are treated
      identically, this will always be 1, since the cell directly to the left
      contains all the information needed. But when expansion gap scores are
      considered, and the starting position and size of the gap can result in
      different scores, a different num_lefts may result.
    diag_score_type: The type of the diag score, indicates what type of pair
      the tokens represented by the row and column of the current cell make.
    best_score: The best alignment score possible when aligning all upstream
      tokens such that there is a pairing between the tokens represented by the
      row and column of the current cell.
    best_nonleft_score: Similar to the best alignment score, except this is the
      best score that excludes inheriting from any cells directly to the left.
    best_direction_sequence: The sequence of directional steps to take through
      the alignment matrix from the current cell to reach the parent cell that
      is associated with the best alignment score for the current cell.
  """
  root_cell: bool = False
  up_score: float = np.inf
  diag_score: float = np.inf
  left_score: float = np.inf
  num_lefts: int = 0
  diag_score_type: Optional[TokenPairScoreType] = None

  @property
  def best_score(self):
    if self.root_cell: return 0.
    return np.min([self.best_nonleft_score, self.left_score])

  @property
  def best_nonleft_score(self):
    return np.min([self.up_score if self.up_score is not None else np.inf,
                   self.diag_score if self.diag_score is not None else np.inf])

  @property
  def best_direction_sequence(self):
    """Returns the best direction sequence."""

    _, best_dir_sequence = sorted(
        [(self.diag_score, [NWMatrixReverseDir.DIAG]),
         (self.left_score, [NWMatrixReverseDir.LEFT] * self.num_lefts),
         (self.up_score, [NWMatrixReverseDir.UP])],
        key=lambda x: x[0])[0]
    return best_dir_sequence


@dataclasses.dataclass
class NeedlemanWunschMatrix:
  """A class for aligning original abbreviated text with its expanded text.

  This implementation assumes that a lower alignment score is more optimal, and
  that the original tokens are lined up along the rows of the alignment matrix
  and the expanded tokens are lined up along the columns. The complexity of
  computing an alignment between an original token sequence of length n and a
  expanded sequence of length m is O(n*m). With the expansion_gap_score, this
  complexity increases to O(n*m^2).

  Attributes:
    tokenizer: The tokenizer used to break apart and join strings to and from
      tokens.
    orig_tokens: The tokens from the original text that potentially contain
      abbreviations.
    expanded_tokens: The tokens from the expanded text that potentially contain
      attempted expansions of each of the abbreviations contained in the
      original text.
    match_score: The score attributed to a pair of matching tokens that have
      been aligned.
    mismatch_score: The score attributed to a pair of mismatching tokens that
      have been aligned.
    indel_score: The score attributed to an insertion/deletion of a token
      between the two text sequences.
    label_abbreviation_expansions: An optional dictionary from each ground truth
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
  """
  tokenizer: tokenizer_lib.Tokenizer
  orig_tokens: Sequence[str]
  expanded_tokens: Sequence[str]
  # Main score settings.
  match_score: float
  mismatch_score: float
  indel_score: float
  # Optional abbreviation expansion dict.
  label_abbreviation_expansions: (
      Optional[AbbrevExpansionDict]) = None
  # Auxiliary score settings.
  same_first_char_score: Optional[float] = None
  expansion_gap_score: Optional[float] = None

  _nw_matrix: Mapping[tuple[int, int], NWMatrixCell] = dataclasses.field(
      init=False)
  _total_rows: int = dataclasses.field(init=False)
  _total_cols: int = dataclasses.field(init=False)
  _expected_expansion_gapsizes: Sequence[int] = dataclasses.field(init=False)

  def __post_init__(self):
    if ((self.expansion_gap_score is None) !=
        (self.label_abbreviation_expansions is None)):
      raise ValueError(
          "Both label_abbreviation_expansions and expansion_gap_score must "
          "either be provided or omitted.")
    self._populate_expected_expansion_gapsizes()
    self._nw_matrix = {}
    self._total_rows = len(self.orig_tokens) + 1
    self._total_cols = len(self.expanded_tokens) + 1
    for row in range(self._total_rows):
      for col in range(self._total_cols):
        self._nw_matrix[row, col] = self._compute_matrix_cell(row, col)

  def get_alignment_score(self):
    """Returns the optimal score for the entire sequence alignment."""
    return (self._nw_matrix[(self._total_rows - 1, self._total_cols - 1)]
            .best_score)

  def _populate_expected_expansion_gapsizes(self):
    """Collects the expansion lengths for each token in the unexpanded text."""
    self._expected_expansion_gapsizes = [0] * len(self.orig_tokens)
    if self.expansion_gap_score is None:
      return
    expansion_indices = collections.defaultdict(int)
    for i, token in enumerate(self.orig_tokens):
      if token not in self.label_abbreviation_expansions:
        continue
      expansion_list = self.label_abbreviation_expansions[token]
      try:
        expansion = expansion_list[expansion_indices[token]]
      except IndexError as ie:
        raise IndexError(
            f"There are more instances of abbreviation '{token}' found in the "
            "text than there are expansions listed for that abbreviation in "
            "label_abbreviation_expansions.") from ie
      # Since first expansion token will be aligned with the abbreviation,
      # expansion gap should be len(expansion tokens) - 1
      expanded_tokens = self.tokenizer.tokenize(
          expansion,
          string_type=tokenizer_lib.TokenizationStringType.EXPANDED)
      self._expected_expansion_gapsizes[i] = len(expanded_tokens) - 1
      if len(expansion_list) > 1:
        expansion_indices[token] += 1
    for abbrev, expansion_index in expansion_indices.items():
      expansions_in_dict = self.label_abbreviation_expansions[abbrev]
      if (expansion_index > 0 and
          expansion_index < len(expansions_in_dict)):
        raise ValueError(
            f"There are less instances of abbreviation '{abbrev}' found in the "
            "text than there are expansions listed for that abbreviation in "
            "label_abbreviation_expansions.")

  def _compute_diag_score_and_type(
      self, row, col):
    """Compute the diagonal score and type for this matrix cell.

    The diagonal score is the score attributed to reaching the current token
    pair from the up-left diagonal cell, which entails a pairing of the tokens
    represented by the row and column of the current cell. As such, this score
    depends on the comparison between these tokens. The result of this
    comparison is captured by the returned score type.

    Args:
      row: The row index of the current matrix cell.
      col: The column index of the current matrix cell.
    Returns:
      A tuple containing the diagonal score and the score type.
    """
    row_token, col_token = (
        self.orig_tokens[row - 1], self.expanded_tokens[col - 1])
    if row_token == col_token:
      diag_type = TokenPairScoreType.MATCH
      diag_score = self.match_score
    elif (self.same_first_char_score is not None
          and row_token[0] == col_token[0]):
      diag_type = TokenPairScoreType.SAME_FIRST_CHAR
      diag_score = self.same_first_char_score
    else:
      diag_type = TokenPairScoreType.MISMATCH
      diag_score = self.mismatch_score
    diag_score = (
        self._nw_matrix[row - 1, col - 1].best_score + diag_score)
    return diag_score, diag_type

  def _score_lefts_from_gap_start(
      self,
      num_lefts,
      expected_gapsize):
    """Computes the score for consecutive lefts taken from a gap start cell."""
    num_expansion_steps = min(expected_gapsize, num_lefts)
    num_other_steps = num_lefts - num_expansion_steps
    abbrev_expansion_gap_score = self.expansion_gap_score * num_expansion_steps
    other_insert_score = self.indel_score * num_other_steps
    return abbrev_expansion_gap_score + other_insert_score

  def _compute_best_left_score_and_num_lefts(
      self, row, col):
    """Computes the best scoring leftward movement starting at this cell."""
    expected_gapsize = self._expected_expansion_gapsizes[row - 1]
    # If there is no expected gapsize, or no expansion_gap_score was
    # provided, or we are still aligning the first token (gap must follow
    # aligned token), return standard left score and num lefts of 1.
    if ((not expected_gapsize) or
        (self.expansion_gap_score is None) or
        col == 1):
      return self._nw_matrix[row, col - 1].best_score + self.indel_score, 1
    left_score_by_num_lefts = {}
    for num_lefts in range(1, col):
      left_score_by_num_lefts[num_lefts] = (
          self._nw_matrix[row, col - num_lefts].best_nonleft_score +
          self._score_lefts_from_gap_start(num_lefts, expected_gapsize))
    numlefts, best_left_score = sorted(
        left_score_by_num_lefts.items(), key=lambda x: (x[1], -x[0]))[0]
    return best_left_score, numlefts

  def _compute_matrix_cell(
      self, row, col):
    """Computes a cell including score and direction information."""
    if row == 0 and col == 0:
      return NWMatrixCell(root_cell=True)
    elif col == 0:
      return NWMatrixCell(up_score=self.indel_score * row)
    elif row == 0:
      return NWMatrixCell(left_score=self.indel_score * col, num_lefts=1)
    diag_score, diag_score_type = self._compute_diag_score_and_type(row, col)
    up_score = self._nw_matrix[row - 1, col].best_score + self.indel_score
    left_score, num_lefts = (
        self._compute_best_left_score_and_num_lefts(row, col))
    return NWMatrixCell(
        root_cell=False,
        up_score=up_score,
        diag_score=diag_score,
        diag_score_type=diag_score_type,
        left_score=left_score,
        num_lefts=num_lefts)

  def get_aligned_pairs(self):
    """Generates a sequence of aligned token pairs and their score types."""
    row, col = self._total_rows - 1, self._total_cols - 1
    token_pairs = []
    while row != 0 or col != 0:
      matrix_cell = self._nw_matrix[row, col]
      best_direction_sequence = matrix_cell.best_direction_sequence
      seq_len = len(best_direction_sequence)
      expected_expansion_len = self._expected_expansion_gapsizes[row - 1] + 1
      for i, direction in enumerate(best_direction_sequence):
        if direction == NWMatrixReverseDir.DIAG:
          token_pairs.insert(0, (
              self.orig_tokens[row - 1],
              self.expanded_tokens[col - 1],
              matrix_cell.diag_score_type))
          row -= 1
          col -= 1
        elif direction == NWMatrixReverseDir.LEFT:
          score_type = (
              TokenPairScoreType.EXPANSION_INSERTION if
              (seq_len - i < expected_expansion_len)
              else TokenPairScoreType.INSERTION)
          token_pairs.insert(0, (
              "", self.expanded_tokens[col - 1], score_type))
          col -= 1
        elif direction == NWMatrixReverseDir.UP:
          token_pairs.insert(0, (
              self.orig_tokens[row - 1], "", TokenPairScoreType.DELETION
              ))
          row -= 1
    return token_pairs
