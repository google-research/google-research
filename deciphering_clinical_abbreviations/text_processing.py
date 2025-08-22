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

"""Helper functions for text processing and evaluation data generation."""

import collections
from collections.abc import Mapping, Sequence
import dataclasses
import re
from typing import Optional, Union
import numpy as np
import pandas as pd

from deciphering_clinical_abbreviations import tokenizer as tokenizer_lib

# Type aliases.
Span = tuple[int, int]
SpanList = list[Span]


@dataclasses.dataclass(order=True, frozen=True)
class ReverseSubstitution:
  expansion_start_index: int
  expansion_end_index: int
  expansion: str
  abbreviation: str
  substitute_prob: float


def generate_snippets_from_notes(notes,
                                 min_char_len = 0,
                                 max_char_len = 200):
  """Converts a Series of notes into a Series of short text snippets.

  Snippets are approximately sentences, although splitting on '. ' is a crude
  heuristic.

  Args:
    notes: pd.Series of text documents to split into snippets.
    min_char_len: minimum length of snippets to keep, in characters.
    max_char_len: maximum length of snippets to keep, in characters.

  Returns:
    pd.Series containing the split and filtered snippets.
  """

  def note_to_snippets(note):
    note_reduce_whitespace = ' '.join(note.split())
    snippets = note_reduce_whitespace.split('. ')
    snippets = [
        s for s in snippets if len(s) > min_char_len and len(s) < max_char_len
    ]
    return snippets

  snippets = notes.map(note_to_snippets).explode(ignore_index=True)
  snippets = snippets.str.lower()
  snippets = snippets.dropna().drop_duplicates()
  snippets = snippets.apply(lambda x: x.strip())
  return snippets


def find_query_spans_in_text(
    text, query_regex):
  """Finds all matches for a list of words or phrases in the given text.

  Note that this function does not perform pure substring matching; the query
  is only matched when it is a discrete word or phrase, i.e. delimited by non-
  alphanumeric characters on either side.

  Args:
    text: string in which to search for the queries.
    query_regex: A regex pattern which matches the query words or phrases to
      find in the given text.

  Returns:
    A dict mapping each query to the list of spans containing it in the text.
  """
  query_matches = collections.defaultdict(list)
  for match in query_regex.finditer(text):
    query_matches[match.group(1)].append((match.start(1), match.end(1)))
  return dict(query_matches)


def reverse_substitute(
    target_string,
    expansions,
    abbreviations_by_expansion,
    substitute_probs_for_expansion,
    seed = None):
  """Substitutes abbreviations for expansions to generate training examples.

  Note that this function only replaces one randomly chosen instance of each
  expansion in the given target string.

  Args:
    target_string: the string in which expansions should be reverse-substituted.
    expansions: the expansions to replace in target_string. Can be provided
      either as a mapping of expansions to character spans, or simply as a
      sequence of expansions (in which case this function will search for all
      corresponding spans in target_string).
    abbreviations_by_expansion: a mapping from expansions to their possible
      abbreviations.
    substitute_probs_for_expansion: a mapping from expansions to the probability
      that they should be substituted with an abbreviation in the text.
    seed: optional seed for RNG determinism.

  Returns:
    A 2-tuple containing:
      - the input string for training (with expansions replaced with
        abbreviations)
      - a dict mapping the abbreviation span in the input string to the original
        expansion that was substituted
  """
  rng = np.random.default_rng(seed=seed)
  input_string = target_string
  reverse_substitutions = []
  if isinstance(expansions, Sequence):
    expansions_re = tokenizer_lib.create_word_finder_regex(expansions)
    expansion_matches = find_query_spans_in_text(target_string, expansions_re)
  else:
    expansion_matches = expansions
  for expansion, matches in expansion_matches.items():
    abbreviation_candidates = abbreviations_by_expansion[expansion]
    abbreviation = rng.choice(abbreviation_candidates)
    query_start, query_stop = rng.choice(matches)
    substitute_prob = substitute_probs_for_expansion[expansion]
    # Ensure the selected substitution doesn't overlap with any prior chosen
    # substitutions.
    if all(query_start > substitution.expansion_end_index or
           query_stop < substitution.expansion_start_index
           for substitution in reverse_substitutions):
      reverse_substitutions.append(
          ReverseSubstitution(
              expansion_start_index=query_start,
              expansion_end_index=query_stop,
              expansion=expansion,
              abbreviation=abbreviation,
              substitute_prob=substitute_prob))
  reverse_substitutions = sorted(reverse_substitutions, reverse=True)
  expansion_by_span = {}
  # We substitute spans in reverse order to ensure the start and stop indices
  # of the span being replaced isn't affected by previous substitutions.
  # However, this does mean that previous replacement spans must be updated each
  # time a new replacement is made.
  for candidate_substitution in reverse_substitutions:
    if rng.uniform() < candidate_substitution.substitute_prob:
      substitution_start = candidate_substitution.expansion_start_index
      substitution_end = candidate_substitution.expansion_end_index
      abbrev_exp_length_diff = (len(candidate_substitution.expansion) -
                                len(candidate_substitution.abbreviation))
      expansion_by_span = {
          (start - abbrev_exp_length_diff, stop - abbrev_exp_length_diff): span
          for (start, stop), span in expansion_by_span.items()}
      new_abbrev_end = (
          substitution_start + len(candidate_substitution.abbreviation))
      expansion_by_span[(substitution_start, new_abbrev_end)] = (
          candidate_substitution.expansion)
      input_string = (
          input_string[:substitution_start] +
          candidate_substitution.abbreviation +
          input_string[substitution_end:])
  return input_string, expansion_by_span


def _spans_overlap(span1, span2):
  """Checks if two spans overlap."""
  return not (span1[1] < span2[0] or span2[1] < span1[0])


def _any_span_overlap(span, spans):
  """Checks if one span overlaps with any span from the list."""
  return any(_spans_overlap(span, candidate_span) for candidate_span in spans)


def generate_abbreviation_expansion_pair_labels(
    abbreviated_snippet_text,
    label_span_expansions):
  """Generates label abbrev-exp pairs, and identity pairs for unlabeled abbrevs.

  When more instances of a labeled abbreviation exist than there are labels for
  that abbreviation, a simple identity label is used to fill in the list.
  This allows the list of expansions produced by the model for each abbreviation
  to be compared to a equal-sized list of labels. For example,

      abbreviated_snippet_text: "The pt is here. The pt's family is waiting."
      label_span_expansions: {(4, 6): "patient"}
      expected output: {"pt": ["patient", "pt"]}

  Args:
    abbreviated_snippet_text: The abbreviated text that will be provided as
      input to the model.
    label_span_expansions: A mapping from abbreviation span to the intended
      expansion label for that span.
  Returns:
    A dictionary mapping each abbreviation to a list of expansion labels whose
      length is identical to the number of instances of that abbreviation in the
      abbreviated snippet.
  """
  span_expansions_by_abbrev = collections.defaultdict(list)
  labeled_spans = []
  for span, expansion in label_span_expansions.items():
    abbrev = abbreviated_snippet_text[span[0]:span[1]]
    span_expansions_by_abbrev[abbrev].append((span, expansion))
    labeled_spans.append(span)
  # Fill in labels for all detected abbreviations which have at least one label.
  abbrev_re = tokenizer_lib.create_word_finder_regex(
      sorted(span_expansions_by_abbrev.keys(), key=len, reverse=True))
  for abbrev_match in abbrev_re.finditer(abbreviated_snippet_text):
    abbrev = abbrev_match.group(1)
    match_span = abbrev_match.start(1), abbrev_match.end(1)
    if not _any_span_overlap(match_span, labeled_spans):
      span_expansions_by_abbrev[abbrev].append((match_span, abbrev))
  return {abbrev: [exp for (_, exp) in sorted(span_expansions)]
          for abbrev, span_expansions in span_expansions_by_abbrev.items()}
