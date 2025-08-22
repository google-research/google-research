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

"""Identity annotator implementation."""

from typing import Dict, List, Optional, Tuple

import pandas as pd

from tide_nlp import core


class IdentityAnnotator:
  """Class for annotating input string with identity context mentions based on SCR."""

  def __init__(
      self,
      lexicon,
      tokenizer = None,
      entity_annotators = None,
      non_entity_annotators = None,

  ):
    """Constructs a SocietalContextAnnotator object used to run the annotators."""

    self._lexicon = lexicon
    self._tokenizer = tokenizer
    self._entity_annotators = entity_annotators
    self._non_entity_annotators = non_entity_annotators
    self._drop_token_span_matches = tokenizer is None

  def annotate(
      self, text
  ):
    """Annotate text with identity context."""
    entity_annotations = []
    non_entity_annotations = []
    entity_annotation_df = pd.DataFrame()
    non_entity_annotation_df = pd.DataFrame()

    if self._tokenizer:
      # Tokenize
      tokens_df, spans_df = self._tokenizer.tokenize(text)

      # Match tokens with SCR
      token_span_matches_df = self._lexicon.match(tokens_df, spans_df)

      if token_span_matches_df.empty:
        return ([], [], {}, pd.DataFrame())

      # Annotate non-identity entities
      if self._non_entity_annotators:
        non_entity_annotations = [annotator.annotate(text,
                                                     tokens_df,
                                                     token_span_matches_df)
                                  for annotator in self._non_entity_annotators]
      non_entity_annotations = [x for x in non_entity_annotations
                                if not x.empty]

      # Concat non-identity annotations
      non_entity_annotation_df = (
          pd.DataFrame()
          if not non_entity_annotations
          else pd.concat(non_entity_annotations, axis=1)
      )

      if token_span_matches_df.empty:
        return ([], [], {}, pd.DataFrame())

      # Annotate identity entities
      if self._entity_annotators:
        entity_annotations = [annotator.annotate(text,
                                                 tokens_df,
                                                 token_span_matches_df)
                              for annotator in self._entity_annotators]
      entity_annotations = [x for x in entity_annotations if not x.empty]

      if self._drop_token_span_matches:
        if not entity_annotations:
          return ([], [], {}, pd.DataFrame())
      else:
        entity_annotations.append(token_span_matches_df)

      # Concat identity annotations
      entity_annotation_df = pd.concat(entity_annotations, ignore_index=True)

      annotation_df_unfiltered = pd.concat([entity_annotation_df,
                                            non_entity_annotation_df])
    else:
      annotation_df_unfiltered = self._lexicon.match_terms(text)

    if not annotation_df_unfiltered.empty:
      if self._tokenizer:
        if non_entity_annotation_df.empty:
          annotation_df = annotation_df_unfiltered
        else:
          # filter out non-identity annotations
          merge_df = pd.merge(
              annotation_df_unfiltered,
              non_entity_annotation_df,
              on=['token.index', 'token.pos'],
              how='left',
              indicator=True,
          )

          merge_df = merge_df.loc[merge_df['_merge'] == 'left_only',
                                  'token.index']

          annotation_df = entity_annotation_df[
              entity_annotation_df['token.index'].isin(merge_df)
          ]
      else:
        annotation_df = annotation_df_unfiltered

      identity_groups = annotation_df['IdentityGroup'].unique()
      identity_terms = annotation_df['text'].unique()
      identity_group_term_df = (
          annotation_df.groupby('IdentityGroup')
          .agg({'text': lambda x: list(set([y for y in x if y == y]))})
          .reset_index()
      )
      identity_group_term_dict = {
          row['IdentityGroup']: row['text']
          for _, row in identity_group_term_df.iterrows()
      }

      # TODO: implement confidence measure and don't filter out low confidence.
      annotation_tuple = (
          identity_groups,
          identity_terms,
          identity_group_term_dict,
          annotation_df_unfiltered,
      )

      return annotation_tuple

    return ([], [], {}, pd.DataFrame())

  def annotate_terms(self, text):
    """Annotate text with identity terms."""
    _, terms, _, _ = self.annotate(text)
    return terms

  def annotate_groups(self, text):
    """Annotate text with identity groups."""
    groups, _, _, _ = self.annotate(text)
    return groups
