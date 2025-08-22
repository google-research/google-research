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

"""Annotator for non-identity entities using person-term combinations."""

import pandas as pd


from tide_nlp import core
from tide_nlp.entity_annotator import ptc_helper as ptc


class NonPtcAnnotator(core.EntityAnnotator):
  """Annotator for non-ptc entity mentions."""

  def __init__(self, helper):
    self._helper = helper

  def annotate(
      self,
      text,
      tokens_df,
      token_span_matches_df,
  ):
    """Annotate text with non-identity entities using person-term combinations."""
    mentions_df = self._helper.get_mentions_person(text,
                                                   tokens_df,
                                                   token_span_matches_df)

    filtered_matches_df = token_span_matches_df.dropna(
        subset=['token.dependencyLabel']
    )

    if filtered_matches_df.empty:
      return pd.DataFrame()

    possible_unambiguous_non_identity_df = filtered_matches_df[
        (
            (filtered_matches_df['token.dependencyLabel'].str.endswith('mod'))
            & (filtered_matches_df['token.pos'] == 'ADJ')
            & (filtered_matches_df['token.dependencyHead.index'] !=
               filtered_matches_df['token.index'])
            & (filtered_matches_df['HasNonIdentityMeaning'])
        )
    ]

    unambiguous_non_identity_df = possible_unambiguous_non_identity_df[
        possible_unambiguous_non_identity_df.apply(
            lambda x: self._unambiguous_non_identity(x, tokens_df, mentions_df),
            axis=1,
        )
    ]
    unambiguous_non_identity_df['PossibleNonIdentity'] = True
    return unambiguous_non_identity_df

  def _unambiguous_non_identity(self, row, tokens_df, mentions_person_df):
    """Checks if a row has an unambiguous non-identity term."""
    potential_person_row = tokens_df[
        (tokens_df['token.index'] == row['token.dependencyHead.index'])
    ].iloc[0]
    if not mentions_person_df.empty:
      for _, person_row in mentions_person_df.iterrows():
        if potential_person_row['text'] in person_row['text']:
          return False

    return True
