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

"""Annotator for person-term combinations."""

import pandas as pd

from tide_nlp import core
from tide_nlp.entity_annotator import ptc_helper as ptc


class PtcAnnotator(core.EntityAnnotator):
  """Annotator for person-term combinations."""

  def __init__(self, helper):
    self._helper = helper

  def _find_person_token(self, row, tokens_df):
    """Annotate a person token if found."""
    for idx in range(row['mention.tokens.start'], row['mention.tokens.limit']):
      token = tokens_df.loc[(tokens_df['token.index'] == idx)].iloc[0]
      if token['text'] in row['text']:
        row['ptc.person_token'] = token['token.index']
        row['ptc.person_term'] = token['text']
        return row
    return row

  def _find_identity_term_token(self, row, token_span_matches_df):
    """Annotate identity token modifiying a person token."""
    row['ptc.text'] = None
    target_idx = row['ptc.person_token']
    for _, token in token_span_matches_df.dropna(
        subset=['token.pos']
    ).iterrows():
      if (token['token.dependencyHead.index'] == target_idx and
          token['token.index'] != target_idx):
        dep_label = token['token.dependencyLabel']
        if dep_label.endswith('mod') or dep_label.endswith('nn'):
          row['ptc.identity_token'] = token['token.index']
          row['ptc.identity_term'] = token['text']
          row['ptc.text'] = token['text']
          return row
    return row

  def _find_inferred_term_token(self, row, ptc_df, tokens_df):
    """Annotate inferred identity tokens from other itdentity tokens."""
    if not row['ptc.text']:
      person_token = tokens_df.loc[
          (tokens_df['token.index'] == row['ptc.person_token'])
      ].iloc[0]
      dep_idx = person_token['token.dependencyHead.index']
      conj_ptc_df = ptc_df.loc[(ptc_df['ptc.person_token'] == dep_idx)]
      if len(conj_ptc_df) == 1:
        inferred_from_row = conj_ptc_df.iloc[0]
        if ('ptc.identity_token' in inferred_from_row and
            row['ptc.person_token'] != inferred_from_row['ptc.identity_token']):
          row['ptc.identity_token'] = inferred_from_row['ptc.identity_token']
          row['ptc.identity_term'] = inferred_from_row['ptc.identity_term']
          row['ptc.text'] = inferred_from_row['ptc.text']
          row['ptc.inferred_from_person_token'] = inferred_from_row[
              'ptc.person_token'
          ]
          row['ptc.inferred_from_person_term'] = inferred_from_row[
              'ptc.person_term'
          ]

    if row['ptc.text']:
      row['ptc.ptc_term'] = (
          str(row['ptc.identity_term']) + ' ' + str(row['ptc.person_term'])
      )
    return row

  def annotate(
      self,
      text,
      tokens_df,
      token_span_matches_df,
  ):
    """Annotate text with person-term combinations."""
    mentions_df = self._helper.get_mentions_person(text,
                                                   tokens_df,
                                                   token_span_matches_df)
    ptc_df = mentions_df.apply(
        lambda x: self._find_person_token(x, tokens_df), axis=1
    )

    ptc_df = pd.DataFrame(ptc_df)

    if 'ptc.person_token' not in ptc_df.columns:
      return pd.DataFrame()

    ptc_df = ptc_df.dropna(subset=['ptc.person_token']).apply(
        lambda x: self._find_identity_term_token(x, token_span_matches_df),
        axis=1,
    )

    inferred_ptc_df = ptc_df.apply(
        lambda x: self._find_inferred_term_token(x, ptc_df, tokens_df), axis=1
    )
    inferred_ptc_df = pd.DataFrame(inferred_ptc_df)

    if (
        inferred_ptc_df.empty
        or 'ptc.identity_token' not in inferred_ptc_df.columns
        or 'ptc.text' not in inferred_ptc_df.columns
        or 'ptc.text' not in inferred_ptc_df.dropna(subset=['ptc.text']).columns
    ):
      return pd.DataFrame()

    ptc_wo_expansion_match_df = pd.merge(
        inferred_ptc_df,
        token_span_matches_df,
        left_on=['ptc.text', 'ptc.identity_token'],
        right_on=['text', 'token.index'],
        suffixes=['_drop', ''],
    ).dropna(subset=['HasNonIdentityMeaning'])

    ptc_wo_expansion_match_df = ptc_wo_expansion_match_df[
        [
            col
            for col in ptc_wo_expansion_match_df.columns
            if not col.endswith('_drop')
        ]
    ]

    ptc_wo_expansion_match_df = ptc_wo_expansion_match_df.loc[
        ptc_wo_expansion_match_df['HasNonIdentityMeaning']]
    ptc_wo_expansion_match_df['PossibleNonIdentity'] = False

    return ptc_wo_expansion_match_df
