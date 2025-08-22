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

"""Lexicon from TIDAL."""

import re
from typing import Optional

import pandas as pd

from tide_nlp import core


class TidalLexicon(core.Lexicon):
  """Lexicon for the TIDAL dataset."""

  def __init__(self,
               lexicon_df,
               use_lemma = False,
               use_root_terms = False,
               nonidentity_ambiguous_subset = False,
               ):
    self._data = lexicon_df[lexicon_df.apply(
        lambda x: x['IdentityGroup_Connotation_ConvergenceScore'] > 0.5,
        axis=1)]
    self._use_lemma = use_lemma
    self._use_root_terms = use_root_terms

    # If set, keep only terms which also have a non-identity meaning
    if nonidentity_ambiguous_subset:
      self._data = self._data.loc[self._data['HasNonIdentityMeaning']]

    # If set, keep only RootTerms in the Lexicon
    if use_root_terms:
      self._data = self._data.loc[self._data['IsRootTerm']]

    self._data_ptc = self._data.dropna(subset=['IsPTCTerm'])
    self._data_ptc = self._data_ptc.loc[self._data_ptc['IsPTCTerm']]

    self._regex_pattern = re.compile('(%s)' % '|'.join(
        self._data['Term'].tolist()))

  def match(
      self, tokens_df, spans_df
  ):
    token_match_df = pd.merge(
        tokens_df,
        self._data,
        left_on=['text'],
        right_on=['Term'],
    )
    span_match_df = pd.DataFrame()
    if not spans_df.empty:
      span_match_df = pd.merge(
          spans_df,
          self._data,
          left_on=['text'],
          right_on=['Term'],
      )

    if self._use_lemma:
      token_match_lemma_df = pd.merge(
          tokens_df.loc[tokens_df['text'] != tokens_df['token.lemma']],
          self._data,
          left_on=['token.lemma'],
          right_on=['Term'],
      )
      token_match_df = pd.concat([token_match_df, token_match_lemma_df])

      span_match_lemma_df = pd.DataFrame()
      if not spans_df.empty:
        span_match_lemma_df = pd.merge(
            spans_df.loc[spans_df['text'] != spans_df['text_lemma']],
            self._data,
            left_on=['text_lemma'],
            right_on=['Term'],
        )
      span_match_df = pd.concat([span_match_df, span_match_lemma_df])

    cols_wo_ambiguous = list(set(span_match_df.columns) -
                             set(['HasNonIdentityMeaning']))
    span_match_df = span_match_df[cols_wo_ambiguous]

    if span_match_df.empty:
      token_span_match_df = token_match_df
    else:
      token_span_match_df = pd.concat([token_match_df, span_match_df])

    if token_span_match_df.empty:
      return pd.DataFrame()

    token_span_match_df = token_span_match_df.groupby(
        list(set(token_span_match_df.columns) - set(['Connotation'])),
        as_index=False,
        dropna=False,
    ).agg({'Connotation': tuple})

    return token_span_match_df

  def match_terms(self, text):
    annotations = list(set(self._regex_pattern.findall(text)))
    if annotations:
      return pd.merge(
          pd.DataFrame(annotations, columns=['text']),
          self._data,
          left_on=['text'],
          right_on=['Term'],
      )
    return pd.DataFrame()
