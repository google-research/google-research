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

"""Annotator for spaCy entity mentions."""

import pandas as pd
import spacy

from tide_nlp import core


_MENTION_COLUMN_ORDER = [
    'text', 'type', 'tokens.start', 'tokens.limit', 'bytes.start', 'bytes.limit'
]


class SpacyAnnotator(core.EntityAnnotator):
  """Annotator for spaCy entity mentions."""

  def __init__(self, nlp):
    self._nlp = nlp

  def _doc_to_df(self, doc, tokens_df):
    """Converts spaCy document to a pandas DataFrame."""
    def _mention_entry(ent):
      return (ent.text,
              ent.label_,
              ent.start,
              ent.end,
              tokens_df.loc[
                  tokens_df['token.index'] == ent.start].iloc[0]['bytes.start'],
              tokens_df.loc[
                  tokens_df['token.index'] == ent.end-1].iloc[0]['bytes.limit'])

    mentions = [_mention_entry(ent) for ent in doc.ents]
    mentions_df = pd.DataFrame(mentions, columns=_MENTION_COLUMN_ORDER)

    mentions_df = mentions_df.add_prefix('mention.')
    mentions_df['text'] = mentions_df['mention.text']
    mentions_df['bytes.start'] = mentions_df['mention.bytes.start']
    mentions_df['bytes.limit'] = mentions_df['mention.bytes.limit']
    mentions_df.drop(columns=['mention.text',
                              'mention.bytes.start',
                              'mention.bytes.limit'],
                     inplace=True)

    return mentions_df

  def annotate(
      self,
      text,
      tokens_df,
      token_span_matches_df,
  ):
    """Annotate text with spaCy entity mentions."""
    doc = self._nlp(text)
    mentions_df = self._doc_to_df(doc, tokens_df)

    if mentions_df.empty:
      return pd.DataFrame()

    keys = ['text']
    i1 = mentions_df.set_index(keys).index
    i2 = token_span_matches_df.set_index(keys).index

    mentions_df = mentions_df[i1.isin(i2)]
    mentions_df = pd.merge(
        mentions_df,
        token_span_matches_df,
        on=['text', 'bytes.start', 'bytes.limit'],
    )

    return mentions_df
