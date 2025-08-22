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

"""Tokenizer using spaCy."""

from typing import Tuple

import pandas as pd
import spacy

from tide_nlp import core


_TOKEN_COLUMN_ORDER = [
    'text', 'index', 'lemma',
    'tag', 'dependencyHead.index', 'dependencyLabel',
    'pos', 'bytes.start', 'bytes.limit',
]


class SpacyTokenizer(core.Tokenizer):
  """Class for spaCy-based tokenizer."""

  def __init__(self, nlp):
    self._nlp = nlp

  def _generate_spans(self, tokens_df):
    """Generate spans from sliding windows of tokens."""
    def _span_entry(i, limit, token):
      return {
          'text': ' '.join(list(token)),
          'tokens.start': i,
          'tokens.limit': i + limit,
          'bytes.start': tokens_df.loc[
              tokens_df['token.index'] == i].iloc[0]['bytes.start'],
          'bytes.limit': tokens_df.loc[
              tokens_df['token.index'] == i+limit-1].iloc[0]['bytes.limit']
          }

    def _split_on_window(tokens, limit):
      iterators = [iter(tokens[index:]) for index in range(limit)]
      return [_span_entry(i, limit, token)
              for i, token in enumerate(zip(*iterators))
              ]

    tokens = tokens_df['text'].tolist()
    tokens_lemma = tokens_df['token.lemma'].tolist()
    size = len(tokens) + 1

    window_min = 2
    window_max = 3 + 1
    lookup_tokens = []
    lookup_tokens_lemma = []

    for i in range(min(window_min, size), min(window_max, size)):
      lookup_tokens.extend(_split_on_window(tokens, i))
      lookup_tokens_lemma.extend(_split_on_window(tokens_lemma, i))

    spans_df = pd.DataFrame.from_dict(lookup_tokens)
    spans_df_lemma = pd.DataFrame.from_dict(lookup_tokens_lemma)

    if spans_df.empty:
      return pd.DataFrame()

    spans_df = pd.merge(spans_df,
                        spans_df_lemma.rename({'text': 'text_lemma'}, axis=1),
                        on=['tokens.start',
                            'tokens.limit',
                            'bytes.start',
                            'bytes.limit'])

    spans_df = spans_df.add_prefix('span.')
    spans_df['text'] = spans_df['span.text']
    spans_df['text_lemma'] = spans_df['span.text_lemma']
    spans_df['bytes.start'] = spans_df['span.bytes.start']
    spans_df['bytes.limit'] = spans_df['span.bytes.limit']
    spans_df.drop(columns=['span.text',
                           'span.text_lemma',
                           'span.bytes.start',
                           'span.bytes.limit'],
                  inplace=True)
    spans_df['token.index'] = spans_df['span.tokens.start']

    return spans_df

  def _doc_to_df(self, doc):
    """Convert spaCy document to pandas DataFrame."""
    tokens = [
        (i.text, i.i, i.lemma_, i.tag_,
         i.head.i if i.head.i != i.i else -1, i.dep_, i.pos_, i.idx,
         i.idx + len(i)) for i in doc
    ]
    tokens_df = pd.DataFrame(tokens, columns=_TOKEN_COLUMN_ORDER)

    # NOTE: The following is a hack since TIDAL doesn't have PROPN
    # granularity in POS attributes. We need a better way to do this join
    tokens_df['pos'] = tokens_df['pos'].str.replace('PROPN', 'NOUN')

    tokens_df = tokens_df.add_prefix('token.')
    tokens_df['text'] = tokens_df['token.text']
    tokens_df['bytes.start'] = tokens_df['token.bytes.start']
    tokens_df['bytes.limit'] = tokens_df['token.bytes.limit']
    tokens_df.drop(columns=['token.text',
                            'token.bytes.start',
                            'token.bytes.limit'],
                   inplace=True)
    return tokens_df

  def tokenize(
      self,
      text,
  ):
    """Tokenize text with spaCy."""
    doc = self._nlp(text)
    tokens_df = self._doc_to_df(doc)
    if tokens_df.empty:
      return pd.DataFrame(), pd.DataFrame()

    spans_df = self._generate_spans(tokens_df)
    return tokens_df, spans_df
