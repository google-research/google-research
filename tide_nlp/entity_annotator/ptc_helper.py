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

"""Person-mention helper."""

from typing import Optional

import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import pandas as pd
import spacy


nltk.download('stopwords')
nltk.download('wordnet')


_MENTION_COLUMN_ORDER = [
    'text', 'type', 'tokens.start', 'tokens.limit', 'bytes.start', 'bytes.limit'
]
_SPACY_PERSON_TYPES = [
    'PERSON', 'NORP', 'GPE'
]
_PERSON_TERMS = ['person', 'people', 'life']
_NONPERSON_TERMS = ['animal', 'object', 'thing', 'plant',
                    'nonhuman', 'idea', 'abstract', 'concept']
_POSSIBLE_PERSON_TYPES = ['attribute', 'group', 'object',
                          'Tops', 'body', 'state']


class PersonMentionHelper:
  """Helper for person mentions."""

  def __init__(self,
               nlp,
               person_lexicon_df = None,
               use_nltk_similarity = False):
    self._nlp = nlp
    self._use_nltk_similarity = use_nltk_similarity
    if self._use_nltk_similarity:
      self._stopwords = stopwords.words('english')
    else:
      if person_lexicon_df is None:
        raise ValueError('either person_lexicon_df or '
                         'use_nltk_similarity must be specified')
      self._person_nouns = person_lexicon_df['noun'].to_list()

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
    return pd.DataFrame(mentions, columns=_MENTION_COLUMN_ORDER)

  def _lexicon_mentions(self, text, tokens_df):
    """Retrieves person mentions from text using person lexicon."""
    mentions = []
    for noun in self._person_nouns:
      doc = self._nlp(noun)
      tokens = [i.text for i in doc]
      if ((tokens_df['text'].eq(tokens[0]).any() and
           tokens_df['text'].eq(tokens[-1]).any()) or
          (tokens_df['token.lemma'].eq(tokens[0]).any() and
           tokens_df['token.lemma'].eq(tokens[-1]).any())):
        start = tokens_df.loc[
            (tokens_df['text'] == tokens[0]) |
            (tokens_df['token.lemma'] == tokens[0])].iloc[0]
        end = tokens_df.loc[
            (tokens_df['text'] == tokens[-1]) |
            (tokens_df['token.lemma'] == tokens[-1])].iloc[0]
        mentions.append((text[start['bytes.start']:end['bytes.limit']],
                         'PERSON_LEXICON',
                         start['token.index'],
                         end['token.index'] + 1,
                         start['bytes.start'],
                         end['bytes.limit']))
    return pd.DataFrame(mentions, columns=_MENTION_COLUMN_ORDER)

  def _is_similar(self, text):
    """Checks if text is similar to wordnet person definitions."""
    lemma = text
    for token in self._nlp(text):
      lemma = token.lemma_
      break
    words = wordnet.synsets(text)
    if not words:
      return False
    words_lemma = wordnet.synsets(lemma)
    person_sim = [wordnet.path_similarity(words[0], wordnet.synsets(word_)[0])
                  for word_ in _PERSON_TERMS]
    non_person_sim = [wordnet.path_similarity(words[0],
                                              wordnet.synsets(word_)[0])
                      for word_ in _NONPERSON_TERMS]

    return (str(words[0].lexname()).startswith('noun') and
            ('noun.person' in str(words[0].lexname()) or
             'noun.person' in str(words_lemma[0].lexname()) or
             (any(['noun.%s' % word_ in str(words[0].lexname())
                   for word_ in _POSSIBLE_PERSON_TYPES]) and
              max(person_sim) >= max(non_person_sim))))

  def _similarity_mentions(self, tokens_df):
    """Retrieves person mentions from text using similarity checks."""
    mentions = []
    for _, row in tokens_df.iterrows():
      if row['text'] not in self._stopwords and self._is_similar(row['text']):
        mentions.append((row['text'],
                         'PERSON_SIMILARITY',
                         row['token.index'],
                         row['token.index'] + 1,
                         row['bytes.start'],
                         row['bytes.limit']))
    return pd.DataFrame(mentions, columns=_MENTION_COLUMN_ORDER)

  def _convert_identity_matches(self, token_span_matches_df):
    """Converts token and span matches to mentions."""
    mentions = []
    for _, row in token_span_matches_df.iterrows():
      if 'token.pos' in row.dropna():
        mentions.append((row['text'],
                         'IDENTITY_LEXICON',
                         row['token.index'],
                         row['token.index'] + 1,
                         row['bytes.start'],
                         row['bytes.limit']))
    return pd.DataFrame(mentions, columns=_MENTION_COLUMN_ORDER)

  def get_mentions_person(self, text, tokens_df,
                          token_span_matches_df):
    """Retrieves person mentions from text."""
    doc = self._nlp(text)
    dfs = []
    spacy_df = self._doc_to_df(doc, tokens_df)
    spacy_person_df = spacy_df[spacy_df['type'].isin(_SPACY_PERSON_TYPES)]
    dfs.append(spacy_person_df)

    if self._use_nltk_similarity:
      similarity_mentions_person_df = self._similarity_mentions(tokens_df)
      dfs.append(similarity_mentions_person_df)
    else:
      lexicon_mentions_person_df = self._lexicon_mentions(text, tokens_df)
      dfs.append(lexicon_mentions_person_df)

    dfs.append(self._convert_identity_matches(token_span_matches_df))

    mentions_df = pd.concat(dfs)

    mentions_df = mentions_df.add_prefix('mention.')
    mentions_df['text'] = mentions_df['mention.text']
    mentions_df['bytes.start'] = mentions_df['mention.bytes.start']
    mentions_df['bytes.limit'] = mentions_df['mention.bytes.limit']
    mentions_df.drop(columns=['mention.text',
                              'mention.bytes.start',
                              'mention.bytes.limit'],
                     inplace=True)
    return mentions_df
