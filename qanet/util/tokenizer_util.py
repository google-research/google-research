# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

# -*- coding: utf-8 -*
"""SQuAD tokenizer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tempfile

import nltk

import tensorflow.compat.v1 as tf
from tensor2tensor.data_generators import text_encoder

__all__ = ['NltkAndPunctTokenizer', 'clean_text', 'convert_to_spans',
           'get_answer']


# New Tokenizer
extra_split_chars = (u'-', u'£', u'€', u'¥', u'¢', u'₹', u'\u2212', u'\u2014',
                     u'\u2013', u'/', u'~', u'"', u"'", u'\ud01C', u'\u2019',
                     u'\u201D', u'\u2018', u'\u00B0')
extra_split_tokens = (
    u'``',
    # dashes w/o a preceding or following dash, so --wow--- -> --- wow ---
    u'(?<=[^_])_(?=[^_])',
    u"''",
    u'[' + u''.join(extra_split_chars) + ']')
extra_split_chars_re = re.compile(u'(' + u'|'.join(extra_split_tokens) + u')')
double_quote_re = re.compile(u"\"|``|''")
space_re = re.compile(u'[ \u202f]')

_replaced_tokens = [u'"', u'``', u'\'\'', u'\u2212', u'-', u'\u2014', u'\u2013']


def flatten_iterable(listoflists):
  return [item for sublist in listoflists for item in sublist]


def post_split_tokens(tokens):
  """Apply a small amount of extra splitting to the given tokens.

  This is in particular to avoid UNK tokens due to contraction, quotation, or
  other forms of puncutation.  Avoids some common UNKs in SQuAD/TriviaQA

  Args:
    tokens: A list of tokens to split further

  Returns:
    List of tokens
  """
  return flatten_iterable(
      [x for x in extra_split_chars_re.split(token) if x] for token in tokens)


def clean_text(word, old_method=True):
  """Quote normalization; replace u2014 and u2212."""
  if old_method:
    # NOTE(thangluong): this version doesn't work as intended in Python 2.
    #   '\u2014' is different than the Unicode symbol u'\u2014'
    # docqa code might have worked in Python 3
    #   https://github.com/allenai/document-qa/blob/master/docqa/data_processing/text_utils.py#L127
    return word.replace("''", "\"").replace('``', "\"").replace(
        '\u2212', '-').replace('\u2014', '\u2013')
  else:
    return word.replace(u"''", u"\"").replace(u'``', u"\"").replace(
        u'\u2212', u'-').replace(u'\u2014', u'\u2013')




def get_answer(context, context_words, word_answer_start, word_answer_end,
               has_answer=True, is_byte=False):
  """Get answer given context, context_words, and span.

  Args:
    context: an untokenized context string.
    context_words: a list of tokenized context words.
    word_answer_start: An int for word-level answer start.
    word_answer_end: An int for word-level answer end.
    has_answer: whether there is an answer for this example (for SQuAD 2.0).
    is_byte: if True, we expect context and each element in context_words to be
        a list of bytes which need to be decoded with utf-8 to become string.
  Returns:
    An answer string or a list of bytes (encoded with utf-8) if is_byte=True
  """
  if not has_answer:
    return ''

  if is_byte:
    context = context.decode('utf-8')
    context_words = [word.decode('utf-8') for word in context_words]

  spans = convert_to_spans(raw_text=context, tokens=context_words)
  start_char = spans[word_answer_start][0]
  end_char = spans[word_answer_end][1]
  answer = context[start_char:end_char]
  if is_byte:
    answer = answer.encode('utf-8')
  return answer


def convert_to_spans(raw_text, tokens):
  """Convert tokenized version of `raw_text` to character spans into it.

  Args:
    raw_text: untokenized text, e.g., "Convert to spans."
    tokens: tokenized list of words, e.g., ["Convert", "to", "spans", "."]

  Returns:
    spans: list of char-level (start, end) indices of tokens in raw_text
      e.g., [(0, 7), (8, 10), (11, 16), (16, 17)]

  Raises:
    ValueError: if there is a non-empty token that has length 0.
  """
  cur_char_pos = 0
  spans = []
  for token in tokens:
    # Tokenizer might transform double quotes, so consider multiple cases
    if double_quote_re.match(token):
      span = double_quote_re.search(raw_text[cur_char_pos:])
      cur_char_pos = cur_char_pos + span.start()
      token_len = span.end() - span.start()
    else:
      if token in _replaced_tokens:
        # Handle case where we did some replacement after tokenization
        # e.g. \u2014 --> \u2013
        best = sys.maxsize
        found_token = None
        for replaced_token in [token] + _replaced_tokens:
          found = raw_text.find(replaced_token, cur_char_pos)
          if found >= 0 and found < best:
            found_token = replaced_token
            best = found
        assert found_token is not None

        token_len = len(found_token)
        cur_char_pos = best
      else:
        cur_char_pos = raw_text.find(token, cur_char_pos)
        token_len = len(token)

    if token and not token_len:
      raise ValueError('Non-empty toten %s with zero length' % token)

    # Add to span
    spans.append((cur_char_pos, cur_char_pos + token_len))
    cur_char_pos += token_len

  return spans


# Utilities for dealing with subwords.
def partition_subtokens(subtokens):
  """Return list of (start, end , [list of subtokens]) for each token."""
  words = []
  last = 0
  for i, token in enumerate(subtokens):
    if token[-1] == '_':
      words.append((last, i + 1, subtokens[last:i + 1]))
      last = i + 1
  return words


def match_subtokens_to_string(context, subtokens):
  """For each subtoken, return a start and end index into context bytes."""
  words = partition_subtokens(subtokens)
  idx = 0
  per_word_spans = []
  for _, _, word in words:
    substring = text_encoder._unescape_token(''.join(word))
    if u'<pad>_' in word:
      break
    found_idx = context.find(substring, idx)
    if found_idx < 0:
      raise ValueError(
          'could not find %s: %s | %s' % (substring, subtokens, context))
    span = (found_idx, found_idx + len(substring))
    # copy the span n times, where n = number of subtokens in this token
    per_word_spans.extend([span] * len(word))
    idx = found_idx

  return per_word_spans
