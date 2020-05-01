# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Utility to handle tasks related to string encoding.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re
import threading

import nltk
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import tokenizer as t2t_tokenizer
import tensorflow.compat.v1 as tf  # tf

from seq2act.data_generation import create_token_vocab
from seq2act.data_generation import resources

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_enum(
    'token_type', 't2t_subtoken',
    ['simple', 'nltk_token', 't2t_subtoken', 't2t_token'],
    'The way to represent words: by using token and char or by subtoken')


embed_dict = {}

# Singleton encoder to do subtokenize, which loads vocab file only once.
# Please use _get_subtoken_encoder() to get this singleton instance.
_subtoken_encoder = None
_token_vocab = None

lock = threading.Lock()


class EmptyTextError(ValueError):
  pass


class CharPosError(ValueError):
  pass


class UnknownTokenError(ValueError):
  pass


def _get_subtoken_encoder():
  with lock:
    global _subtoken_encoder
    if not _subtoken_encoder:
      _subtoken_encoder = text_encoder.SubwordTextEncoder(
          resources.get_vocab_file())
    return _subtoken_encoder


def _get_token_vocab():
  with lock:
    global _token_vocab
    if not _token_vocab:
      _token_vocab = {}
      tokens, _, _ = create_token_vocab.read_vocab(resources.get_vocab_file())
      _token_vocab = dict(zip(tokens, range(len(tokens))))
    return _token_vocab


def subtokenize_to_ids(text):
  """Subtokenizes text string to subtoken ids according to vocabulary."""
  return _get_subtoken_encoder().encode(text)


def t2t_tokenize_to_ids(text):
  """Tokenize text string with tensor2tensor tokenizer."""
  token_vocab = _get_token_vocab()
  tokens = t2t_tokenizer.encode(text)
  token_ids = []
  for token in tokens:
    if token not in token_vocab:
      raise UnknownTokenError('Unknown token %s' % token)
    else:
      token_ids.append(token_vocab[token])
  return token_ids, tokens


stat_fix_dict = collections.defaultdict(int)


def _fix_char_position(text, start, end):
  """Fixes char position by extending the substring.

  In text_encoder.SubwordTextEncoder, alphanumeric chars vs non-alphanumeric
  will be splited as 2 different categories in token level, like:
  abc "settings" def ->
    0) abc
    1) space"
    2) settings
    3) "space
    4) def
  So if the substring specified by start/end is <"settings">, then its tokens:
    0) "
    1) settings
    2) "
  will mismatch the tokens of whole text, because <"> != <space">
  Solution is extenting the substring: if the first char is non-alphanumeric and
  the previous char is also non-alphanumeric, then move start backforward. Do
  same on the end position.

  Args:
    text: whole text.
    start: char level start position.
    end: char level end position (exclusive).
  Returns:
    start: fixed start position.
    end: fixed end position (exclusive).
  """
  original_start, original_end = start, end
  if text[start: end].strip():  # Do trim if the subtext is more than spaces
    while text[start] == ' ':
      start += 1
    while text[end-1] == ' ':
      end -= 1

  def same_category(a, b):
    return a.isalnum() and b.isalnum() or not a.isalnum() and not b.isalnum()

  while start > 0 and same_category(text[start-1], text[start]):
    start -= 1
  while end < len(text) and same_category(text[end-1], text[end]):
    end += 1

  edit_distance = abs(start - original_start) + abs(end - original_end)
  stat_fix_dict[edit_distance] += 1
  return start, end


def get_t2t_token_pos_from_char_pos(text, start, end):
  """Converts char level position to t2t token/subtoken level position.

  Example: please click "settings" app.
                         |       |
      char-level:        start   end

  Tokens: [u'please', u'click', u' "', u'settings', u'app', u'"', u'.']
             |____________________|      |
               prev tokens               curr tokens

  The start/end position of curr tokens should be (3, 4).
  '3' is calculated by counting the tokens of prev tokens.

  Args:
    text: whole text.
    start: char level start position.
    end: char level end position (exclusive).
  Returns:
    token_start, token_end: token level start/end position.
  Raises:
    ValueError: Empty token or wrong index to search in text.
  """
  if start < 0 or end > len(text):
    raise CharPosError('Position annotation out of the boundaries of text.')

  start, end = _fix_char_position(text, start, end)
  tokens, _ = tokenize_to_ids(text)
  prev, _ = tokenize_to_ids(text[0:start])
  curr, _ = tokenize_to_ids(text[start:end])

  if curr == tokens[len(prev): len(prev) + len(curr)]:
    return len(prev), len(prev) + len(curr)

  space = 1535

  # try ignore the last token(' ') of prev tokens.
  if prev[-1] == space and curr == tokens[len(prev)-1: len(prev) + len(curr)-1]:
    return len(prev)-1, len(prev) + len(curr)-1

  if text[start: end] == ' ':
    raise EmptyTextError('Single space between words will be ignored.')

  assert False, 'Fail to locate start/end positions in text'


def text_sequence_to_ids(text_seq, vocab_idx_dict):
  """Encodes list of words into word id sequence and character id sequence.

  Retrieves words' index and char's ascii code as encoding. If word is not
  contained in vocab_idx_dict, len(vocab_idx_dict) is the word's encoding
  number.

  For Example:
    vocab_idx_dict = {'hi':0, 'hello':1, 'apple':2}
    text_sequence_to_ids(['hello', 'world'], vocab_idx_dict) returns:
      word_ids = [1, 3]
      char_ids = [[104, 101, 108, 108, 111], [119, 111, 114, 108, 100]]

  Args:
    text_seq: list of words to be encoded
    vocab_idx_dict: a dictionary, keys are vocabulary, values are words' index

  Returns:
    word_ids: A 1d list of intergers, encoded word id sequence
    char_ids: A 2d list of integers, encoded char id sequence
  """
  word_ids = [
      vocab_idx_dict[word.lower()]
      if word.lower() in vocab_idx_dict else len(vocab_idx_dict)
      for word in text_seq
  ]
  char_ids = []
  for word in text_seq:
    char_ids.append([ord(ch) for ch in word.lower()])
  return word_ids, char_ids


def tokenizer_with_punctuation(origin_string):
  """Extracts tokens including punctuation from origial string."""
  tokens = nltk.word_tokenize(origin_string)

  # Note: nltk changes: left double quote to `` and right double quote to ''.
  # As we don't need this feature, so change them back to origial quotes
  tokens = ['"' if token == '``' or token == '\'\'' else token
            for token in tokens]

  result = []
  for token in tokens:
    # nltk will separate " alone, which is good. But:
    # nltk will keep ' together with neightbor word, we need split the ' in head
    # tai. If ' is in middle of a word, leave it unchanged, like n't.
    # Example:
    #   doesn't    -> 2 tokens: does, n't.
    #   'settings' -> 3 tokens: ', setting, '.
    if token == '\'':
      result.append(token)
    elif token.startswith('\'') and token.endswith('\''):
      result.extend(['\'', token[1:-1], '\''])
    elif token.startswith('\''):
      result.extend(['\'', token[1:]])
    elif token.endswith('\''):
      result.extend([token[:-1], '\''])

    # nltk keeps abbreviation like 'ok.' as single word, so split tailing dot.
    elif len(token) > 1 and token.endswith('.'):
      result.extend([token[:-1], '.'])
    else:
      result.append(token)

  # Now nltk will split https://caldav.calendar.yahoo.com to
  # 'https', ':', '//caldav.calendar.yahoo.com'
  # Combine them together:
  tokens = result
  result = []
  i = 0
  while i < len(tokens):
    if (i < len(tokens) -2 and
        tokens[i] in ['http', 'https'] and
        tokens[i+1] == ':' and
        tokens[i+2].startswith('//')):
      result.append(tokens[i] + tokens[i+1] + tokens[i+2])
      i += 3
    else:
      result.append(tokens[i])
      i += 1

  return result


def tokenizer(action_str):
  """Extracts token from action string.

  Removes punctuation, extra space and changes all words to lower cases.

  Args:
    action_str: the action string.

  Returns:
    action_str_tokens: A list of clean tokens.

  """
  action_str_no_punc = re.sub(r'[^\w\s]|\n', ' ', action_str).strip()
  tokens = action_str_no_punc.split(' ')
  action_str_tokens = [token for token in tokens if token]
  return action_str_tokens


def is_ascii_str(token_str):
  """Checks if the given token string is construced with all ascii chars.

  Args:
    token_str: A token string.

  Returns:
    A boolean to indicate if the token_str is ascii string or not.
  """
  return all(ord(token_char) < 128 for token_char in token_str)


def replace_non_ascii(text, replace_with=' '):
  """Replaces all non-ASCII chars in strinng."""
  return ''.join([i if ord(i) < 128 else replace_with for i in text])


def get_index_of_list_in_list(base_list, the_sublist,
                              start_pos=0, lookback_pivot=None):
  """Gets the start and end(exclusive) indexes of a sublist in base list.

  Examples:
    call with (['00', '.', '22', '33', '44'. '.' '66'], ['22', '33'], 3)
      raise ValueError  # Search from 3rd and never lookback.
    call with (['00', '.', '22', '33', '44'. '.' '66'], ['22', '33'], 3, '.')
      return (2, 4)  # Search from 3rd and lookback until previous dot('.')
  Args:
    base_list: list of str (or any other type), the base list.
    the_sublist: list of str (or any other type), the sublist search for.
    start_pos: the index to start search.
    lookback_pivot: string. If not None, the start_pos will be moved backforward
      until an item equal to lookback_pivot. If no previous item matchs
      lookback_pivot, start_pos will be set at the beginning of base_list.
  Returns:
    int, int: the start and end indexes(exclusive) of the sublist in base list.
  Raises:
    ValueError: when sublist not found in base list.
  """
  if lookback_pivot is not None:
    current = start_pos -1
    while current >= 0:
      if base_list[current] == lookback_pivot:
        break
      current -= 1
    start_pos = current + 1

  if not base_list or not the_sublist:
    return ValueError('Empty base_list or sublist.')
  for i in range(start_pos, len(base_list) - len(the_sublist) + 1):
    if the_sublist == base_list[i: i + len(the_sublist)]:
      return i, i + len(the_sublist)
  raise ValueError('Sublist not found in list')


def tokenize(text):
  """Totenizes text to subtext with specific granularity."""
  global embed_dict
  if FLAGS.token_type == 't2t_subtoken':
    ids = _get_subtoken_encoder().encode(text)
    return [_get_subtoken_encoder().decode([the_id]) for the_id in ids]
  else:
    assert False, 'Unknown tokenize mode'


def tokenize_to_ids(text):
  """Totenizes text to ids of subtext with specific granularity."""
  if FLAGS.token_type == 't2t_subtoken':
    ids = _get_subtoken_encoder().encode(text)
    subtokens = [_get_subtoken_encoder().decode([the_id]) for the_id in ids]
    char_ids = []
    for subtoken in subtokens:
      char_ids.append([ord(ch) for ch in subtoken.lower()])
    return ids, char_ids
  else:
    assert False, 'Unknown tokenize mode'


def get_token_pos_from_char_pos(text, start, end):
  if FLAGS.token_type == 'simple':
    raise NotImplementedError()
  elif FLAGS.token_type == 'nltk_token':
    raise NotImplementedError()
  elif FLAGS.token_type == 't2t_subtoken' or FLAGS.token_type == 't2t_token':
    return get_t2t_token_pos_from_char_pos(text, start, end)
  else:
    assert False, 'Unknown tokenize mode'
