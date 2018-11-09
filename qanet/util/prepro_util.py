# coding=utf-8
# Copyright 2018 The Google Research Authors.
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

"""Utils for preprocessing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

import itertools

try:
  unicode        # Python 2
except NameError:
  unicode = str  # Python 3

# Special constants used when byte encoding
# char ids 0-255 come from utf-8 encoding bytes
# assign 256-300 to special chars
_BOS_CHAR_ID = 256  # <begin sentence>
_EOS_CHAR_ID = 257  # <end sentence>
_BOW_CHAR_ID = 258  # <begin word>
_EOW_CHAR_ID = 259  # <end word>
_PAD_CHAR_ID = 260  # <padding>

_BOS_CHAR = '<BOS>'
_EOS_CHAR = '<EOS>'
_BOW_CHAR = '<BOW>'
_EOW_CHAR = '<EOW>'
_PAD_CHAR = '<PAD>'  # Note the lack of unicode

__all__ = ['char2idx', 'generate_char_bos_eos', 'words2chars',
           'get_answer_index']


def char2idx(char):
  """Converts given character string to numeric ID.

  Args:
    char: String to encode. Must be one of the special marker strings or
      a single byte character
  Returns:
    Numeric ID
  """
  chars_map = {
      _BOS_CHAR: _BOS_CHAR_ID,
      _EOS_CHAR: _EOS_CHAR_ID,
      _BOW_CHAR: _BOW_CHAR_ID,
      _EOW_CHAR: _EOW_CHAR_ID,
      _PAD_CHAR: _PAD_CHAR_ID
  }
  if char in chars_map:
    result = chars_map[char]
  else:
    result = ord(char)
  return result + 1


def generate_char_bos_eos(num_chars_per_word):
  """Generate char sequences for bos / eos."""
  bos_seq = [_PAD_CHAR] * num_chars_per_word
  eos_seq = [_PAD_CHAR] * num_chars_per_word

  # BOS_SEQ = [<BOW>, <BOS>, <EOW>, <PAD> * N]
  bos_seq[0] = _BOW_CHAR
  bos_seq[1] = _BOS_CHAR
  bos_seq[2] = _EOW_CHAR

  # BOS_SEQ = [<BOW>, <EOS>, <EOW>, <PAD> * N]
  eos_seq[0] = _BOW_CHAR
  eos_seq[1] = _EOS_CHAR
  eos_seq[2] = _EOW_CHAR

  return bos_seq, eos_seq


def words2chars(words, num_chars_per_word, bos_seq, eos_seq):
  """Convert words to chars & add the beginning and end of sequence markers."""
  assert words
  result = [bos_seq]
  for word in words:
    word_bytes = _word2bytes(num_chars_per_word, word)
    result.append(word_bytes)
  result.append(eos_seq)

  # We have to store it as a since sequence of characters instead of
  # a sequence of lists because of tf.Example limitations.
  # So [[t, h, i, s], [i, s, PAD, PAD]]
  # is mapped to [t, h, i, s, i, s, PAD, PAD].
  # They are converted back to a list of lists upon load.
  return list(itertools.chain(*result))


def _word2bytes(num_chars_per_word, word):
  """Converts word to sequence of bytes."""
  chars = list(word.encode('utf-8', 'ignore'))
  if len(chars) > num_chars_per_word - 2:
    chars = chars[:num_chars_per_word - 2]
  chars = [_BOW_CHAR] + chars + [_EOW_CHAR]

  diff = num_chars_per_word - len(chars)
  if diff > 0:
    chars += [_PAD_CHAR] * diff
  assert len(chars) == num_chars_per_word
  return chars


def get_answer_index(context, context_words, answer_start, answer):
  """Get word-level answer index.

  Args:
    context: `unicode`, representing the context of the question.
    context_words: a list of `unicode`, tokenized context.
    answer_start: `int`, the char-level start index of the answer.
    answer: `unicode`, the answer that is substring of context.
  Returns:
    a tuple of `(word_answer_start, word_answer_end)`, representing the start
    and end indices of the answer in respect to `context_words`.
  """
  assert answer, 'Encountered length-0 answer.'
  assert isinstance(answer, unicode)
  assert isinstance(context, unicode)
  assert isinstance(context_words[0], unicode)
  answer_end = answer_start + len(answer)
  char_idxs = _tokens2idxs(context, context_words)
  word_answer_start = None
  word_answer_end = None
  for word_idx, char_idx in enumerate(char_idxs):
    if char_idx <= answer_start:
      word_answer_start = word_idx
    if char_idx < answer_end:
      word_answer_end = word_idx
  return word_answer_start, word_answer_end


def _tokens2idxs(text, tokens):
  idxs = []
  idx = 0
  for token in tokens:
    idx = text.find(token, idx)
    assert idx >= 0, (text, tokens, token)
    idxs.append(idx)
    idx += len(token)
  return idxs
