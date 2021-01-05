# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Utilities for converting words to integers and vice versa."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import numpy as np


def load_vocab_list(vocab_path, add_eos=True):
  """Load the vocab list from path."""
  vocab_list = open(vocab_path).read().split()
  if vocab_list[0] != 'eos' and add_eos:
    vocab_list = ['eos'] + vocab_list
  return vocab_list


def create_look_up_table(vocab_list):
  """Create tables for encoding and decoding texts."""
  vocab2int = {word: i for i, word in enumerate(vocab_list)}
  int2vocab = vocab_list
  return vocab2int, int2vocab


def encode_text(text, lookup_table, max_sequence_length=None):
  """Encode a sentence into tokens with a lookup table.

  Args:
    text: a string to be encoded
    lookup_table: a table that maps tokens to integers
    max_sequence_length: maximum length of the token sequences

  Returns:
    encoded text
  """
  if text[-3:] != 'eos':
    text += ' eos'
  sentence = re.findall(r"[\w']+|[.,!?;]", text)
  encoded_sentence = []
  for w in sentence:
    if w == 'that': continue
    encoded_sentence.append(lookup_table[w.lower()])
  if max_sequence_length:
    return encoded_sentence[:max_sequence_length]
  else:
    return encoded_sentence


def decode_int(int_array, lookup_table, delimeter=' '):
  decoded_sentence = []
  for i in int_array:
    decoded_sentence.append(lookup_table[i])
  return delimeter.join(decoded_sentence)


def encode_text_with_lookup_table(look_up_table, max_sequence_length=None):
  return lambda text: encode_text(text, look_up_table, max_sequence_length)


def decode_with_lookup_table(look_up_table):
  return lambda int_array: decode_int(int_array, look_up_table)


def encode_bow_with_vocab_list(vocab_list):
  """Make function that encode sentence as bag of words.

  Args:
    vocab_list: all possible vocabularies

  Returns:
    a function that returns the bag of words representation
      of a text, i.e. frequency count for each word
  """
  def _encode(text):
    sentence = re.findall(r"[\w']+|[.,!?;]", text)
    bow_repr = {w: 0 for w in vocab_list}
    for w in sentence:
      if w == 'eos': continue
      bow_repr[w.lower()] += 1
    return np.float32([bow_repr[w] for w in vocab_list])
  return _encode


def decode_bow_with_vocab_list(vocab_list):
  """Make function that decode sentence represented as bag of words.

  Args:
    vocab_list: all possible vocabularies

  Returns:
    a function that returns the string representation of a bow representation
      of a text, i.e. a string with each word repeated for amount of times
      specified by the bow
  """
  def _decode(bow_repr):
    base_string = ''
    for c, w in zip(bow_repr, vocab_list):
      for _ in range(int(c)):
        base_string += w + ' '
    return base_string
  return _decode
