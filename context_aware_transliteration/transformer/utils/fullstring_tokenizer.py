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

# pylint: disable=unused-variable
# pylint: disable=unused-argument
# pylint: disable=line-too-long
"""Defines Subtokenizer class to encode and decode strings."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import unicodedata

from absl import logging
import numpy as np
import six
from six.moves import xrange
import tensorflow as tf
import tensorflow_text as text
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

PAD = "<pad>"
PAD_ID = 0
EOS = "<EOS>"
EOS_ID = 1
UNK = "<UNK>"
UNK_ID = 2
RESERVED_TOKENS = [PAD, EOS, UNK]


def alphanumeric_char_set():
  return set(
      six.unichr(i)
      for i in xrange(sys.maxunicode)
      if (
          unicodedata.category(six.unichr(i)).startswith("L")
          or unicodedata.category(six.unichr(i)).startswith("N")
      )
  )


# Set contains all letter and number characters.
_ALPHANUMERIC_CHAR_SET = alphanumeric_char_set()

# min_count is the minimum number of times a subtoken must appear in the data
# before before it is added to the vocabulary. The value is found using binary
# search to obtain the target vocabulary size.
_MIN_MIN_COUNT = 1  # min value to use when binary searching for min_count
_MAX_MIN_COUNT = 1000  # max value to use when binary searching for min_count


class Subtokenizer(object):
  """Encodes and decodes strings to/from integer IDs."""

  def __init__(self, vocab_file, reserved_tokens=None, master_char_set=None):
    """Initializes class, creating a vocab file if data_files is provided."""
    logging.info("Initializing Subtokenizer from file %s.", vocab_file)

    if master_char_set is None:
      master_char_set = _ALPHANUMERIC_CHAR_SET

    if reserved_tokens is None:
      reserved_tokens = RESERVED_TOKENS

    self.subtoken_list = _load_vocab_file(vocab_file, reserved_tokens)
    self.tokenizer = text.BertTokenizer(vocab_file, lower_case=False)

    self.char2idx = {}
    self.idx2char = {}
    for i, s in enumerate(self.subtoken_list):
      self.char2idx[s] = i
      self.idx2char[i] = s

  @staticmethod
  def init_from_files(
      vocab_file,
      files,
      target_vocab_size,
      threshold,
      min_count=None,
      file_byte_limit=1e6,
      reserved_tokens=None,
      correct_strip=True,
      master_char_set=None,
  ):
    """Create subtoken vocabulary based on files, and save vocab to file.

    Args:
      vocab_file: String name of vocab file to store subtoken vocabulary.
      files: List of file paths that will be used to generate vocabulary.
      target_vocab_size: target vocabulary size to generate.
      threshold: int threshold of vocabulary size to accept.
      min_count: int minimum count to use for generating the vocabulary. The min
        count is the minimum number of times a subtoken should appear in the
        files before it is added to the vocabulary. If set to none, this value
        is found using binary search.
      file_byte_limit: (Default 1e6) Maximum number of bytes of sample text that
        will be drawn from the files.
      reserved_tokens: List of string tokens that are guaranteed to be at the
        beginning of the subtoken vocabulary list.
      correct_strip: Whether to convert text to unicode before strip.
      master_char_set: the char set.

    Returns:
      Subtokenizer object
    """
    if master_char_set is None:
      master_char_set = _ALPHANUMERIC_CHAR_SET
    if reserved_tokens is None:
      reserved_tokens = RESERVED_TOKENS

    if tf.io.gfile.exists(vocab_file):
      logging.info("Vocab file already exists (%s)", vocab_file)
    else:
      logging.info("Begin steps to create subtoken vocabulary...")

      bert_tokenizer_params = dict(lower_case=False)
      bert_vocab_args = dict(
          # The target vocabulary size
          vocab_size=target_vocab_size,
          # Reserved tokens that must be included in the vocabulary
          reserved_tokens=reserved_tokens,
          # Arguments for `text.BertTokenizer`
          bert_tokenizer_params=bert_tokenizer_params,
          # Arguments for `wordpiece_vocab.wordpiece_tokenizer_learner_lib.learn`
          learn_params={},
      )

      dataset = tf.data.TextLineDataset(files)

      subtoken_list = bert_vocab.bert_vocab_from_dataset(
          dataset.batch(1000).prefetch(2), **bert_vocab_args
      )

      logging.info(
          "Generated vocabulary with %d subtokens.", len(subtoken_list)
      )
      _save_vocab_file(vocab_file, subtoken_list)
    return Subtokenizer(vocab_file, master_char_set=master_char_set)

  def encode(self, raw_string, add_eos=False):
    """Encodes a string into a list of int subtoken ids."""
    ret = self.tokenizer.tokenize([raw_string])
    ret = ret.merge_dims(-2, -1).to_list()[0]
    if add_eos:
      assert (
          EOS in self.subtoken_list
      ), "Can't append 'EOS' because it is not in list of known subtokens."
      ret.append(EOS_ID)
    return ret

  def decode(self, subtokens):
    """Converts list of int subtokens ids into a string."""
    if isinstance(subtokens, np.ndarray):
      # Note that list(subtokens) converts subtokens to a python list, but the
      # items remain as np.int32. This converts both the array and its items.
      subtokens = subtokens.tolist()

    if not subtokens:
      return ""

    assert isinstance(subtokens, list) and isinstance(
        subtokens[0], int
    ), "Subtokens argument passed into decode() must be a list of integers."

    words = self.tokenizer.detokenize([subtokens])
    result = tf.strings.reduce_join(words, separator=" ", axis=-1)
    result = result.numpy()[0].decode("utf8")
    # print("RTYPE:", type(result), result)
    # result = native_to_unicode(result)
    # print("RTYPE2:", type(result), result)

    return result


def _save_vocab_file(vocab_file, subtoken_list):
  """Save subtokens to file."""
  with tf.io.gfile.GFile(vocab_file, mode="w") as f:
    for subtoken in subtoken_list:
      f.write(subtoken + "\n")


def _load_vocab_file(vocab_file, reserved_tokens=None):
  """Load vocabulary while ensuring reserved tokens are at the top."""
  if reserved_tokens is None:
    reserved_tokens = RESERVED_TOKENS

  subtoken_list = []
  with tf.io.gfile.GFile(vocab_file, mode="r") as f:
    for line in f:
      subtoken = line.strip()
      if subtoken in reserved_tokens:
        continue
      subtoken_list.append(subtoken)
  return reserved_tokens + subtoken_list


def native_to_unicode(s):
  """Convert string to unicode (required in Python 2)."""
  try:  # Python 2
    return s if isinstance(s, unicode) else s.decode("utf-8")  # pylint: disable=undefined-variable
  except NameError:  # Python 3
    return s
