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

"""BERT utils.

Originally from
https://github.com/google-research/language/blob/master/language/orqa/utils/bert_utils.py
Modified not to have to re-load a module we've already loaded with TF2.
"""
import logging
from typing import Any, Dict

from bert import tokenization  # pytype: disable=import-error
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
import tensorflow_text as tf_text


LOGGER = logging.getLogger(__name__)

_DELIM_REGEX = [
    r"\s+",
    r"|".join([
        r"[!-/]",
        r"[:-@]",
        r"[\[-`]",
        r"[{-~]",
        r"[\p{P}]",
    ]),
    r"|".join([
        r"[\x{4E00}-\x{9FFF}]",
        r"[\x{3400}-\x{4DBF}]",
        r"[\x{20000}-\x{2A6DF}]",
        r"[\x{2A700}-\x{2B73F}]",
        r"[\x{2B740}-\x{2B81F}]",
        r"[\x{2B820}-\x{2CEAF}]",
        r"[\x{F900}-\x{FAFF}]",
        r"[\x{2F800}-\x{2FA1F}]",
    ]),
]

_DELIM_REGEX_PATTERN = "|".join(_DELIM_REGEX)


def get_tokenization_info(module_handle):
  """Loads the `tokenization_info` object from the tf-Hub module.
  """
  with tf.Graph().as_default():
    bert_module = hub.Module(module_handle)
    with tf.Session() as sess:
      return sess.run(bert_module(signature="tokenization_info", as_dict=True))


def get_tokenizer(module_handle):
  """Creates the BERT tokenizer.
  """
  tokenization_info = get_tokenization_info(module_handle)

  return tokenization.FullTokenizer(
      vocab_file=tokenization_info["vocab_file"],
      do_lower_case=tokenization_info["do_lower_case"])


def get_tf_tokenizer(module_handle,
                     tokenization_info = None):

  """Creates a preprocessing function."""
  LOGGER.debug("(get_tf_tokenizer): get_tokenization_info")
  # We get tokenization info to know where the vocab is and if the model
  # is lower cased
  if tokenization_info is None:
    tokenization_info = get_tokenization_info(module_handle=module_handle)

  LOGGER.debug("(get_tf_tokenizer): tf.lookup.TextFileInitializer")
  # Create a lookup table initializer from a text file (the vocab file)
  table_initializer = tf.lookup.TextFileInitializer(
      filename=tokenization_info["vocab_file"],
      key_dtype=tf.string,
      key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
      value_dtype=tf.int64,
      value_index=tf.lookup.TextFileIndex.LINE_NUMBER)

  LOGGER.debug("(get_tf_tokenizer): tf.lookup.StaticVocabularyTable")
  # Make the table itself
  vocab_lookup_table = tf.lookup.StaticVocabularyTable(
      initializer=table_initializer,
      num_oov_buckets=1,
      lookup_key_dtype=tf.string)

  LOGGER.debug("(get_tf_tokenizer): tf_text.BertTokenizer")
  # Build the tokenizer
  tokenizer = tf_text.BertTokenizer(
      vocab_lookup_table=vocab_lookup_table,
      lower_case=tokenization_info["do_lower_case"])

  LOGGER.debug("(get_tf_tokenizer): Done")
  return tokenizer, vocab_lookup_table
