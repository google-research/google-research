# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Util functions for loading word embedding data."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading
import numpy as np
import tensorflow.compat.v1 as tf
gfile = tf.gfile
flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "vocab_file", "",
    "Full path to the directory containing the data files for a set of tasks.")
flags.DEFINE_string(
    "input_candiate_file", "",
    "Full path to the directory for saving the tf record file.")


_candidate_words = None
lock = threading.Lock()

token_id_map = None
token_id_map_lock = threading.Lock()

id_token_map = None
id_token_map_lock = threading.Lock()


# This func is ONLY for converting subtoken to id by reading vocab file directly
# Do not use this func as tokenizer for raw string, use string_utils.py instead
def tokens_to_ids(token_list):
  """Gets line numbers as ids for tokens accroding to vocab file."""
  with token_id_map_lock:
    global token_id_map
    if not token_id_map:
      token_id_map = {}
      with gfile.Open(get_vocab_file(), "r") as f:
        for idx, token in enumerate(f.read().split("\n")):
          # Remove head and tail apostrophes of token
          if token[1:-1]:
            token_id_map[token[1:-1]] = idx
  return [token_id_map[token] for token in token_list]


def ids_to_tokens(id_list):
  """Gets tokens from id list accroding to vocab file."""
  with id_token_map_lock:
    global id_token_map
    if not id_token_map:
      id_token_map = {}
      with gfile.Open(get_vocab_file(), "r") as f:
        for idx, token in enumerate(f.read().split("\n")):
          # Remove head and tail apostrophes of token
          if token[1:-1]:
            id_token_map[idx] = token[1:-1]
  return [id_token_map[the_id] for the_id in id_list]


def _get_candidate_words():
  with lock:
    global _candidate_words
    if not _candidate_words:
      candidate_file = FLAGS.input_candidate_file
      with gfile.Open(candidate_file, "r") as f:
        _candidate_words = f.read().split("\n")
    return _candidate_words


def get_random_words(sample_size):
  candidate_words = _get_candidate_words()
  return np.random.choice(candidate_words, sample_size, replace=False)


def get_vocab_file():
  return FLAGS.vocab_file
