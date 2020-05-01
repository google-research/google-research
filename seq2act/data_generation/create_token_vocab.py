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

"""Creates token vocabulary using tensor2tensor tokenizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import operator
import os

from tensor2tensor.data_generators import tokenizer
import tensorflow.compat.v1 as tf  # tf

_INPUT_DIR = "/tmp"
_OUTPUT_DIR = "/tmp"

flags = tf.flags
FLAGS = flags.FLAGS
gfile = tf.gfile

flags.DEFINE_string(
    "corpus_dir", _INPUT_DIR,
    "Full path to the directory containing the data files for a set of tasks.")
flags.DEFINE_string(
    "vocab_dir", _OUTPUT_DIR,
    "Full path to the directory for saving the tf record file.")
flags.DEFINE_string("mode", "write",
                    "Flag to indicate read vocab csv or write token csv.")


word_count = collections.Counter()
freq_count = collections.Counter()


def create_token_id_files(corpus_dir, output_vocab_dir):
  """Creates token id csv  files.

  Args:
    corpus_dir: input corpus directory
    output_vocab_dir: output token vocabulary csv file directory
  """
  walking_iter = gfile.Walk(corpus_dir)
  for iter_rst in walking_iter:
    valid_filenames = [
        filename for filename in iter_rst[2]
        if ".txt" in filename or "wadata" in filename
    ]
    if not valid_filenames:
      continue
    input_file_dir = iter_rst[0]
    for filename in valid_filenames:
      path = os.path.join(input_file_dir, filename)
      with gfile.Open(path, "r") as f:
        for line in f.read().lower().split("\n"):
          tokens = tokenizer.encode(line)
          for token in tokens:
            word_count[token] += 1

  sorted_vocab = sorted(word_count.items(), key=operator.itemgetter(1))
  tf.logging.info("%d items in vocb", sum(word_count.values()))

  csv_file = gfile.Open(os.path.join(output_vocab_dir, "vocab.csv"), "w+")
  csv_writter = csv.writer(csv_file)

  rows = [["<PAD>", 0, 0], ["<EOS>", 0, 1], ["<UKN>", 0, 2], ["<START>", 0, 3]]
  for row in rows:
    csv_writter.writerow(row)
  start_index = len(rows)
  for word_freq in reversed(sorted_vocab):
    row = [word_freq[0], word_freq[1], start_index]
    freq_count[word_freq[1]] += 1
    start_index += 1
    csv_writter.writerow(row)
  tf.logging.info("vocab_size=%d", start_index)
  tf.logging.info("token frequency count")
  tf.logging.info(sorted(freq_count.items(), key=operator.itemgetter(1)))
  csv_file.close()


def read_vocab(vocab_path):
  """Reads vocabulary csv file.

  Args:
    vocab_path: full path of the vocabulary csv file

  Returns:
    tokens: list of token strings
    freqs: list of token frequencies
    ids: list of token ids
  """
  csv_file = gfile.Open(vocab_path, "r")
  csv_reader = csv.reader(csv_file)
  tokens, freqs, ids = [], [], []

  for row in csv_reader:
    tokens.append(row[0])
    freqs.append(int(row[1]))
    ids.append(int(row[2]))
  tf.logging.info("Totally %d vocabs", len(tokens))
  return tokens, freqs, ids
