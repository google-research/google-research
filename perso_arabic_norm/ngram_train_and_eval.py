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

r"""Utility for building and evaluating n-gram models.

Example:
--------
  python ngram_train_and_eval.py \
    --corpus_file ${CORPUS_DIR}/kswiki.txt.bz2 \
    --line_diffs_file ${CORPUS_DIR}/kswiki_line_diffs.pickle \
    --num_trials 100 \
    --order 6 \
    --output_model_dir ${RESULTS_DIR}

Dependencies:
-------------
  absl
  kenlm
  numpy
  pandas
"""

from typing import Sequence

import datetime
import logging
import math
import os
import pathlib
import pickle
import subprocess
import tempfile

from absl import app
from absl import flags

import kenlm
import numpy as np
import pandas as pd

import utils

flags.DEFINE_string(
    "corpus_file", "",
    "Path to the input corpus file in text format or bzip2/gzip format.")

flags.DEFINE_string(
    "line_diffs_file", "",
    "Binary pickle file containing line IDs that have diffs.")

flags.DEFINE_float(
    "train_test_ratio", 0.8,
    "Ratio of the training set to select.")

flags.DEFINE_integer(
    "num_trials", 10,
    "Number of trials that split the data, train and evaluate the model.")

flags.DEFINE_integer(
    "order", 3,
    "Order of the n-grams.")

flags.DEFINE_string(
    "output_model_dir", "",
    "Output directory for storing the language models.")

flags.DEFINE_boolean(
    "keep_model", False,
    "When enabled, the model at each iteration is kept in `output_model_dir`.")

flags.DEFINE_boolean(
    "word_models", False,
    "If enabled, build word models instead of character models.")

FLAGS = flags.FLAGS


def _timestamp():
  """Returns current timestamp as string."""
  current_time = datetime.datetime.now()
  return str(current_time.timestamp())


def _num_tokens(sentences):
  """Returns number of tokens in a sentence (tokens can be chars or words)."""
  num_tokens = 0
  for sent in sentences:
    if not FLAGS.word_models:
      num_tokens += len(sent)  # Characters.
    else:
      num_tokens += len(list(filter(None, sent.split())))  # Words.
  return num_tokens


def _split(indices, line_diffs):
  num_total_lines = len(indices)
  max_train_lines = math.floor(FLAGS.train_test_ratio * num_total_lines)
  train_ind = list(line_diffs)
  np.random.shuffle(train_ind)
  test_ind = []
  for i in indices:
    if i in line_diffs:
      continue
    if len(train_ind) <= max_train_lines:
      train_ind.append(i)
    else:
      test_ind.append(i)
  logging.info("Split %d sentences into %d (train) and %d (test)",
               num_total_lines, len(train_ind), len(test_ind))
  return train_ind, test_ind


def _tokenize(sentences):
  """Tokenizes a list of sentences into characters or words (no-op)."""
  logging.info(f"Tokenizing {len(sentences)} sentences ...")
  for sent in sentences:
    if FLAGS.word_models:
      yield sent.strip()
    else:
      sent = sent.strip().lower().replace(" ", "|")
      yield " ".join([c for c in sent])


def _train_model(sentences, model_name):
  """Trains language model using KenLM from tokenized sentences."""
  output_model_file = os.path.join(FLAGS.output_model_dir, model_name) + ".lm"
  with tempfile.NamedTemporaryFile(prefix=_timestamp(), mode="w",
                                   encoding="utf-8") as corpus_f:
    # Prepare training data.
    num_tokens = _num_tokens(sentences)
    logging.info(f"Training set has {num_tokens} tokens ...")
    corpus_f.writelines(s + "\n" for s in sentences)
    corpus_f.flush()

    # Train.
    with tempfile.NamedTemporaryFile(prefix=_timestamp(), mode="w",
                                     encoding="utf-8") as arpa_f:
      logging.info("Training %d-gram LM to %s ...", FLAGS.order, arpa_f.name)
      proc = subprocess.run(["lmplz", "--order", str(FLAGS.order),
                             "--text", corpus_f.name,
                             "--arpa", arpa_f.name,
                             # Skip special symbols (e.g., `<s>`) found in
                             # the original data.
                             "--skip_symbols"],
                            check=True,   # Will raise if the command fails.
                            capture_output=True,
                            encoding="utf-8")

      logging.info(f"Converting ARPA LM to binary in {output_model_file} ...")
      proc = subprocess.run(["build_binary", "-s",
                             arpa_f.name, output_model_file],
                            check=True,
                            capture_output=True,
                            encoding="utf-8")

  model = kenlm.LanguageModel(output_model_file)
  if not FLAGS.keep_model:
    os.remove(output_model_file)
  return model


def _process_corpus(corpus_lines, line_diffs):
  num_total_lines = len(corpus_lines)
  logging.info(f"Read {num_total_lines} corpus lines.")
  max_train_lines = math.floor(FLAGS.train_test_ratio * num_total_lines)
  indices = np.arange(num_total_lines)
  experiment_name = "%s_%s_%dgram_tr%.2f" % (
      pathlib.Path(FLAGS.corpus_file).stem,
      "word" if FLAGS.word_models else "char",
      FLAGS.order, FLAGS.train_test_ratio)
  perplexities = []
  ntokens_train = []
  ntokens_test  = []
  for i in range(FLAGS.num_trials):
    # Shuffle and split.
    logging.info(f"===> Trial {i} ...")
    np.random.shuffle(indices)
    train_ind, test_ind = _split(indices, line_diffs)

    # Train the model.
    model_name = "%s_%s" % (experiment_name, i)
    train_sentences = [corpus_lines[j] for j in train_ind]
    ntokens_train.append(_num_tokens(train_sentences))
    model = _train_model(train_sentences, model_name)

    # Score the model.
    test_sentences = [corpus_lines[j] for j in test_ind]
    num_test_tokens = _num_tokens(test_sentences)
    logging.info(f"Scoring {len(test_ind)} sentences "
                 f"({num_test_tokens} tokens)...")
    ntokens_test.append(num_test_tokens)
    test_data = " ".join(test_sentences)
    ppl = model.perplexity(test_data)
    logging.info("Perplexity: %f", ppl)
    perplexities.append(ppl)

  # Save the report.
  report_path = os.path.join(FLAGS.output_model_dir,
                             experiment_name + "_report.tsv")
  logging.info(f"Saving the scored sentences to {report_path} ...")
  df = pd.DataFrame(data = {
      "col1" : ntokens_train,
      "col2" : ntokens_test,
      "col3" : perplexities,
  })
  df.to_csv(report_path, sep="\t", index=None, header=None, encoding="utf-8")



def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  if not FLAGS.corpus_file:
    raise app.UsageError("Specify --corpus_file [FILE]!")
  if not FLAGS.output_model_dir:
    raise app.UsageError("Specify --output_model_dir [DIR]!")

  if not os.path.isdir(FLAGS.output_model_dir):
    raise ValueError(f"Directory `{FLAGS.output_model_dir}` "
                     "does not exist")

  # Read line diffs.
  line_diffs = []
  if FLAGS.line_diffs_file:
    logging.info(f"Reading line diffs from {FLAGS.line_diffs_file} ...")
    with open(FLAGS.line_diffs_file, "rb") as f:
      line_diffs = frozenset(pickle.load(f))
    logging.info(f"Loaded {len(line_diffs)} line diffs")

  # Process corpus.
  logging.info(f"Reading corpus from {FLAGS.corpus_file} ...")
  with utils.open_file(FLAGS.corpus_file, "r") as f:
    corpus_lines = list(_tokenize(f.readlines()))
    _process_corpus(corpus_lines, line_diffs)


if __name__ == "__main__":
  app.run(main)
