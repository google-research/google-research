# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Library for scoring canaries using a bigram model."""

import collections
import os

from absl import logging
import bagz
import numpy as np
import tunix


# Use the GEMMA3_TOKENIZER path from environment or config:
class _ParamsLib:
  GEMMA3_TOKENIZER = os.environ.get(
      'GEMMA3_TOKENIZER_PATH', '<PATH_TO_GEMMA3_TOKENIZER>'
  )


params_lib = _ParamsLib()


def _get_ngram_counts(
    synthetic_data_file,
    tokenizer,
):
  """Computes unigram and bigram counts from the synthetic data.

  Args:
    synthetic_data_file: Path to the synthetic data bagz file.
    tokenizer: Tokenizer to use for encoding the text.

  Returns:
    A tuple of unigram and bigram counts.
  """
  unigram_counts = collections.Counter()
  bigram_counts = collections.Counter()
  with bagz.BagReader(synthetic_data_file) as reader:
    num_records = len(reader)
    logging.info(
        'Counting ngrams from %d records from %s',
        num_records,
        synthetic_data_file,
    )
    for record in reader:
      tokens = tokenizer.encode(record.decode('utf8'))
      unigram_counts.update(tokens)
      bigram_counts.update(zip(tokens, tokens[1:]))
  logging.info(
      'Read %d unigrams and %d bigrams', len(unigram_counts), len(bigram_counts)
  )
  return unigram_counts, bigram_counts


class BigramModel:
  """A bigram model for scoring canaries."""

  def __init__(self, synthetic_data_file, canary_len):
    self.tokenizer = tunix.generate.tokenizer_adapter.Tokenizer(
        tokenizer_path=params_lib.GEMMA3_TOKENIZER,
        add_bos=True,
        add_eos=True,
    )
    self.unigram_counts, self.bigram_counts = _get_ngram_counts(
        synthetic_data_file, self.tokenizer
    )
    self.vocab_size = self.tokenizer.GetPieceSize()
    self.canary_len = canary_len

  def _log_prob(self, t1, t2):
    """Returns the log probability of t2 given t1 with Laplace smoothing."""
    bigram_count = self.bigram_counts[(t1, t2)]
    unigram_count = self.unigram_counts[t1]
    return np.log(bigram_count + 1) - np.log(unigram_count + self.vocab_size)

  def score(self, canary):
    """Scores the canary using the bigram model."""
    # First token is BOS. We multiply the probability of the remaining tokens.
    tokens = self.tokenizer.encode(canary)[: self.canary_len + 1]
    return sum(self._log_prob(t1, t2) for t1, t2 in zip(tokens, tokens[1:]))
