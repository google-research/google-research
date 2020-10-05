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

"""Wikitext-103 word level language modeling."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import wikitext103
from tensor2tensor.utils import registry


@registry.register_problem
class Wikitext103Length4k(wikitext103.LanguagemodelWikitext103L4k):
  """Wikitext-103 word level language modeling with context length 4k."""

  @property
  def dataset_splits(self):
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 30,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    }, {
        "split": problem.DatasetSplit.TEST,
        "shards": 1,
    }]

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    samples_by_line = super(wikitext103.LanguagemodelWikitext103L4k,
                            self).generate_samples(data_dir, tmp_dir,
                                                   dataset_split)
    def _generate_samples():
      """Add intermediate windows of overlapping articles."""
      if dataset_split == problem.DatasetSplit.TRAIN:
        samples = collections.deque([])
        tokens = []
        for sample in samples_by_line:
          sample_tokens = sample["targets"].split()
          samples.append(sample["targets"])
          if (len(tokens) + len(sample_tokens)) >= self.sequence_length:
            yield {"targets": " ".join(tokens)}
            sent_len = 0
            while sent_len < len(sample_tokens):
              sent = samples.popleft()
              sent_len += len(sent.split())
            tokens = tokens[sent_len:]
          tokens.extend(sample_tokens)
      else:
        tokens = []
        for sample in samples_by_line:
          sample_tokens = sample["targets"].split()
          if len(tokens) + len(sample_tokens) < self.sequence_length:
            tokens.extend(sample_tokens)
          else:
            yield {"targets": " ".join(tokens)}
            tokens = sample_tokens
    tokens = _generate_samples()
    return tokens


@registry.register_problem
class Wikitext103Length4kOverlap(wikitext103.LanguagemodelWikitext103L4k):
  """Wikitext-103 word level language modeling with context length 4k."""

  @property
  def dataset_splits(self):
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 1000,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    }, {
        "split": problem.DatasetSplit.TEST,
        "shards": 1,
    }]

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    samples_by_line = super(wikitext103.LanguagemodelWikitext103L4k,
                            self).generate_samples(data_dir, tmp_dir,
                                                   dataset_split)

    def _generate_samples():
      """Add intermediate windows of overlapping articles."""
      if dataset_split == problem.DatasetSplit.TRAIN:
        tokens = []
        for sentence in samples_by_line:
          sentence["targets"] += " <EOS>"
          sample_tokens = sentence["targets"].split()
          tokens.extend(sample_tokens)
          while len(tokens) >= self.sequence_length:
            yield {"targets": " ".join(tokens[:self.sequence_length - 1])}
            tokens = tokens[3:]
      else:
        tokens = []
        for sentence in samples_by_line:
          sentence["targets"] += " <EOS>"
          sample_tokens = sentence["targets"].split()
          if len(tokens) + len(sample_tokens) < self.sequence_length:
            tokens.extend(sample_tokens)
          else:
            yield {"targets": " ".join(tokens)}
            tokens = sample_tokens

    tokens = _generate_samples()
    return tokens


@registry.register_problem
class Wikitext103Length2k(Wikitext103Length4k):
  """Wikitext sequence length 2k."""

  @property
  def sequence_length(self):
    """Length of each example (in tokens)."""
    return 2048


@registry.register_problem
class Wikitext103Length1k(Wikitext103Length4k):
  """Wikitext sequence length 1k."""

  @property
  def sequence_length(self):
    """Length of each example (in tokens)."""
    return 1024

