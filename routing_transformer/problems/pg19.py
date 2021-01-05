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

"""PG-19 dataset from https://openreview.net/forum?id=SylKikSYDH.

To facilitate comparisons multiply t2t log ppl with 1.248.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry


def train_data_filenames(tmp_dir):
  dir_path = os.path.join(tmp_dir, "train")
  return [os.path.join(dir_path, f) for f in os.listdir(dir_path)]


def valid_data_filenames(tmp_dir):
  dir_path = os.path.join(tmp_dir, "validation")
  return [os.path.join(dir_path, f) for f in os.listdir(dir_path)]


def test_data_filenames(tmp_dir):
  dir_path = os.path.join(tmp_dir, "test")
  return [os.path.join(dir_path, f) for f in os.listdir(dir_path)]


@registry.register_problem
class Pg19Length8k(text_problems.Text2SelfProblem):
  """PG-19 word level language modeling with context length 8k."""

  @property
  def is_generate_per_split(self):
    return True

  @property
  def vocab_type(self):
    return text_problems.VocabType.SUBWORD

  @property
  def approx_vocab_size(self):
    """Approximate vocab size to generate. Only for VocabType.SUBWORD."""
    return 2**15  # ~32k

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

  @property
  def sequence_length(self):
    """Length of each example (in tokens)."""
    return 8192

  def generate_text_for_vocab(self, data_dir, tmp_dir):
    input_files = train_data_filenames(tmp_dir)
    for input_file in input_files:
      for line in text_problems.txt_line_iterator(input_file):
        yield line

  def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
    encoder = self.get_or_create_vocab(data_dir, tmp_dir)

    def _generator():
      """Inner data generator."""
      if dataset_split == problem.DatasetSplit.TRAIN:
        input_files = train_data_filenames(tmp_dir)
      elif dataset_split == problem.DatasetSplit.TEST:
        input_files = test_data_filenames(tmp_dir)
      elif dataset_split == problem.DatasetSplit.EVAL:
        input_files = valid_data_filenames(tmp_dir)
      else:
        raise ValueError("Undefined dataset_split")

      tokens = []
      for input_file in input_files:
        for line in text_problems.txt_line_iterator(input_file):
          sample_tokens = encoder.encode(line)
          if len(tokens) + len(sample_tokens) < self.sequence_length:
            tokens.extend(sample_tokens)
          else:
            tokens.append(text_encoder.EOS_ID)
            yield {"targets": tokens}
            tokens = sample_tokens
      tokens.append(text_encoder.EOS_ID)
      yield {"targets": tokens}

    return _generator()
