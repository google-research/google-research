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

"""Enwik8 character level language modeling."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import metrics
from tensor2tensor.utils import registry


def train_data_filenames(tmp_dir):
  return [
      os.path.join(tmp_dir,
                   "train.txt")
  ]


def valid_data_filenames(tmp_dir):
  return [
      os.path.join(tmp_dir,
                   "valid.txt")
  ]


def test_data_filenames(tmp_dir):
  return [
      os.path.join(tmp_dir,
                   "test.txt")
  ]


@registry.register_problem
class Enwik8(text_problems.Text2SelfProblem):
  """Enwiki8."""

  @property
  def is_generate_per_split(self):
    return True

  @property
  def vocab_type(self):
    return text_problems.VocabType.CHARACTER

  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each."""
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 100,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    }, {
        "split": problem.DatasetSplit.TEST,
        "shards": 1,
    }]

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    if dataset_split == problem.DatasetSplit.TRAIN:
      input_files = train_data_filenames(tmp_dir)
    elif dataset_split == problem.DatasetSplit.TEST:
      input_files = test_data_filenames(tmp_dir)
    elif dataset_split == problem.DatasetSplit.EVAL:
      input_files = valid_data_filenames(tmp_dir)
    else:
      raise ValueError("Undefined dataset_split")
    for input_file in input_files:
      for line in text_problems.txt_line_iterator(input_file):
        yield {"targets": line}

  def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
    generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
    for sample in generator:
      if not sample["targets"]:
        continue
      inputs = sample["targets"]
      inputs = inputs.strip(" \n").split(" ")
      input_seq = []
      for inp in inputs:
        if inp:
          input_seq.append(int(inp))
      yield {"targets": input_seq}

  def eval_metrics(self):
    return [metrics.Metrics.NEG_LOG_PERPLEXITY]

