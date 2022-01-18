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

"""Base class for tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import csv
import os
import tensorflow.compat.v1 as tf


class Example(object):
  __metaclass__ = abc.ABCMeta

  def __init__(self, task_name):
    self.task_name = task_name


class Task(object):
  """Override this class to add a new task."""

  __metaclass__ = abc.ABCMeta

  def __init__(self, config, name, long_sequences=False):
    self.config = config
    self.name = name
    self.long_sequences = long_sequences

  def get_examples(self, split):
    return self.load_data(split + ".tsv", split)

  def get_test_splits(self):
    return ["test"]

  def load_data(self, fname, split):
    examples = self._create_examples(
        read_tsv(os.path.join(self.config.raw_data_dir(self.name), fname),
                 max_lines=50 if self.config.debug else None),
        split)
    return examples

  @abc.abstractmethod
  def _create_examples(self, lines, split):
    pass

  @abc.abstractmethod
  def get_scorer(self):
    pass

  @abc.abstractmethod
  def get_feature_specs(self):
    pass

  @abc.abstractmethod
  def featurize(self, example, is_training):
    pass

  @abc.abstractmethod
  def get_prediction_module(self, bert_model, features, is_training,
                            percent_done):
    pass

  def __repr__(self):
    return "Task(" + self.name + ")"


def read_tsv(input_file, quotechar=None, max_lines=None):
  """Reads a tab separated value file."""
  with tf.gfile.Open(input_file, "r") as f:
    reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
    lines = []
    for i, line in enumerate(reader):
      if max_lines and i >= max_lines:
        break
      lines.append(line)
    return lines
