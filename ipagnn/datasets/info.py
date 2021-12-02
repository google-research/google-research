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

"""Defines the DatasetInfo available for a Learned Interpreters dataset."""

import tensorflow_datasets as tfds


class LearnedInterpretersDatasetInfo(tfds.core.DatasetInfo):
  """DatasetInfo for a Learned Interpreters datasets."""

  def __init__(self, *args, **kwargs):
    self.builder_config = kwargs.pop("builder_config")
    self.max_diameter = kwargs.pop("max_diameter")
    self.program_generator_config = kwargs.pop("program_generator_config")
    self.program_encoder = kwargs.pop("program_encoder")
    self.state_encoder = kwargs.pop("state_encoder")
    self.branch_encoder = kwargs.pop("branch_encoder")
    super(LearnedInterpretersDatasetInfo, self).__init__(*args, **kwargs)

  @property
  def output_vocab_size(self):
    if self.supervised_keys[-1] == "error_type":
      return self.features["error_type"].num_classes
    else:
      return self.features["target_output"].vocab_size


class ExpressionsDatasetInfo(tfds.core.DatasetInfo):
  """DatasetInfo for a Learned Interpreters datasets."""

  def __init__(self, *args, **kwargs):
    self.builder_config = kwargs.pop("builder_config")
    self.generator_config = kwargs.pop("generator_config")
    super(ExpressionsDatasetInfo, self).__init__(*args, **kwargs)

  @property
  def encoder(self):
    return self.features["expressions"].encoder
