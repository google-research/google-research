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

"""Utilities for handling model inputs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf


def get_shared_feature_specs(config):
  """Non-task-specific model inputs."""
  return [
      FeatureSpec("input_ids", [config.max_seq_length]),
      FeatureSpec("input_mask", [config.max_seq_length]),
      FeatureSpec("segment_ids", [config.max_seq_length]),
      FeatureSpec("task_id", []),
  ]


class FeatureSpec(object):
  """Defines a feature passed as input to the model."""

  def __init__(self, name, shape, default_value_fn=None, is_int_feature=True):
    self.name = name
    self.shape = shape
    self.default_value_fn = default_value_fn
    self.is_int_feature = is_int_feature

  def get_parsing_spec(self):
    return tf.FixedLenFeature(
        self.shape, tf.int64 if self.is_int_feature else tf.float32)

  def get_default_value(self):
    if self.default_value_fn:
      return self.default_value_fn(self.shape)
    else:
      return np.zeros(
          self.shape, np.int64 if self.is_int_feature else np.float32)
