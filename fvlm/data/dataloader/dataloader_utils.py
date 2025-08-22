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

# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Utility functions for dataloader."""

from data.dataloader import input_utils
import tensorflow.compat.v1 as tf


def process_source_id(source_id):
  """Processes source_id to the right format."""
  if source_id.dtype == tf.string:
    source_id = tf.string_to_number(source_id, out_type=tf.int64)
  with tf.control_dependencies([source_id]):
    source_id = tf.cond(tf.equal(tf.size(source_id), 0),
                        lambda: tf.cast(tf.constant(-1), tf.int64),
                        lambda: tf.identity(source_id))
  return source_id


def pad_groundtruths_to_fixed_size(gt, n):
  """Pads the first dimension of groundtruths labels to the fixed size."""
  gt['boxes'] = input_utils.clip_or_pad_to_fixed_size(gt['boxes'], n, -1)
  gt['is_crowds'] = input_utils.clip_or_pad_to_fixed_size(gt['is_crowds'], n, 0)
  gt['areas'] = input_utils.clip_or_pad_to_fixed_size(gt['areas'], n, -1)
  gt['classes'] = input_utils.clip_or_pad_to_fixed_size(gt['classes'], n, -1)
  return gt
