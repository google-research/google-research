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

"""Preprocessors for Tasks."""
import seqio
import tensorflow.compat.v2 as tf


def tf_negate_before_pattern(arr, pattern):
  """Negate token IDs before a certain pattern."""
  pattern_len = len(pattern)
  arr_len = tf.shape(arr)[0]
  pattern_stack = tf.stack(
      [tf.roll(arr, -i, 0) for i in range(pattern_len)], axis=1
      )[:arr_len-pattern_len+1,:]
  pattern_match = tf.reduce_all(tf.math.equal(pattern_stack, pattern), axis=1)
  pattern_idx = tf.squeeze(
      tf.concat([tf.constant([[0]], dtype=tf.int64), tf.where(pattern_match)],
                axis=0)[-1])
  new_arr = tf.concat(
      [tf.math.negative(arr[:pattern_idx]), arr[pattern_idx:]], axis=0)
  return new_arr


@seqio.map_over_dataset
def tf_negate_inputs(features, pattern):
  """Negate token IDs before a certain pattern for feature inputs."""
  assert 'inputs' in features
  feature = features['inputs']
  feature = tf_negate_before_pattern(feature, pattern)
  return {
      'inputs': feature, **{k: features[k] for k in features if k != 'inputs'}
  }


@seqio.map_over_dataset
def tf_negate_inputs_by_mask(features):
  """Negate inputs IDs according to the inputs_mask."""
  assert 'inputs' in features
  assert 'inputs_mask' in features
  feature = features['inputs']
  mask = features['inputs_mask']
  return {
      'inputs': feature * (2 * mask - 1),
      **{k: features[k] for k in features if k != 'inputs'}
  }
