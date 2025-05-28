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

"""Preprocessors for D3PM tasks."""

import tensorflow as tf


def add_bos(batch, key_name='inputs', bos_tok=0):
  """Adds a BOS token."""

  inputs = batch[key_name]
  bos = tf.constant(bos_tok, dtype=tf.int32, shape=inputs.shape[:-1] + (1,))
  inputs = tf.concat([bos, inputs[Ellipsis, :-1]], axis=-1)
  batch[key_name] = inputs
  return batch


def rekey(batch, key_map):
  """Rekeys a batch using key_map.

  key_map specifies new_key: old_key pairs.

  Args:
    batch: a dictionary to modify.
    key_map: a dictionary that new keys to old keys. So if you want your new
      dataset to have keys 'inputs' and 'targets', where the old task just had
      'targets, pass {'inputs': 'targets', 'targets': 'targets'}.

  Returns:
    a new batch dict.
  """
  return {key: batch[value] for key, value in key_map.items()}
