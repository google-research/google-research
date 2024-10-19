# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Data Utilities."""

import enum

import jax


class DataSplit(enum.Enum):
  """Dataset split."""

  TRAIN = 'train'
  TEST = 'test'


def shard(xs):
  """Split data into shards for multiple devices along the first dimension."""
  return jax.tree_util.tree_map(
      lambda x: x.reshape((jax.local_device_count(), -1) + x.shape[1:]), xs
  )


def unshard(x, padding=0):
  """Collect the sharded tensor to the shape before sharding."""
  y = x.reshape([x.shape[0] * x.shape[1]] + list(x.shape[2:]))
  if padding > 0:
    y = y[:-padding]
  return y
