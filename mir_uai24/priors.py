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

"""Prior definitions."""

import numpy as np
import tensorflow as tf

from mir_uai24 import enum_utils


def uniform(dataset_info):
  prior = -np.ones(shape=(dataset_info.n_instances, 1), dtype=np.float32)
  for bag_id in dataset_info.memberships.bags:
    bag_size = len(dataset_info.memberships.bags[bag_id])
    instances = [
        instance.bag_id_x_instance_id
        for instance in dataset_info.memberships.bags[bag_id]
    ]
    prior[instances] = 1 / bag_size
  assert not (prior == -1).any()
  return prior
