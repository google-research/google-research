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

"""Several objective functions."""
import jax
import numpy as np


def draw_states(num_states, howmany, key):
  """Draws howmany state indices from 0 ... num_states."""
  key, subkey = jax.random.split(key)
  states = jax.random.randint(subkey, (howmany,), 0, num_states)
  return states, key


def compute_max_feature_norm(features):
  """Computes the maximum norm (squared) of a collection of feature vectors."""
  feature_norm = np.linalg.norm(features, axis=1)
  feature_norm = np.max(feature_norm)**2

  return feature_norm
