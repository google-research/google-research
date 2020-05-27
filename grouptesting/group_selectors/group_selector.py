# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# Lint as: python3
"""Abstract class for a group selection strategies."""

from absl import logging
import jax.numpy as np


class GroupSelector:
  """Base class for all group selectors."""

  NEEDS_POSTERIOR = False

  def __call__(self, rng, state):
    new_groups = self.get_groups(rng, state)
    state.add_groups_to_test(new_groups)
    logging.warning('Added %i groups to test', new_groups.shape[0])
    logging.debug(new_groups.astype(np.int32))
    return state

  def get_groups(self, rng, state):
    raise NotImplementedError


