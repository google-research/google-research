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

"""Example config file for collecting data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# data_config: list of [policy_id, policy_config, proportion_in_mixture] tuples.
# policy_config [policy_type, ckpt, policy_wrapper, model_params]
# policy_wrapper [wrapper_type, *params]

model_params = (200, 200)

default_policy_root_dir = os.path.join(
    os.getenv('HOME', '/'),
    'tmp/offlinerl/policies')


def get_data_config(env_name, policy_root_dir=None):
  """Gets data config."""
  if not policy_root_dir:
    policy_root_dir = default_policy_root_dir
  ckpt_file = os.path.join(
      policy_root_dir,
      env_name,
      'sac/0/agent_partial_target',
      )
  randwalk = ['randwalk', '', ['none'], ()]
  p1_pure = ['load', ckpt_file, ['none',], model_params]
  p1_eps = ['load', ckpt_file, ['eps', 0.3], model_params]
  data_config = [
      ['randwalk', randwalk, 2],
      ['p1_pure', p1_pure, 4],
      ['p1_eps', p1_eps, 4],
  ]
  return data_config
