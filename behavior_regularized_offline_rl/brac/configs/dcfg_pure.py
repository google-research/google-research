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

"""Config file for collecting policy data without noise."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


model_params = (200, 200)

default_policy_root_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../../trained_policies')


def get_data_config(env_name, policy_root_dir=None):
  if not policy_root_dir:
    policy_root_dir = default_policy_root_dir
  ckpt_file = os.path.join(
      policy_root_dir,
      env_name,
      'agent_partial_target',
      )
  p1_pure = ['load', ckpt_file, ['none',], model_params]
  data_config = [
      ['p1_pure', p1_pure, 1],
  ]
  return data_config
