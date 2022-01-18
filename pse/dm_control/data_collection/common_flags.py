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

# Lint as: python3
"""Common flags."""

from absl import flags

flags.DEFINE_integer('trial_id', 0, 'The trial ID from 0 to num_trials-1.')
flags.DEFINE_integer('max_episode_len', 1000, 'Number of steps in an episode.')
flags.DEFINE_string('env_name', 'cartpole-swingup', 'Name of the environment.')
flags.DEFINE_string('root_dir', None,
                    'Path to output trajectories from data collection')
flags.DEFINE_multi_string('gin_files', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_bindings', None, 'Gin binding parameters.')
flags.DEFINE_integer('seed', None, 'Random Seed for model_dir/data_dir')
