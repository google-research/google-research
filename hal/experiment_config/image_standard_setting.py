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

# pylint: skip-file
"""Experiment setting for standard settings with image."""


def image_standard_setting():
  setting = {
      'action_type': 'discrete',
      'obs_type': 'image',
      'reward_shape_val': 0.25,
      'use_subset_instruction': False,
      'frame_skip': 20,
      'use_polar': False,
      'suppress': False,
      'diverse_scene_content': False,
      'buffer_size': int(2e6),
      'use_movement_bonus': True,
      'img_resolution': 64,

      # agent
      'masking_q': False,
      'discount': 0.9,
      'intruction_repr': 'language',
      'learning_rate': 1e-3,
      'polyak_rate': 0.95,
      'trainable_encoder': True,

      # learner
      'num_epoch': 50,
      'num_cycle': 50,
      'num_episode': 50,
      'optimization_steps': 100,
      'batchsize': 128,
      'sample_new_scene_prob': 0.,
      'max_episode_length': 100,
      'record_atomic_instruction': True,
      'paraphrase': True,
      'relabeling': True,
      'k_immediate': 3,
      'future_k': 4,
      'negate_unary': False,
      'min_epsilon': 0.05,
      'epsilon_decay': 0.995,
      'collect_cycle': 2,
  }
  return setting
