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

"""Utilities for creating policy based on user config."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from absl import logging
import tensorflow.compat.v1 as tf
from behavior_regularized_offline_rl.brac import networks
from behavior_regularized_offline_rl.brac import policies


PolicyConfig = collections.namedtuple(
    'PolicyConfig', 'ptype, ckpt, wrapper, model_params')


PTYPES = [
    'randwalk',
    'randinit',
    'load',
]

WRAPPER_TYPES = [
    'none',
    'eps',
    'gaussian',
    'gaussianeps',
]

# params: (wrapper_type, *wrapper_params)
# wrapper_type: none, eps, gaussian, gaussianeps


def wrap_policy(a_net, wrapper):
  """Wraps actor network with desired randomization."""
  if wrapper[0] == 'none':
    policy = policies.RandomSoftPolicy(a_net)
  elif wrapper[0] == 'eps':
    policy = policies.EpsilonGreedyRandomSoftPolicy(
        a_net, wrapper[1])
  elif wrapper[0] == 'gaussian':
    policy = policies.GaussianRandomSoftPolicy(
        a_net, std=wrapper[1])
  elif wrapper[0] == 'gaussianeps':
    policy = policies.GaussianEpsilonGreedySoftPolicy(
        a_net, std=wrapper[1], eps=wrapper[2])
  return policy


def load_policy(policy_cfg, action_spec):
  """Loads policy based on config."""
  if policy_cfg.ptype not in PTYPES:
    raise ValueError('Unknown policy type %s.' % policy_cfg.ptype)
  if policy_cfg.ptype == 'randwalk':
    policy = policies.ContinuousRandomPolicy(action_spec)
  elif policy_cfg.ptype in ['randinit', 'load']:
    a_net = networks.ActorNetwork(
        action_spec,
        fc_layer_params=policy_cfg.model_params)
    if policy_cfg.ptype == 'load':
      logging.info('Loading policy from %s...', policy_cfg.ckpt)
      policy_ckpt = tf.train.Checkpoint(policy=a_net)
      policy_ckpt.restore(policy_cfg.ckpt).expect_partial()
    policy = wrap_policy(a_net, policy_cfg.wrapper)
  return policy


def parse_policy_cfg(policy_cfg):
  return PolicyConfig(*policy_cfg)
