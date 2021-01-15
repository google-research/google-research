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

"""This contains the general topology blackbox object, which allows topology updates to the policy."""
from es_optimization import blackbox_functions
from es_optimization import blackbox_objects


class GeneralTopologyBlackboxObject(blackbox_objects.BlackboxObject):
  """This BlackboxObject handles topology_str's in addition to normal weight parameters."""

  def __init__(self, config, **kwargs):
    self.config = config
    self.horizon = self.config.horizon
    self.env = config.environment_fn()
    self.policy = config.policy_fn_for_object()

  def get_initial(self):
    return (self.policy).get_initial()

  def execute_with_topology(self, params, topology_str, hyperparams):
    self.policy.update_topology(topology_str)
    return self.execute(params, hyperparams)

  def execute(self, params, hyperparams):
    (self.env).deterministic_start()
    self.policy.update_weights(params)
    return blackbox_functions.rl_extended_rollout(self.policy, hyperparams,
                                                  self.env, self.horizon)

  def get_metaparams(self):
    return [(self.env).state_dimensionality()]
