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

# python3
"""BC agent."""

from jrl.agents.bc import config
from jrl.agents.bc import networks
from jrl.agents.bc.builder import BCBuilder
from jrl.agents.bc.learning import BCLearner
from jrl.agents.bc.networks import apply_policy_and_sample
from jrl.utils.agent_utils import RLComponents


class BCRLComponents(RLComponents):
  def __init__(self, logger_fn, spec, create_data_iter_fn):
    self._logger_fn = logger_fn
    self._spec = spec
    self._config = config.BCConfig()
    self._create_data_iter_fn = create_data_iter_fn

  def make_builder(self):
    return BCBuilder(
        config=self._config,
        make_demonstrations=self._create_data_iter_fn,
        logger_fn=self._logger_fn)

  def make_networks(self):
    return networks.make_networks(
        self._spec,
        build_actor_fn=self._config.build_actor_fn,
        img_encoder_fn=self._config.img_encoder_fn,)

  def make_behavior_policy(self, network):
    return networks.apply_policy_and_sample(network, eval_mode=False)

  def make_eval_behavior_policy(self, network):
    return networks.apply_policy_and_sample(network, eval_mode=True)
