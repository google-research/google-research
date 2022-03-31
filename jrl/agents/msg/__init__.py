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
"""MSG agent."""

from acme.agents.jax.sac.config import target_entropy_from_env_spec

from jrl.agents.msg import config
from jrl.agents.msg import networks
from jrl.agents.msg.builder import MSGBuilder
from jrl.agents.msg.learning import MSGLearner
from jrl.agents.msg.networks import apply_policy_and_sample
from jrl.utils.agent_utils import RLComponents


class MSGRLComponents(RLComponents):
  def __init__(self, logger_fn, spec, create_data_iter_fn):
    self._logger_fn = logger_fn
    self._spec = spec
    self._config = config.MSGConfig(
        target_entropy=target_entropy_from_env_spec(spec))
    # self._config = config.MSGConfig(
    #     target_entropy=0.,
    #     entropy_coefficient=1.)
    self._create_data_iter_fn = create_data_iter_fn

  def make_builder(self):
    return MSGBuilder(
        config=self._config,
        make_demonstrations=self._create_data_iter_fn,
        logger_fn=self._logger_fn)

  def make_networks(self):
    return networks.make_networks(
        self._spec,
        actor_hidden_layer_sizes=self._config.actor_network_hidden_sizes,
        critic_hidden_layer_sizes=self._config.critic_network_hidden_sizes,
        init_type=self._config.networks_init_type,
        critic_init_scale=self._config.critic_init_scale,
        use_double_q=self._config.use_double_q,
        img_encoder_fn=self._config.img_encoder_fn,
        build_kernel_fn=(
            self._config.behavior_regularization_type == 'spec_norm'),
        ensemble_method=self._config.ensemble_method,
        ensemble_size=self._config.ensemble_size, # not used for making deep ensembles
        mimo_using_obs_tile=self._config.mimo_using_obs_tile,
        mimo_using_act_tile=self._config.mimo_using_act_tile,
    )

  def make_behavior_policy(self, network):
    return networks.apply_policy_and_sample(network, eval_mode=False)

  def make_eval_behavior_policy(self, network, force_eval_with_q_filter=False, q_filter_with_unif=True,):
    # return networks.apply_policy_and_sample(network, eval_mode=True)
    if self._config.eval_with_q_filter or force_eval_with_q_filter:
      return networks.build_q_filtered_actor(
          networks=network,
          beta=self._config.beta,
          num_samples=self._config.num_eval_samples,
          use_img_encoder=self._config.use_img_encoder,
          with_uniform=q_filter_with_unif,
          ensemble_method=self._config.ensemble_method,
          ensemble_size=self._config.ensemble_size,
          mimo_using_obs_tile=self._config.mimo_using_obs_tile,
          mimo_using_act_tile=self._config.mimo_using_act_tile,)
    else:
      return networks.apply_policy_and_sample(
          network,
          eval_mode=True,
          use_img_encoder=self._config.use_img_encoder)
