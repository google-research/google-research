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
"""MSG Builder."""

import dataclasses
from typing import Callable, Iterator, List, Optional
import acme
from acme import adders
from acme import core
from acme import specs
from acme.agents.jax import actors
from acme.agents.jax import builders
from acme.jax import networks as networks_lib
from acme.jax import variable_utils
from acme.utils import counting
from acme.utils import loggers
import optax
import reverb
from jrl.agents.msg import config
from jrl.agents.msg import learning
from jrl.agents.msg import networks as msg_networks


class MSGBuilder(builders.ActorLearnerBuilder):
  """MSG Builder."""

  def __init__(
      self,
      config,
      # make_demonstrations: Callable[[int], Iterator[types.Transition]],
      make_demonstrations,
      logger_fn = lambda: None,):
    self._config = config
    self._logger_fn = logger_fn
    self._make_demonstrations = make_demonstrations

  def make_learner(
      self,
      random_key,
      networks,
      dataset,
      replay_client = None,
      counter = None,
      checkpoint = False,
  ):
    del dataset # Offline RL

    data_iter = self._make_demonstrations()

    return learning.MSGLearner(
        ensemble_size=self._config.ensemble_size,
        beta=self._config.beta,
        networks=networks,
        rng=random_key,
        iterator=data_iter,
        ensemble_method=self._config.ensemble_method,
        perform_sarsa_q_eval=self._config.perform_sarsa_q_eval,
        num_q_repr_pretrain_iters=self._config.num_q_repr_pretrain_iters,
        pretrain_temp=self._config.pretrain_temp,
        use_sass=self._config.use_sass,
        num_bc_iters=self._config.num_bc_iters,
        use_random_weighting_in_critic_loss=self._config.use_random_weighting_in_critic_loss,
        use_ema_target_critic_params=self._config.use_ema_target_critic_params,
        entropy_coefficient=self._config.entropy_coefficient,
        target_entropy=self._config.target_entropy,
        use_entropy_regularization=self._config.use_entropy_regularization,
        behavior_regularization_type=self._config.behavior_regularization_type,
        behavior_regularization_alpha=self._config.behavior_regularization_alpha,
        num_cql_actions=self._config.num_cql_actions,
        td_target_method=self._config.td_target_method,
        critic_random_init=self._config.critic_random_init,
        use_img_encoder=self._config.use_img_encoder,
        img_encoder_params_ckpt_path=self._config.img_encoder_params_ckpt_path,
        rem_mode=self._config.rem_mode,
        mimo_using_adamw=self._config.mimo_using_adamw,
        mimo_using_obs_tile=self._config.mimo_using_obs_tile,
        mimo_using_act_tile=self._config.mimo_using_act_tile,
        policy_lr=self._config.policy_lr,
        q_lr=self._config.q_lr,
        counter=counter,
        logger=self._logger_fn(),
        num_sgd_steps_per_step=self._config.num_sgd_steps_per_step,)

  def make_actor(
      self,
      random_key,
      policy_network,
      adder = None,
      variable_source = None,
      force_eval_with_q_filter=False):
    assert variable_source is not None
    if self._config.eval_with_q_filter or force_eval_with_q_filter:
      params_to_get = ['policy', 'all_q']
      if self._config.use_img_encoder:
        params_to_get.append('img_encoder')
      # return actors.GenericActor(
      #     policy_network,
      #     random_key=random_key,
      #     # Inference happens on CPU, so it's better to move variables there too.
      #     variable_client=variable_utils.VariableClient(
      #         variable_source, params_to_get, device='cpu'),
      #     adder=adder,
      #     backend='cpu')
      return actors.GenericActor(
          actor=policy_network,
          random_key=random_key,
          # Inference happens on CPU, so it's better to move variables there too.
          variable_client=variable_utils.VariableClient(
              variable_source, params_to_get, device='cpu'),
          adder=adder,
      )
    else:
      params_to_get = ['policy']
      if self._config.use_img_encoder:
        params_to_get.append('img_encoder')
      return actors.GenericActor(
          actor=policy_network,
          random_key=random_key,
          # Inference happens on CPU, so it's better to move variables there too.
          variable_client=variable_utils.VariableClient(
              variable_source, params_to_get, device='cpu'),
          adder=adder,
      )

  def make_replay_tables(
      self,
      environment_spec,
  ):
    """Create tables to insert data into."""
    return []

  def make_dataset_iterator(
      self,
      replay_client):
    """Create a dataset iterator to use for learning/updating the agent."""
    return None

  def make_adder(self,
                 replay_client):
    """Create an adder which records data generated by the actor/environment."""
    return None
