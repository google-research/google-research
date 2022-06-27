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

#!/bin/bash
echo "!!!!! THIS IS JUST A COMMAND FOR QUICKLY TESTING TO MAKE SURE THE CODEBASE ISN'T BROKEN. FOR INSTRUCTIONS PLEASE REFER TO README.md !!!!!"
python3 -m jrl.localized.runner \
--pdb_post_mortem \
--debug_nans=False \
--create_saved_model_actor=False \
--num_steps 100 \
--eval_every_steps 50 \
--episodes_per_eval 10 \
--batch_size 64 \
--root_dir '/tmp/test_msg_deep_ensembles' \
--seed 42 \
--algorithm 'msg' \
--task_class 'd4rl' \
--task_name 'antmaze-large-diverse-v0' \
--gin_bindings='msg.config.MSGConfig.num_sgd_steps_per_step=1' \
--gin_bindings='msg.config.MSGConfig.ensemble_size=64' \
--gin_bindings='msg.config.MSGConfig.ensemble_method="deep_ensembles"' \
--gin_bindings='msg.config.MSGConfig.td_target_method="independent"' \
--gin_bindings='msg.config.MSGConfig.beta=-8' \
--gin_bindings='msg.config.MSGConfig.behavior_regularization_alpha=0.1' \
--gin_bindings='msg.config.MSGConfig.behavior_regularization_type="v1"' \
--gin_bindings='msg.config.MSGConfig.num_cql_actions=1' \
--gin_bindings='msg.config.MSGConfig.use_random_weighting_in_critic_loss=True' \
--gin_bindings='msg.config.MSGConfig.num_bc_iters=50' \
--gin_bindings='msg.config.MSGConfig.num_q_repr_pretrain_iters=0' \
--gin_bindings='msg.config.MSGConfig.pretrain_temp=1' \
--gin_bindings='msg.config.MSGConfig.use_sass=True' \
--gin_bindings='msg.config.MSGConfig.q_lr=3e-4' \
--gin_bindings='msg.config.MSGConfig.policy_lr=3e-5' \
--gin_bindings='msg.config.MSGConfig.use_ema_target_critic_params=True' \
--gin_bindings='msg.config.MSGConfig.use_entropy_regularization=True' \
--gin_bindings='msg.config.MSGConfig.actor_network_hidden_sizes=(256, 256, 256)' \
--gin_bindings='msg.config.MSGConfig.critic_network_hidden_sizes=(256, 256, 256)' \
--gin_bindings='msg.config.MSGConfig.use_double_q=False' \
--gin_bindings='msg.config.MSGConfig.networks_init_type="glorot_also_dist"' \
--gin_bindings='msg.config.MSGConfig.critic_random_init=False' \
--gin_bindings='msg.config.MSGConfig.perform_sarsa_q_eval=False' \
--gin_bindings='msg.config.MSGConfig.eval_with_q_filter=False' \
--gin_bindings='msg.config.MSGConfig.num_eval_samples=32' \
--gin_bindings='msg.config.MSGConfig.mimo_using_adamw=False' \
--gin_bindings='msg.config.MSGConfig.mimo_using_obs_tile=False' \
--gin_bindings='msg.config.MSGConfig.mimo_using_act_tile=False'
