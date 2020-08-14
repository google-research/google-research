# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Setup the experiment components such as agents and environment."""
from __future__ import absolute_import
from __future__ import division


from clevr_robot_env import ClevrEnv

from hal.agent.DQN.dqn import DoubleDQN
from hal.agent.RND.rnd import StateRND
from hal.agent.UVFA_repr.uvfa_image import ImageUVFA
from hal.agent.UVFA_repr.uvfa_state import StateUVFA
from hal.agent.UVFA_repr_tf2.uvfa_image import ImageUVFA2
from hal.agent.UVFA_repr_tf2.uvfa_state import StateUVFA2
from hal.learner.hir import HIR
from hal.learner.hir_fn_approx import FnApproxHIR
from hal.learner.hir_maxent_irl import MaxentIrlHIR
from hal.learner.hir_maxent_irl_fn_approx import FnApproxMaxentIrlHIR
from hal.learner.hir_rnd import RndHIR
from hal.utils.replay_buffer import ReplayBuffer


def experiment_setup(cfg, use_tf2, use_nn_relabeling):
  """Set up the experiment by creating agent, learner etc.

  Args:
    cfg: a Config object that specifies experimental configuration
    use_tf2:  Use Tensorflow
    use_nn_relabeling: Use neural network instead of environment to relabel

  Returns:
    env: the reinforcement learning environment
    learner: the learner in charge of doing the learning with the agent
    replay_buffer: the buffer where past experience is sotred
    agent: the RL agent that interacts with the environment
    extra_components: extra artifcats that may be needed such as a mirror agent
  """
  # miscellaneous parameters
  res = 64
  cfg.descriptor_length = 64
  cfg.inner_product_length = 32
  cfg.encoder_n_unit = 32
  cfg.embedding_size = 8

  # environment
  env = get_env(cfg)

  s = list(env.reset().shape)

  if cfg.obs_type == 'order_invariant':
    obs_dim = [None, s[-1]]
  elif cfg.obs_type == 'image':
    obs_dim = [res, res, 3]
  elif cfg.obs_type == 'state':
    obs_dim = [10]
  else:
    raise ValueError('Obs type {} is not recognized'.format(cfg.obs_type))
  cfg.obs_dim = obs_dim

  if cfg.action_type == 'discrete':
    ac_dim = [800]
    per_input_ac_dim = None
  elif cfg.action_type == 'perfect':
    ac_dim = [40]
    per_input_ac_dim = 8
  else:
    raise ValueError('Action type {} is not recognized'.format(cfg.action_type))

  cfg.ac_dim = ac_dim
  cfg.per_input_ac_dim = per_input_ac_dim

  # architecture settings
  cfg.conv_layer_config = [(48, 8, 2), (128, 5, 2), (64, 3, 1)]

  dense_layer_config = [256, 512, 1024]
  if cfg.obs_type == 'image':
    dense_layer_config = [512, 512]
  cfg.dense_layer_config = dense_layer_config

  cfg.max_sequence_length = 21  # TODO(ydjiang): conditioned on environment

  # learner
  if use_nn_relabeling:
    if 'maxent_irl' in cfg.keys() and cfg.maxent_irl:
      learner = FnApproxMaxentIrlHIR(cfg)
    else:
      learner = FnApproxHIR(cfg)
  elif 'use_rnd' in cfg.keys() and cfg.use_rnd:
    learner = RndHIR(cfg)
  elif 'maxent_irl' in cfg.keys() and cfg.maxent_irl:
    learner = MaxentIrlHIR(cfg)
  else:
    learner = HIR(cfg)

  vocab_list = learner.vocab_list
  cfg.vocab_list = vocab_list
  cfg.vocab_size = len(vocab_list)

  # replay buffer
  replay_buffer = ReplayBuffer(cfg.buffer_size)

  # agent
  if cfg.obs_type == 'order_invariant':
    if use_tf2:
      agent = StateUVFA2(cfg)
    else:
      agent = StateUVFA(cfg)
  elif cfg.obs_type == 'image':
    if use_tf2:
      agent = ImageUVFA2(cfg)
    else:
      agent = ImageUVFA(cfg)
  else:
    raise ValueError('Unrecognized obs type: {}'.format(cfg.obs_type))

  extra_components = {}
  if 'use_rnd' in cfg.keys() and cfg.use_rnd:
    rnd_module = StateRND()
    rnd_agent = DoubleDQN()
    extra_components = {
        'rnd_module': rnd_module,
        'rnd_agent': rnd_agent
    }

  return env, learner, replay_buffer, agent, extra_components


def get_env(cfg):
  """Get a Clevr-robot environment for training."""
  res = cfg.img_resolution
  direct_obs = cfg.obs_type == 'order_invariant' or cfg.obs_type == 'state'
  r_scale = 1.0 if 'reward_scale' not in cfg.keys() else cfg.reward_scale
  if 'agent_type' in cfg.keys() and cfg.agent_type == 'franka':
    model_scale = .27
    zoom_factor = 0.25
    azimuth = 45
  else:
    model_scale = 1.0
    zoom_factor = 1.0
    azimuth = None
  return ClevrEnv(num_object=5,
                  agent_type=cfg.agent_type,
                  env_type=cfg.scenario_type,
                  franka_scaling_factor=model_scale,
                  random_start=True,
                  maximum_episode_steps=50,
                  description_num=0,
                  action_type=cfg.action_type,
                  obs_type=cfg.obs_type,
                  direct_obs=direct_obs,
                  reward_scale=r_scale,
                  shape_val=cfg.reward_shape_val,
                  resolution=res,
                  use_subset_instruction=cfg.use_subset_instruction,
                  frame_skip=cfg.frame_skip,
                  use_polar=cfg.use_polar,
                  suppress_other_movement=cfg.suppress,
                  variable_scene_content=cfg.diverse_scene_content,
                  use_movement_bonus=cfg.use_movement_bonus,
                  zoom_factor=zoom_factor,
                  azimuth=azimuth)
