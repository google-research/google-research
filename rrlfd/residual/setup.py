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
"""Setup utilities for residual training."""

import collections
import datetime
import os
from typing import Sequence

from absl import flags
from acme import specs
from acme import wrappers
from acme.agents.tf import d4pg
from acme.agents.tf import dmpo
from acme.agents.tf import mpo
from acme.tf import networks as tf_networks
from acme.tf import utils as tf_utils
from acme.utils import counting
from acme.utils import loggers
import gym
from mime.envs import utils as mime_env_utils
import numpy as np
import sonnet as snt
import tensorflow as tf

from rrlfd import adroit_ext  # pylint: disable=unused-import
from rrlfd import adroit_utils
from rrlfd import env_wrappers
from rrlfd import mime_utils
from rrlfd.bc import bc_agent
from rrlfd.bc import train_utils
from rrlfd.residual import agents
from rrlfd.residual import environment_loop
from rrlfd.residual import networks
from tensorflow.io import gfile


# Agent flags
flags.DEFINE_string('agent', 'DMPO', 'Acme agent to train.')
flags.DEFINE_float('critic_vmin', -2.0, 'Vmin to use in distributional critic.')
flags.DEFINE_float('critic_vmax', 2.0, 'Vmax to use in distributional critic.')
flags.DEFINE_float('discount', 0.99, 'Discount factor for TD updates.')
flags.DEFINE_integer('critic_num_atoms', 51,
                     'Number of atoms to use in distributional critic.')

flags.DEFINE_list('rl_policy_layer_sizes', [256, 256, 256],
                  'Sizes of fully connected layers in policy network.')
flags.DEFINE_list('rl_critic_layer_sizes', [512, 512, 512],
                  'Sizes of fully connected layers in policy network.')
flags.DEFINE_integer('rl_batch_size', 64, 'Batch size for RL updates.')
flags.DEFINE_boolean('write_acme_checkpoints', True,
                     'Checkpoint argument to pass to acme learners.')
flags.DEFINE_float('policy_init_std', 0.01,
                   'Initial standard devation to use for gaussian residual '
                   'policy.')
flags.DEFINE_float('policy_weights_init_scale', 1e-5,
                   'Scale parameter of VarianceScaling initialization for '
                   '(D4PG) policy network.')
flags.DEFINE_integer('min_replay_size', 1000,
                     'Minimum size of the replay buffer before using it for '
                     'updates.')
flags.DEFINE_integer('max_replay_size', 1_000_000,
                     'Maximum size of the replay buffer.')
flags.DEFINE_float('policy_lr', 1e-4, 'Learning rate for policy optimizer.')
flags.DEFINE_float('critic_lr', 1e-4, 'Learning rate for critic optimizer.')
# flags.DEFINE_float('dual_lr', 1e-2, 'Learning rate for dual optimizer.')


# Environment flags
flags.DEFINE_integer('max_episode_steps', None,
                     'If set, override environment default for max episode '
                     'length during training.')
flags.DEFINE_boolean('dense_reward', False, 'If True, use dense reward signal.')
flags.DEFINE_float('dense_reward_multiplier', 1.0,
                   'Multiplier for dense rewards.')
flags.DEFINE_float('lateral_friction', 0.5, 'Friction coefficient for cube.')
flags.DEFINE_boolean('render', False, 'If True, render the environment.')
flags.DEFINE_boolean('use_egl', False, 'If True, use EGL for rendering.')


FLAGS = flags.FLAGS


def setup_counting():
  counter = counting.Counter()
  # Add keys to counter (needed for CSV column names).
  counter.increment(steps=0, walltime=0)
  return counter


def set_job_id():
  """Define job id for output paths.

  Returns:
    job_id: Identifier for output paths.
  """
  job_id = FLAGS.job_id
  if not job_id:
    job_id = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
  return job_id


def setup_logging(top_logdir):
  """Initialize CSV and TensorBoard loggers."""
  logdir = env_logger = agent_logger = summary_writer = summary_dir = None
  if top_logdir is not None:
    job_id = set_job_id()
    logdir = os.path.join(top_logdir, job_id)
    summary_dir = os.path.join(top_logdir, 'tb', job_id)
    summary_writer = tf.summary.create_file_writer(summary_dir)
    env_logger = loggers.CSVLogger(logdir, 'env_loop')
    agent_logger = loggers.CSVLogger(logdir, 'learner')

  return logdir, env_logger, agent_logger, summary_writer, summary_dir


def get_original_demos_path(bc_ckpt_path):
  r"""Get dataset used to train BC policy at bc_ckpt_path.

  E.g.
  /path/to/topdir/bc_policy/subdirectories/dataset_name/... ->
  /path/to/topdir/bc_demos/subdirectories/dataset_name.pkl

  Args:
    bc_ckpt_path: Path to BC policy checkpoint.

  Returns:
    demos_path: Path to demonstration dataset.
  """
  demos_path = FLAGS.original_demos_path
  if demos_path is None:
    topdir, subdirs = bc_ckpt_path.split('bc_policy/')
    demos_path = os.path.join(topdir, 'bc_demos')

    subdirs = subdirs.split('/')
    while subdirs:
      if subdirs[0] + '.pkl' in gfile.ListDir(demos_path):
        demos_path = os.path.join(demos_path, subdirs[0] + '.pkl')
      if subdirs[0] in gfile.ListDir(demos_path):
        demos_path = os.path.join(demos_path, subdirs[0])
      subdirs = subdirs[1:]
  return demos_path


def load_saved_bc_agent(
    ckpt_to_load, network_type, late_fusion, input_type, domain,
    binary_grip_action, num_input_frames, crop_frames, full_image_size,
    crop_margin_size, target_offsets, visible_state_features, action_norm,
    signals_norm, last_activation, fc_layer_sizes, weight_decay,
    max_demos_to_load, max_demo_length, val_size, val_full_episodes, split_seed,
    env, task,
    # For preprocessing demos (residual BC) only
    grip_action_from_state=False,
    zero_action_keeps_state=False,
    early_closing=False,
    ):
  """Load a trained behavioral cloning agent."""
  # TODO(minttu): No need to set split seed, val full episodes etc. since split
  # is always saved.
  # TODO(minttu): Save full model definition.
  if ckpt_to_load is None and FLAGS.policy_init_path is None:
    network_type = None
  if input_type == 'full_state':
    input_type = None
  agent = bc_agent.BCAgent(
      network_type=network_type,
      input_type=input_type,
      binary_grip_action=binary_grip_action,
      grip_action_from_state=grip_action_from_state,
      zero_action_keeps_state=zero_action_keeps_state,
      early_closing=early_closing,
      num_input_frames=num_input_frames,
      crop_frames=crop_frames,
      full_image_size=full_image_size,
      crop_size=full_image_size - crop_margin_size,
      target_offsets=target_offsets,
      visible_state_features=visible_state_features,
      action_norm=action_norm,
      signals_norm=signals_norm,
      action_space='tool_lin' if domain == 'mime' else task,
      last_activation=last_activation,
      fc_layer_sizes=fc_layer_sizes,
      # TODO(minttu): Might not actually need if not training.
      weight_decay=weight_decay,
      env=env.env,
      late_fusion=late_fusion)
  split_dir = None
  if ckpt_to_load is not None and network_type is not None:
    print('Loading from', ckpt_to_load)
    split_dir = os.path.dirname(ckpt_to_load)
    agent.restore_from_ckpt(ckpt_to_load, compile_model=True)

  # Set correct action normalization stats for base agent.
  if action_norm == 'zeromean_unitvar':
    if FLAGS.original_demos_file is not None:
      original_demos_file = FLAGS.original_demos_file
    else:
      original_demos_file = get_original_demos_path(ckpt_to_load)
    train_utils.reset_action_stats(
        original_demos_file, max_demos_to_load, max_demo_length, val_size,
        val_full_episodes, split_seed, agent, split_dir)
  return agent


def set_visible_features(domain, task, visible_state):
  # TODO(minttu): bc/train.py has this very same functionality: share.
  if not isinstance(visible_state, str) and isinstance(visible_state, Sequence):
    return visible_state
  domain_utils = adroit_utils if domain == 'adroit' else mime_utils
  features = domain_utils.get_visible_features_for_task(task, visible_state)
  return features


def create_env_copies(
    domain, task, seed, input_type, num_input_frames, image_size, visible_state,
    use_base_agent_image_shape, late_fusion, max_episode_steps=(None, None)):
  """Create train and eval envs of possibly different max episode lengths."""
  envs = [None, None]
  for i in range(2):
    if domain == 'mime':
      # Configure and initialize mime environment.
      env = env_wrappers.MimeWrapper(
          task, FLAGS.use_egl, seed, input_type, image_size, FLAGS.render,
          FLAGS.lateral_friction, max_episode_steps[i])

      env = env_wrappers.MimeRewardWrapper(
          env, sparse=not FLAGS.dense_reward,
          dense_reward_multiplier=FLAGS.dense_reward_multiplier)

      # Only expose visible keys.
      visible_keys = visible_state
      if input_type != 'full_state':
        visible_keys = [input_type] + visible_keys
      env = env_wrappers.MimeVisibleKeysWrapper(env, visible_keys)

      # Convert from gym to dm_env format (specs, TimeSteps);
      # retain relevant info fields.
      env = env_wrappers.GymMimeAdapter(env)

      # Flatten action dictionaries by sorted keys.
      env = env_wrappers.FlatActionWrapper(env)

    else:
      # Configure and initialize Adroit environment.
      env = env_wrappers.AdroitWrapper(task, image_size, max_episode_steps[i])

      env = env_wrappers.AdroitRewardWrapper(
          env, sparse=not FLAGS.dense_reward,
          dense_reward_multiplier=FLAGS.dense_reward_multiplier)

      # Convert from gym to dm_env format (specs, TimeSteps);
      # retain relevant info fields.
      env = env_wrappers.GymAdroitAdapter(
          env, end_on_success=FLAGS.end_on_success)
    env = wrappers.SinglePrecisionWrapper(env)
    # Record all evaluation videos.
    # record_every = 100 if i == 0 else 1
    # path = logdir if i == 0 else os.path.join(logdir, 'eval')
    # env = env_wrappers.KeyedVideoWrapper(
    #     env, visual_key=input_type, frame_rate=10, record_every=record_every,
    #     path=path)
    if input_type in ['depth', 'rgb', 'rgbd']:
      stack_length = {input_type: num_input_frames}
      env = env_wrappers.CustomStackingWrapper(env, stack_length)
    if input_type == 'rgb':
      if use_base_agent_image_shape:
        # BCAgent expects frames in this shape.
        env = env_wrappers.TransposeImageWrapper(env, input_type)
      elif not late_fusion:
        # ObservationNet expects frames in this shape.
        env = env_wrappers.EarlyFusionImageWrapper(env, input_type)
    envs[i] = env
  return envs


def make_environment_loop(
    domain, task, seed, input_type, num_input_frames, visible_state, image_size,
    use_base_agent_image_shape, late_fusion, max_train_episode_steps, agent,
    counter, env_logger, summary_writer):
  """Initialize environment loop."""
  env, eval_env = create_env_copies(
      domain=domain,
      task=task,
      seed=seed,
      input_type=input_type,
      num_input_frames=num_input_frames,
      image_size=image_size,
      visible_state=visible_state,
      use_base_agent_image_shape=use_base_agent_image_shape,
      late_fusion=late_fusion,
      max_episode_steps=(max_train_episode_steps, None))
  cam_env = None
  cam_eval_env = None
  if input_type not in ['depth', 'rgb', 'rgbd']:
    # Create second environment with camera observations to write videos.
    cam_env, cam_eval_env = create_env_copies(
        domain=domain,
        task=task,
        seed=seed,
        input_type='rgb',  # was previously set to depth for mime
        num_input_frames=1,
        image_size=image_size,
        visible_state=visible_state,
        use_base_agent_image_shape=use_base_agent_image_shape,
        late_fusion=late_fusion,
        max_episode_steps=(max_train_episode_steps, None))
  env_loop = environment_loop.EnvironmentLoop(
      environment=env,
      eval_environment=eval_env,
      cam_environment=cam_env,
      cam_eval_environment=cam_eval_env,
      actor=agent,
      counter=counter,
      logger=env_logger,
      summary_writer=summary_writer)
  return env_loop


def define_residual_spec(
    rl_features,
    env,
    base_agent,
    action_norm,
    action_norm_scale=1.0,
    include_base_action=True,
    include_base_feats=True,
    base_network=None):
  # TODO(minttu): pass in GymWrapper(env) without any other wrapper classes.
  """Defines environment observation and action spaces as seen by the RL agent.

  Args:
    rl_features: A list of state features visible to the agent. If set, they
      replace any visual features.
    env: The environment which defines the action space, rewards and discounts.
    base_agent: base agent to use in residual training.
    action_norm: bc_agent.ActionSpace object defining action normalization.
    action_norm_scale: Scalar by which to scale residual action normalization.
    include_base_action: If True, add base agent action to spec.
    include_base_feats: If True, add features given by base agent to spec.
    base_network: Network type used by the base agent, if applicable.

  Returns:
    residual_spec: An acme.specs.EnvironmentSpec instance defining the residual
      spec.
  """
  feats_spec = collections.OrderedDict()
  visible_state_dim = 0
  # This check allows train_bc to use this function to set residual spec
  # without using env wrappers.
  if isinstance(env, gym.Env):
    for k, v in env.observation_space.spaces.items():
      if k in rl_features:
        visible_state_dim += v.shape[0] if v.shape else 1
  else:
    if FLAGS.domain == 'mime':
      obs_space = mime_env_utils.make_dict_space(env.scene, *rl_features).spaces
    else:
      obs_space = env.observation_spec()
    for k, v in obs_space.items():
      if k in rl_features:
        visible_state_dim += v.shape[0] if v.shape else 1
  if include_base_feats:
    base_feat_size = {
        'resnet18_narrow32': 256,
        'hand_vil': 200,
    }[base_network]
    feats_spec['feats'] = specs.Array([base_feat_size], np.float32, 'feats')
  if visible_state_dim > 0:
    feats_spec['visible_state'] = (
        specs.Array([visible_state_dim], np.float32, 'visible_state'))
  if include_base_action:
    feats_spec['base_action'] = specs.Array(
        [base_agent.action_target_dim], np.float32, 'base_action')
  if FLAGS.rl_observation_network is not None:
    # TODO(minttu): Get image size from env observation spec.
    if FLAGS.input_type == 'depth':
      feats_spec['depth'] = specs.Array(
          [FLAGS.image_size, FLAGS.image_size, 3], np.uint8, 'depth')
    elif FLAGS.input_type == 'rgb':
      image_size = FLAGS.image_size
      rgb_shape = (
          [3, image_size, image_size, 3] if FLAGS.late_fusion
          else [image_size, image_size, 9])
      feats_spec['rgb'] = specs.Array(rgb_shape, np.uint8, 'rgb')
  if isinstance(env, gym.Env):
    env_action_spec = env.action_space
    env_action_spec.minimum = env_action_spec.low
    env_action_spec.maximum = env_action_spec.high
    env_action_spec.name = 'action'
    # Concatenating fields here since it is non-trivial to use dictionary
    # observations with DemoReader's generator.
    concat_shape = np.sum([a.shape for a in feats_spec.values()])
    feats_spec = collections.OrderedDict()
    feats_spec['residual_obs'] = specs.Array(
        (concat_shape,), np.float32, 'residual_obs')
  else:
    env_action_spec = env.action_spec()
  env_min = env_action_spec.minimum
  env_max = env_action_spec.maximum
  # Allow (at the extreme) to fully reverse a base action (from one action
  # space limit to the opposite limit).
  min_residual = env_min - env_max if include_base_action else env_min
  max_residual = env_max - env_min if include_base_action else env_max
  print('min residual', min_residual, 'max residual', max_residual)
  residual_action_space = bc_agent.ActionSpace(
      action_norm, env=env, scale=action_norm_scale)
  if action_norm in ['centered', 'zeromean_unitvar']:
    # Reuse stats; normalization scheme may still be different.
    residual_action_space.mean = base_agent.action_space.mean
    residual_action_space.std = base_agent.action_space.std
  norm_min = residual_action_space.normalize_flat(min_residual)
  norm_max = residual_action_space.normalize_flat(max_residual)
  norm_action_spec = specs.BoundedArray(
      shape=env_action_spec.shape,
      dtype=env_action_spec.dtype,
      minimum=norm_min,
      maximum=norm_max,
      name=env_action_spec.name)
  print(env_action_spec)
  print(norm_action_spec)

  if isinstance(env, gym.Env):
    reward_spec = specs.BoundedArray(
        shape=(), dtype=float, minimum=env.reward_range[0],
        maximum=env.reward_range[1], name='reward')
  else:
    reward_spec = env.reward_spec()
  if isinstance(env, gym.Env):
    discount_spec = specs.BoundedArray(
        shape=(), dtype=float, minimum=0., maximum=1., name='discount')
  else:
    discount_spec = env.discount_spec()
  # residual_spec = specs.make_environment_spec(env)
  # Use same normalization for base agent and residual agent.
  residual_spec = specs.EnvironmentSpec(
      observations=feats_spec,
      actions=norm_action_spec,
      rewards=reward_spec,
      discounts=discount_spec)
  print('Residual spec', residual_spec)
  return residual_spec


def make_residual_bc_agent(
    residual_spec,
    base_agent,
    action_norm,
    action_norm_scale=1.0,
    binary_grip_action=False,
    env=None,
    visible_state_features=None):
  """Initialize behavioral cloning for residual control."""
  agent_networks = networks.make_bc_network(
      action_spec=residual_spec.actions,
      policy_layer_sizes=FLAGS.rl_policy_layer_sizes,
      policy_init_std=FLAGS.policy_init_std,
      binary_grip_action=binary_grip_action,
      )
  # TODO(minttu): binary_grip_action
  residual_agent = bc_agent.ResidualBCAgent(
      base_agent=base_agent,
      residual_spec=residual_spec,
      # observation_network=agent_networks['observation'],
      policy_network=agent_networks['policy'],
      action_norm=action_norm,
      action_norm_scale=action_norm_scale,
      env=env,
      visible_state_features=visible_state_features)
  return residual_agent


def make_acme_agent(
    environment_spec,
    residual_spec,
    obs_network_type,
    crop_frames,
    full_image_size,
    crop_margin_size,
    late_fusion,
    binary_grip_action=False,
    input_type=None,
    counter=None,
    logdir=None,
    agent_logger=None):
  """Initialize acme agent based on residual spec and agent flags."""
  # TODO(minttu): Is environment_spec needed or could we use residual_spec?
  del logdir  # Setting logdir for the learner ckpts not currently supported.
  obs_network = None
  if obs_network_type is not None:
    obs_network = agents.ObservationNet(
        network_type=obs_network_type,
        input_type=input_type,
        add_linear_layer=False,
        crop_frames=crop_frames,
        full_image_size=full_image_size,
        crop_margin_size=crop_margin_size,
        late_fusion=late_fusion)

  eval_policy = None
  if FLAGS.agent == 'MPO':
    agent_networks = networks.make_mpo_networks(
        environment_spec.actions,
        policy_init_std=FLAGS.policy_init_std,
        obs_network=obs_network)

    rl_agent = mpo.MPO(
        environment_spec=residual_spec,
        policy_network=agent_networks['policy'],
        critic_network=agent_networks['critic'],
        observation_network=agent_networks['observation'],
        discount=FLAGS.discount,
        batch_size=FLAGS.rl_batch_size,
        min_replay_size=FLAGS.min_replay_size,
        max_replay_size=FLAGS.max_replay_size,
        policy_optimizer=snt.optimizers.Adam(FLAGS.policy_rl),
        critic_optimizer=snt.optimizers.Adam(FLAGS.critic_lr),
        counter=counter,
        logger=agent_logger,
        checkpoint=FLAGS.write_acme_checkpoints,
    )
  elif FLAGS.agent == 'DMPO':
    agent_networks = networks.make_dmpo_networks(
        environment_spec.actions,
        policy_layer_sizes=FLAGS.rl_policy_layer_sizes,
        critic_layer_sizes=FLAGS.rl_critic_layer_sizes,
        vmin=FLAGS.critic_vmin,
        vmax=FLAGS.critic_vmax,
        num_atoms=FLAGS.critic_num_atoms,
        policy_init_std=FLAGS.policy_init_std,
        binary_grip_action=binary_grip_action,
        obs_network=obs_network)

    # spec = residual_spec if obs_network is None else environment_spec
    spec = residual_spec
    rl_agent = dmpo.DistributionalMPO(
        environment_spec=spec,
        policy_network=agent_networks['policy'],
        critic_network=agent_networks['critic'],
        observation_network=agent_networks['observation'],
        discount=FLAGS.discount,
        batch_size=FLAGS.rl_batch_size,
        min_replay_size=FLAGS.min_replay_size,
        max_replay_size=FLAGS.max_replay_size,
        policy_optimizer=snt.optimizers.Adam(FLAGS.policy_lr),
        critic_optimizer=snt.optimizers.Adam(FLAGS.critic_lr),
        counter=counter,
        # logdir=logdir,
        logger=agent_logger,
        checkpoint=FLAGS.write_acme_checkpoints,
    )
    # Learned policy without exploration.
    eval_policy = (
        tf.function(
            snt.Sequential([
                tf_utils.to_sonnet_module(agent_networks['observation']),
                agent_networks['policy'],
                tf_networks.StochasticMeanHead()])
            )
        )
  elif FLAGS.agent == 'D4PG':
    agent_networks = networks.make_d4pg_networks(
        residual_spec.actions,
        vmin=FLAGS.critic_vmin,
        vmax=FLAGS.critic_vmax,
        num_atoms=FLAGS.critic_num_atoms,
        policy_weights_init_scale=FLAGS.policy_weights_init_scale,
        obs_network=obs_network)

    # TODO(minttu): downscale action space to [-1, 1] to match clipped gaussian.
    rl_agent = d4pg.D4PG(
        environment_spec=residual_spec,
        policy_network=agent_networks['policy'],
        critic_network=agent_networks['critic'],
        observation_network=agent_networks['observation'],
        discount=FLAGS.discount,
        batch_size=FLAGS.rl_batch_size,
        min_replay_size=FLAGS.min_replay_size,
        max_replay_size=FLAGS.max_replay_size,
        policy_optimizer=snt.optimizers.Adam(FLAGS.policy_lr),
        critic_optimizer=snt.optimizers.Adam(FLAGS.critic_lr),
        sigma=FLAGS.policy_init_std,
        counter=counter,
        logger=agent_logger,
        checkpoint=FLAGS.write_acme_checkpoints,
    )

    # Learned policy without exploration.
    eval_policy = tf.function(
        snt.Sequential([
            tf_utils.to_sonnet_module(agent_networks['observation']),
            agent_networks['policy']]))

  else:
    raise NotImplementedError('Supported agents: MPO, DMPO, D4PG.')
  return rl_agent, eval_policy


def init_policy_networks(init_network, rl_agent):
  """Initialize an RL agent's policy network weights from an initial network.

  Args:
    init_network: Network weights to copy.
    rl_agent: Acme agent with a learner, with policy and target policy networks.
  """

  def init_mlp(policy_network, init_network):
    if isinstance(init_network.linear, snt.Sequential):
      for v, var in enumerate(policy_network.variables):
        if len(var.shape) == 1:
          var.assign(init_network.linear.variables[v][:var.shape[0]])
        else:
          var.assign(init_network.linear.variables[v][:, :var.shape[1]])
    else:
      for var in policy_network.variables:
        print(var.name)
        # Init network may include action target augmentation.
        if var.name == 'ArmPolicyNormalDiagHead/mean/w:0':
          var.assign(init_network.linear.kernel[:, :var.shape[1]])
        elif var.name == 'ArmPolicyNormalDiagHead/mean/b:0':
          var.assign(init_network.linear.bias[:var.shape[0]])
        # TODO(minttu): Handle networks with more layers.

  learner = rl_agent._learner  # pylint: disable=protected-access

  for net in [learner._policy_network, learner._target_policy_network]:  # pylint: disable=protected-access
    init_mlp(net, init_network)


def init_observation_networks(init_network, rl_agent):
  learner = rl_agent._learner  # pylint: disable=protected-access

  for network in [
      learner._observation_network, learner._target_observation_network]:  # pylint: disable=protected-access
    for w in init_network.variables[-3:]:
      print('Skipping', w.name)
    # log std, weight & bias
    network._transformation.network.set_weights(init_network.get_weights()[:-3])  # pylint: disable=protected-access


def load_acme_agent(agent, ckpt_path):
  print('Loading policy weights from', ckpt_path)
  checkpoint = tf.train.Checkpoint(module=agent._learner._policy_network)  # pylint: disable=protected-access
  checkpoint.restore(ckpt_path)


def load_agent(agent, ckpt_path):
  """Load agent policy weights and the number of steps trained from ckpt."""
  load_acme_agent(agent.rl_agent, ckpt_path)
  ckpt_parts = os.path.basename(ckpt_path).split('_')
  if len(ckpt_parts) > 1:
    loaded_step = ckpt_parts[1]
    print('Loaded step', loaded_step)
  else:
    loaded_step = ''
  return loaded_step


def save_acme_agent(agent, logdir):
  """Save tf.train.Checkpoints for policy and observation networks."""
  out_path = os.path.join(logdir, 'policy_net')
  print('Saving policy weights to', out_path)
  checkpoint = tf.train.Checkpoint(
      module=agent.rl_agent._learner._policy_network)  # pylint: disable=protected-access
  checkpoint.save(out_path)
  if agent.rl_observation_network_type is not None:
    out_path = os.path.join(logdir, 'observation_net')
    print('Saving observation weights to', out_path)
    checkpoint = tf.train.Checkpoint(
        module=agent.rl_agent._learner._observation_network)  # pylint: disable=protected-access
    checkpoint.save(out_path)
