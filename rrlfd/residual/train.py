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
"""Train a residual policy on top of a learned agent.

Usage:

Use case --> flags to set

 1) Use base agent
   a) Use feats from base agent --> network && bc_ckpt_to_load
   b) Learn new feats  --> network && bc_ckpt_to_load && rl_observation_network
   c) Init feats from base agent but finetune
     --> network && bc_ckpt_to_load && rl_observation_network
     && init_feats_from_bc && predict_residual

 2) Use RL only
   a) Learn new feats --> rl_observation_network (if input type is visual)
   b) Init feats & policy from base agent but finetune
     --> network && bc_ckpt_to_load && rl_observation_network && init_from_bc
   c) Init feats from base agent but finetune
     --> network && bc_ckpt_to_load && rl_observation_network
     && init_feats_from_bc

  3) Use base controller + rl observation net from scratch
     --> base_controller && rl_observation_network
"""

import os

from absl import app
from absl import flags
from acme import specs
import numpy as np
import tensorflow as tf

from rrlfd.residual import agents
from rrlfd.residual import eval_utils
from rrlfd.residual import setup
from tensorflow.io import gfile


flags.DEFINE_string('domain', None, 'Domain from which to load task.')
flags.DEFINE_string('task', None, 'Task to solve.')
flags.DEFINE_enum('input_type', 'depth', ['depth', 'rgb', 'rgbd', 'full_state'],
                  'Input modality.')

flags.DEFINE_integer('num_episodes', 10000, 'Number of episodes to run for.')
flags.DEFINE_integer('seed', 2, 'Experiment seed.')
flags.DEFINE_integer('eval_seed', 1, 'Environtment seed for evaluation.')
flags.DEFINE_boolean('increment_eval_seed', False,
                     'If True, increment eval seed after each eval episode.')

flags.DEFINE_integer('num_eval_episodes', 100,
                     'Number of episodes to evaluate.')
flags.DEFINE_boolean('collapse_in_eval', True,
                     'If True, collapse RL policy to its mean in evaluation.')
flags.DEFINE_boolean('stop_if_stuck', False,
                     'If True, end episode if observations and actions are '
                     'stuck.')
flags.DEFINE_boolean('end_on_success', False,
                     'If True, end episode early if success criteria is met.')
flags.DEFINE_integer('eval_freq', 100_000,
                     'Frequency (in environment training steps) with which to '
                     'evaluate policy.')
flags.DEFINE_boolean('eval_only', False,
                     'If True, evaluate policy ckpts of trained policy.')

# Flags for BC agent.
flags.DEFINE_boolean('binary_grip_action', True,
                     'If True, use open/close action space for gripper. Else '
                     'use gripper velocity.')
flags.DEFINE_enum('action_norm', 'unit', ['unit', 'zeromean_unitvar'],
                  'Which normalization to apply to actions.')
flags.DEFINE_enum('residual_action_norm', 'unit',
                  ['none', 'unit', 'zeromean_unitvar', 'centered'],
                  'Which normalization to apply to residual actions.')
flags.DEFINE_float('residual_action_norm_scale', 1.0,
                   'Factor by which to scale residual actions. Applied to raw '
                   'predictions in none, unit and centered normalisation, and '
                   'to standard deviation in the case of zeromean_unitvar.')
flags.DEFINE_enum('signals_norm', 'none', ['none', 'unit', 'zeromean_unitvar'],
                  'Which normalization to apply to scalar observations.')
flags.DEFINE_string('original_demos_file', None,
                    'Dataset used to compute stats for action normalization.')
flags.DEFINE_integer('max_demos_to_load', None,
                     'Maximum number of demos from demos_file (in order) to '
                     'use to compute action stats.')
flags.DEFINE_integer('max_demo_length', None,
                     'If set, trim demonstrations to this length.')
flags.DEFINE_float('val_size', 0.05,
                   'Amount of data to exlude from action normalisation stats. '
                   'If < 1, the fraction of total loaded data points. Else the '
                   'number of data points.')
flags.DEFINE_boolean('val_full_episodes', True,
                     'If True, split data into train and validation on an '
                     'episode basis. Else split by individual time steps.')

flags.DEFINE_string('last_activation', None,
                    'Activation function to apply to network output, if any.')
flags.DEFINE_list('fc_layer_sizes', [],
                  'Sizes of fully connected layers to add on top of bottleneck '
                  'layer, if any.')
flags.DEFINE_integer('num_input_frames', 3,
                     'Number of frames to condition base policy on.')
flags.DEFINE_integer('image_size', None, 'Size of rendered images.')
flags.DEFINE_integer('crop_margin_size', 16,
                     'If crop_frames is True, the number of pixels to crop '
                     'from each dimension.')

flags.DEFINE_boolean('crop_frames', True,
                     'If True, crop input frames by 16 pixels in H and W.')
flags.DEFINE_list('target_offsets', [0, 10, 20, 30],
                  'Offsets in time for actions to predict in behavioral '
                  'cloning.')
flags.DEFINE_enum('network', None,
                  ['resnet18', 'resnet18_narrow32', 'resnet50', 'simple_cnn',
                   'hand_vil'],
                  'Policy network of base policy.')
flags.DEFINE_boolean('bn_before_concat', False,
                     'If True, add a batch norm layer before concatenating '
                     'scalar featuses to visual features.')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight decay for training.')

flags.DEFINE_boolean('predict_residual', True,
                     'If True, train a residual agent. Else train RL from '
                     'scratch without base agent.')
flags.DEFINE_enum('rl_observation_network', None,
                  ['resnet18', 'resnet18_narrow32', 'resnet50', 'simple_cnn',
                   'hand_vil'],
                  'Observation network of residual policy. If None, '
                  'observation network of base agent is reused.')
flags.DEFINE_boolean('late_fusion', False,
                     'If True, fuse stacked frames after convolutional layers. '
                     'If False, fuse at network input.')
flags.DEFINE_string('policy_init_path', None,
                    'If set, initialize network weights from a pickle file at '
                    'this path.')
flags.DEFINE_string('rl_observation_network_ckpt', None,
                    'If set, checkpoint from which to load observation network '
                    'weights.')
flags.DEFINE_string('base_controller', None,
                    'If set, a black-box controller to use for base actions.')

flags.DEFINE_string('bc_ckpt_to_load', None,
                    'If set, checkpoint from which to load base policy.')
flags.DEFINE_string('rl_ckpt_to_load', None,
                    'If set, checkpoint from which to load residual policy.')
flags.DEFINE_string('original_demos_path', None,
                    'If set, path to the original demonstration dataset (to '
                    'restore normalization statistics). If not set, inferred '
                    'from BC checkpoint path.')

flags.DEFINE_boolean('init_from_bc', False,
                     'If True, use BC agent loaded from bc_ckpt_to_load as '
                     'initialization for RL observation and policy nets.')
flags.DEFINE_boolean('init_feats_from_bc', False,
                     'If True, initialize RL observation network with BC.')


flags.DEFINE_string('logdir', None, 'Location to log results to.')
flags.DEFINE_boolean('load_saved', False,
                     'If True, load saved model from checkpoint. Else train '
                     'from scratch.')
flags.DEFINE_string('base_visible_state', 'robot',
                    'State features on which to condition the base policy.')
flags.DEFINE_string('residual_visible_state', 'robot',
                    'State features on which to condition the residual policy. '
                    'If using full state, the BC net features are replaced '
                    'with these true state features in input to RL policy.')
flags.DEFINE_float('bernoulli_rate', 0.,
                   'Fraction of time to use bernoulli exploration for gripper '
                   'action.')
flags.DEFINE_float('sticky_rate', 0.,
                   'Stickiness rate of bernoulli exploration for gripper '
                   'action.')
flags.DEFINE_string('job_id', None,
                    'Subdirectory to add to logdir to identify run. Set '
                    'automatically to XM id or datetime if None.')
flags.DEFINE_integer('base_policy_success', None,
                     'No-op flag used to identify base policy.')
flags.DEFINE_boolean('freeze_rl_observation_network', False,
                     'If True, do not update acme observation network weights. '
                     'Else train critic and observation net jointly.')

FLAGS = flags.FLAGS


def train_residual(
    env_loop, num_episodes, logdir, eval_freq, num_eval_episodes,
    collapse_in_eval, eval_seed, increment_eval_seed, stop_if_stuck):
  """Train residual for num_episodes episodes."""
  # TODO(minttu): Should bernoulli rate and sticky rate be defined here instead?
  total_steps = env_loop.run(
      num_episodes=num_episodes,
      out_dir=logdir,
      ckpt_freq=min(50_000, eval_freq),
      eval_freq=eval_freq,
      num_eval_episodes=num_eval_episodes,
      collapse_in_eval=collapse_in_eval,
      eval_seed=eval_seed,
      increment_eval_seed=increment_eval_seed,
      stop_if_stuck=stop_if_stuck)
  if logdir is not None:
    setup.save_acme_agent(env_loop.actor, logdir)
  return total_steps


def main(_):
  np.random.seed(FLAGS.seed)
  tf.random.set_seed(FLAGS.seed)

  counter = setup.setup_counting()
  logdir, env_logger, agent_logger, summary_writer, _ = setup.setup_logging(
      FLAGS.logdir)
  base_state = setup.set_visible_features(
      FLAGS.domain, FLAGS.task, FLAGS.base_visible_state)
  residual_state = setup.set_visible_features(
      FLAGS.domain, FLAGS.task, FLAGS.residual_visible_state)
  print('Base policy state features', base_state)
  print('Residual policy state features', residual_state)

  image_size = FLAGS.image_size
  if image_size is None:
    # Default sizes.
    image_size = {
        'adroit': 128,
        'mime': 240,
    }[FLAGS.domain]

  # Whether BCAgent's network is used for visual features (expects frames in a
  # certain shape).
  use_base_agent_image_shape = (
      FLAGS.predict_residual or FLAGS.freeze_rl_observation_network)
  visible_state = (
      list(set(base_state + residual_state)) if FLAGS.predict_residual
      else residual_state)
  env_loop = setup.make_environment_loop(
      domain=FLAGS.domain,
      task=FLAGS.task,
      seed=FLAGS.seed,
      input_type=FLAGS.input_type,
      num_input_frames=FLAGS.num_input_frames,
      visible_state=visible_state,
      image_size=image_size,
      use_base_agent_image_shape=use_base_agent_image_shape,
      late_fusion=FLAGS.late_fusion,
      max_train_episode_steps=FLAGS.max_episode_steps,
      agent=None,
      counter=counter,
      env_logger=env_logger,
      summary_writer=summary_writer)
  env = env_loop._environment    # pylint: disable=protected-access
  environment_spec = specs.make_environment_spec(env)
  print('Environment spec', environment_spec)

  base_agent = None
  # Create BC agent. In residual RL, it is used as the base agent, and in
  # standalone RL it may be used for action and observation space normalization.
  if FLAGS.bc_ckpt_to_load or FLAGS.original_demos_file:
    base_agent = setup.load_saved_bc_agent(
        ckpt_to_load=FLAGS.bc_ckpt_to_load,
        network_type=FLAGS.network,
        late_fusion=FLAGS.late_fusion,
        input_type=FLAGS.input_type,
        domain=FLAGS.domain,
        binary_grip_action=FLAGS.binary_grip_action,
        num_input_frames=FLAGS.num_input_frames,
        crop_frames=FLAGS.crop_frames,
        full_image_size=image_size,
        crop_margin_size=FLAGS.crop_margin_size,
        target_offsets=[int(t) for t in FLAGS.target_offsets],
        visible_state_features=base_state,
        action_norm=FLAGS.action_norm,
        signals_norm=FLAGS.signals_norm,
        last_activation=FLAGS.last_activation,
        fc_layer_sizes=[int(i) for i in FLAGS.fc_layer_sizes],
        weight_decay=FLAGS.weight_decay,
        max_demos_to_load=FLAGS.max_demos_to_load,
        max_demo_length=FLAGS.max_demo_length,
        val_size=FLAGS.val_size,
        val_full_episodes=FLAGS.val_full_episodes,
        split_seed=FLAGS.split_seed,
        env=env,
        task=FLAGS.task)
    print('action normalization mean\n', base_agent.action_space.mean)
    print('action normalization std\n', base_agent.action_space.std)

  obs_network_type = None
  include_base_feats = True
  if ((FLAGS.bc_ckpt_to_load is None and FLAGS.policy_init_path is None)
      or (FLAGS.init_from_bc and not FLAGS.freeze_rl_observation_network)
      or FLAGS.init_feats_from_bc):
    obs_network_type = FLAGS.rl_observation_network
    include_base_feats = False
  if FLAGS.residual_visible_state == 'full':
    include_base_feats = False
  include_base_action = FLAGS.predict_residual
  residual_spec = setup.define_residual_spec(
      residual_state, env, base_agent,
      action_norm=FLAGS.residual_action_norm,
      action_norm_scale=FLAGS.residual_action_norm_scale,
      include_base_action=include_base_action,
      include_base_feats=include_base_feats,
      base_network=FLAGS.network)

  binary_grip_action = FLAGS.init_from_bc and FLAGS.binary_grip_action
  residual_agent, eval_policy = setup.make_acme_agent(
      environment_spec=environment_spec,
      residual_spec=residual_spec,
      obs_network_type=obs_network_type,
      crop_frames=FLAGS.crop_frames,
      full_image_size=image_size,
      crop_margin_size=FLAGS.crop_margin_size,
      late_fusion=FLAGS.late_fusion,
      binary_grip_action=binary_grip_action,
      input_type=FLAGS.input_type,
      counter=counter,
      logdir=logdir,
      agent_logger=agent_logger)
  if FLAGS.init_from_bc:
    setup.init_policy_networks(base_agent.network, residual_agent)
    if not FLAGS.freeze_rl_observation_network:
      setup.init_observation_networks(base_agent.network, residual_agent)
  if FLAGS.init_feats_from_bc:
    setup.init_observation_networks(base_agent.network, residual_agent)

  # agent_class = (
  # agents.ResidualAgent if FLAGS.predict_residual else agents.RLAgent)
  if FLAGS.predict_residual:
    agent_class = agents.ResidualAgent
  else:
    if FLAGS.freeze_rl_observation_network:
      agent_class = agents.FixedObservationAgent
    else:
      agent_class = agents.RLAgent
  agent = agent_class(
      base_agent=base_agent,
      rl_agent=residual_agent,
      action_space='tool_lin' if FLAGS.domain == 'mime' else FLAGS.task,
      action_norm=FLAGS.residual_action_norm,
      action_norm_scale=FLAGS.residual_action_norm_scale,
      signals_norm=FLAGS.signals_norm,
      rl_eval_policy=eval_policy,
      feats_spec=residual_spec.observations,
      state_keys=residual_state,
      bernoulli_rate=FLAGS.bernoulli_rate,
      sticky_rate=FLAGS.sticky_rate,
      rl_observation_network_type=FLAGS.rl_observation_network,
      rl_input_type=FLAGS.input_type,
      rl_num_input_frames=FLAGS.num_input_frames,
      base_controller=FLAGS.base_controller,
      env=env)
  env_loop.actor = agent

  if FLAGS.eval_only:
    ckpts = gfile.Glob(os.path.join(logdir, 'policy_*.index'))
    print(os.path.join(logdir, 'policy_*.index'))
    print(ckpts)
    for ckpt in ckpts:
      ckpt = ckpt.replace('.index', '')
      loaded_steps = setup.load_agent(agent, ckpt)
      total_steps = loaded_steps

      eval_utils.eval_agent(
          env_loop, FLAGS.task, FLAGS.eval_seed, FLAGS.increment_eval_seed,
          FLAGS.num_eval_episodes, loaded_steps, FLAGS.collapse_in_eval,
          FLAGS.stop_if_stuck, FLAGS.num_episodes, total_steps, logdir,
          summary_writer=None, eval_id='late')
  else:
    if FLAGS.rl_ckpt_to_load is None:
      total_steps = train_residual(
          env_loop, FLAGS.num_episodes, logdir, FLAGS.eval_freq,
          FLAGS.num_eval_episodes, FLAGS.collapse_in_eval, FLAGS.eval_seed,
          FLAGS.increment_eval_seed, FLAGS.stop_if_stuck)
      loaded_steps = 'final'
    else:
      loaded_steps = setup.load_agent(agent, FLAGS.rl_ckpt_to_load)
      total_steps = loaded_steps
      logdir = os.path.dirname(FLAGS.rl_ckpt_to_load)

    eval_utils.eval_agent(
        env_loop, FLAGS.task, FLAGS.eval_seed, FLAGS.increment_eval_seed,
        FLAGS.num_eval_episodes, loaded_steps, FLAGS.collapse_in_eval,
        FLAGS.stop_if_stuck, FLAGS.num_episodes, total_steps, logdir,
        summary_writer)


if __name__ == '__main__':
  app.run(main)
