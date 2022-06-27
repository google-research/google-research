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

"""Train residual and re-train base policy periodically with new data."""

import os

from absl import app
from absl import flags
from acme import specs
import gym
import tensorflow as tf

from rrlfd.bc import bc_agent
from rrlfd.bc import eval_loop
from rrlfd.bc import pickle_dataset
from rrlfd.bc import train_utils
from rrlfd.residual import agents
from rrlfd.residual import eval_utils
from rrlfd.residual import setup
from tensorflow.io import gfile


flags.DEFINE_string('task', None, 'Mime task.')
flags.DEFINE_enum('input_type', 'depth', ['depth', 'rgb', 'rgbd', 'position'],
                  'Input modality.')

flags.DEFINE_list('retrain_after', [500],
                  'Episodes at which to retrain base agent.')
flags.DEFINE_integer('num_episodes', 1000,
                     'Number of episodes to run for each iteration.')
flags.DEFINE_integer('num_trajectories_to_add', 100,
                     'Number of trajectories to collect between updates to the '
                     'base policy.')
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

# Flags for BC agent.
flags.DEFINE_boolean('binary_grip_action', True,
                     'If True, use open/close action space for gripper. Else '
                     'use gripper velocity.')
flags.DEFINE_enum('action_norm', 'unit', ['unit', 'zeromean_unitvar'],
                  'Which normalization to apply to actions.')
flags.DEFINE_boolean('normalize_signals', False,
                     'If True, normalize scalar inputs to be in unit range.')
flags.DEFINE_string('last_activation', None,
                    'Activation function to apply to network output, if any.')
flags.DEFINE_list('fc_layer_sizes', [],
                  'Sizes of fully connected layers to add on top of bottleneck '
                  'layer, if any.')
flags.DEFINE_integer('max_demos_to_load', None,
                     'Maximum number of demos from demos_file (in order) to '
                     'use to compute action stats.')
flags.DEFINE_integer('num_input_frames', 3,
                     'Number of frames to condition base policy on.')
flags.DEFINE_boolean('crop_frames', True,
                     'If True, crop input frames to 224x224.')
flags.DEFINE_list('target_offsets', [1, 10, 20, 30],
                  'Offsets in time for actions to predict in behavioral '
                  'cloning.')

# Flags for BC training.
flags.DEFINE_boolean('retrain_from_scratch', False,
                     'If True, reinitialize BC network and train from scratch.'
                     'Else continue training with new dataset.')
flags.DEFINE_boolean('grip_action_from_state', False,
                     'If True, use gripper state as gripper action.')
flags.DEFINE_boolean('zero_action_keeps_state', True,
                     'If True, convert a zero-action in a demonstration to '
                     'maintain gripper state (as opposed to opening). Only '
                     'makes sense when not using grip_action_from_state.')
flags.DEFINE_boolean('early_closing', False,
                     'If True, clone gripper closing action in advance.')
flags.DEFINE_float('l2_weight', 0.9,
                   'How much relative weight to give to linear velocity loss.')
flags.DEFINE_boolean('augment_frames', True,
                     'If True, augment images by scaling, cropping and '
                     'rotating.')
flags.DEFINE_integer('num_epochs', 100,
                     'Number of epochs to train for after each addition.')
flags.DEFINE_enum('optimizer', 'adam', ['adam', 'rmsprop'],
                  'Keras optimizer for training.')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate for training.')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight decay for training.')
flags.DEFINE_float('val_fraction', 0.05, 'Fraction of data to validate on.')
flags.DEFINE_boolean('val_full_episodes', True,
                     'If True, split data into train and validation on an '
                     'episode basis. Else split by individual time steps.')
flags.DEFINE_integer('batch_size', 64, 'Batch size for training.')

flags.DEFINE_enum('network', None,
                  ['resnet18', 'resnet18_narrow32', 'resnet50', 'simple_cnn'],
                  'Policy network of base policy.')
flags.DEFINE_enum('rl_observation_network', None,
                  ['resnet18', 'resnet18_narrow32', 'resnet50', 'simple_cnn'],
                  'Policy network of base policy.')
flags.DEFINE_string('rl_observation_network_ckpt', None,
                    'If set, checkpoint from which to load observation network '
                    'weights.')

flags.DEFINE_string('bc_ckpt_to_load', None,
                    'If set, checkpoint from which to load base policy.')
flags.DEFINE_string('rl_ckpt_to_load', None,
                    'If set, checkpoint from which to load residual policy.')


flags.DEFINE_string('logdir', None, 'Location to log results to.')
flags.DEFINE_boolean('load_saved', False,
                     'If True, load saved model from checkpoint. Else train '
                     'from scratch.')
flags.DEFINE_list('bc_visible_state_features', [],
                  'Additional features on which to condition the policy.')
flags.DEFINE_list('rl_visible_state_features', [],
                  'If not empty, replace BC net features with these true state '
                  'features in input to RL policy.')
flags.DEFINE_float('bernoulli_rate', 0.,
                   'Fraction of time to use bernoulli exploration for gripper '
                   'action.')
flags.DEFINE_float('sticky_rate', 0.,
                   'Fraction of time to use bernoulli exploration for gripper '
                   'action.')
flags.DEFINE_string('job_id', None,
                    'Subdirectory to add to logdir to identify run. Set '
                    'automatically to XM id or datetime if None.')
flags.DEFINE_string('eval_id', '', 'ID to add to evaluation output path.')
flags.DEFINE_integer('eval_episodes_to_save', 10,
                     'The number of eval episodes whose frames to write to '
                     'file.')

FLAGS = flags.FLAGS


def train_residual(
    env_loop, num_episodes, num_successes, success_writer, logdir,
    num_eval_episodes, collapse_in_eval, eval_seed, stop_if_stuck,
    start_with_eval, trajectory_filter, summary_writer=None):
  """Train residual and save num_successes successful training trajectories."""
  # TODO(minttu): should bernoulli rate and sticky rate be defined here instead?
  env_loop.run(
      num_episodes=num_episodes,
      num_successes=num_successes,
      success_writer=success_writer,
      out_dir=logdir,
      start_with_eval=start_with_eval,
      num_eval_episodes=num_eval_episodes,
      collapse_in_eval=collapse_in_eval,
      eval_seed=eval_seed,
      stop_if_stuck=stop_if_stuck,
      trajectory_filter=trajectory_filter,
      summary_writer=summary_writer)
  setup.save_acme_agent(env_loop.actor, logdir)


def main(_):
  """Main function for alternate training.

  base_agent = ...
  rl_agent = ...
  train_demos, val_demos = ...
  for i in range(3):
    new_data = train_residual(base_agent, rl_agent, num_episodes)
    train_demos = train_demos + new_data

    # until validation error converges?
    # for N epochs?
    base_agent = continue_training_base(base_agent, train_demos, val_demos)
  """
  tf.random.set_seed(FLAGS.seed)

  logdir, env_logger, agent_logger, _, summary_dir = setup.setup_logging(
      FLAGS.logdir)
  new_demos_dir = os.path.join(logdir, 'successes')
  if not gfile.makedirs(new_demos_dir):
    gfile.makedirs(new_demos_dir)
  new_base_ckpt_dir = os.path.join(logdir, 'retrain_bc')
  new_base_summary_dir = os.path.join(summary_dir, 'retrain_bc')
  eval_id = FLAGS.eval_id
  increment_str = 'i' if FLAGS.increment_eval_seed else ''
  eval_str = (
            f'{FLAGS.task}_s{FLAGS.eval_seed}{increment_str}'
            f'_e{FLAGS.num_eval_episodes}{eval_id}')

  bc_state, rl_state = setup.set_visible_features(
      FLAGS.bc_visible_state_features, FLAGS.rl_visible_state_features)
  print('BC state', bc_state)
  print('RL state', rl_state)
  # TODO(minttu): Augment viewpoints?
  env_loop = setup.make_environment_loop(
      FLAGS.task, FLAGS.seed, FLAGS.input_type, FLAGS.num_input_frames,
      bc_state, rl_state, agent=None, logdir=logdir, env_logger=env_logger,
      summary_writer=None)
  env = env_loop._environment  # pylint: disable=protected-access
  environment_spec = specs.make_environment_spec(env)
  print(environment_spec)

  # Create BC agent. In residual RL, it is used as the base agent, and in
  # standalone RL for action space normalization.
  # TODO(minttu): Save dataset stats for zeromean_unitvar normalization.
  base_agent = setup.load_saved_bc_agent(
      FLAGS.bc_ckpt_to_load, FLAGS.network, FLAGS.input_type,
      FLAGS.binary_grip_action, FLAGS.num_input_frames, FLAGS.crop_frames,
      FLAGS.target_offsets, FLAGS.bc_visible_state_features, FLAGS.action_norm,
      FLAGS.normalize_signals, FLAGS.last_activation, FLAGS.fc_layer_sizes,
      FLAGS.weight_decay, FLAGS.max_demos_to_load, env.env)
  original_demos_file = setup.get_original_demos_path(FLAGS.bc_ckpt_to_load)
  split_dir = os.path.dirname(FLAGS.bc_ckpt_to_load)

  residual_spec = setup.define_residual_spec(
      FLAGS.rl_visible_state_features, env, base_agent.action_space,
      include_base_agent=FLAGS.bc_ckpt_to_load is not None)

  obs_network_type = (
      FLAGS.rl_observation_network if FLAGS.bc_ckpt_to_load is None else None)
  rl_agent, eval_policy = setup.make_rl_agent(
      environment_spec=environment_spec,
      residual_spec=residual_spec,
      obs_network_type=obs_network_type,
      obs_network_ckpt=FLAGS.rl_observation_network_ckpt,
      input_type=FLAGS.input_type,
      agent_logger=agent_logger)
  agent_class = (
      agents.RLAgent if FLAGS.bc_ckpt_to_load is None else agents.ResidualAgent)
  agent = agent_class(
      base_agent=base_agent,
      rl_agent=rl_agent,
      rl_eval_policy=eval_policy,
      feats_spec=residual_spec.observations,
      state_keys=rl_state,
      bernoulli_rate=FLAGS.bernoulli_rate,
      sticky_rate=FLAGS.sticky_rate,
      rl_observation_network_type=FLAGS.rl_observation_network,
      rl_input_type=FLAGS.input_type,
      rl_num_input_frames=FLAGS.num_input_frames)

  env_loop.actor = agent
  total_steps = 0
  num_new_demos = FLAGS.num_trajectories_to_add
  new_demos_paths = []

  retrain_after = FLAGS.retrain_after
  num_iterations = len(retrain_after) + 1
  if FLAGS.rl_ckpt_to_load is None:
    new_demos_paths = []
    for i in range(num_iterations):
      # TODO(minttu): Train for longer? Subsample N?
      # Save in rl_policy/.../successes/
      new_demos_path = os.path.join(
          new_demos_dir, str(i), f'e{num_new_demos}.pkl')
      success_writer = pickle_dataset.DemoWriter(new_demos_path)
      logdir_i = os.path.join(logdir, str(i))
      if not gfile.exists(logdir_i):
        gfile.makedirs(logdir_i)
      summary_dir_i = os.path.join(summary_dir, str(i))
      if not gfile.exists(summary_dir_i):
        gfile.makedirs(summary_dir_i)
      summary_writer = tf.summary.create_file_writer(summary_dir_i)
      if i == 0:
        num_episodes = int(retrain_after[i])
      elif i < num_iterations - 1:
        num_episodes = (
            int(FLAGS.retrain_after[i]) - int(FLAGS.retrain_after[i - 1]))
      else:
        num_episodes = FLAGS.num_episodes - env_loop.episodes
      train_residual(
          env_loop, num_episodes, num_new_demos, success_writer, logdir_i,
          FLAGS.num_eval_episodes, FLAGS.collapse_in_eval, FLAGS.eval_seed,
          FLAGS.stop_if_stuck, start_with_eval=True, trajectory_filter='latest',
          summary_writer=summary_writer)
      eval_utils.eval_agent(
          env_loop=env_loop,
          task=FLAGS.task,
          eval_seed=FLAGS.eval_seed,
          num_eval_episodes=FLAGS.num_eval_episodes,
          loaded_ckpt='',  # env_loop.steps, # 'final',
          collapse_in_eval=FLAGS.collapse_in_eval,
          stop_if_stuck=FLAGS.stop_if_stuck,
          num_trained_episodes=env_loop.episodes,
          total_steps=env_loop.steps,
          logdir=logdir_i,
          summary_writer=summary_writer)

      new_demos_paths = [new_demos_path] + new_demos_paths
      # merged_dataset = pickle_dataset.DemoReader(
      #     old_demos, new_demos_paths, in_memory=True)
      # dataset = train_utils.prepare_data(
      #     original_demos_file, base_agent, split_dir, in_memory=True)

      # TODO(minttu): Last stage should train RL only.
      if i < num_iterations - 1:
        if FLAGS.retrain_from_scratch:
          base_agent = bc_agent.BCAgent(
              network_type=FLAGS.network,
              input_type=FLAGS.input_type,
              binary_grip_action=FLAGS.binary_grip_action,
              grip_action_from_state=FLAGS.grip_action_from_state,
              zero_action_keeps_state=FLAGS.zero_action_keeps_state,
              early_closing=FLAGS.early_closing,
              num_input_frames=FLAGS.num_input_frames,
              crop_frames=FLAGS.crop_frames,
              target_offsets=[int(t) for t in FLAGS.target_offsets],
              visible_state_features=FLAGS.bc_visible_state_features,
              action_norm=FLAGS.action_norm,
              signals_norm=FLAGS.normalize_signals,
              last_activation=FLAGS.last_activation,
              fc_layer_sizes=FLAGS.fc_layer_sizes,
              weight_decay=FLAGS.weight_decay,
              env=env.env)  # Changed to env.env

        dataset = train_utils.prepare_data(
            original_demos_file, FLAGS.input_type, FLAGS.max_demos_to_load,
            FLAGS.augment_frames, base_agent, split_dir, FLAGS.val_fraction,
            FLAGS.val_full_episodes)
        for path in new_demos_paths:
          # TODO(minttu): Add in same ratio to validation set?
          dataset.add_demos(path)

        new_base_summary_dir_i = os.path.join(new_base_summary_dir, str(i))
        new_base_ckpt_dir_i = os.path.join(new_base_ckpt_dir, str(i))
        new_base_eval_path = os.path.join(new_base_ckpt_dir_i,
                                          f'eval{eval_str}')
        new_base_summary_writer = tf.summary.create_file_writer(
            new_base_summary_dir_i)
        print(i, 'Saving BC checkpoints to', new_base_ckpt_dir_i)
        print(i, 'Writing BC evaluation to', new_base_eval_path)
        print(i, 'Writing BC summaries to', new_base_summary_dir_i)
        if not gfile.exists(new_base_ckpt_dir_i):
          gfile.makedirs(new_base_ckpt_dir_i)
        if not gfile.exists(new_base_summary_dir_i):
          gfile.makedirs(new_base_summary_dir_i)
        # Continue training. TODO(minttu): Save optimizer state.
        # Alternatively, retrain from scratch.
        best_epoch = train_utils.train(
            dataset, base_agent, new_base_ckpt_dir_i, FLAGS.optimizer,
            FLAGS.learning_rate, FLAGS.batch_size, FLAGS.num_epochs,
            FLAGS.l2_weight, summary_dir=new_base_summary_dir_i)
        print('best epoch', best_epoch)

        egl_str = '-EGL' if FLAGS.use_egl else ''
        bc_eval_env = gym.make(f'UR5{egl_str}-{FLAGS.task}CamEnv-v0')
        summary_key = f'{FLAGS.task}_s{FLAGS.eval_seed}{increment_str}'

        eval_loop.eval_policy(
            bc_eval_env, FLAGS.eval_seed, FLAGS.increment_eval_seed, base_agent,
            FLAGS.num_eval_episodes, new_base_eval_path,
            FLAGS.eval_episodes_to_save, new_base_summary_writer,
            summary_key=summary_key, stop_if_stuck=FLAGS.stop_if_stuck)
        del bc_eval_env
    loaded_ckpt = 'final'
    total_episodes = env_loop.episodes
    total_steps = env_loop.steps
  else:
    loaded_ckpt = setup.load_agent(agent, FLAGS.rl_ckpt_to_load)
    total_episodes = None
    total_steps = int(loaded_ckpt)
    logdir = os.path.dirname(FLAGS.rl_ckpt_to_load)

    eval_utils.eval_agent(
        env_loop=env_loop,
        task=FLAGS.task,
        eval_seed=FLAGS.eval_seed,
        num_eval_episodes=FLAGS.num_eval_episodes,
        loaded_ckpt=loaded_ckpt,
        collapse_in_eval=FLAGS.collapse_in_eval,
        stop_if_stuck=FLAGS.stop_if_stuck,
        num_trained_episodes=total_episodes,
        total_steps=total_steps,
        logdir=logdir,
        summary_writer=summary_writer)

if __name__ == '__main__':
  app.run(main)
