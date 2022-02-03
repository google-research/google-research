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
"""Train a residual BC policy on top of a learned agent.
"""

import os
import pickle

from absl import app
from absl import flags
import numpy as np
import tensorflow.compat.v2 as tf

from rrlfd.bc import train_utils
from rrlfd.residual import setup
from tensorflow.io import gfile


flags.DEFINE_string('domain', None, 'Domain from which to load task.')
flags.DEFINE_string('demo_task', None,
                    'Task used to gather demos in dataset, if different from '
                    'eval_task.')
flags.DEFINE_string('eval_task', None,
                    'If set, evaluate trained policy on this task.')
flags.DEFINE_enum('input_type', 'depth', ['depth', 'rgb', 'rgbd', 'position'],
                  'Input modality.')
flags.DEFINE_integer('test_set_size', 0,
                     'Number of additional demonstrations on which to evaluate '
                     'final model.')
flags.DEFINE_integer('test_set_start', None,
                     'Where in the dataset to start test set.')

flags.DEFINE_integer('batch_size', 64, 'Batch size for training.')
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

flags.DEFINE_integer('residual_max_demos_to_load', 100,
                     'Number of demonstrations (in order, starting after the '
                     'last demo used by the base agent) to use for residual '
                     'training.')
flags.DEFINE_float('residual_val_size', 0.05,
                   'Val size applies to residual training.')

flags.DEFINE_string('last_activation', None,
                    'Activation function to apply to network output, if any.')
flags.DEFINE_list('fc_layer_sizes', [],
                  'Sizes of fully connected layers to add on top of bottleneck '
                  'layer, if any.')
flags.DEFINE_enum('regression_loss', 'l2', ['l2', 'nll'],
                  'Loss function to minimize for continuous action dimensions.')
flags.DEFINE_float('l2_weight', 0.9,
                   'How much relative weight to give to linear velocity loss.')
flags.DEFINE_integer('num_input_frames', 3,
                     'Number of frames to condition base policy on.')
flags.DEFINE_boolean('crop_frames', True,
                     'If True, crop input frames by 16 pixels in H and W.')
flags.DEFINE_boolean('augment_frames', True,
                     'If True, augment images by scaling, cropping and '
                     'rotating.')
flags.DEFINE_list('target_offsets', [1, 10, 20, 30],
                  'Offsets in time for actions to predict in behavioral '
                  'cloning.')
flags.DEFINE_enum('network', None,
                  ['resnet18', 'resnet18_narrow32', 'resnet50', 'simple_cnn',
                   'hand_vil'],
                  'Policy network of base policy.')
flags.DEFINE_integer('num_epochs', 100, 'Number of epochs to train for.')
flags.DEFINE_list('epochs_to_eval', [],
                  'Epochs at which to evaluate checkpoint with best validation '
                  'error so far.')
flags.DEFINE_enum('optimizer', 'adam', ['adam', 'rmsprop'],
                  'Keras optimizer for training.')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate for training.')
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
flags.DEFINE_string('eval_id', '', 'ID to add to evaluation output path.')
flags.DEFINE_boolean('render_eval', False,
                     'If True, render environment during evaluation.')
# TODO(minttu): Consolidate flags with bc/train_utils and bc/train.
flags.DEFINE_integer('eval_episodes_to_save', 0,
                     'The number of eval episodes whose frames to write to '
                     'file.')

flags.DEFINE_boolean('init_from_bc', False,
                     'If True, use BC agent loaded from bc_ckpt_to_load as '
                     'initialization for RL observation and policy nets.')
flags.DEFINE_boolean('init_feats_from_bc', False,
                     'If True, initialize RL observation network with BC.')
flags.DEFINE_boolean('clip_actions', False,
                     'If True, clip actions to unit interval before '
                     'normalization.')

flags.DEFINE_string('logdir', None, 'Location to log results to.')
flags.DEFINE_boolean('load_saved', False,
                     'If True, load saved model from checkpoint. Else train '
                     'from scratch.')
flags.DEFINE_enum('base_visible_state', 'robot', ['image', 'robot', 'full'],
                  'State features on which to condition the base policy.')
flags.DEFINE_enum('residual_visible_state', 'robot', ['image', 'robot', 'full'],
                  'State features on which to condition the residual policy. '
                  'If using full state, the BC net features are replaced  with '
                  'these true state features in input to RL policy.')
flags.DEFINE_float('bernoulli_rate', 0.,
                   'Fraction of time to use bernoulli exploration for gripper '
                   'action.')
flags.DEFINE_float('sticky_rate', 0.,
                   'Stickiness rate of bernoulli exploration for gripper '
                   'action.')
flags.DEFINE_string('exp_id', '', 'Experiment ID to add to output paths.')
flags.DEFINE_string('job_id', '', 'Job ID to add to output paths.')

FLAGS = flags.FLAGS


def reset_dir(directory):
  if gfile.exists(directory):
    gfile.DeleteRecursively(directory)


def set_paths(ckpt_to_load, network):
  """Set output paths based on base agent path and network."""
  demos_file = setup.get_original_demos_path(ckpt_to_load)
  # For now, assume the same dataset has unused demonstrations to use in
  # residual training.
  new_demos_file = demos_file

  ckpt_dir = os.path.dirname(ckpt_to_load)
  new_ckpt_dir = ckpt_dir.replace('/bc_policy/', '/residual_bc_policy/')
  new_ckpt_dir = os.path.join(new_ckpt_dir, FLAGS.exp_id)
  job_id = train_utils.set_job_id()
  if job_id is not None:
    new_ckpt_dir = os.path.join(new_ckpt_dir, job_id)
  assert new_ckpt_dir != ckpt_dir
  if not FLAGS.load_saved:
    reset_dir(new_ckpt_dir)

  summary_dir = train_utils.get_summary_dir_for_ckpt_dir(ckpt_dir, network)
  new_summary_dir = summary_dir.replace('/bc_policy/', '/residual_bc_policy/')
  if job_id is not None:
    new_summary_dir = os.path.join(new_summary_dir, job_id)
  assert new_summary_dir != summary_dir
  if not FLAGS.load_saved:
    reset_dir(new_summary_dir)

  return new_demos_file, new_ckpt_dir, new_summary_dir


def main(_):
  np.random.seed(FLAGS.seed)
  tf.random.set_seed(FLAGS.seed)

  demo_task = FLAGS.demo_task or FLAGS.eval_task
  if demo_task is None:
    raise ValueError('eval_task or demo_task must be set')
  base_state = setup.set_visible_features(
      FLAGS.domain, demo_task, FLAGS.base_visible_state)
  residual_state = setup.set_visible_features(
      FLAGS.domain, demo_task, FLAGS.residual_visible_state)
  print('Base policy state features', base_state)
  print('Residual policy state features', residual_state)

  env = train_utils.make_environment(FLAGS.domain, demo_task)
  demos_file, ckpt_dir, summary_dir = set_paths(
      FLAGS.bc_ckpt_to_load, FLAGS.network)

  # Create BC agent. In residual RL, it is used as the base agent, and in
  # standalone RL for action and observation space normalization.
  base_agent = setup.load_saved_bc_agent(
      ckpt_to_load=FLAGS.bc_ckpt_to_load,
      network_type=FLAGS.network,
      late_fusion=FLAGS.late_fusion,
      input_type=FLAGS.input_type,
      domain=FLAGS.domain,
      binary_grip_action=FLAGS.binary_grip_action,
      num_input_frames=FLAGS.num_input_frames,
      crop_frames=FLAGS.crop_frames,
      target_offsets=[int(t) for t in FLAGS.target_offsets],
      visible_state_features=base_state,
      action_norm=FLAGS.action_norm,
      signals_norm=FLAGS.signals_norm,
      last_activation=FLAGS.last_activation,
      fc_layer_sizes=FLAGS.fc_layer_sizes,
      weight_decay=FLAGS.weight_decay,
      max_demos_to_load=FLAGS.max_demos_to_load,
      max_demo_length=FLAGS.max_demo_length,
      val_size=FLAGS.val_size,
      val_full_episodes=FLAGS.val_full_episodes,
      split_seed=FLAGS.split_seed,
      env=env,
      task=demo_task,
      grip_action_from_state=False,
      zero_action_keeps_state=False,
      early_closing=False,
      )

  print('action normalization mean\n', base_agent.action_space.mean)
  print('action normalization std\n', base_agent.action_space.std)

  # Verify base agent is loaded correctly (repro success).
  # print('Evaluating standalone base agent')
  # success_rates = eval_loop.eval_policy(
  #     env, FLAGS.eval_seed, FLAGS.increment_eval_seed, base_agent,
  #     FLAGS.num_eval_episodes)

  include_base_feats = True
  if ((FLAGS.bc_ckpt_to_load is None and FLAGS.policy_init_path is None)
      or FLAGS.init_from_bc
      or FLAGS.init_feats_from_bc):
    include_base_feats = False
  if FLAGS.residual_visible_state == 'full':
    include_base_feats = False
  include_base_action = FLAGS.predict_residual
  # TODO(minttu): Scale residual spec minima and maxima according to residual
  # action normalization.
  residual_spec = setup.define_residual_spec(
      residual_state, env, base_agent,
      include_base_action=include_base_action,
      include_base_feats=include_base_feats,
      base_network=FLAGS.network)

  # TODO(minttu): Allow predicting continuous actions even if base agent uese
  # binary grip actions.
  # TODO(minttu): Pass in action target dimension or action pred dimemsion?
  binary_grip_action = FLAGS.binary_grip_action
  # Action normalization of residual BC:
  # -> centered (original demos std), with same scaling as residual exps
  agent = setup.make_residual_bc_agent(
      residual_spec=residual_spec,
      base_agent=base_agent,
      action_norm=FLAGS.residual_action_norm,
      action_norm_scale=FLAGS.residual_action_norm_scale,
      binary_grip_action=binary_grip_action,
      env=env,
      visible_state_features=residual_state)

  dataset = None
  if demos_file is not None:
    dataset = train_utils.prepare_demos(
        demos_file=demos_file,
        input_type=FLAGS.input_type,
        max_demos_to_load=FLAGS.residual_max_demos_to_load,
        max_demo_length=FLAGS.max_demo_length,
        augment_frames=FLAGS.augment_frames,
        agent=None,  # Do not reset action stats.
        split_dir=ckpt_dir,
        val_size=FLAGS.residual_val_size,
        val_full_episodes=FLAGS.val_full_episodes,
        skip=FLAGS.max_demos_to_load)
    dataset.agent = agent  # Use for demo normalization.

    # Assumes dataset fits in RAM.
    # Transform dataset once (to avoid transforming at each epoch).
    # Replace observations with concatenated visible state, base action and
    # base feats. Replace demo actions with residuals (given the base agent).
    agent.preprocess_dataset(dataset, FLAGS.batch_size)

  epochs_to_eval = [int(epoch) for epoch in FLAGS.epochs_to_eval]
  if not FLAGS.load_saved:
    if ckpt_dir is not None:
      gfile.makedirs(ckpt_dir)
      with gfile.GFile(os.path.join(ckpt_dir, 'train_split.pkl'), 'wb') as f:
        pickle.dump(dataset.train_split, f)
      with gfile.GFile(
          os.path.join(ckpt_dir, 'episode_train_split.pkl'), 'wb') as f:
        pickle.dump(dataset.episode_train_split, f)
    train_utils.train(
        dataset=dataset,
        agent=agent,
        ckpt_dir=ckpt_dir,
        optimizer_type=FLAGS.optimizer,
        learning_rate=FLAGS.learning_rate,
        batch_size=FLAGS.batch_size,
        num_epochs=FLAGS.num_epochs,
        loss_fn=FLAGS.regression_loss,
        l2_weight=FLAGS.l2_weight,
        summary_dir=summary_dir,
        epochs_to_eval=epochs_to_eval)

  test_set_size = FLAGS.test_set_size
  test_dataset = None
  if test_set_size > 0:
    test_set_start = FLAGS.test_set_start or FLAGS.max_demos_to_load
    test_dataset = train_utils.prepare_demos(
        demos_file, FLAGS.input_type, test_set_start + test_set_size,
        FLAGS.max_demo_length, augment_frames=False, agent=agent,
        split_dir=None,
        val_size=test_set_size / (test_set_start + test_set_size),
        val_full_episodes=True)
  num_eval_episodes = FLAGS.num_eval_episodes
  if FLAGS.eval_task is not None and num_eval_episodes > 0:
    env = train_utils.make_environment(
        FLAGS.domain, FLAGS.eval_task, FLAGS.use_egl, FLAGS.render_eval)
    summary_writer = train_utils.make_summary_writer(summary_dir)
    ckpts_to_eval = train_utils.get_checkpoints_to_evaluate(
        ckpt_dir, epochs_to_eval)
    print('Evaluating ckpts', ckpts_to_eval)

    epoch_to_success_rates = {}
    for ckpt in ckpts_to_eval:
      train_utils.evaluate_checkpoint(
          ckpt=ckpt,
          ckpt_dir=ckpt_dir,
          agent=agent,
          env=env,
          num_eval_episodes=num_eval_episodes,
          epoch_to_success_rates=epoch_to_success_rates,
          summary_writer=summary_writer,
          test_dataset=test_dataset)


if __name__ == '__main__':
  app.run(main)
