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
"""Learn a policy from logged behaviour data using behavioral cloning."""

import os
import pickle
import random

from absl import app
from absl import flags
import mime  # pylint: disable=unused-import
import numpy as np
import tensorflow as tf

from rrlfd import adroit_ext  # pylint: disable=unused-import
from rrlfd import adroit_utils
from rrlfd import mime_utils
from rrlfd.bc import bc_agent
from rrlfd.bc import eval_loop
from rrlfd.bc import train_utils
from tensorflow.io import gfile


flags.DEFINE_string('input_type', 'depth', 'Input modality.')
flags.DEFINE_boolean('binary_grip_action', False,
                     'If True, use open/close action space for gripper. Else '
                     'use gripper velocity.')
flags.DEFINE_boolean('grip_action_from_state', False,
                     'If True, use gripper state as gripper action.')
flags.DEFINE_boolean('zero_action_keeps_state', False,
                     'If True, convert a zero-action in a demonstration to '
                     'maintain gripper state (as opposed to opening). Only '
                     'makes sense when not using grip_action_from_state.')
flags.DEFINE_boolean('early_closing', False,
                     'If True, clone gripper closing action in advance.')
flags.DEFINE_enum('action_norm', 'unit', ['none', 'unit', 'zeromean_unitvar'],
                  'Which normalization to apply to actions.')
flags.DEFINE_enum('signals_norm', 'none', ['none', 'unit', 'zeromean_unitvar'],
                  'Which normalization to apply to signal observations.')

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
                     'Number of frames to condition policy on.')
flags.DEFINE_integer('image_size', None, 'Size of rendered images.')
flags.DEFINE_integer('crop_margin_size', 16,
                     'If crop_frames is True, the number of pixels to crop '
                     'from each dimension.')

flags.DEFINE_boolean('crop_frames', True,
                     'If True, crop input frames by crop_margin_size pixels in '
                     'H and W.')
flags.DEFINE_boolean('augment_frames', True,
                     'If True, augment images by scaling, cropping and '
                     'rotating.')
flags.DEFINE_list('target_offsets', [0, 10, 20, 30],
                  'Offsets in time for actions to predict in behavioral '
                  'cloning.')
flags.DEFINE_string('visible_state', 'image',
                    'Which scalar features to condition the policy on.')
flags.DEFINE_string('network', 'resnet18_narrow32', 'Policy network to train.')
flags.DEFINE_boolean('bn_before_concat', False,
                     'If True, add a batch norm layer before concatenating '
                     'scalar featuses to visual features.')
flags.DEFINE_string('weight_init_scheme', 'v1',
                    'Which initializers to use for policy network. See '
                    'network.py for details.')
flags.DEFINE_integer('num_epochs', 100, 'Number of epochs to train for.')
flags.DEFINE_list('epochs_to_eval', [],
                  'Epochs at which to evaluate checkpoint with best validation '
                  'error so far.')
flags.DEFINE_list('ckpts_to_eval', [],
                  'Epochs at which to evaluate a trained checkpoint.')
flags.DEFINE_enum('optimizer', 'adam', ['adam', 'rmsprop'],
                  'Keras optimizer for training.')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate for training.')
flags.DEFINE_float('weight_decay', 0., 'Weight decay for training.')
flags.DEFINE_float('val_size', 0.05,
                   'Amount of data to validate on. If < 1, the fraction of '
                   'total loaded data points. Else the number of data points.')
flags.DEFINE_boolean('val_full_episodes', True,
                     'If True, split data into train and validation on an '
                     'episode basis. Else split by individual time steps.')
flags.DEFINE_integer('test_set_size', 0,
                     'Number of additional demonstrations on which to evaluate '
                     'final model.')
flags.DEFINE_integer('test_set_start', None,
                     'Where in the dataset to start test set.')

flags.DEFINE_integer('batch_size', 64, 'Batch size for training.')
flags.DEFINE_integer('seed', 0, 'Experiment seed.')
flags.DEFINE_integer('eval_seed', 1,
                     'Seed for environment in which trained policy is '
                     'evaluated.')
flags.DEFINE_boolean('increment_eval_seed', False,
                     'If True, increment eval seed after each eval episode.')


# Flags for setting paths automatically.
flags.DEFINE_string('top_dir', None,
                    'If set, unset paths will be set relative to this '
                    'directory.')
flags.DEFINE_string('dataset_origin', '', 'Name of subdirectory for dataset.')
flags.DEFINE_string('dataset', None,
                    'Filename of demonstration dataset, of form '
                    's<seed>_e<num_episodes>.')
flags.DEFINE_boolean('load_saved', False,
                     'If True, load saved model from checkpoint relative to '
                     'top_dir. Else train from scratch.')
flags.DEFINE_boolean('eval_trained_weights', False,
                     'If True, eval weights in policy_init_path.')
flags.DEFINE_boolean('load_demos', True,
                     'If False and loading saved model, will skip loading demo '
                     'data.')
flags.DEFINE_string('exp_id', '', 'Experiment ID to add to output paths.')
flags.DEFINE_string('job_id', '', 'Job ID to add to output paths.')
# Flags for setting paths manually.
flags.DEFINE_string('demos_file', None,
                    'Pickle file from which to read demonstration data.')
flags.DEFINE_string('ckpt_dir', None,
                    'If set, directory for model checkpoints.')


flags.DEFINE_enum('domain', 'mime', ['mime', 'adroit'],
                  'Domain of task.')
flags.DEFINE_integer('max_demos_to_load', None,
                     'Maximum number of demos from demos_file (in order) to '
                     'use.')
flags.DEFINE_integer('max_demo_length', None,
                     'If set, trim demonstrations to this length.')
flags.DEFINE_string('demo_task', None,
                    'Task used to gather demos in dataset, if different from '
                    'eval_task.')
flags.DEFINE_string('eval_task', None,
                    'If set, evaluate trained policy on this mime task.')
flags.DEFINE_integer('num_eval_episodes', 100,
                     'If eval_task is set, number of episodes to evaluate '
                     'trained policy for.')
flags.DEFINE_boolean('stop_if_stuck', False,
                     'If True, end episode if observations and actions are '
                     'stuck.')
flags.DEFINE_string('eval_id', '', 'ID to add to evaluation output path.')
flags.DEFINE_boolean('render_eval', False,
                     'If True, render environment during evaluation.')
flags.DEFINE_boolean('use_egl', False, 'If True, use EGL for rendering mime.')

flags.DEFINE_integer('eval_episodes_to_save', 0,
                     'The number of eval episodes whose frames to write to '
                     'file.')
flags.DEFINE_boolean('late_fusion', False,
                     'If True, fuse stacked frames after convolutional layers. '
                     'If False, fuse at network input.')
flags.DEFINE_string('policy_init_path', None,
                    'If set, initialize network weights from a pickle file at '
                    'this path.')
flags.DEFINE_boolean('clip_actions', False,
                     'If True, clip actions to unit interval before '
                     'normalization.')

FLAGS = flags.FLAGS


def set_train_str():
  """Set parameter string identifying training configuration."""
  val_full_episodes_str = 'e' if FLAGS.val_full_episodes else ''
  grip_action_str = 'b' if FLAGS.binary_grip_action else 'v'
  if FLAGS.early_closing:
    grip_action_str += 'e'
  if FLAGS.grip_action_from_state:
    grip_action_str += 's'
  if FLAGS.zero_action_keeps_state:
    grip_action_str += '0'
  if FLAGS.action_norm == 'zeromean_unitvar':
    grip_action_str += 'n'
  if FLAGS.signals_norm == 'zeromean_unitvar':
    grip_action_str += 'o'
  input_frames_str = str(FLAGS.num_input_frames)
  if FLAGS.crop_frames:
    input_frames_str += 'c'
  if FLAGS.augment_frames:
    input_frames_str += 'a'
  demos_str = ''
  clip_str = 'c' if FLAGS.clip_actions else ''
  if FLAGS.max_demos_to_load is not None:
    demos_str = f'_{FLAGS.max_demos_to_load}demos'
  train_str = '{}{}_w{}_b{}_val{}{}_{}{}t{}_{}epochs_{}frames_g{}_s{}'.format(
      FLAGS.optimizer, FLAGS.learning_rate, FLAGS.weight_decay,
      FLAGS.batch_size, FLAGS.val_size, val_full_episodes_str, clip_str,
      ','.join([str(t) for t in FLAGS.target_offsets]), demos_str,
      FLAGS.num_epochs, input_frames_str, grip_action_str, FLAGS.seed)
  return train_str


def set_paths(demo_task):
  """Set input data path as well as output paths for checkpoints and summaries.

  Args:
    demo_task: Task for which to load demonstrations.
  Returns:
    demos_file: Path to demonstration dataset.
    ckpt_dir: Directory to which to write model checkpoints.
    summary_dir: Directory to which to write TensorBoard summaries.
  """
  demos_file = FLAGS.demos_file
  ckpt_dir = FLAGS.ckpt_dir
  summary_dir = None
  if FLAGS.top_dir is not None:
    dataset_origin = FLAGS.dataset_origin

    if FLAGS.dataset is not None:  # Manually set for mime.
      dataset = f'{demo_task}/{FLAGS.dataset}'
      dataset_filename = f'{dataset}.pkl'
    else:  # Automatically set for Adroit.
      dataset = f'{demo_task}-v0_demos'
      # dataset = f'train_paths_hand_{demo_task}_batch_0'
      # dataset = dataset.replace('_relocate_', '_pickup_')
      dataset_filename = f'{dataset}.pickle'

    demos_file = os.path.join(
        FLAGS.top_dir, 'bc_demos', dataset_origin, dataset_filename)

    train_str = set_train_str()
    job_id = train_utils.set_job_id()
    network = FLAGS.network
    if network == 'hand_vil' and not FLAGS.late_fusion:
      network = 'e' + network
    ckpt_dir = os.path.join(
        FLAGS.top_dir, 'bc_policy', dataset_origin, FLAGS.exp_id, dataset,
        network, job_id, train_str)
    summary_dir = train_utils.get_summary_dir_for_ckpt_dir(ckpt_dir, network)
    if not FLAGS.load_saved:
      # If training a new model and checkpoints or summaries already exist,
      # overwrite fully.
      if gfile.exists(summary_dir):
        gfile.DeleteRecursively(summary_dir)
      if gfile.exists(ckpt_dir):
        gfile.DeleteRecursively(ckpt_dir)
  if not FLAGS.load_demos:
    demos_file = None
  return demos_file, ckpt_dir, summary_dir


def main(_):
  os.environ['PYTHONHASHSEED'] = str(FLAGS.seed)
  random.seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)
  tf.random.set_seed(FLAGS.seed)
  # Silcence warning about missing gradients for log_std (not used for l2 loss).
  tf.get_logger().setLevel('ERROR')

  demo_task = FLAGS.demo_task or FLAGS.eval_task
  if demo_task is None:
    raise ValueError('eval_task or demo_task must be set to normalize actions')
  # Environment whose limits are used for action and / or signals normalization.
  image_size = FLAGS.image_size
  if image_size is None:
    # Default sizes.
    image_size = {
        'adroit': 128,
        'mime': 240,
    }[FLAGS.domain]
  env = train_utils.make_environment(
      FLAGS.domain, demo_task, image_size=image_size)

  demos_file, ckpt_dir, summary_dir = set_paths(demo_task)
  domain_utils = adroit_utils if FLAGS.domain == 'adroit' else mime_utils
  visible_state_features = domain_utils.get_visible_features_for_task(
      demo_task, FLAGS.visible_state)
  agent = bc_agent.BCAgent(
      network_type=FLAGS.network,
      input_type=FLAGS.input_type,
      binary_grip_action=FLAGS.binary_grip_action,
      grip_action_from_state=FLAGS.grip_action_from_state,
      zero_action_keeps_state=FLAGS.zero_action_keeps_state,
      early_closing=FLAGS.early_closing,
      num_input_frames=FLAGS.num_input_frames,
      crop_frames=FLAGS.crop_frames,
      full_image_size=image_size,
      crop_size=image_size - FLAGS.crop_margin_size,
      target_offsets=[int(t) for t in FLAGS.target_offsets],
      visible_state_features=visible_state_features,
      action_norm=FLAGS.action_norm,
      signals_norm=FLAGS.signals_norm,
      action_space='tool_lin' if FLAGS.domain == 'mime' else demo_task,
      last_activation=FLAGS.last_activation,
      fc_layer_sizes=[int(i) for i in FLAGS.fc_layer_sizes],
      weight_decay=FLAGS.weight_decay,
      env=env,
      late_fusion=FLAGS.late_fusion,
      init_scheme=FLAGS.weight_init_scheme)
  dataset = None
  if demos_file is not None:
    dataset = train_utils.prepare_demos(
        demos_file, FLAGS.input_type, FLAGS.max_demos_to_load,
        FLAGS.max_demo_length, FLAGS.augment_frames, agent, ckpt_dir,
        FLAGS.val_size, FLAGS.val_full_episodes)

  # For evaluating hand_vil trained model (without load_saved flag)
  if FLAGS.eval_trained_weights:
    # Get directory name identifying experiment to use as ckpt_id.
    ckpt_id = FLAGS.policy_init_path
    assert os.path.basename(ckpt_id) == 'trained_policy_ep_1_numpy.pickle'
    ckpt_id = os.path.dirname(ckpt_id)
    if FLAGS.eval_task != 'relocate':
      assert (
          os.path.basename(ckpt_id)
          == f'dagger_hand_{FLAGS.eval_task}_viz_policy')
    ckpt_id = os.path.dirname(ckpt_id)
    ckpt_id = os.path.basename(ckpt_id)
    if FLAGS.eval_task != 'relocate':
      assert ckpt_id.startswith(f'hand_{FLAGS.eval_task}_')
    eval_path = train_utils.set_eval_path(ckpt_dir, FLAGS.eval_id, ckpt_id)
    print('eval path', eval_path)
    if not gfile.exists(os.path.dirname(eval_path)):
      gfile.makedirs(os.path.dirname(eval_path))
    unused_success_rates = eval_loop.eval_policy(
        env, FLAGS.eval_seed, FLAGS.increment_eval_seed, agent,
        FLAGS.num_eval_episodes, eval_path, FLAGS.eval_episodes_to_save,
        summary_writer=None, summary_key=None,
        stop_if_stuck=FLAGS.stop_if_stuck)
    return
  epochs_to_eval = [int(epoch) for epoch in FLAGS.epochs_to_eval]
  if not FLAGS.load_saved:
    if ckpt_dir is not None:
      gfile.makedirs(ckpt_dir)
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
        val_full_episodes=True, reset_agent_stats=False)
  num_eval_episodes = FLAGS.num_eval_episodes
  if FLAGS.eval_task is not None and num_eval_episodes > 0:
    env = train_utils.make_environment(
        FLAGS.domain, FLAGS.eval_task, FLAGS.use_egl, FLAGS.render_eval,
        image_size)
    summary_writer = train_utils.make_summary_writer(summary_dir)
    ckpts_to_eval = train_utils.get_checkpoints_to_evaluate(
        ckpt_dir, epochs_to_eval, FLAGS.ckpts_to_eval)
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
