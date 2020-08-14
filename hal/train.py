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

# Lint as: python3
r"""Train low-level policy."""
# pylint: disable=unused-variable
# pylint: disable=g-import-not-at-top
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import sys

from absl import app
from absl import flags
import tensorflow as tf

from hal.experiment_config import get_exp_config
from hal.experiment_setup import experiment_setup
from hal.utils.config import Config
from hal.utils.logger import Logger
from hal.utils.logger import Logger2

if 'gfile' not in sys.modules:
  import tf.io.gfile as gfile


FLAGS = flags.FLAGS
flags.DEFINE_bool('use_tf2', True, 'use eager execution')
flags.DEFINE_bool('use_nn_relabeling', False,
                  'use function approximators for relabeling')
flags.DEFINE_bool(
    'use_labeler_as_reward', False,
    'use the labeling function reward instead of true environment reward')
flags.DEFINE_bool(
    'use_oracle_instruction', True,
    'use the oracle/environment to generate the relabeling instructions')
flags.DEFINE_string('save_dir', None, 'experiment home directory')
flags.DEFINE_string('agent_type', 'pm', 'Which agent to use')
flags.DEFINE_string('scenario_type', 'fixed_primitive', 'Which env to use')
flags.DEFINE_bool('save_model', False, 'save model and log')
flags.DEFINE_bool('save_video', False, 'save video for evaluation')
flags.DEFINE_integer('save_interval', 50, 'intervals between saving models')
flags.DEFINE_integer('video_interval', 400, 'intervals between videos')
flags.DEFINE_bool('direct_obs', True, 'direct observation')
flags.DEFINE_string('action_type', 'perfect', 'what type of action to use')
flags.DEFINE_string('obs_type', 'order_invariant', 'type of observation')
flags.DEFINE_integer('img_resolution', 64, 'resolution of image observations')
flags.DEFINE_integer('render_resolution', 300, 'resolution of rendered image')
flags.DEFINE_integer('max_episode_length', 50, 'maximum episode duration')
flags.DEFINE_integer('num_epoch', 200, 'number of epoch')
flags.DEFINE_integer('num_cycle', 50, 'number of cycle per epoch')
flags.DEFINE_integer('num_episode', 50, 'number of episode per cycle')
flags.DEFINE_integer('optimization_steps', 100, 'optimization per episode')
flags.DEFINE_integer('collect_cycle', 10, 'cycles for populating buffer')
flags.DEFINE_integer('future_k', 3, 'number of future to put into buffer')
flags.DEFINE_integer('buffer_size', int(1e6), 'size of replay buffer')
flags.DEFINE_integer('batchsize', 128, 'batchsize')
flags.DEFINE_integer('rollout_episode', 10, 'number of episode for testing')
flags.DEFINE_bool('record_trajectory', False,
                  'test using the same epsilon as training and record traj')
flags.DEFINE_float('polyak_rate', 0.95, 'moving average factor for target')
flags.DEFINE_float('discount', 0.5, 'discount factor')
flags.DEFINE_float('initial_epsilon', 1.0, 'initial epsilon')
flags.DEFINE_float('min_epsilon', 0.1, 'minimum epsilon')
flags.DEFINE_float('learning_rate', 1e-4, 'minimum epsilon')
flags.DEFINE_float('epsilon_decay', 0.95, 'decay factor for epsilon')
flags.DEFINE_float('sample_new_scene_prob', 0.1,
                   'Probability of sampling a new scene')
flags.DEFINE_bool('record_atomic_instruction', False, 'record atomic goals')
flags.DEFINE_integer('k_immediate', 2,
                     'number of immediate correct statements added')
flags.DEFINE_bool('masking_q', False, 'mask the end of the episode')
flags.DEFINE_bool('paraphrase', False, 'paraphrase sentences')
flags.DEFINE_bool('relabeling', True, 'use hindsight experience replay')
flags.DEFINE_bool('use_subset_instruction', False,
                  'use a subset of 600 sentences, for generalization')
flags.DEFINE_string('image_action_parameterization', 'regular',
                    'type of action parameterization used by the image model')
flags.DEFINE_integer('frame_skip', 20, 'simulation step for the physics')
flags.DEFINE_bool('use_polar', False,
                  'use polar coordinate for neighbor assignment')
flags.DEFINE_bool('suppress', False,
                  'suppress movements of unnecessary objects')
flags.DEFINE_bool('diverse_scene_content', False,
                  'whether to use variable scene content')
flags.DEFINE_bool('use_synonym_for_rollout', False,
                  'use unseen synonyms for rolling out')
flags.DEFINE_float('reward_shape_val', 0.25, 'Value for reward shaping')
flags.DEFINE_string('instruction_repr', 'language',
                    'representation of the instruction')
flags.DEFINE_string('encoder_type', 'vanilla_rnn', 'type of language encoder')
flags.DEFINE_string('embedding_type', 'random',
                    'type of word embedding to use for training agent')
flags.DEFINE_bool('negate_unary', True, 'Negate unary sentences')
flags.DEFINE_string('experiment_confg', None, 'specific experiment config')
flags.DEFINE_bool('trainable_encoder', True, 'Is encoder trainable for agent.')
flags.DEFINE_bool('use_movement_bonus', False, 'Encourage moving objects.')
flags.DEFINE_string('varying', None, 'What parameters are changing.')
flags.DEFINE_integer('generated_label_num', 50, 'Number of generated label')
flags.DEFINE_float('sampling_temperature', 1.0, 'Sampling temperature')
flags.DEFINE_string('reset_mode', 'regular', 'How the environment is reset')
flags.DEFINE_float('reward_scale', 1.0, 'Reward scale of the environment')
# Maxent IRL
flags.DEFINE_bool('maxent_irl', False, 'Use maximum entropy IRL')
flags.DEFINE_integer('irl_parallel_n', 1, 'Number of parallel inference in irl')
flags.DEFINE_integer('irl_sample_goal_n', 32, 'Number of goals sampled for irl')
flags.DEFINE_float('relabel_proportion', 0.5, 'portion of minibatch to relabel')
flags.DEFINE_float('entropy_alpha', 0.001, 'alpha for max ent')


def main(_):
  if FLAGS.use_tf2:
    tf.enable_v2_behavior()
  config_content = {
      'action_type': FLAGS.action_type,
      'obs_type': FLAGS.obs_type,
      'reward_shape_val': FLAGS.reward_shape_val,
      'use_subset_instruction': FLAGS.use_subset_instruction,
      'frame_skip': FLAGS.frame_skip,
      'use_polar': FLAGS.use_polar,
      'suppress': FLAGS.suppress,
      'diverse_scene_content': FLAGS.diverse_scene_content,
      'buffer_size': FLAGS.buffer_size,
      'use_movement_bonus': FLAGS.use_movement_bonus,
      'reward_scale': FLAGS.reward_scale,
      'scenario_type': FLAGS.scenario_type,
      'img_resolution': FLAGS.img_resolution,
      'render_resolution': FLAGS.render_resolution,

      # agent
      'agent_type': FLAGS.agent_type,
      'masking_q': FLAGS.masking_q,
      'discount': FLAGS.discount,
      'instruction_repr': FLAGS.instruction_repr,
      'encoder_type': FLAGS.encoder_type,
      'learning_rate': FLAGS.learning_rate,
      'polyak_rate': FLAGS.polyak_rate,
      'trainable_encoder': FLAGS.trainable_encoder,
      'embedding_type': FLAGS.embedding_type,

      # learner
      'num_episode': FLAGS.num_episode,
      'optimization_steps': FLAGS.optimization_steps,
      'batchsize': FLAGS.batchsize,
      'sample_new_scene_prob': FLAGS.sample_new_scene_prob,
      'max_episode_length': FLAGS.max_episode_length,
      'record_atomic_instruction': FLAGS.record_atomic_instruction,
      'paraphrase': FLAGS.paraphrase,
      'relabeling': FLAGS.relabeling,
      'k_immediate': FLAGS.k_immediate,
      'future_k': FLAGS.future_k,
      'negate_unary': FLAGS.negate_unary,
      'min_epsilon': FLAGS.min_epsilon,
      'epsilon_decay': FLAGS.epsilon_decay,
      'collect_cycle': FLAGS.collect_cycle,
      'use_synonym_for_rollout': FLAGS.use_synonym_for_rollout,
      'reset_mode': FLAGS.reset_mode,
      'maxent_irl': FLAGS.maxent_irl,

      # relabeler
      'sampling_temperature': FLAGS.sampling_temperature,
      'generated_label_num': FLAGS.generated_label_num,
      'use_labeler_as_reward': FLAGS.use_labeler_as_reward,
      'use_oracle_instruction': FLAGS.use_oracle_instruction
  }

  if FLAGS.maxent_irl:
    assert FLAGS.batchsize % FLAGS.irl_parallel_n == 0
    config_content['irl_parallel_n'] = FLAGS.irl_parallel_n
    config_content['irl_sample_goal_n'] = FLAGS.irl_sample_goal_n
    config_content['relabel_proportion'] = FLAGS.relabel_proportion
    config_content['entropy_alpha'] = FLAGS.entropy_alpha

  cfg = Config(config_content)

  if FLAGS.experiment_confg:
    cfg.update(get_exp_config(FLAGS.experiment_confg))

  save_home = FLAGS.save_dir if FLAGS.save_dir else tf.test.get_temp_dir()
  if FLAGS.varying:
    exp_name = 'exp-'
    for varied_var in FLAGS.varying.split(','):
      exp_name += str(varied_var) + '=' + str(FLAGS[varied_var].value) + '-'
  else:
    exp_name = 'SingleExperiment'
  save_dir = os.path.join(save_home, exp_name)
  try:
    gfile.MkDir(save_home)
  except gfile.Error as e:
    print(e)
  try:
    gfile.MkDir(save_dir)
  except gfile.Error as e:
    print(e)

  cfg.update(Config({'model_dir': save_dir}))

  print('############################################################')
  print(cfg)
  print('############################################################')

  env, learner, replay_buffer, agent, extra_components = experiment_setup(
      cfg, FLAGS.use_tf2, FLAGS.use_nn_relabeling)
  agent.init_networks()

  if FLAGS.use_tf2:
    logger = Logger2(save_dir)
  else:
    logger = Logger(save_dir)

  with gfile.GFile(os.path.join(save_dir, 'config.json'), mode='w+') as f:
    json.dump(cfg.as_dict(), f, sort_keys=True, indent=4)

  if FLAGS.save_model and tf.train.latest_checkpoint(save_dir):
    print('Loading saved weights from {}'.format(save_dir))
    agent.load_model(save_dir)

  if FLAGS.save_model:
    video_dir = os.path.join(save_dir, 'rollout_cycle_{}.mp4'.format('init'))
    print('Saving video to {}'.format(video_dir))
    learner.rollout(
        env,
        agent,
        video_dir,
        num_episode=FLAGS.rollout_episode,
        record_trajectory=FLAGS.record_trajectory)

  success_rate_ema = -1.0

  # Training loop
  for epoch in range(FLAGS.num_epoch):
    for cycle in range(FLAGS.num_cycle):
      stats = learner.learn(env, agent, replay_buffer)

      if success_rate_ema < 0:
        success_rate_ema = stats['achieved_goal']

      loss_dropped = stats['achieved_goal'] < 0.1 * success_rate_ema
      far_along_training = stats['global_step'] > 100000
      if FLAGS.save_model and loss_dropped and far_along_training:
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('Step {}: Loading models due to sudden loss drop D:'.format(
            stats['global_step']))
        print('Dropped from {} to {}'.format(success_rate_ema,
                                             stats['achieved_goal']))
        agent.load_model(save_dir)
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        continue
      success_rate_ema = 0.95 * success_rate_ema + 0.05 * stats['achieved_goal']

      at_save_interval = stats['global_step'] % FLAGS.save_interval == 0
      better_reward = stats['achieved_goal'] > success_rate_ema
      if FLAGS.save_model and at_save_interval and better_reward:
        print('Saving model to {}'.format(save_dir))
        agent.save_model(save_dir)

      if FLAGS.save_model and stats['global_step'] % FLAGS.video_interval == 0:
        video_dir = os.path.join(save_dir, 'rollout_cycle_{}.mp4'.format(cycle))
        print('Saving video to {}'.format(video_dir))
        test_success_rate = learner.rollout(
            env,
            agent,
            video_dir,
            record_video=FLAGS.save_video,
            num_episode=FLAGS.rollout_episode,
            record_trajectory=FLAGS.record_trajectory)
        stats['Test Success Rate'] = test_success_rate
        print('Test Success Rate: {}'.format(test_success_rate))

      stats['ema success rate'] = success_rate_ema
      logger.log(epoch, cycle, stats)


if __name__ == '__main__':
  app.run(main)
