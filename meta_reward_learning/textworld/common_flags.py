# coding=utf-8
# Copyright 2021 The Google Research Authors.
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
"""Flags used in the experiment."""

from absl import flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('num_steps', 1500, 'Number of epochs we want to train our '
                     'RL agent')
flags.DEFINE_string(
    'train_dir', None, 'Directory in which the training '
    'checkpoints and tensorboard summaries are saved')
flags.DEFINE_string(
    'eval_dir', None, 'Directory to be used for restoring '
    'the best checkpoint of a trained agent')
flags.DEFINE_string('train_file', 'textworld-train.pkl', 'Training dataset')
flags.DEFINE_string('dev_file', 'textworld-dev.pkl', 'Validation dataset')
flags.DEFINE_string('test_file', 'textworld-test.pkl', 'Holdout test set')
flags.DEFINE_integer('n_train_envs', 125, 'Number of training envs ')
flags.DEFINE_integer('n_dev_envs', 25, 'Number of validation envs')
flags.DEFINE_float('eps', 0.0, 'Noise for random exploration')
flags.DEFINE_float('entropy_reg_coeff', 0.0, 'Entropy regularization')
flags.DEFINE_integer('units', 16,
                     'Number of hidden units in our policy network')
flags.DEFINE_integer('seed', 42, 'Random seed to be set for reproducibility')
flags.DEFINE_integer('dummy', None, 'Dummy variable')
flags.DEFINE_float('gamma', 1.0, 'Discount factor')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate for training the'
                   'RL agent')
flags.DEFINE_integer('grid_size', 7, 'Grid size of our text environment')
flags.DEFINE_integer('n_train_plants', 14, 'Number of plants to be used'
                     'for training grids')
flags.DEFINE_integer('n_dev_plants', 14, 'Number of plants to be used'
                     'for training grids')
flags.DEFINE_integer('n_test_plants', 14, 'Number of plants to be used'
                     'for test grids')
flags.DEFINE_float('max_grad_norm', 1.0, 'Number of plants to be used'
                   'for test grids')
flags.DEFINE_bool('eval_only', False, 'Whether to do evaluation or training')
flags.DEFINE_bool('is_debug', False, 'Whether to print debugging information')
flags.DEFINE_integer('save_every_n', 500, 'Number of steps after which the '
                     'model is saved')
flags.DEFINE_bool('explore', False, 'Whether to do exploration for generating '
                  'samples')
flags.DEFINE_bool('train_use_gpu', True, 'Whether to use GPU for training')
flags.DEFINE_bool('use_gold_trajs', False, 'Whether to use gold data for '
                  'training')
flags.DEFINE_float('min_replay_weight', 0.1, 'Clipping threshold for the '
                   'replay buffer')
flags.DEFINE_bool(
    'use_top_k_samples', False, 'Whether we want to use the top '
    'k samples from the replay buffer for training or not')
flags.DEFINE_integer('n_replay_samples', 1,
                     'Number of replay samples per context')
flags.DEFINE_bool('meta_learn', False, 'Use meta learning or not')
flags.DEFINE_float('meta_lr', 1e-3, 'Meta Learning rate')
flags.DEFINE_bool(
    'dev_explore', False, 'Whether to do exploration for '
    'generating samples on dev set.')
flags.DEFINE_bool('use_dev_gold_trajs', False, 'Whether to use gold data for '
                  'dev set.')
flags.DEFINE_string('pretrained_ckpt_dir', None, 'Directory from which to load'
                    'a pretrained checkpoint')
flags.DEFINE_bool('pretrained_load_data_only', False,
                  'Whether to just load data from the pretrained ckpt')
flags.DEFINE_bool('log_summaries', False,
                  'Whether to plot extra summaries or not')
FEATURE_KEYS = ['0', '1', '2', '3']
# pylint: disable=g-complex-comprehension
PAIR_FEATURE_KEYS = [
    '{}{}'.format(a1, a2) for a1 in FEATURE_KEYS for a2 in FEATURE_KEYS
]
# pylint: enable=g-complex-comprehension
PAIRWISE_WEIGHTS = ['w1', 'w2']
ALL_FEATURE_KEYS = PAIR_FEATURE_KEYS + PAIRWISE_WEIGHTS

flags.DEFINE_bool(
    'use_buffer_scorer', False, 'To use the buffer scorer with '
    'the given feature weight or not')
for k in ALL_FEATURE_KEYS:
  flags.DEFINE_float(
      'score_{}'.format(k), 0.0, 'Score for correspondence b/w '
      'command {} and action {}'.format(k[0], k[1]))
flags.DEFINE_string(
    'score_fn', 'linear', 'Type of score function to be '
    'used for meta learning: (1) linear or (2) simple_linear')

if __name__ == '__main__':
  pass
