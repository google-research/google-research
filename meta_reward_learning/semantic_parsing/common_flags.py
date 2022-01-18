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

"""Common flags for experiments.

Modules that import these flags can choose to use any number of these flags.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

FLAGS = flags.FLAGS


# Algorithm to run
flags.DEFINE_string(
    'config', 'mapo', 'Config to be used for the experiment'
    'Possible configs: mapo|mml|iml|hard_em')

# Experiment name
flags.DEFINE_string('output_dir', 'output', 'output folder.')
flags.DEFINE_string(
    'experiment_name', None, 'All outputs of this experiment is'
    ' saved under a folder with the same name.')

# Tensorboard logging
flags.DEFINE_string('tb_log_dir', 'tb_log', 'Path for saving tensorboard logs.')

# Tensorflow model checkpoint.
flags.DEFINE_string('saved_model_dir', 'saved_model', 'Path for saving models.')
flags.DEFINE_string('best_model_dir', 'best_model',
                    'Path for saving best models.')
flags.DEFINE_string('init_model_path', '', 'Path for saving best models.')
flags.DEFINE_string('experiment_to_eval', '', '.')

# Loading pretrained model
flags.DEFINE_bool(
    'trainable_only', False, 'Whether to load only trainable'
    'variables or all global variables')
flags.DEFINE_string('experiment_to_load', '', '.')
flags.DEFINE_integer(
    'save_every_n', -1,
    'Save model to a ckpt every n train steps, -1 means save '
    'every epoch.')

# Model
# Computer
flags.DEFINE_integer('max_n_mem', 100,
                     'Max number of memory slots in the "computer".')
flags.DEFINE_integer('max_n_exp', 3,
                     'Max number of expressions allowed in a program.')
flags.DEFINE_integer('max_n_valid_indices', 100,
                     'Max number of valid tokens during decoding.')
flags.DEFINE_bool('use_cache', False,
                  'Use cache to avoid generating the same samples.')
flags.DEFINE_string('en_vocab_file', '', '.')
flags.DEFINE_string('executor', 'wtq', 'Which executor to use, wtq or wikisql.')

# ## neural network
flags.DEFINE_integer('hidden_size', 200, 'Number of hidden units.')
flags.DEFINE_integer('attn_size', 200, 'Size of attention vector.')
flags.DEFINE_integer('attn_vec_size', 200,
                     'Size of the vector parameter for computing attention.')
flags.DEFINE_integer('n_layers', 2, 'Number of layers in decoder.')
flags.DEFINE_integer('en_n_layers', 2, 'Number of layers in encoder.')
flags.DEFINE_integer('en_embedding_size', 200,
                     'Size of encoder input embedding.')
flags.DEFINE_integer('value_embedding_size', 300,
                     'Size of value embedding for the constants.')
flags.DEFINE_bool('en_bidirectional', False,
                  'Whether to use bidirectional RNN in encoder.')
flags.DEFINE_bool('en_attn_on_constants', False, '.')
flags.DEFINE_bool('use_pretrained_embeddings', True,
                  'Whether to use pretrained embeddings.')
flags.DEFINE_integer('pretrained_embedding_size', 300,
                     'Size of pretrained embedding.')

# Features
flags.DEFINE_integer('n_de_output_features', 1,
                     'Number of features in decoder output softmax.')
flags.DEFINE_integer('n_en_input_features', 1,
                     'Number of features in encoder inputs.')

# Data
flags.DEFINE_string('table_file', '',
                    'Path to the file of wikitables, a jsonl file.')
flags.DEFINE_string('train_file', '',
                    'Path to the file of training examples, a jsonl file.')
flags.DEFINE_string('dev_file', '',
                    'Path to the file of training examples, a jsonl file.')
flags.DEFINE_string('eval_file', '',
                    'Path to the file of test examples, a jsonl file.')
flags.DEFINE_string('embedding_file', '',
                    'Path to the file of pretrained embeddings, a npy file.')
flags.DEFINE_string(
    'vocab_file', '', 'Path to the vocab file for the pretrained embeddings, a '
    'json file.')
flags.DEFINE_string('train_shard_dir', '',
                    'Folder containing the sharded training data.')
flags.DEFINE_string('train_shard_prefix', '',
                    'The prefix for the sharded files.')
flags.DEFINE_integer('n_train_shard', 90, 'Number of shards in total.')
flags.DEFINE_integer('shard_start', 0, 'Start id of the shard to use.')
flags.DEFINE_integer('shard_end', 90, 'End id of the shard to use.')

# Load saved samples.
flags.DEFINE_bool('load_saved_programs', False,
                  'Whether to use load saved programs from exploration.')
flags.DEFINE_string('saved_program_file', '', 'Saved program file.')
flags.DEFINE_string(
    'saved_replay_program_files', '',
    'Other saved programs files saved from training replay '
    'buffer of some algorithms. Comma separted list of strings')
flags.DEFINE_string('trigger_words_file', '',
                    'Json file containing the trigger words.')
# Training
## Core Params
flags.DEFINE_integer('n_steps', 25000, 'Maximum number of steps in training.')
flags.DEFINE_integer('batch_size', 10, 'Model batch size.')
flags.DEFINE_integer('n_replay_samples', 5, 'Number of replay samples drawn.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')

flags.DEFINE_string('optimizer', 'sgd', '.')
flags.DEFINE_float('adam_beta1', 0.9, 'adam beta1 parameter.')
flags.DEFINE_float('max_grad_norm', 1.0, 'Maximum gradient norm.')
flags.DEFINE_float('l2_coeff', 0.0, 'l2 regularization coefficient.')
flags.DEFINE_float('dropout', 0.0, 'Dropout rate.')
flags.DEFINE_float('lm_loss_coeff', 0.0, 'Weight for lm loss.')
flags.DEFINE_float('entropy_reg_coeff', 0.0,
                   'Weight for entropy regularization.')
flags.DEFINE_float('momentum', 0.9, 'Weight for entropy regularization.')

# Exploration Samples
flags.DEFINE_integer('n_explore_samples', 0,
                     'Number of exploration samples per env per epoch.')
flags.DEFINE_integer('n_extra_explore_for_hard', 0,
                     'Number of exploration samples for hard envs.')
flags.DEFINE_integer('n_policy_samples', 0,
                     'Number of on-policy samples drawn.')
flags.DEFINE_bool(
    'greedy_exploration', False,
    'Whether to use a greedy policy when doing systematic '
    'exploration.')

# Distributed Training
flags.DEFINE_integer('n_actors', 3, 'Number of actors for generating samples.')

# Data saving related
flags.DEFINE_bool(
    'save_replay_buffer_at_end', False,
    'Whether to save the full replay buffer for each actor at the '
    'end of training or not')
flags.DEFINE_integer('log_buffer_size_every_n_epoch', 0,
                     'Log replay buffer size for each actor every n epochs')
flags.DEFINE_integer('log_samples_every_n_epoch', 0,
                     'Log samples every n epochs.')

# Policy Gradient Related
flags.DEFINE_bool('use_baseline', False,
                  'Whether to use baseline during policy gradient.')
flags.DEFINE_float(
    'min_prob', 0.0, 'Minimum probability of a negative example for it to be '
    'punished to avoid numerical issue.')

# Sampling Procedure (Determines the algorithm used)
flags.DEFINE_bool('use_replay_samples_in_train', False,
                  'Whether to use replay samples for training.')
flags.DEFINE_bool('random_replay_samples', False,
                  'randomly pick a replay samples as ML baseline.')
flags.DEFINE_bool('use_policy_samples_in_train', False,
                  'Whether to use on-policy samples for training.')
flags.DEFINE_bool('use_nonreplay_samples_in_train', False,
                  'Whether to use a negative samples for training.')
flags.DEFINE_bool(
    'use_top_k_replay_samples', False,
    'Whether to use the top k most probable (model probability) '
    'replay samples or to sample from the replay samples.')
flags.DEFINE_bool(
    'use_top_k_policy_samples', False,
    'Whether to use the top k most probable beam search '
    'samples or to sample from the replay samples.')
flags.DEFINE_float('fixed_replay_weight', 0.5,
                   'Weight for replay samples between 0.0 and 1.0.')
flags.DEFINE_bool(
    'use_replay_prob_as_weight', False,
    'Whether or not use replay probability as weight for replay '
    'samples.')
flags.DEFINE_float('min_replay_weight', 0.1, 'minimum replay weight.')
flags.DEFINE_bool('use_memory_weight_clipping', False, 'whether to use the '
                  'replay memory weight clipping or not')
flags.DEFINE_integer(
    'truncate_replay_buffer_at_n', 0,
    'Whether truncate the replay buffer to the top n highest '
    'prob trajs.')
flags.DEFINE_bool('use_trainer_prob', False,
                  'Whether to supply all the replay buffer for training.')

# Evaluation
flags.DEFINE_integer(
    'num_val_shards', 0, 'The number of training shards to '
    'be used for validation')
flags.DEFINE_bool('eval_only', False, 'only run evaluator on test.')
flags.DEFINE_string(
    'eval_dev', '', 'The type of dev set is to be used, '
    'Valid options include: 1)meta 2)validation')
flags.DEFINE_integer('eval_beam_size', 5,
                     'Beam size when evaluating on development set.')
flags.DEFINE_integer('eval_batch_size', 100,
                     'Batch size when evaluating on development set.')
flags.DEFINE_bool('debug_model', False, 'Whether to output debug information.')

# Device placement.
flags.DEFINE_bool('train_use_gpu', True,
                  'Whether to use gpu for training or not.')
flags.DEFINE_integer('train_gpu_id', 0, 'Id of the gpu used for training.')

flags.DEFINE_bool('eval_use_gpu', False, 'Whether to output debug information.')
flags.DEFINE_integer('eval_gpu_id', 0, 'Id of the gpu used for eval.')

flags.DEFINE_bool('actor_use_gpu', False,
                  'Whether to output debug information.')
flags.DEFINE_integer(
    'actor_gpu_start_id', 0,
    'Id of the gpu for the first actor, gpu for other actors '
    'will follow.')

# Testing/Debugging
flags.DEFINE_bool('show_log', True, 'Whether to show logging info.')
flags.DEFINE_bool('unittest', False, '.')
flags.DEFINE_integer('n_opt_step', 1,
                     'Number of optimization steps per training batch.')

# Summary plotting related
flags.DEFINE_bool(
    'plot_summaries', True,
    'Whether to plot the graph summaries useful for debugging '
    'purposes. If set to False, only a minimal set of summaries'
    '(avg return/length/prob) is plotted.')

# Max Sequence Lengths
flags.DEFINE_integer('maxlen', 25, 'Maximum length of a program')
flags.DEFINE_integer('en_maxlen', 70, 'Maximum length of a query')

# Meta Learning
flags.DEFINE_bool('meta_learn', False, 'Whether to use meta learning or not!')
flags.DEFINE_float('meta_lr', 1e-3, 'Learning rate to be used for meta tuning')
flags.DEFINE_string(
    'score_norm_fn', None, 'Function to be used to process the scores '
    'Valid values: softmax, sigmoid or None')
flags.DEFINE_float('score_temperature', 1.0,
                   'Temperature parameter for score softmax')
flags.DEFINE_integer(
    'val_batch_size', 512,
    'No of samples to use from validation set for meta tuning')
flags.DEFINE_string('saved_val_program_file', '',
                    'Json file containing high reward programs for val data')
flags.DEFINE_integer('max_programs', 50, 'Max number of progs for a question')
flags.DEFINE_integer(
    'max_val_programs', 1, 'Max number of programs to be '
    'added to replay buffer for each question in the meta '
    'training set')
flags.DEFINE_string('init_score_path', '', 'Restore the score function '
                    'from a specific path.')
flags.DEFINE_bool('ckpt_from_another', False, 'Whether we are checkpointing '
                  'from a cloud model or not')
flags.DEFINE_string(
    'sampling_strategy', 'probs',
    'Whether to use rewards for sampling trajs, "probs": Use only model probs '
    'for sampling, "reward": Just use rewards, "probs_and_reward": Product of '
    'rewards and probs, "st_estimator"')
flags.DEFINE_string('score_model', 'tabular', 'What score function to use')
flags.DEFINE_string(
    'val_objective', 'mapo', 'whether to use the mml vs mapo '
    'objective for indirect optimization on the validation set')
flags.DEFINE_boolean(
    'use_model_weight_init', False,
    'When doing meta learning, whether to initialize the '
    'weights using the model probability, so that the initial sampling '
    'looks like what the initial ckpt model would have done')
flags.DEFINE_boolean(
    'use_validation_for_meta_train', False,
    'Whether to use the validation set for meta learning or not')

if __name__ == '__main__':
  pass
