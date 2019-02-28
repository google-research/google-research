# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Helper for building arguments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
from absl import flags

from meta_reward_learning.semantic_parsing import common_flags

FLAGS = flags.FLAGS
flags.adopt_module_key_flags(common_flags)


def build_eval_args(dataset, data_dir):
  """Build args for the test evaluation worker."""
  data_dir = osp.join(data_dir, 'projects/data/{}'.format(dataset))
  output_dir = osp.join(data_dir, 'output')
  if dataset == 'wikitable':
    input_dir = osp.join(data_dir, 'processed_input/preprocess_14')
    executor = 'wtq'
  elif dataset == 'wikisql':
    input_dir = osp.join(data_dir, 'processed_input/preprocess_2')
    executor = 'wikisql'
  eval_file = FLAGS.eval_file if FLAGS.eval_file else 'test_split'
  if (eval_file != 'test_split') and dataset == 'wikitable':
    # For example, when we want to use dev_split to run evaluation
    eval_file = osp.join(input_dir, 'data_split_1',
                         '{}.jsonl').format(eval_file)
  else:
    eval_file = osp.join(input_dir, '{}.jsonl').format(eval_file)

  exp_name = 'eval_beam{}/{}'.format(FLAGS.eval_beam_size,
                                     FLAGS.experiment_name)
  exec_args = dict(
      eval_only=True,
      eval_use_gpu=FLAGS.eval_use_gpu,
      eval_batch_size=FLAGS.eval_batch_size,
      eval_gpu_id=0,
      show_log=FLAGS.show_log,
      experiment_to_eval=FLAGS.experiment_name,
      eval_beam_size=FLAGS.eval_beam_size,
      output_dir=output_dir,
      executor=executor,
      experiment_name=exp_name,
      ckpt_from_another=FLAGS.ckpt_from_another,
      eval_file=eval_file)
  return exec_args


def build_train_args(dataset, data_dir, experiment_name=None):
  """Build args for the training/dev_evaluation worker."""
  if experiment_name is None:
    experiment_name = FLAGS.experiment_name
  data_dir = osp.join(data_dir, 'projects/data/{}'.format(dataset))
  output_dir = osp.join(data_dir, 'output')

  if dataset == 'wikitable':
    program_file_dir = osp.join(data_dir, 'processed_input')
    input_dir = osp.join(data_dir, 'processed_input/preprocess_14')
    train_shard_dir = osp.join(input_dir, 'data_split_1')
    shard_end = 90
    dropout = 0.2
    num_val_shards = 15
    total_envs = 11321
    programs_per_shard = 125  # true for the last few last shards
    max_n_exp = 3
    en_maxlen = 45
    executor = 'wtq'
    maxlen = 22
    trigger_words_file = osp.join(data_dir, 'raw_input/trigger_word_all.json')
  elif dataset == 'wikisql':
    input_dir = osp.join(data_dir, 'processed_input/preprocess_2')
    program_file_dir = input_dir
    train_shard_dir = input_dir
    shard_end = 30
    dropout = 0.1
    num_val_shards = 5
    total_envs = 56355
    programs_per_shard = 1878  # true for the last few last shards
    max_n_exp = 4
    en_maxlen = 54
    executor = 'wikisql'
    maxlen = 28
    trigger_words_file = None
  else:
    raise (ValueError, '{} doesn\'t refer to any dataset'.format(dataset))

  # If meta learning is set to be False, don't remove any training data.
  if FLAGS.use_validation_for_meta_train or (not FLAGS.meta_learn):
    num_val_shards = 0
  # If num_val_shards is passed explicitly by the user, use that value instead.
  if FLAGS.num_val_shards > 0:
    num_val_shards = FLAGS.num_val_shards

  num_envs = total_envs - num_val_shards * programs_per_shard
  train_shard_prefix = 'train_split_shard_{}-'.format(shard_end)
  main_dev_file = 'dev_split' if not FLAGS.dev_file else FLAGS.dev_file
  dev_file = osp.join(train_shard_dir, '{}.jsonl'.format(main_dev_file))
  embedding_file = osp.join(
      data_dir, 'raw_input/{}_glove_embedding_mat.npy'.format(dataset))
  vocab_file = osp.join(data_dir,
                        'raw_input/{}_glove_vocab.json'.format(dataset))
  table_file = osp.join(input_dir, 'tables.jsonl')
  en_vocab_file = osp.join(input_dir, 'en_vocab_min_count_5.json')

  use_replay_prob_as_weight = False
  use_top_k_replay_samples = FLAGS.use_top_k_replay_samples
  random_replay_samples = False
  use_trainer_prob = False
  ent_reg = 0.0

  config = FLAGS.config
  if config not in ['mapo', 'mml', 'iml', 'hard_em']:
    raise (ValueError, 'Config {} is not valid'.format(config))
  if config == 'mapo':
    use_replay_prob_as_weight = True
    if dataset == 'wikitable':
      ent_reg = 0.01
  elif config == 'hard_em':
    use_top_k_replay_samples = True
  elif config == 'iml':
    random_replay_samples = True

  if FLAGS.entropy_reg_coeff:
    ent_reg = FLAGS.entropy_reg_coeff
  if FLAGS.save_every_n > 0:
    save_every_n = FLAGS.save_every_n
  else:
    save_every_n = (num_envs * FLAGS.n_replay_samples) // FLAGS.batch_size

  if FLAGS.saved_replay_program_files:
    replay_files = FLAGS.saved_replay_program_files.split(',')
    for i, filename in enumerate(replay_files):
      replay_files[i] = osp.join(program_file_dir, '{}.json').format(filename)
    saved_replay_program_files = ','.join(replay_files)
  else:
    saved_replay_program_files = ''

  if FLAGS.dropout:
    dropout = FLAGS.dropout  # Override the default value

  exec_args = dict(
      debug_model=FLAGS.debug_model,
      eval_only=False,
      output_dir=output_dir,
      dev_file=dev_file,
      train_shard_dir=train_shard_dir,
      train_shard_prefix=train_shard_prefix,
      shard_start=0,
      shard_end=shard_end,
      num_val_shards=num_val_shards,
      load_saved_programs=True,
      embedding_file=embedding_file,
      vocab_file=vocab_file,
      table_file=table_file,
      en_vocab_file=en_vocab_file,
      trigger_words_file=trigger_words_file,
      n_explore_samples=FLAGS.n_explore_samples,
      n_extra_explore_for_hard=FLAGS.n_extra_explore_for_hard,
      greedy_exploration=FLAGS.greedy_exploration,
      use_cache=True,
      dropout=dropout,
      hidden_size=200,
      attn_size=200,
      attn_vec_size=200,
      en_embedding_size=200,
      en_bidirectional=True,
      n_layers=2,
      en_n_layers=2,
      use_pretrained_embeddings=True,
      pretrained_embedding_size=300,
      value_embedding_size=300,
      n_policy_samples=FLAGS.n_policy_samples,
      use_replay_samples_in_train=True,
      use_nonreplay_samples_in_train=FLAGS.use_nonreplay_samples_in_train,
      use_policy_samples_in_train=FLAGS.use_policy_samples_in_train,
      use_replay_prob_as_weight=use_replay_prob_as_weight,
      use_top_k_replay_samples=use_top_k_replay_samples,
      fixed_replay_weight=1.0,
      random_replay_samples=random_replay_samples,
      use_trainer_prob=use_trainer_prob,
      min_replay_weight=FLAGS.min_replay_weight,
      truncate_replay_buffer_at_n=FLAGS.truncate_replay_buffer_at_n,
      max_n_exp=max_n_exp,
      executor=executor,
      use_baseline=FLAGS.use_baseline,
      max_n_mem=60,
      max_n_valid_indices=60,
      en_maxlen=en_maxlen if not FLAGS.unittest else 40,
      maxlen=maxlen,
      saved_program_file=osp.join(program_file_dir,
                                  '{}.json'.format(FLAGS.saved_program_file)),
      saved_replay_program_files=saved_replay_program_files,
      saved_val_program_file=osp.join(
          program_file_dir, '{}.json'.format(FLAGS.saved_val_program_file)),
      entropy_reg_coeff=ent_reg,
      eval_beam_size=FLAGS.eval_beam_size,
      score_norm_fn=FLAGS.score_norm_fn,
      show_log=FLAGS.show_log,
      unittest=FLAGS.unittest,
      trainable_only=FLAGS.trainable_only,
      experiment_name=experiment_name,
      meta_learn=FLAGS.meta_learn,
      meta_lr=FLAGS.meta_lr,
      optimizer=FLAGS.optimizer,
      max_programs=FLAGS.max_programs,
      max_val_programs=FLAGS.max_val_programs,
      # Won't work for with hyperparameter exploration for batch_size
      save_every_n=save_every_n,
      batch_size=FLAGS.batch_size,
      n_replay_samples=FLAGS.n_replay_samples,
      learning_rate=FLAGS.learning_rate,
      n_steps=FLAGS.n_steps,
      momentum=FLAGS.momentum,
      init_score_path=FLAGS.init_score_path,
      ckpt_from_another=FLAGS.ckpt_from_another,
      sampling_strategy=FLAGS.sampling_strategy,
      score_model=FLAGS.score_model,
      max_grad_norm=FLAGS.max_grad_norm,
      val_batch_size=FLAGS.val_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      val_objective=FLAGS.val_objective,
      use_model_weight_init=FLAGS.use_model_weight_init,
      use_memory_weight_clipping=FLAGS.use_memory_weight_clipping,
      score_temperature=FLAGS.score_temperature,
      plot_summaries=FLAGS.plot_summaries,
      use_validation_for_meta_train=FLAGS.use_validation_for_meta_train,
      experiment_to_load=FLAGS.experiment_to_load)

  return exec_args
