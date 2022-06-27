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

"""Arg builder for the experiment launcher."""

import os.path as osp
from absl import flags
from meta_reward_learning.textworld import common_flags

FLAGS = flags.FLAGS
flags.adopt_module_key_flags(common_flags)


def build_args(main_dir, work_unit_name):
  """Builds the arguments to be passed to the XM executable."""
  data_dir = osp.join(main_dir, 'datasets')
  train_file = osp.join(data_dir, FLAGS.train_file)
  dev_file = osp.join(data_dir, FLAGS.dev_file)
  test_file = osp.join(data_dir, FLAGS.test_file)
  train_dir = osp.join(main_dir, 'experiments', work_unit_name)
  if FLAGS.eval_dir is not None:
    eval_dir = osp.join(main_dir, 'experiments', FLAGS.eval_dir)
  else:
    eval_dir = None
  if FLAGS.pretrained_ckpt_dir is not None:
    pretrained_ckpt_dir = osp.join(main_dir, 'experiments',
                                   FLAGS.pretrained_ckpt_dir)
  else:
    pretrained_ckpt_dir = None

  exec_args = dict(
      train_dir=train_dir,
      eval_dir=eval_dir,
      train_file=train_file,
      test_file=test_file,
      dev_file=dev_file,
      meta_learn=FLAGS.meta_learn,
      meta_lr=FLAGS.meta_lr,
      dev_explore=FLAGS.dev_explore,
      use_dev_gold_trajs=FLAGS.use_dev_gold_trajs,
      pretrained_ckpt_dir=pretrained_ckpt_dir,
      pretrained_load_data_only=FLAGS.pretrained_load_data_only,
      use_top_k_samples=FLAGS.use_top_k_samples,
      min_replay_weight=FLAGS.min_replay_weight,
      n_replay_samples=FLAGS.n_replay_samples,
      train_use_gpu=FLAGS.train_use_gpu,
      n_train_envs=FLAGS.n_train_envs,
      n_dev_envs=FLAGS.n_dev_envs,
      seed=FLAGS.seed,
      eval_only=FLAGS.eval_only,
      use_gold_trajs=FLAGS.use_gold_trajs,
      n_train_plants=FLAGS.n_train_plants,
      n_test_plants=FLAGS.n_test_plants,
      n_dev_plants=FLAGS.n_dev_plants,
      grid_size=FLAGS.grid_size,
      explore=FLAGS.explore,
      is_debug=FLAGS.is_debug,
      max_grad_norm=FLAGS.max_grad_norm,
      num_steps=FLAGS.num_steps,
      save_every_n=FLAGS.save_every_n,
      units=FLAGS.units,
      eps=FLAGS.eps,
      gamma=FLAGS.gamma,
      entropy_reg_coeff=FLAGS.entropy_reg_coeff,
      dummy=FLAGS.dummy,
      score_fn=FLAGS.score_fn,
      log_summaries=FLAGS.log_summaries,
      learning_rate=FLAGS.learning_rate)

  # Code related to score function initialization
  if FLAGS.use_buffer_scorer:
    exec_args['use_buffer_scorer'] = True
    score_fn_keys = [
        'score_{}'.format(i) for i in common_flags.ALL_FEATURE_KEYS
    ]
    score_fn_dict = {k: getattr(FLAGS, k) for k in score_fn_keys}
    exec_args.update(score_fn_dict)

  return exec_args
