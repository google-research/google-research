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

"""Evaluation utilities for residual agent."""

import os
import tensorflow as tf

from tensorflow.io import gfile


def format_eval_filename(
    task, seed, num_eval_episodes, loaded_ckpt, collapse_in_eval,
    stop_if_stuck, eval_id):
  """Add evaluation settings to output file name."""
  collapse_str = 'c' if collapse_in_eval else ''
  stuck_str = 's' if stop_if_stuck else ''
  if loaded_ckpt and str(loaded_ckpt)[0] != '_':
    loaded_ckpt = '_' + str(loaded_ckpt)
  filename = (
      f'{task}_s{seed}_e{num_eval_episodes}{loaded_ckpt}{collapse_str}'
      f'{stuck_str}')
  if eval_id:
    filename += '_' + eval_id
  return filename


def eval_agent(env_loop,
               task,
               eval_seed,
               increment_eval_seed,
               num_eval_episodes,
               loaded_ckpt,
               collapse_in_eval=True,
               stop_if_stuck=False,
               num_trained_episodes=None,
               total_steps=None,
               logdir=None,
               summary_writer=None,
               eval_id=''):
  """Evaluate success rate of a trained agent in an environment loop."""
  if num_eval_episodes <= 0:
    return
  eval_path = None
  if logdir is not None or summary_writer is not None:
    eval_filename = format_eval_filename(
        task, eval_seed, num_eval_episodes, loaded_ckpt, collapse_in_eval,
        stop_if_stuck, eval_id)
  if logdir is not None:
    eval_path = os.path.join(logdir, 'eval', f'eval{eval_filename}')
    if gfile.exists(eval_path + '_success.txt'):
      print('Evaluation', eval_path, 'already exists; skipping')
      return
    print('Writing evaluation to', eval_path)
  finished_eval = False
  while not finished_eval:
    success_rate, finished_eval = env_loop.eval_policy(
        num_episodes=num_eval_episodes,
        collapse_policy=collapse_in_eval,
        eval_path=eval_path,
        num_videos_to_save=num_eval_episodes,
        seed=eval_seed,
        increment_seed=increment_eval_seed,
        stop_if_stuck=stop_if_stuck)
  if summary_writer is not None:
    with summary_writer.as_default():
      if num_trained_episodes is not None:
        tf.summary.scalar(
            f'{eval_id}_success_rate', success_rate, step=num_trained_episodes)
      if total_steps is not None:
        tf.summary.scalar(
            f'{eval_id}_success_rate_env_steps', success_rate, step=total_steps)
