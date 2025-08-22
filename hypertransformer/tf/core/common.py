# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Datatypes, classes and functions used across `ca_supp`."""

import dataclasses
import os
import time

from typing import Any, Callable, Dict, List, Optional, Text, Union

from absl import flags
from absl import logging

import tensorflow.compat.v1 as tf

from hypertransformer.tf.core import util

FLAGS = flags.FLAGS

# Padding in image summaries.
IMAGE_PADDING = 4
BENCHMARK_STEPS = 100
# Number of images to save into the summary.
NUM_OUTPUT_IMAGES = 4
KEEP_CHECKPOINT_EVERY_N_HOURS = 3

field = dataclasses.field


@dataclasses.dataclass
class TrainConfig(object):
  """Training configuration."""
  train_steps: int = 10000
  steps_between_saves: int = 5000
  small_summaries_every: int = 100
  large_summaries_every: int = 500


def get_default_summary_writer(dump_graph=True, suffix='train'):
  """Creates summary writers."""
  train_writer = util.MultiFileWriter(
      os.path.join(FLAGS.train_log_dir, suffix),
      graph=tf.get_default_graph() if dump_graph else None)
  return train_writer


def get_default_saver():
  return tf.train.Saver(
      keep_checkpoint_every_n_hours=KEEP_CHECKPOINT_EVERY_N_HOURS,
      max_to_keep=3)


@dataclasses.dataclass
class TrainState(object):
  """Model state."""
  train_op: tf.Tensor
  init_op: Optional[tf.Operation] = None
  step_initializer: Optional[tf.Operation] = None
  update_op: tf.Operation = field(default_factory=tf.no_op)
  small_summaries: Union[List[Any], Dict[Text, List[Any]]] = field(
      default_factory=list)
  large_summaries: Union[List[Any], Dict[Text, List[Any]]] = field(
      default_factory=list)

  global_step: tf.Tensor = dataclasses.field(
      default_factory=tf.train.get_or_create_global_step)

  summary_writer: tf.summary.FileWriter = None
  saver: tf.train.Saver = field(default_factory=get_default_saver)

  record_graph_in_summary: dataclasses.InitVar[bool] = True

  tensors_to_eval: Optional[Any] = None
  per_step_fn: Optional[Callable[[tf.Tensor, Any, Any], None]] = None
  checkpoint_suffix: str = 'model'

  def __post_init__(self, record_graph_in_summary):
    if self.summary_writer is None:
      self.summary_writer = get_default_summary_writer(
          dump_graph=record_graph_in_summary)


@dataclasses.dataclass
class OptimizerConfig(object):
  """Optimizer configuration."""
  learning_rate: float = 1e-3
  lr_decay_steps: int = 10000
  lr_decay_rate: float = 1.0


@dataclasses.dataclass
class ModelOutputs:
  """Model outputs."""
  predictions: tf.Tensor
  accuracy: tf.Tensor
  labels: Optional[tf.Tensor] = None
  test_accuracy: Optional[tf.Tensor] = None


def merge_summaries(summaries):
  if isinstance(summaries, dict):
    return {k: tf.summary.merge(v) for k, v in summaries.items()}
  elif summaries:
    return tf.summary.merge(summaries)
  else:
    return None


def _is_not_empty(tensor):
  if isinstance(tensor, tf.Tensor):
    return True
  return tensor is not None and len(tensor) > 0  # pylint: disable=g-explicit-length-test


def train(train_config, state,
          run_options = None):
  """Train loop."""
  sess = tf.get_default_session()

  step = sess.run(state.global_step)
  first_step = step
  starting_time = time.time()
  def save():
    state.saver.save(
        sess, os.path.join(FLAGS.train_log_dir, state.checkpoint_suffix),
        write_meta_graph=False, global_step=step)

  while step <= train_config.train_steps:
    if state.step_initializer is not None:
      sess.run(state.step_initializer, options=run_options)

    to_run = {'step': state.global_step,
              'train': state.train_op,
              'update': state.update_op}
    if state.tensors_to_eval is not None:
      to_run['tensors'] = state.tensors_to_eval
    if (step - first_step) % 100 == 1:
      logging.info('Step: %d, Time per step: %f',
                   step, (time.time() - starting_time) / (step - first_step))

    if (train_config.steps_between_saves > 0 and
        step % train_config.steps_between_saves == 0):
      save()

    if (step % train_config.small_summaries_every == 0 and
        _is_not_empty(state.small_summaries)):
      to_run['small'] = state.small_summaries

    if (step % train_config.large_summaries_every == 0 and
        _is_not_empty(state.large_summaries)):
      to_run['large'] = state.large_summaries

    results = sess.run(to_run, options=run_options)
    if state.per_step_fn:
      state.per_step_fn(step, sess, results)

    step = results['step']

    if 'small' in results:
      state.summary_writer.add_summary(
          util.normalize_tags(results['small']), global_step=step)
    if 'large' in results:
      state.summary_writer.add_summary(
          util.normalize_tags(results['large']), global_step=step)
  # Final save.
  save()


def init_training(state):
  """Initializes the training loop.

  Args:
    state: Training state.

  Returns:
    True if checkpoint was found and false otherwise.
  """
  state.large_summaries = merge_summaries(state.large_summaries)  # pytype: disable=annotation-type-mismatch
  state.small_summaries = merge_summaries(state.small_summaries)  # pytype: disable=annotation-type-mismatch
  sess = tf.get_default_session()
  init_op = state.init_op
  if state.init_op is None:
    init_op = [tf.initializers.global_variables(),
               tf.initializers.local_variables()]

  sess.run(init_op)
  restore_path = os.path.dirname(os.path.join(FLAGS.train_log_dir,
                                              state.checkpoint_suffix))
  checkpoint = util.latest_checkpoint(restore_path)
  if checkpoint is None:
    logging.warning('No checkpoint found.')
    return False
  else:
    logging.info('Restoring from "%s".', checkpoint)
    state.saver.restore(sess, checkpoint)
    return True
