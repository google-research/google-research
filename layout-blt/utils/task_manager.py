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

"""Auxiliary class and functions for eval jobs."""

import csv
import os
import time
from typing import Any, Dict, Iterable, Optional, TypeVar

from absl import logging
import flax
from flax.training import checkpoints as flax_checkpoints
import jax
import tensorflow as tf

T = TypeVar("T")


def restore_checkpoint(state, workdir):
  return flax_checkpoints.restore_checkpoint(workdir, state)


def save_checkpoint(state, workdir):
  if jax.process_index() == 0:
    # get train state from the first replica
    state = jax.device_get(jax.tree.map(lambda x: x[0], state))
    step = int(state.step)
    flax_checkpoints.save_checkpoint(workdir, state, step, keep=5,
                                     overwrite=True)


def restore_from_path(state, path):
  with tf.io.gfile.GFile(path, "rb") as f:
    state = flax.serialization.from_bytes(state, f.read())
  return state


def load_state_dict(path):
  with tf.io.gfile.GFile(path, "rb") as f:
    return flax.serialization.msgpack_restore(f.read())


class TaskManager:
  """Class for checking the model folder repeately for evaluation."""

  def __init__(self, model_dir):
    self._model_dir = model_dir

  @property
  def model_dir(self):
    return self._model_dir

  def mark_training_done(self):
    with tf.io.gfile.GFile(
        os.path.join(self.model_dir, "TRAIN_DONE"), "w") as f:
      f.write("")

  def is_training_done(self):
    return tf.io.gfile.exists(os.path.join(self.model_dir, "TRAIN_DONE"))

  def add_eval_result(
      self,
      checkpoint_path,
      result_dict,
      default_value = -1):
    pass

  def _get_checkpoints_with_results(self):
    return set()

  def unevaluated_checkpoints(self,
                              timeout = 3600 * 8,
                              num_batched_steps = 1,
                              eval_every_steps = None,
                              ):
    """Generator for checkpoints without evaluation results.

    Args:
      timeout: Optional timeout for waiting for new checkpoints. Set this to
        do continious evaluation.
      num_batched_steps: Steps that are batched into a single tf.function.
        Required for computing correct evaluation checkpoints.
      eval_every_steps: Only evaluate checkpoints from steps divisible by this
                         integer.

    Yields:
      Path to checkpoints that have not yet been evaluated.
    """
    logging.info("Looking for checkpoints in %s", self._model_dir)
    evaluated_checkpoints = self._get_checkpoints_with_results()
    last_eval = time.time()
    while True:
      # Check if directory exists. The train job may only create the directory
      # some time after the test job starts.
      if not tf.io.gfile.exists(self.model_dir):
        logging.info("Directory %s does not exist!", self.model_dir)
      else:
        logging.info("what is in %s:  are  %s", self.model_dir,
                     tf.io.gfile.listdir(self.model_dir))
        unevaluated_checkpoints = []
        checkpoints = tf.io.gfile.glob(
            os.path.join(self._model_dir, "checkpoint*"))
        checkpoints = set([x for x in checkpoints if "tmp" not in x])
        logging.info("checkpoints: %s", checkpoints)
        unevaluated_checkpoints = checkpoints - evaluated_checkpoints
        step_and_ckpt = sorted(
            (int(x.split("_")[-1]), x) for x in unevaluated_checkpoints)

        unevaluated_checkpoints = []
        for step, ckpt in step_and_ckpt:
          if eval_every_steps:
            if step > num_batched_steps and (
                step % eval_every_steps < num_batched_steps):
              unevaluated_checkpoints.append(ckpt)
          else:
            unevaluated_checkpoints.append(ckpt)

        logging.info(
            "Found checkpoints: %s\nEvaluated checkpoints: %s\n"
            "Unevaluated checkpoints: %s", checkpoints, evaluated_checkpoints,
            unevaluated_checkpoints)
        for checkpoint_path in unevaluated_checkpoints:
          yield checkpoint_path

        if unevaluated_checkpoints:
          evaluated_checkpoints |= set(unevaluated_checkpoints)
          last_eval = time.time()
          continue
      if time.time() - last_eval > timeout or self.is_training_done():
        break
      time.sleep(5)


class TaskManagerWithCsvResults(TaskManager):
  """Task Manager that writes results to a CSV file."""

  def __init__(self,
               model_dir,
               score_file = None):
    super().__init__(model_dir)
    if score_file is None:
      score_file = os.path.join(self._model_dir, "scores.csv")
    else:
      score_file = os.path.join(self._model_dir, score_file)
    self._score_file = score_file

  def _get_checkpoints_with_results(self):
    """Return the checkpoints as set."""
    if not tf.io.gfile.exists(self._score_file):
      return set()
    with tf.io.gfile.GFile(self._score_file) as f:
      reader = csv.DictReader(f)
      return {r["checkpoint_path"] for r in reader}
    return set()

  def add_eval_result(self,
                      checkpoint_path,
                      result_dict,
                      default_value):
    """Add eval result to the CSV file."""
    if jax.process_index() == 0:
      step = int(os.path.basename(checkpoint_path).split("_")[-1])
      csv_header = (
          ["checkpoint_path", "step"] + sorted(result_dict))
      write_header = not tf.io.gfile.exists(self._score_file)
      if write_header:
        with tf.io.gfile.GFile(self._score_file, "w") as f:
          writer = csv.DictWriter(
              f, fieldnames=csv_header, extrasaction="ignore")
          writer.writeheader()
      row = dict(checkpoint_path=checkpoint_path, step=str(step))
      for k, v in result_dict.items():
        if isinstance(v, float):
          v = "{:.3f}".format(v)
        row[k] = v
      with tf.io.gfile.GFile(self._score_file, "a") as f:
        writer = csv.DictWriter(f, fieldnames=csv_header, extrasaction="ignore")
        writer.writerow(row)
