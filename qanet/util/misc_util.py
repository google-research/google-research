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

"""Miscellaneous util functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re
import sys
import time

import numpy as np
import tensorflow as tf

__all__ = ["init_from_checkpoint", "print_out",
           "get_device_str", "debug_tensor", "get_trainable_vars",
           "print_vars", "log_scope"]


def init_from_checkpoint(init_checkpoint, new_scope, pattern="Net"):
  """Initialize model from a checkpoint.

  Args:
    init_checkpoint: checkpoint to init from.
    new_scope: a value to change the top variable scope.
    pattern: to find the top variable scope and replace with new_scope.

  Raises:
    ValueError: if top variable score doesn't contain the keyword "Net"
  """
  # Get checkpoint
  if tf.gfile.IsDirectory(init_checkpoint):
    checkpoint = tf.train.latest_checkpoint(init_checkpoint)
  else:
    checkpoint = init_checkpoint
  tf.logging.info("# Initializing %s from checkpoint %s" % (
      new_scope, checkpoint))

  # Find the existing top variable scope with pattern
  current_scope = None
  reader = tf.train.NewCheckpointReader(checkpoint)
  variable_map = reader.get_variable_to_shape_map()

  for var in variable_map:
    if pattern in var.split("/")[0]:
      current_scope = var.split("/")[0]
      tf.logging.info("  found current_scope %s" % current_scope)
      break
  if not current_scope:
    raise ValueError("Can\'t find scope with pattern %s in %s" % (
        pattern, checkpoint))

  # Build assignment map
  trainable_vars = {}
  for v in tf.trainable_variables():
    trainable_vars[v.name.split(":")[0]] = v.shape.as_list()
  assignment_map = {}
  scope_skip = {}
  ema_skip = 0
  skip_messages = []
  for var in variable_map:
    if not var.startswith(current_scope):  # Not the same scope
      # count
      other_scope = var.split("/")[0]
      if other_scope not in scope_skip:
        scope_skip[other_scope] = 0
      scope_skip[other_scope] += 1
    elif var.endswith("ExponentialMovingAverage"):  # EMA variables
      ema_skip += 1
    else:
      new_var = new_scope + var[len(current_scope):]
      if new_var not in trainable_vars:  # Not trainable
        skip_messages.append("  not in trainable, skip %s" % new_var)
      elif trainable_vars[new_var] != variable_map[var]:  # Shape mismatch
        skip_messages.append("  shape mismatch %s vs %s, skip %s" % (
            trainable_vars[new_var], variable_map[var], new_var))
      else:  # This is good!
        assignment_map[var] = new_var
        tf.logging.info("  load %s, %s" % (new_var, str(variable_map[var])))

  # Logging of variables skipped
  for msg in skip_messages:
    tf.logging.info("%s" % msg)
  tf.logging.info("# Scopes skipped %s" % str(scope_skip))
  tf.logging.info("# EMA variables skipped %d" % ema_skip)
  tf.logging.info("# Checkpoint has %d entries, map %d entries" % (
      len(variable_map), len(assignment_map)))

  # Init
  tf.train.init_from_checkpoint(checkpoint, assignment_map)


def print_out(s, f=None, new_line=True):
  """Similar to print but with support to flush and output to a file."""
  if isinstance(s, bytes):
    s = s.decode("utf-8")

  if f:
    f.write(s.encode("utf-8"))
    if new_line:
      f.write(b"\n")

  # stdout
  out_s = s.encode("utf-8")
  if not isinstance(out_s, str):
    out_s = out_s.decode("utf-8")
  print(out_s, end="", file=sys.stdout)

  if new_line:
    sys.stdout.write("\n")
  sys.stdout.flush()


def get_device_str(device_id, num_gpus):
  """Return a device string for multi-GPU setup."""
  if num_gpus == 0:
    return "/cpu:0"
  device_str_output = "/gpu:%d" % (device_id % num_gpus)
  return device_str_output


def debug_tensor(s, msg=None, summarize=10, other_tensors=None):
  """Print the shape and value of a tensor at test time. Return a new tensor."""
  if not msg:
    msg = s.name
  outputs = [tf.shape(s), s]

  # print info of other tensors
  if other_tensors:
    for tensor in other_tensors:
      outputs.extend([tf.shape(tensor), tensor])

  return tf.Print(s, outputs, msg + " ", summarize=summarize)


def _count_total_params(all_vars):
  """Count total number of variables."""
  return np.sum([np.prod(v.get_shape().as_list()) for v in all_vars])


def print_vars(all_vars=None, label="Variables"):
  """Print info about a list of variables."""
  if not all_vars:
    all_vars = tf.all_variables()
  num_params = _count_total_params(all_vars)
  tf.logging.info("# %s, num_params=%d" % (label, num_params))
  tf.logging.info("Format: <name>, <shape>, <(soft) device placement>")
  for var in all_vars:
    tf.logging.info("  %s, %s, %s" % (var.name, str(var.get_shape()),
                                      var.op.device))


def _build_regex(pattern):
  """Compile regex pattern turn comma-separated list into | in regex."""
  compiled_pattern = None
  if pattern:
    # Trip ' " at the beginning and end
    pattern = re.sub("(^\"|^\'|\"$|\'$)", "", pattern)

    # Escape
    pattern = re.sub("/", r"\/", pattern)

    # Change "a,b,c" into "(a|b|c)"
    pattern = "(" + re.sub(",", "|", pattern) + ")"

    # Compile
    tf.logging.info("regex pattern %s" % pattern)
    compiled_pattern = re.compile(pattern)
  return compiled_pattern


def get_trainable_vars(all_vars=None, keep_pattern=None, exclude_pattern=None):
  """Get trainable vars, frozen vars, check partition, count total params."""
  if not all_vars:
    all_vars = tf.trainable_variables()

  # Split variables
  trainable_vars = []
  frozen_vars = []
  has_partition = False
  keep_regex = _build_regex(keep_pattern)
  exclude_regex = _build_regex(exclude_pattern)
  for var in all_vars:
    if keep_regex and keep_regex.search(var.name):
      tf.logging.info("  keeping %s, %s, %s" % (
          var.name, str(var.get_shape()), var.op.device))
      trainable_vars.append(var)
    elif exclude_regex and exclude_regex.search(var.name):
      tf.logging.info("  excluding %s, %s, %s" % (
          var.name, str(var.get_shape()), var.op.device))
      frozen_vars.append(var)
    else:
      trainable_vars.append(var)

    # Checked for partition variables
    if "/part_" in var.name:
      has_partition = True

  # Print variables
  print_vars(trainable_vars, label="Trainable variables")
  print_vars(frozen_vars, label="Frozen variables")

  return trainable_vars, frozen_vars, has_partition


def log_scope(msg):
  """Print log messages with current variable scope."""
  current_scope_name = tf.get_variable_scope().name
  print("%s, %s" % (current_scope_name, msg))


class FixedRuntimeHook(tf.train.SessionRunHook):
  """Run model for a fixed time.

  Estimates runtime based on median of most recent step times. Must have at
  least `min_window_size` samples stored before it will stop a job, and it
  stores at most `max_window_size` times.
  """

  def __init__(self, seconds, tuner=None, window_size=50):
    """Set internal state.

    Args:
      seconds: Number of seconds to run for
      tuner: Optional. Vizier tuner used to request that a trial stop
      window_size: Optional. Num samples for model time estimate.
    """
    tf.train.SessionRunHook.__init__(self)
    self._seconds = seconds
    self._window_size = window_size
    self._tuner = tuner

  def begin(self):
    self._global_step_tensor = tf.train.get_global_step()
    self._step_times = collections.deque()

  def before_run(self, run_context):
    self._step_start = time.time()
    return tf.train.SessionRunArgs(self._global_step_tensor)

  def after_run(self, run_context, run_values):
    global_step = run_values.results
    diff = time.time() - self._step_start
    self._step_times.append(diff)
    # NOTE: Could remove this to change to minimum, but that could end up taking
    # a long time on long running models.
    if len(self._step_times) > self._window_size:
      self._step_times.popleft()

    samples = len(self._step_times)
    if samples >= self._window_size:
      median = sorted(self._step_times)[samples / 2]
      if median * global_step > self._seconds:
        tf.logging.info(
            "Model has trained for estimated %s seconds - stopping. "
            "Estimated median step time is %s.", median * global_step, median)
        run_context.request_stop()
        if self._tuner is not None:
          self._tuner._hp_tuner.request_trial_stop()  # pylint: disable=protected-access
