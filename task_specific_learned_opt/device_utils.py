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

"""Device utils for working with distributed training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
from absl import flags
from absl import logging
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS


def get_local_remote_device_fn(ps_tasks):
  """Get the device setter for local vars, remote vars and resources.

  Args:
    ps_tasks: int number of ps tasks.

  Returns:
    local_device: string / fn
      Use with tf.device. Device setter sets all variables and compute to the
      local device.
    remote_device: string / fn
      Use with tf.device. Device setter sets all variables to a remote device
      and all non variables (compute) to the local device.
    index_remote_device: string / fn
      Use with tf.device. Device setter for the index data store op.
      This usually is assigned to its own PS to give it more compute and
      network bandwidth.
  """
  ps_ops = ["Variable", "VariableV2", "VarHandleOp", "Mutex", "IndexDataStore"]
  if ps_tasks != 0:
    # Force everything to be on this device -- explicitly.
    def local_device(op):
      d1 = tf.DeviceSpec.from_string(op.device if op.device else "")
      d2 = tf.DeviceSpec.from_string("/job:%s" % FLAGS.brain_job_name)
      d2.merge_from(d1)
      return d2.to_string()

    n_variable_tasks = ps_tasks - 1 if ps_tasks > 1 else 1

    ps_strategy = tf.contrib.training.RandomStrategy(n_variable_tasks, seed=42)

    remote_device = tf.train.replica_device_setter(
        n_variable_tasks,
        ps_ops=ps_ops,
        worker_device="/job:%s" % FLAGS.brain_job_name,
        ps_strategy=ps_strategy)

    index_remote_device = "/job:ps/task:%d/cpu" % (ps_tasks - 1)

  else:
    # these are functions and not strings
    # to prevent auto merging. These strings
    # are used to sanity check device placement.
    def local_device(op):
      if op.device:
        return op.device
      else:
        # do not change this.
        # it is used in debug_utils to ensure local device placement.
        return "/job:*/replica:0"

    def remote_device(op):
      d1 = tf.DeviceSpec.from_string(op.device if op.device else "")
      d2 = tf.DeviceSpec.from_string("/job:*")
      d2.merge_from(d1)
      return d2.to_string()

    index_remote_device = remote_device
  return local_device, remote_device, index_remote_device


def check_variables_accounting(local_vars, remote_vars):
  """Ensure that all variables are accounted for.

  Check the global varibles and compare to the sum of local and remote vars.
  Args:
    local_vars: list
    remote_vars: list
  """
  shared = set(local_vars).intersection(remote_vars)
  if shared:
    logging.error("Variables in both remote and local -- %d", len(shared))
    for v in shared:
      logging.error("    %s --> %s", str(v), v.op.device)

    raise ValueError(
        "Variables are shared in remote vs local. Scroll up for errors!")

  global_vars = tf.global_variables()
  not_local = set(global_vars) - set(local_vars)
  if set(remote_vars) != not_local:
    logging.error(
        "There is a mismatch in variable device placement and accounting.")
    vs = set(remote_vars) - not_local
    logging.error("Variables in remote_vars but not in local: %d", len(vs))
    for v in vs:
      logging.error("    %s", str(v))
    vs = not_local - set(remote_vars)
    logging.info("Variables in not local but not remote_vars: %d", len(vs))
    for v in vs:
      logging.error("    %s", str(v))
    raise ValueError("Unaccounted for variable. Scroll up for errors!")


def is_on_local(v):
  return v.op.device == "/job:*/replica:0" or "job:%s" % FLAGS.brain_job_name in v.op.device


def check_variables_are_local(local_vars):
  correct = [is_on_local(v) for v in local_vars]
  if not all(correct):
    logging.error("All local variables MUST be on Worker. Errors: %d",
                  len([x for x in correct if not x]))
    for v in local_vars:
      if not is_on_local(v):
        logging.error("    %s --> %s", str(v), v.op.device)

    raise ValueError(
        "All local variables MUST be on worker. Scroll up for errors!")


def is_on_remote(v):
  if FLAGS.ps_tasks == 0:
    return "/job:*" in v.op.device
  else:
    return "job:ps" in v.op.device


def check_variables_are_remote(remote_vars):
  correct = [is_on_remote(v) for v in remote_vars]
  if not all(correct):
    logging.error("All remote variables MUST be on ps. Errors: %d",
                  len([x for x in correct if not x]))
    for v in remote_vars:
      if not is_on_remote(v):
        logging.error("    %s --> %s", str(v), v.op.device)

    raise ValueError(
        "All remote variables MUST be on ps. Scroll up for errors!")


def tf_device_wrap(f):
  """Wrapper for class methods to place the variables in a specific device."""

  @functools.wraps(f)
  def wrapper(self, *args, **kwargs):
    if not hasattr(self, "device"):
      raise ValueError("To use tf_device_wrap one must have a device"
                       "member variable")
    with tf.device(self.device):
      return f(self, *args, **kwargs)

  return wrapper
