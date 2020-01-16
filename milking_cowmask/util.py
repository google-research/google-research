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

"""Utility functions for logging and recording training metrics."""
import logging
from absl import logging as absl_logging
import jax
from jax import lax
import jax.numpy as jnp
import numpy as onp


def add_log_file(logfile):
  """Replicate logs to an additional logfile.

  The caller is responsible for closing the logfile.
  Args:
    logfile: Open file to write log to
  """
  handler = logging.StreamHandler(logfile)
  handler.setFormatter(absl_logging.PythonFormatter())

  absl_logger = logging.getLogger('absl')
  absl_logger.addHandler(handler)


def to_state_list(obj):
  """Return the state of the model as a flattened list.

  Restore with `load_state_list`.

  Args:
    obj: the object to extract state from

  Returns:
    State as a list of jax.numpy arrays
  """
  return jax.device_get(
      [x[0] for x in jax.tree_leaves(obj)])


def restore_state_list(obj, state_list):
  """Restore model state from a state list.

  Args:
    obj: the object that is to be duplicated with the
      restored state
    state_list: state as a list of jax.numpy arrays

  Returns:
    a copy of `self` with the parameters from state_list loaded

  >>> restored = restore_state_list(model, state_list)
  """
  state_list = replicate(state_list)
  structure = jax.tree_util.tree_structure(obj)
  return jax.tree_unflatten(structure, state_list)


def replicate(xs, n_devices=None):
  if n_devices is None:
    n_devices = jax.local_device_count()
  return jax.pmap(
      lambda _: xs, axis_name='batch')(jnp.arange(n_devices))


def shard(xs):
  local_device_count = jax.local_device_count()
  return jax.tree_map(
      lambda x: x.reshape((local_device_count, -1) + x.shape[1:]), xs)


def onehot(labels, num_classes):
  x = (labels[Ellipsis, None] == jnp.arange(num_classes)[None])
  return x.astype(jnp.float32)


def pmean(tree, axis_name='batch'):
  num_devices = lax.psum(1., axis_name)
  return jax.tree_map(lambda x: lax.psum(x, axis_name) / num_devices, tree)


def psum(tree, axis_name='batch'):
  return jax.tree_map(lambda x: lax.psum(x, axis_name), tree)


def pad_classification_batch(batch, batch_size):
  """Pad a classification batch so that it has `batch_size` samples.

  The batch should be a dictionary of the form:

  {
    'image': <image>,
    'label': <GT label>
  }

  Args:
    batch: the batch to pad
    batch_size: the desired number of elements

  Returns:
    Padded batch as a dictionary
  """
  actual_size = len(batch['image'])
  if actual_size < batch_size:
    padding = batch_size - actual_size
    padded = {
        'label': onp.pad(batch['label'], [[0, padding]],
                         mode='constant', constant_values=-1),
        'image': onp.pad(batch['image'], [[0, padding], [0, 0], [0, 0], [0, 0]],
                         mode='constant', constant_values=0),
    }
    return padded
  else:
    return batch


def stack_forest(forest):
  stack_args = lambda *args: onp.stack(args)
  return jax.tree_multimap(stack_args, *forest)


def get_metrics(device_metrics):
  device_metrics = jax.tree_map(lambda x: x[0], device_metrics)
  metrics_np = jax.device_get(device_metrics)
  return stack_forest(metrics_np)
