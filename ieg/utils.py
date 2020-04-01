# coding=utf-8
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS


def get_latest_checkpoint(path):
  """Gets latest checkpoint.

  Args:
    path: the path to search for checkpoint.
  Returns:
    Path for the latest checkpoint.
  """
  tf.logging.info('Reading {} for getting latest checkpoint'.format(path))
  cands = tf.gfile.Glob('{}/checkpoint*.meta'.format(path))
  tf.logging.info('available checkpoints \n\t {}'.format(cands))

  def _get_iter(file_str):
    """Filter the iteration number of a checkpoint file according to string.

    Args:
      file_str: path of the checkpoint file.

    Returns:
      iteration integer
    """
    basename = os.path.splitext(file_str)[0]
    iter_n = int(basename.split('-')[-1])
    return iter_n

  if len(cands) > 0:  # pylint: disable=g-explicit-length-test
    ckpt = sorted(cands, key=_get_iter)[-1]
    ckpt = os.path.splitext(ckpt)[0]
    return ckpt
  else:
    return None


def get_var(list_of_tensors, prefix_name=None, with_name=None):
  """Gets specific variable.

  Args:
    list_of_tensors: A list of candidate tensors
    prefix_name:  Variable name starts with prefix_name
    with_name: with_name in the variable name

  Returns:
    Obtained tensor list
  """
  if prefix_name is None:
    return list_of_tensors
  else:
    specific_tensor = []
    specific_tensor_name = []
    if prefix_name is not None:
      for var in list_of_tensors:
        if var.name.startswith(prefix_name):
          if with_name is None or with_name in var.name:
            specific_tensor.append(var)
            specific_tensor_name.append(var.name)
    return specific_tensor


def print_flags(flags_v):
  """Verboses flags."""

  print('-' * 20)
  for k, v in sorted(flags_v.flag_values_dict().items(), key=lambda x: x[0]):
    print('{}: {}'.format(k, v))
  print('-' * 20)


def create_session():
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  session = tf.Session(config=config)
  return session


def clear_tensorboard_files(directory, warning):
  """Cleans tensorboard files in a directory."""

  if not tf.gfile.IsDirectory(directory):
    return
  for fil in tf.gfile.Glob(os.path.join(directory, 'events.*')):
    if warning:
      v = input('Are you sure to remove existing event {} (y/n)'.format(fil))
      if v == 'y':
        pass
      else:
        exit()
    tf.gfile.Remove(fil)


def make_dir_if_not_exists(directory):
  """Makes a directory if not exists."""

  if not tf.gfile.Exists(directory):
    tf.gfile.MakeDirs(directory)


def topk_accuracy(logits,
                  labels,
                  topk,
                  ignore_label_above=None,
                  return_counts=False):
  """Top-k accuracy."""
  if ignore_label_above is not None:
    logits = logits[labels < ignore_label_above, :]
    labels = labels[labels < ignore_label_above]

  prds = np.argsort(logits, axis=1)[:, ::-1]
  prds = prds[:, :topk]
  total = np.any(prds == np.tile(labels[:, np.newaxis], [1, topk]), axis=1)
  acc = total.mean()
  if return_counts:
    return acc, labels.shape[0]
  return acc
