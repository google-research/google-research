# coding=utf-8
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

import json
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

  cands = tf.gfile.Glob("{}/checkpoint*.meta".format(path))
  tf.logging.info("Reading {} for getting latest checkpoint: {}".format(
      path, cands))

  def _get_iter(file_str):
    """Filter the iteration number of a checkpoint file according to string.

    Args:
      file_str: path of the checkpoint file.

    Returns:
      iteration integer
    """
    basename = os.path.splitext(file_str)[0]
    iter_n = int(basename.split("-")[-1])
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
  tf.logging.info("All flags values")
  for k, v in sorted(flags_v.flag_values_dict().items(), key=lambda x: x[0]):
    tf.logging.info("FLAGS.{}: {}".format(k, v))


def create_session():
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  session = tf.Session(config=config)
  return session


def clear_tensorboard_files(directory, warning):
  """Cleans tensorboard files in a directory."""

  if not tf.gfile.IsDirectory(directory):
    return
  for fil in tf.gfile.Glob(os.path.join(directory, "events.*")):
    if warning:
      v = input("Are you sure to remove existing event {} (y/n)".format(fil))
      if v == "y":
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


def _collective_communication(all_reduce_alg):
  """Return a CollectiveCommunication based on all_reduce_alg.

  Args:
    all_reduce_alg: a string specifying which collective communication to pick,
      or None.

  Returns:
    tf.distribute.experimental.CollectiveCommunication object
  Raises:
    ValueError: if `all_reduce_alg` not in [None, "ring", "nccl"]
  """
  collective_communication_options = {
      None: tf.distribute.experimental.CollectiveCommunication.AUTO,
      "ring": tf.distribute.experimental.CollectiveCommunication.RING,
      "nccl": tf.distribute.experimental.CollectiveCommunication.NCCL
  }
  if all_reduce_alg not in collective_communication_options:
    raise ValueError(
        "When used with `multi_worker_mirrored`, valid values for "
        "all_reduce_alg are [`ring`, `nccl`].  Supplied value: {}".format(
            all_reduce_alg))
  return collective_communication_options[all_reduce_alg]


def _mirrored_cross_device_ops(all_reduce_alg, num_packs):
  """Return a CrossDeviceOps based on all_reduce_alg and num_packs.

  Args:
    all_reduce_alg: a string specifying which cross device op to pick, or None.
    num_packs: an integer specifying number of packs for the cross device op.

  Returns:
    tf.distribute.CrossDeviceOps object or None.
  Raises:
    ValueError: if `all_reduce_alg` not in [None, "nccl", "hierarchical_copy"].
  """
  if all_reduce_alg is None:
    return None
  mirrored_all_reduce_options = {
      "nccl": tf.distribute.NcclAllReduce,
      "hierarchical_copy": tf.distribute.HierarchicalCopyAllReduce
  }
  if all_reduce_alg not in mirrored_all_reduce_options:
    raise ValueError(
        "When used with `mirrored`, valid values for all_reduce_alg are "
        "[`nccl`, `hierarchical_copy`].  Supplied value: {}".format(
            all_reduce_alg))
  cross_device_ops_class = mirrored_all_reduce_options[all_reduce_alg]
  return cross_device_ops_class(num_packs=num_packs)


def tpu_initialize(tpu_address):
  """Initializes TPU for TF 2.x training.

  Args:
    tpu_address: string, bns address of master TPU worker.

  Returns:
    A TPUClusterResolver.
  """
  cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
      tpu=tpu_address)
  # if tpu_address not in ("", "local"):
  #   tf.config.experimental_connect_to_cluster(cluster_resolver)
  tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
  return cluster_resolver


def get_distribution_strategy(distribution_strategy="mirrored",
                              tpu_address=None,
                              **kwargs):
  """Returns a DistributionStrategy for running the model.

  Args:
    distribution_strategy: a string specifying which distribution strategy to
      use. Accepted values are "off", "one_device", "mirrored",
      "parameter_server", "multi_worker_mirrored", and "tpu" -- case
      insensitive. "off" means not to use Distribution Strategy; "tpu" means to
      use TPUStrategy using `tpu_address`.
    tpu_address: Optional. String that represents TPU to connect to. Must not be
      None if `distribution_strategy` is set to `tpu`.
    **kwargs: Additional kwargs for internal usages.

  Returns:
    tf.distribute.DistibutionStrategy object.
  Raises:
    ValueError: if `distribution_strategy` is "off" or "one_device" and
      `num_gpus` is larger than 1; or `num_gpus` is negative or if
      `distribution_strategy` is `tpu` but `tpu_address` is not specified.
  """
  del kwargs

  if distribution_strategy == "tpu":
    # When tpu_address is an empty string, we communicate with local TPUs.
    cluster_resolver = tpu_initialize(tpu_address)
    tf_config_env = {
        "session_master": cluster_resolver.get_master(),
        "eval_session_master": cluster_resolver.get_master()
    }
    os.environ["TF_CONFIG"] = json.dumps(tf_config_env)
    return tf.distribute.experimental.TPUStrategy(cluster_resolver)

  if distribution_strategy == "mirrored":
    return tf.distribute.MirroredStrategy()

  raise ValueError("Unrecognized Distribution Strategy: %r" %
                   distribution_strategy)
