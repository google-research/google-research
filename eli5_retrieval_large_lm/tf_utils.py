# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Various utilities that involve Tensorflow and accelerator devices.
"""
import logging
from typing import Dict, List, Optional, Tuple, Union  # pylint: disable=g-import-not-at-top

import colorama
import dataclasses
import numpy as np
import tensorflow as tf
import tensorflow.python.distribute.values as values
import tensorflow.python.eager.context as context
import tensorflow.python.framework.ops as ops
import tensorflow.python.tpu.topology as topology
import utils

LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class TpuConfigType:
  resolver: tf.distribute.cluster_resolver.TPUClusterResolver
  topology: topology.Topology


@dataclasses.dataclass
class DevicesMapType:
  # pylint: disable=invalid-name
  TPUs: List[context.LogicalDevice]
  GPUs: List[context.LogicalDevice]
  CPUs: List[context.LogicalDevice]
  # pylint: enable=invalid-name


def init_tpus(tpu_name = None):
  """Initializes the connection with the TPUs."""
  try:
    if tpu_name:
      resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu_name)
    else:
      resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    if resolver:
      tf.config.experimental_connect_to_cluster(resolver)
      topology_ = tf.tpu.experimental.initialize_tpu_system(resolver)
      return TpuConfigType(resolver=resolver, topology=topology_)
  except ValueError:
    LOGGER.warning(
        "%(red_bg)s%(white)s%(bold)s WORKING WITH CPUS %(reset)s",
        dict(
            red_bg=colorama.Back.RED,
            white=colorama.Fore.WHITE,
            bold=colorama.Style.BRIGHT,
            reset=colorama.Style.RESET_ALL
            )
    )
    return None


def devices_to_use():
  """Returns the device objects for the accel. we are the most likely to use.

  Returns:
    List of logical devices of the accelerators we will use.
  """
  if tf.config.list_logical_devices("TPU"):
    devices = tf.config.list_logical_devices("TPU")
  elif tf.config.list_logical_devices("GPU"):
    devices = tf.config.list_logical_devices("GPU")
  else:
    devices = tf.config.list_logical_devices("CPU")
  devices.sort()
  return devices


def current_accelerator_type():
  """Returns the type of accelerator we are using.

  devices_to_use guaranties that all the accelerators are of the same type.
  """
  return devices_to_use()[0].device_type


def device_mapping():
  """Gives a dict with the different types of logical devices."""
  return DevicesMapType(
      TPUs=sorted(tf.config.list_logical_devices("TPU")),
      GPUs=sorted(tf.config.list_logical_devices("GPU")),
      CPUs=sorted(tf.config.list_logical_devices("CPU"))
  )


def make_dict_distribute_fn(batch):
  """Builds the dict distribution function."""
  def dict_distribute_fn(ctx):
    """Assumes all the tensors in the dict are of the same batch size."""
    quanta = len(next(iter(batch.values()))) // ctx.num_replicas_in_sync
    start = quanta * ctx.replica_id_in_sync_group
    end = start + quanta
    new = {}
    for k, v in batch.items():
      new[k] = v[start:end]
    return new
  return dict_distribute_fn


def deal_w_entry(strategy_outputs):
  output = strategy_outputs.values  # pytype: disable=attribute-error
  if isinstance(strategy_outputs, tuple):
    output = tf.concat(output, axis=0)
  return output


def process_strat_output(
    strategy_outputs,
    name,
    strategy,
    current_batch_size,
):
  """Uniformizes the different outputs of strategy.run calls."""
  if isinstance(strategy_outputs, values.PerReplica):
    strategy_outputs: values.PerReplica
    # LOGGER.debug("process_strat_output: %s: %s", name, str(strategy_outputs))
    output = deal_w_entry(strategy_outputs)
    utils.check_equal(output.shape, current_batch_size)
  elif (isinstance(strategy_outputs, tuple) and
        isinstance(strategy_outputs[0], values.PerReplica)):
    strategy_outputs: Tuple[values.PerReplica, Ellipsis]
    output = []
    for indiv_val in strategy_outputs:
      output.append(deal_w_entry(indiv_val))
    output = tuple(output)
  elif (isinstance(strategy_outputs, dict) and
        isinstance(next(iter(strategy_outputs.values())), values.PerReplica)):
    strategy_outputs: Dict[str, values.PerReplica]
    output = {}
    for k, indiv_val in strategy_outputs.items():
      output[k] = deal_w_entry(indiv_val)
  elif isinstance(
      strategy_outputs,
      ops.EagerTensor) or (isinstance(strategy_outputs, tuple) and
                           isinstance(strategy_outputs[0], ops.EagerTensor)):
    output = strategy_outputs
  else:
    raise RuntimeError(
        f"{name}: {type(strategy_outputs)}, {type(strategy)}"
    )

  return output


def load_reference_db(
    checkpoint_path, variable_name
):
  """Load the reference database for retrieval.

  This is mostly for compatibility with the REALM code.

  Args:
    checkpoint_path: The path of the checkpoint to use.
    variable_name: The variable name of the database inside of the checkpoint.

  Returns:
    A numpy array with the reference database.

  """
  ckpt = tf.train.load_checkpoint(str(checkpoint_path))
  try:
    reference_db = ckpt.get_tensor(variable_name)
  except tf.errors.NotFoundError:
    reference_db = ckpt.get_tensor(
        variable_name + "/.ATTRIBUTES/VARIABLE_VALUE")

  return reference_db


def mips_exact_search(
    vectors, num_neighbors, db
):
  """Does exact retrieval over a database.

  Args:
    vectors: The key vectors to retrieve with.
    num_neighbors: The number of neighbors to extract.
    db: The vector datase to retrieve from.

  Returns:
    top_k: The top_k indices, the retrieved neighbors.
    inners: The inner products of each neighbor.
  """
  product = tf.linalg.matmul(vectors, db, transpose_b=True)
  inners, top_k = tf.math.top_k(product, k=num_neighbors, sorted=sorted)
  return top_k, inners


def sample_without_replacement(logits, k):
  """Samples k values without replacement, from a set of logits.

  Courtesy of https://github.com/tensorflow/tensorflow/issues/9260#issuecomment-437875125  # pylint: disable=line-too-long
  and https://timvieira.github.io/blog/post/2014/08/01/gumbel-max-trick-and-weighted-reservoir-sampling/  # pylint: disable=line-too-long

  Arguments:
    logits: The logits for the probabilities of the distribution.
    k: The number of samples to take.

  Returns:
    The indices of the values that were chose.
  """
  z = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(logits), 0, 1)))
  _, indices = tf.nn.top_k(logits + z, k)
  return indices


@dataclasses.dataclass
class REALMSave:
  query_embedder_path: utils.PathType
  text_records: utils.PathType
  num_block_records: int
  description: str


# TODO(julesgm): This part needs work
# class InformationOnDevices:
#   """Information about the task to device distribution of the devices.
#
#   This doesn't make the assumption that each device has the same quantity of
#   tasks. Always true for TPUs. Maybe more resistant to weird configurations.
#   """
#
#   name_parse_pat = re.compile(
#       r"/job:worker/replica:0/task:([0-9]+)/device:(\w+):([0-9]+)"
#   )
#
#   def __init__(self):
#     self.devices_by_device_id = None
#     self.devices_by_task_id = None
#     self.num_tasks: int = 0
#     self.num_devices: int = 0
#     self.refresh()
#
#   def refresh(self) -> None:
#     """Refreshes the information.
#
#     Raises:
#       RuntimeError:
#         If one of the device names is in a format we can't parse.
#     """
#     devices_by_device_id = collections.defaultdict(list)
#     devices_by_task_id = collections.defaultdict(list)
#     for device in devices_to_use():
#       matches = self.name_parse_pat.match(device.name)
#       if matches is None:
#         raise RuntimeError(device.name)
#       task_no = int(matches.group(1))
#       device_no = int(matches.group(3))
#       devices_by_device_id[device_no].append((task_no, device))
#       devices_by_task_id[task_no].append((device_no, device))
#
#     LOGGER.debug("first devices_by_task_id:   %s", devices_by_task_id)
#     LOGGER.debug("first devices_by_device_id: %s", devices_by_device_id)
#
#     num_devices = len(devices_by_device_id)
#     num_tasks = len(devices_by_task_id)
#     LOGGER.debug("num_devices: %s", num_devices)
#     LOGGER.debug("num_tasks:   %s", num_tasks)
#
#     for k in devices_by_device_id:
#       devices_by_device_id[k].sort(key=lambda pair: pair[0])
#       # Remove the task no from the pair, as it is now equivalent to
#       # the position in the list.
#       devices_by_device_id[k] = [pair[1] for pair in devices_by_device_id[k]]
#
#     for k in devices_by_task_id:
#       devices_by_task_id[k].sort(key=lambda pair: pair[0])
#       # Remove the device no from the pair, as it is now equivalent to
#       # the position in the list.
#       devices_by_task_id[k] = [pair[1] for pair in devices_by_task_id[k]]
#
#     LOGGER.debug("second devices_by_task_id:   %s", devices_by_task_id)
#     LOGGER.debug("second devices_by_device_id: %s", devices_by_device_id)
#
#     self.devices_by_task_id = devices_by_task_id
#     self.devices_by_device_id = devices_by_device_id
#     self.num_devices = num_devices
#     self.num_tasks = num_tasks
