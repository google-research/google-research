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

# Copyright 2020 The Google Research Authors.
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
"""Implementation of utilities used in the training scripts."""

import os
import jax
import time
import numpy as onp
from jax import numpy as jnp
import tensorflow.compat.v2 as tf

from collections import OrderedDict

from bnn_hmc.utils import models
from bnn_hmc.utils import data_utils
from bnn_hmc.utils import train_utils
from bnn_hmc.utils import precision_utils
from bnn_hmc.utils import losses
from bnn_hmc.utils import checkpoint_utils
from bnn_hmc.utils import cmd_args_utils
from bnn_hmc.utils import tree_utils


def print_visible_devices():
  print("JAX sees the following devices:", jax.devices())
  print("TF sees the following devices:", tf.config.get_visible_devices())


def prepare_logging(subdirname, args):
  dirname = os.path.join(args.dir, subdirname)
  os.makedirs(dirname, exist_ok=True)
  tf_writer = tf.summary.create_file_writer(dirname)
  cmd_args_utils.save_cmd(dirname, tf_writer)
  return dirname, tf_writer


def get_data_model_fns(args):
  dtype = jnp.float64 if args.use_float64 else jnp.float32
  train_set, test_set, task, data_info = data_utils.make_ds_pmap_fullbatch(
      args.dataset_name, dtype, truncate_to=args.subset_train_to)

  net_apply, net_init = models.get_model(args.model_name, data_info)
  net_apply = precision_utils.rewrite_high_precision(net_apply)

  (likelihood_factory, predict_fn, ensemble_upd_fn, metrics_fns,
   tabulate_metrics) = train_utils.get_task_specific_fns(task, data_info)
  log_likelihood_fn = likelihood_factory(args.temperature)
  log_prior_fn, log_prior_diff_fn = losses.make_gaussian_log_prior(
      args.weight_decay, args.temperature)

  key, net_init_key = jax.random.split(jax.random.PRNGKey(args.seed), 2)
  init_data = jax.tree_map(lambda elem: elem[0][:1], train_set)
  params, net_state = net_init(net_init_key, init_data, True)

  param_types = tree_utils.tree_get_types(params)
  assert all([
      p_type == dtype for p_type in param_types
  ]), ("Params data types {} do not match specified data type {}".format(
      param_types, dtype))

  return (train_set, test_set, net_apply, params, net_state, key,
          log_likelihood_fn, log_prior_fn, log_prior_diff_fn, predict_fn,
          ensemble_upd_fn, metrics_fns, tabulate_metrics)


def get_num_batches_total_steps(args, train_set):
  num_data = jnp.size(train_set[1])
  num_batches = num_data // args.batch_size
  total_steps = num_batches * args.num_epochs
  return num_batches, total_steps


def get_initialization_dict(dirname, args, init_dict):
  """Loads checkpoint if available.

  This function is used in training scripts to initialize variables, it handles
  resuming training from checkpoints and starting from provided initialization.

  If `args.init_checkpoint` is provided, it is used to load the initial value
  of the `params` and `net_state`; other variables are loaded from `init_dict`.

  If `args.init_checkpoint` is None and the directory `dirname` has checkpoints,
  all the variables are loaded from the last checkpoint.

  If `args.init_checkpoint` is None and the directory `dirname` has no
  checkpoints, all variables are loaded from `init_dict`.

  """
  checkpoint_dict, status = checkpoint_utils.initialize(dirname,
                                                        args.init_checkpoint)
  if status == checkpoint_utils.InitStatus.LOADED_PREEMPTED:
    print("Continuing the run from the last saved checkpoint")
    return checkpoint_dict
  if status == checkpoint_utils.InitStatus.INIT_RANDOM:
    print("Starting from random initialization with provided seed")
    return init_dict
  if status == checkpoint_utils.InitStatus.INIT_CKPT:
    print("Starting the run from the provided init_checkpoint")
    init_dict.update({"params": checkpoint_dict["params"]})
    init_dict.update({"net_state": checkpoint_dict["net_state"]})
    return init_dict
  raise ValueError("Unknown initialization status: {}".format(status))


def evaluate(net_apply, params, net_state, train_set, test_set, predict_fn,
             metrics_fns, log_prior_fn):
  net_state, test_predictions = onp.asarray(
      predict_fn(net_apply, params, net_state, test_set))
  net_state, train_predictions = onp.asarray(
      predict_fn(net_apply, params, net_state, train_set))
  test_stats = train_utils.evaluate_metrics(test_predictions, test_set[1],
                                            metrics_fns)
  train_stats = train_utils.evaluate_metrics(train_predictions, train_set[1],
                                             metrics_fns)
  train_stats["prior"] = log_prior_fn(params)
  return (net_state, test_predictions, train_predictions, test_stats,
          train_stats)


def time_fn(fn):

  def timed_fn(*args, **kwargs):
    start_time = time.time()
    output = fn(*args, **kwargs)
    iteration_time = time.time() - start_time
    return output, iteration_time

  return timed_fn


def get_common_logs(iteration, iteration_time, args):
  logs = {
      "telemetry/iteration": iteration,
      "telemetry/iteration_time": iteration_time,
      "hypers/weight_decay": args.weight_decay,
      "hypers/temperature": args.temperature,
  }
  return logs


def write_to_tensorboard(tf_writer, logging_dict, iteration):
  with tf_writer.as_default():
    for stat_name, stat_val in logging_dict.items():
      tf.summary.scalar(stat_name, stat_val, step=iteration)


def get_tabulate_dict(tabulate_metrics, logging_dict):
  tabulate_dict = OrderedDict()
  tabulate_dict["i"] = logging_dict["telemetry/iteration"]
  tabulate_dict["t"] = logging_dict["telemetry/iteration_time"]
  for metric_name in tabulate_metrics:
    if metric_name in logging_dict:
      tabulate_dict[metric_name] = logging_dict[metric_name]
    else:
      tabulate_dict[metric_name] = None
  return tabulate_dict
