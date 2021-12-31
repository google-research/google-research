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
"""Run SGD training on a cloud TPU. We are not using data augmentation."""

import os
from jax import numpy as jnp
import jax
import tensorflow.compat.v2 as tf
import argparse

from bnn_hmc.utils import checkpoint_utils
from bnn_hmc.utils import cmd_args_utils
from bnn_hmc.utils import logging_utils
from bnn_hmc.utils import train_utils
from bnn_hmc.utils import optim_utils
from bnn_hmc.utils import script_utils

parser = argparse.ArgumentParser(description="Run SGD on a cloud TPU")
cmd_args_utils.add_common_flags(parser)
cmd_args_utils.add_sgd_flags(parser)

args = parser.parse_args()
train_utils.set_up_jax(args.tpu_ip, args.use_float64)


def get_dirname_tfwriter(args):
  method_name = "sgd_mom_{}".format(args.momentum_decay)
  lr_schedule_name = "lr_sch_i_{}".format(args.init_step_size)
  hypers_name = "_epochs_{}_wd_{}_batchsize_{}_temp_{}".format(
      args.num_epochs, args.weight_decay, args.batch_size, args.temperature)
  subdirname = "{}__{}__{}__seed_{}".format(method_name, lr_schedule_name,
                                            hypers_name, args.seed)
  dirname, tf_writer = script_utils.prepare_logging(subdirname, args)
  return dirname, tf_writer


def train_model():
  # Initialize training directory
  dirname, tf_writer = get_dirname_tfwriter(args)

  # Initialize data, model, losses and metrics
  (train_set, test_set, net_apply, params, net_state, key, log_likelihood_fn,
   log_prior_fn, _, predict_fn, ensemble_upd_fn, metrics_fns,
   tabulate_metrics) = script_utils.get_data_model_fns(args)

  # Initialize step-size schedule and optimizer
  num_batches, total_steps = script_utils.get_num_batches_total_steps(
      args, train_set)
  num_devices = len(jax.devices())
  lr_schedule = optim_utils.make_cosine_lr_schedule(args.init_step_size,
                                                    total_steps)
  optimizer = optim_utils.make_sgd_optimizer(
      lr_schedule, momentum_decay=args.momentum_decay)

  # Initialize variables
  opt_state = optimizer.init(params)
  net_state = jax.pmap(lambda _: net_state)(jnp.arange(num_devices))
  key = jax.random.split(key, num_devices)
  init_dict = checkpoint_utils.make_sgd_checkpoint_dict(-1, params, net_state,
                                                        opt_state, key)
  init_dict = script_utils.get_initialization_dict(dirname, args, init_dict)
  start_iteration, params, net_state, opt_state, key = (
      checkpoint_utils.parse_sgd_checkpoint_dict(init_dict))
  start_iteration += 1

  # Define train epoch
  sgd_train_epoch = script_utils.time_fn(
      train_utils.make_sgd_train_epoch(net_apply, log_likelihood_fn,
                                       log_prior_fn, optimizer, num_batches))

  # Train
  for iteration in range(start_iteration, args.num_epochs):

    (params, net_state, opt_state, logprob_avg, key), iteration_time = (
        sgd_train_epoch(params, net_state, opt_state, train_set, key))

    # Evaluate the model
    train_stats, test_stats = {"log_prob": logprob_avg}, {}
    if (iteration % args.eval_freq == 0) or (iteration == args.num_epochs - 1):
      _, test_predictions, train_predictions, test_stats, train_stats_ = (
          script_utils.evaluate(net_apply, params, net_state, train_set,
                                test_set, predict_fn, metrics_fns,
                                log_prior_fn))
      train_stats.update(train_stats_)

    # Save checkpoint
    if iteration % args.save_freq == 0 or iteration == args.num_epochs - 1:
      checkpoint_name = checkpoint_utils.make_checkpoint_name(iteration)
      checkpoint_path = os.path.join(dirname, checkpoint_name)
      checkpoint_dict = checkpoint_utils.make_sgd_checkpoint_dict(
          iteration, params, net_state, opt_state, key)
      checkpoint_utils.save_checkpoint(checkpoint_path, checkpoint_dict)

    # Log results
    other_logs = script_utils.get_common_logs(iteration, iteration_time, args)
    other_logs["hypers/step_size"] = lr_schedule(opt_state[-1].count)
    other_logs["hypers/momentum"] = args.momentum_decay
    logging_dict = logging_utils.make_logging_dict(train_stats, test_stats, {})
    logging_dict.update(other_logs)
    script_utils.write_to_tensorboard(tf_writer, logging_dict, iteration)

    tabulate_dict = script_utils.get_tabulate_dict(tabulate_metrics,
                                                   logging_dict)
    tabulate_dict["lr"] = lr_schedule(opt_state[-1].count)
    table = logging_utils.make_table(tabulate_dict, iteration - start_iteration,
                                     args.tabulate_freq)
    print(table)


if __name__ == "__main__":
  script_utils.print_visible_devices()
  train_model()
